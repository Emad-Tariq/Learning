"""Gallery-trained metric and graph label propagation for open-set retrieval."""
import numpy as np
import pandas as pd
from scipy import linalg, sparse
from sklearn.decomposition import PCA

NEW = 'new_individual'


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-10)


def fit_coordinates(data, train):
    result = np.empty((len(data['x']), 384), 'f')
    for sp in ('hyena', 'leopard'):
        rows = np.flatnonzero(data['species'] == sp)
        ref = train[data['species'][train] == sp]
        p = PCA(384, svd_solver='randomized', random_state=71).fit(data['x'][ref])
        result[rows] = p.transform(data['x'][rows])
    return result


def neighbors(s, k):
    k = min(k, s.shape[1])
    top = np.argpartition(s, -k, axis=1)[:, -k:]
    return np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, 1), axis=1), 1)


def reduce_gallery(mat, inv, n):
    first, second = [], []
    for j in range(n):
        v = mat[:, inv == j]
        if v.shape[1] > 1:
            best = np.partition(v, -2, axis=1)[:, -2:]
            first.append(best.max(1))
            second.append(best.mean(1))
        else:
            first.append(v[:, 0])
            second.append(v[:, 0])
    return np.stack(first, 1), np.stack(second, 1)


def discriminant(z, inv, counts):
    # Gallery labels determine within-animal variation. Two regularizers
    # separate stable identity directions from directions with scant support.
    ng = len(inv)
    z = z - z[:ng].mean(0)
    means = np.stack([z[:ng][inv == i].mean(0) for i in range(len(counts))])
    residual = z[:ng] - means[inv]
    cov = residual.T @ residual / ng
    e, u = linalg.eigh(cov)
    scaled = np.maximum(e, 0) / max(e.mean(), 1e-8)
    basis, centers = z @ u, means @ u
    results = []
    for shrink in (.27, 1.3):
        w = 1 / np.sqrt(shrink + scaled)
        weighted = (centers * w) * np.sqrt(counts / ng)[:, None]
        _, _, directions = linalg.svd(weighted, full_matrices=False)
        results.append(unit((basis * w) @ directions[:min(len(counts)-1, 200)].T))
    # Ridge classification uses signed evidence against confusable animals.
    cov_total = z[:ng].T @ z[:ng]
    ridge = linalg.solve(cov_total + np.eye(z.shape[1], dtype='f') *
                         (np.trace(cov_total) / z.shape[1] * .6), means.T, assume_a='pos')
    scores = (z @ ridge) * np.sqrt(counts)[None, :]
    return *results, scores.astype('f')


def evidence(data, coords, g, q):
    rows = np.r_[g, q]
    ng, nq = len(g), len(q)
    x = data['x'][rows]
    catalog, inv, counts = np.unique(data['identity'][g], return_inverse=True, return_counts=True)
    n = len(counts)
    f, h, ridge = discriminant(coords[rows], inv, counts)
    raw, fisher = x[ng:] @ x.T, f[ng:] @ f.T
    other = h[ng:] @ h[:ng].T
    sift = data['inliers'][q][:, rows].toarray()
    good = data['good'][q][:, g].toarray()
    local = {}
    for name, metric in [('raw',raw[:,:ng]), ('fisher',fisher[:,:ng]), ('shrink',other),
                         ('geometry',sift[:,:ng]), ('good',good)]:
        local[name], local[name+'_two'] = reduce_gallery(metric, inv, n)
    local['ridge'] = ridge[ng:]
    local['centroid'] = f[ng:] @ unit(np.stack([f[:ng][inv==i].mean(0) for i in range(n)])).T
    # Reciprocal neighborhoods penalize popular gallery images. Sparse
    # neighborhood overlap provides a second notion of similarity.
    # Block computations avoid building an all-gallery square matrix.
    ngbr, similarities = [], []
    for start in range(0,len(rows),192):
        stop = min(start+192,len(rows))
        s = .65*(f[start:stop] @ f.T) + .35*(x[start:stop] @ x.T)
        s[np.arange(stop-start), np.arange(start,stop)] = -2
        idx = neighbors(s, 24)
        ngbr.append(idx)
        similarities.append(np.take_along_axis(s, idx, 1))
    idx, sims = np.vstack(ngbr), np.vstack(similarities)
    rr = np.repeat(np.arange(len(rows)), idx.shape[1])
    weights = np.exp((sims-sims[:,:1])*12)
    adj = sparse.csr_matrix((weights.ravel(), (rr,idx.ravel())), shape=(len(rows),len(rows)))
    norm = np.sqrt(np.maximum(adj.multiply(adj).sum(1).A1,1e-8))
    norm_adj = adj.multiply((1/norm)[:,None]).tocsr()
    overlap = (norm_adj[ng:] @ norm_adj.T).toarray().astype('f')
    joint = .65*fisher + .35*raw
    corrected = joint - .48*sims[:,-1][None,:]
    local['reciprocal'], _ = reduce_gallery(corrected[:,:ng],inv,n)
    context_match = joint + .26*overlap
    local['context'], _ = reduce_gallery(context_match[:,:ng],inv,n)
    local['overlap'], _ = reduce_gallery(overlap[:,:ng],inv,n)
    # Propagate votes through a query's neighbors with the gallery as a
    # fixed boundary. Geometric matches can bridge missing embedding links.
    affinity = corrected + .16*overlap + .30*np.tanh(np.maximum(sift-4,0)/18)
    affinity[np.arange(nq),ng+np.arange(nq)] = -3
    nn = neighbors(affinity,20)
    v = np.take_along_axis(affinity,nn,1)
    gallery_labels = np.eye(n, dtype='f')[inv]
    logits = 19*(v-v[:,:1])
    w = np.exp(logits)
    w /= w.sum(1,keepdims=True)
    trans = sparse.csr_matrix((w.ravel(),(np.repeat(np.arange(nq),nn.shape[1]),nn.ravel())),
                              shape=(nq,len(rows)))
    anchor = trans[:,:ng] @ gallery_labels
    walk = anchor.copy()
    for _ in range(5):
        walk = anchor + .90*(trans[:,ng:] @ walk)
    local['direct_mass'], local['walk'] = anchor, walk
    qq = affinity[:,ng:]
    nb = neighbors(qq,5)
    strength = np.take_along_axis(qq,nb,1)
    for key in ('fisher','ridge','geometry','centroid'):
        orig = local[key]
        trust = np.exp(np.minimum((strength - .35)*9,0))
        local['neighbor_'+key] = (orig[nb] * trust[:,:,None]).max(1)
    views = data['flank'][rows]
    same = np.where((views[ng:,None] == views[None,:ng]) & (views[ng:,None] >= 0),
                     fisher[:,:ng], -1)
    local['same_flank'], _ = reduce_gallery(same,inv,n)
    context = dict(flank=views[ng:], query_strength=strength[:,0], query_third=strength[:,2],
                   geometry_neighbors=(sift[:,ng:]>=7).sum(1), graph_mass=walk.sum(1),
                   gallery_size=np.full(nq,np.log(ng)), neighbors_in_gallery=(nn<ng).mean(1))
    for key,s in [('raw',raw),('fisher',fisher),('geometry',sift)]:
        qq = s[:,ng:].copy()
        qq[np.arange(nq),np.arange(nq)] = -2
        context[key+'_query'] = qq.max(1)
        context[key+'_relative'] = qq.max(1) - local[key].max(1)
    selected = np.zeros((nq,n), bool)
    for key,k in [('fisher',16),('ridge',10),('raw',8),('context',10),('geometry',6),
                  ('walk',10),('centroid',8),('neighbor_fisher',6)]:
        idx = neighbors(local[key],k)
        r = np.arange(nq)[:,None]
        selected[r,idx] |= local[key][r,idx] > 0 if key in ('walk','geometry') else True
    ri, ci = np.nonzero(selected)
    feat = dict(gallery_count=counts[ci])
    for key, mat in local.items():
        top = np.take_along_axis(mat,neighbors(mat,2),1)
        feat[key] = mat[ri,ci]
        feat[key+'_gap'] = mat[ri,ci]-top[ri,0]
        if key in ('raw','fisher','geometry','context','ridge','walk','centroid','neighbor_geometry'):
            context['best_'+key] = top[:,0]
            context['margin_'+key] = top[:,0]-top[:,1]
    common = pd.DataFrame(context)
    known = pd.concat([common.iloc[ri].reset_index(drop=True),pd.DataFrame(feat)],axis=1)
    known['name'], known['pos'], known['is_new'] = catalog[ci], ri, False
    novel = common.assign(name=NEW,pos=np.arange(nq),is_new=True)
    return pd.concat([known,novel],ignore_index=True)


def make_frame(data, coords, g, q, truth=None):
    parts = []
    for sp in ('hyena','leopard'):
        take = np.flatnonzero(data['species'][q]==sp)
        df = evidence(data,coords,g[data['species'][g]==sp],q[take])
        df['row'] = take[df.pop('pos').to_numpy()]
        df['species_hyena'] = sp=='hyena'
        parts.append(df)
    frame = pd.concat(parts,ignore_index=True).sort_values('row',kind='stable').reset_index(drop=True)
    if truth is not None:
        frame['hit'] = frame.name.to_numpy() == truth[frame.row.to_numpy()]
    for col in frame.select_dtypes('float').columns:
        frame[col] = frame[col].astype('f')
    return frame
