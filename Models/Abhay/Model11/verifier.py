"""Full-vector pair MLP and gallery-aware unknown calibration. No raw images."""
import copy
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, roc_auc_score


def pair_features(x, a, c):
    """Symmetric, full 2152-D descriptor interactions, without identity inputs."""
    return np.c_[np.abs(x[a]-x[c]),x[a]*x[c],np.sum(x[a]*x[c],axis=1)].astype(np.float32)


def sample_pairs(x,y,sp,indices,seed,max_pairs=24000):
    rng=np.random.default_rng(seed); pairs=[]; labels=[]
    for species in ('hyena','leopard'):
        ix=indices[sp[indices]==species]; sim=x[ix]@x[ix].T
        groups=np.zeros(len(ix),int); offset=0
        for name in np.unique(y[ix]):
            loc=np.flatnonzero(y[ix]==name)
            _,gg=connected_components(csr_matrix(sim[np.ix_(loc,loc)]>=.97),directed=False)
            groups[loc]=gg+offset; offset+=gg.max()+1
        # Cap anchor appearances per identity so large identities do not dominate.
        anchors=[]
        for name in np.unique(y[ix]):
            loc=np.flatnonzero(y[ix]==name)
            anchors.extend(rng.choice(loc,min(24,len(loc)),replace=False))
        for j in anchors:
            pos=np.flatnonzero((y[ix]==y[ix[j]])&(groups!=groups[j]))
            neg=np.flatnonzero(y[ix]!=y[ix[j]])
            if not len(pos) or not len(neg):continue
            hard=neg[np.argsort(-sim[j,neg])[:30]]
            for _ in range(2):
                for candidate,label in ((rng.choice(pos),1),(rng.choice(hard),0),(rng.choice(neg),0)):
                    pairs.append((ix[j],ix[candidate])); labels.append(label)
    if not pairs:raise ValueError('No usable training pairs')
    pairs=np.asarray(pairs); labels=np.asarray(labels)
    # Deduplicate unordered pairs before sampling to avoid duplicate weights.
    pairs=np.sort(pairs,axis=1); _,keep=np.unique(pairs,axis=0,return_index=True)
    keep=rng.permutation(keep)[:max_pairs]
    if len(np.unique(labels[keep]))!=2:raise ValueError('Pair classes missing')
    return pairs[keep],labels[keep]


def fit_verifier(x,y,sp,indices,seed=2052,epochs=40,max_pairs=24000):
    rng=np.random.default_rng(seed); fit=[]; stop=[]
    for species in ('hyena','leopard'):
        names=np.unique(y[indices[sp[indices]==species]]); rng.shuffle(names)
        n=max(2,round(.2*len(names)))
        stop.extend(indices[(sp[indices]==species)&np.isin(y[indices],names[:n])])
        fit.extend(indices[(sp[indices]==species)&np.isin(y[indices],names[n:])])
    fit=np.asarray(fit); stop=np.asarray(stop)
    assert not set(y[fit]) & set(y[stop])
    fp,fy=sample_pairs(x,y,sp,fit,seed,max_pairs)
    ep,ey=sample_pairs(x,y,sp,stop,seed+1,min(8000,max_pairs))
    scaler=StandardScaler().fit(pair_features(x,*fp.T))
    transform=lambda pairs:np.clip(scaler.transform(pair_features(x,*pairs.T)),-6,6).astype(np.float32)
    xx,ee=transform(fp),transform(ep)
    net=MLPClassifier(hidden_layer_sizes=(32,16),alpha=1.,batch_size=256,
        learning_rate_init=.0003,random_state=seed,early_stopping=False)
    best=None; best_loss=np.inf; stale=0; history=[]
    for epoch in range(1,epochs+1):
        order=rng.permutation(len(fy)); net.partial_fit(xx[order],fy[order],classes=[0,1])
        pr=net.predict_proba(ee)[:,1]; loss=log_loss(ey,pr,labels=[0,1])
        history.append(dict(epoch=epoch,stop_logloss=float(loss),stop_auc=float(roc_auc_score(ey,pr))))
        print('MLP epoch',epoch,'identity-stop loss',round(loss,5),flush=True)
        if loss<best_loss-1e-4:best_loss=loss; best=copy.deepcopy(net); stale=0
        else:stale+=1
        if stale>=6:break
    parameters=sum(v.size for v in best.coefs_+best.intercepts_)
    return dict(network=best,scaler=scaler,history=history,parameters=parameters,
                fit_indices=fit,stop_indices=stop,fit_pairs=len(fy),stop_pairs=len(ey))


def pair_scores(model,x,a,c,batch=512):
    result=[]
    for start in range(0,len(a),batch):
        f=pair_features(x,a[start:start+batch],c[start:start+batch])
        z=np.clip(model['scaler'].transform(f),-6,6)
        result.append(model['network'].predict_proba(z)[:,1])
    return np.concatenate(result) if result else np.array([])


def query_features(model,x,y,gallery,s,r):
    """No query truth is used. Score every gallery photo of all 15 candidates."""
    n=len(s['q']); k=15; ids=s['c'].ident.to_numpy().reshape(n,k)
    blocks={name:gallery[y[gallery]==name] for name in np.unique(ids)}
    aa=[]; cc=[]; sizes=[]
    for i,query in enumerate(s['order']):
        for name in ids[i]:
            g=blocks[name]; aa.extend([query]*len(g)); cc.extend(g); sizes.append(len(g))
    pp=pair_scores(model,x,np.asarray(aa),np.asarray(cc)); statistics=[]; offset=0
    for size in sizes:
        v=np.sort(pp[offset:offset+size])[::-1]; offset+=size
        statistics.append([v[0],v[:2].mean(),v[:3].mean(),v.mean(),(v>.5).mean()])
    statistics=np.asarray(statistics).reshape(n,k,5)
    count=np.asarray(sizes).reshape(n,k); top=r['pc'].argmax(1); ix=np.arange(n)
    get=lambda key:s['c'][key].to_numpy().reshape(n,k)[ix,top]
    sorted_pc=np.sort(r['pc'],axis=1)
    # The control and MLP arm have identical baseline context and calibration model.
    context=np.c_[r['pn'],r['pb'],np.clip(1-r['raw'].sum(1),0,1),sorted_pc[:,-1],sorted_pc[:,-2],
        -(r['pc']*np.log(np.maximum(r['pc'],1e-12))).sum(1),np.log1p(count[ix,top]),
        get('mxr'),get('mxl'),np.log1p(get('sx')),np.log1p(get('n_flank')),
        np.log1p(get('n_view')),s['q'].sp.to_numpy(),s['q'].view.to_numpy()<0]
    sorted_best=np.sort(statistics[:,:,0],axis=1)
    richest=statistics[:,:,0].argmax(1)
    extra=np.c_[statistics[ix,top],statistics.max(1),
        (statistics*r['pc'][:,:,None]).sum(1),sorted_best[:,-1]-sorted_best[:,-2],
        np.log1p(count[ix,richest]),np.log1p(count.sum(1))]
    return np.nan_to_num(context),np.nan_to_num(np.c_[context,extra])


def calibrator(features,unknown):
    # Query probabilities are learned on separate calibration identities;
    # pair scores themselves are never treated as calibrated P(known).
    return make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1000)).fit(features,unknown)


def rank(candidates,pc,pnew):
    labels=np.c_[candidates,np.full(len(pc),'new_individual',dtype=object)]
    scores=np.c_[(1-pnew[:,None])*pc,pnew]
    return np.take_along_axis(labels,np.argsort(-scores,axis=1)[:,:5],axis=1)


def rr5(pred,truth):
    return np.array([next((1/(i+1) for i,label in enumerate(row) if label==target),0.) for row,target in zip(pred,truth)])
