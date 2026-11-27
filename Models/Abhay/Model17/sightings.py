"""Fit same-animal links between unlabelled sightings, then pool their posteriors."""
import sys
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from experiment import (OUT, CACHE, inputs, prepare, benchmarks, episodes, metrics, score,
                        fit_coordinates, cached_frame, reconstruction)
from graph_model import discriminant, neighbors, NEW


def links(data,z,g,q,training=False):
    parts = []
    for sp in inputs.SPECIES:
        qi = np.flatnonzero(data['species'][q]==sp)
        peer = q[qi]
        ref = g[data['species'][g]==sp]
        ix = np.r_[ref,peer]
        _,lab,count = np.unique(data['identity'][ref],return_counts=True,return_inverse=True)
        f,_,_ = discriminant(z[ix],lab,count)
        fq = f[len(ref):]
        raw = data['x'][peer] @ data['x'][peer].T
        fisher = fq @ fq.T
        inl = data['inliers'][peer][:,peer].toarray()
        good = data['good'][peer][:,peer].toarray()
        combined = .65*fisher + .35*raw
        for x in (combined,fisher,raw):
            np.fill_diagonal(x,-2)
        near = neighbors(combined,15)
        csr = sparse.csr_matrix((np.ones(near.size,'f'),(np.repeat(np.arange(len(peer)),near.shape[1]),
                              near.ravel())),shape=combined.shape)
        shared = (csr @ csr.T).toarray() / near.shape[1]
        a,b = [],[]
        for metric,k in [(combined,10),(raw,6),(inl,7)]:
            j = neighbors(metric,k)
            a.append(np.repeat(np.arange(len(peer)),j.shape[1]))
            b.append(j.ravel())
        a,b = np.unique(np.sort(np.column_stack((np.concatenate(a),np.concatenate(b))),axis=1),axis=0).T
        keep = (a!=b) & ((inl[a,b]>=4) | (combined[a,b] >= np.minimum(
            combined[np.arange(len(peer)),near[:,-1]][a],combined[np.arange(len(peer)),near[:,-1]][b])))
        a,b = a[keep],b[keep]
        vals = dict(raw=raw[a,b],fisher=fisher[a,b],inliers=inl[a,b],good=good[a,b],
                    overlap=shared[a,b],species_hyena=sp=='hyena',
                    same_view=data['view'][peer[a]]==data['view'][peer[b]],
                    same_flank=data['flank'][peer[a]]==data['flank'][peer[b]])
        for name,metric in [('raw',raw),('fisher',fisher),('inl',inl)]:
            top = np.take_along_axis(metric,neighbors(metric,6),1)
            for k in (0,1,5):
                vals[f'{name}_near{k}'] = metric[a,b]-np.maximum(top[a,k],top[b,k])
                vals[f'{name}_far{k}'] = metric[a,b]-np.minimum(top[a,k],top[b,k])
        frame = pd.DataFrame(vals).assign(a=qi[a],b=qi[b])
        if training:
            frame['target'] = data['identity'][peer[a]] == data['identity'][peer[b]]
        parts.append(frame)
    return pd.concat(parts,ignore_index=True)


def link_model(data,z,pool,simulate,prefix,count=18):
    path = (OUT/'sighting_verifier.joblib' if prefix=='prod' and count==18 else
            CACHE / (prefix+('_links.pkl' if count==18 else f'_links_{count}.pkl')))
    if path.exists():
        return joblib.load(path)
    frames = [links(data,z,g,q,True) for g,q,y in episodes(data,pool,simulate,count)]
    df = pd.concat(frames,ignore_index=True)
    cols = [c for c in df if c not in ('a','b','target')]
    model = lgb.LGBMClassifier(n_estimators=270,num_leaves=18,min_child_samples=100,
        learning_rate=.055,reg_lambda=10,n_jobs=4,verbosity=-1,random_state=715)
    model.fit(df[cols],df.target)
    joblib.dump((model,cols),path)
    return model,cols


def linked_ranks(data,z,g,q,frame,model,pairmodel):
    logits = score(model,frame)
    i = frame.row.to_numpy()
    cat,j = np.unique(frame.name,return_inverse=True)
    posterior = np.zeros((len(q),len(cat)),dtype='f')
    posterior[i,j] = np.exp(logits)
    posterior /= posterior.sum(1,keepdims=True)
    edges = links(data,z,g,q)
    learner,cols = pairmodel
    reliability = learner.predict_proba(edges[cols])[:,1]
    a,b = edges.a.to_numpy(),edges.b.to_numpy()
    # Suppress ambiguous edges. Repeated encounters share evidence, so
    # their influence saturates rather than increasing linearly with count.
    weights = np.maximum((reliability-.40)/.60,0)**2
    matrix = sparse.csr_matrix((np.tile(weights,2),(np.r_[a,b],np.r_[b,a])),
                               shape=(len(q),len(q)))
    total = matrix.sum(1).A1
    conf = matrix.max(axis=1).toarray().ravel()
    neighbors_p = (matrix @ posterior) / np.maximum(total[:,None],1e-8)
    alpha = .65*conf
    combined = (1-alpha[:,None])*posterior + alpha[:,None]*neighbors_p
    # A second sighting can introduce an identity missing from the shortlist.
    return cat[neighbors(combined,5)]


def main(production,count=24):
    data,dev,audit,simulate = prepare()
    pool = np.arange(len(data['train'])) if production else dev
    prefix = 'prod' if production else 'dev'
    z = fit_coordinates(data,pool)
    pairmodel = link_model(data,z,pool,simulate,prefix,min(18,count))
    path = OUT/'identity_rankers.joblib' if production and count==24 else CACHE/f'{prefix}_model_{count}.pkl'
    model = joblib.load(path)
    packets = [('test',0,pool,np.arange(len(pool),len(data['x'])),None)] if production else benchmarks(dev,audit,simulate)
    result = []
    for size,seed,g,q,y in packets:
        f = cached_frame('test' if production else f'hold_{seed}',data,z,g,q,y)
        pred = linked_ranks(data,z,g,q,f,model,pairmodel)
        if production:
            sample = pd.read_csv(inputs.DATA/'sample_submission.csv')
            order = data['index'].get_indexer(sample.id)-len(pool)
            assert len(order)==len(q) and np.array_equal(np.sort(order),np.arange(len(q)))
            lookup = data['train'].groupby('species').identity.agg(set).to_dict()
            for row,sp in zip(pred,data['species'][q]):
                assert len(set(row))==5 and set(row)<=(lookup[sp]|{NEW})
            sample['predictions'] = [' '.join(row) for row in pred[order]]
            dest = OUT/'submission_model17.csv'
            sample.to_csv(dest,index=False)
            assert pd.read_csv(dest).equals(sample)
            print(dest,flush=True)
        else:
            result.append(pd.DataFrame(dict(size=size,seed=seed,id=data['ids'][q],
                animal=data['identity'][q],truth=y,**{f'p{i+1}':pred[:,i] for i in range(5)})))
    if result:
        df = pd.concat(result,ignore_index=True)
        df.to_csv(OUT/f'linked_holdout_{count}.csv',index=False)
        report = {k:metrics(s[[f'p{i}' for i in range(1,6)]].to_numpy(),s.truth.to_numpy())
                  for k,s in df.groupby('size')}
        (OUT/f'linked_metrics_{count}.json').write_text(json.dumps(report,indent=2))
        print(report,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=3,user_api='blas'):
        main(sys.argv[1]=='production',int(sys.argv[2]) if len(sys.argv)>2 else 24)
