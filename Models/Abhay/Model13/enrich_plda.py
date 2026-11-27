"""Append gallery-fitted probabilistic embedding evidence to frozen frames.

All fitting inputs are episode gallery vectors and their supplied labels.
The query targets are copied as bookkeeping, never passed to metric fitting.
Base frames/checkpoints stay unchanged for the paired comparison.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
import experiment as e
import plda_features as pl
b,p=e.b,e.p

PARAMS=dict(components=128,within_shrinkage=.3)

def enrich(frame, gallery):
    out=dict(frame,q=frame['q'].copy(),c=frame['c'].copy())
    order=frame['order'];diagnostics={}
    assert not set(order)&set(gallery)
    for species in ['hyena','leopard']:
        loc=np.flatnonzero(b.SP[order]==species)
        gi=gallery[b.SP[gallery]==species]
        ci=np.flatnonzero(np.isin(frame['c'].q.to_numpy(),loc))
        expected=np.repeat(loc,b.K)
        assert np.array_equal(frame['c'].q.to_numpy()[ci],expected)
        candidate_labels=frame['c'].ident.to_numpy()[ci].reshape(len(loc),b.K)
        mapper=pl.fit(b.X[gi],b.Y[gi],**PARAMS)
        cf,qf=pl.candidate_features(mapper,b.X[order[loc]],candidate_labels)
        assert len(cf)==len(ci) and len(qf)==len(loc)
        for dest,idx,features in [('q',loc,qf),('c',ci,cf)]:
            for col in features:
                assert col.startswith('pl_') and col not in frame[dest]
                if col not in out[dest]:out[dest][col]=np.nan
                out[dest].loc[idx,col]=features[col].to_numpy()
        diagnostics[species]=dict(gallery_photos=len(gi),gallery_identities=len(np.unique(b.Y[gi])),queries=len(loc))
    for key in ['q','c']:
        for col in frame[key]:pd.testing.assert_series_equal(frame[key][col],out[key][col])
        new=[c for c in out[key] if c not in frame[key]]
        assert np.isfinite(out[key][new].to_numpy()).all()
    return out,diagnostics

def prepare(out,seed,development=False,which='all'):
    p.load_data(e.DATA);b.NX=b.X
    manifest=pd.read_csv(e.OLD/'training_episode_splits.csv' if development else out/'training_episode_splits.csv')
    base_seed=2052 if development else seed
    diagnostics={}
    if which in ['all','train']:
        for ep in range(base_seed+3000,base_seed+3012):
            path=out/f'plda_train_{ep}.joblib'
            if path.exists():continue
            g,_,_=e.from_manifest(manifest,ep)
            frame=joblib.load(out/f'train_{ep}.joblib')
            enriched,diag=enrich(frame,g);e.dump(enriched,path);diagnostics[str(ep)]=diag
            print('PLDA training',ep,flush=True)
    if which in ['all','evaluation']:
        for path in sorted(out.glob('eval_*.joblib')):
            dest=path.with_name('plda_'+path.name)
            if dest.exists():continue
            bundle=joblib.load(path)
            frame,diag=enrich(bundle['frame'],bundle['gallery'])
            e.dump(dict(bundle,frame=frame),dest);diagnostics[str(bundle['episode'])]=diag
            print('PLDA evaluation features',bundle['episode'],flush=True)
    e.write_json(out/f'plda_preparation_{which}.json',dict(seed=base_seed,parameters=PARAMS,
        source_sha256=e.sha(__file__),metric_source_sha256=e.sha(Path(pl.__file__)),episodes=diagnostics))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--development',action='store_true');ap.add_argument('--seed',type=int,default=27613)
    ap.add_argument('--which',choices=['all','train','evaluation'],default='all');args=ap.parse_args()
    out=e.HERE/'local_runs'/('development_2052' if args.development else f'fresh_{args.seed}')
    with threadpool_limits(limits=2):prepare(out,args.seed,args.development,args.which)
