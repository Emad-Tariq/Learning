"""Reproducible development and fresh-identity comparison against Model12.

Only pair measurements, never trained weights, are reused across fresh splits.
The final-stage learning inputs are produced by the earlier identity-disjoint
stages. All candidates and comparators share episode galleries and query pools.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import sys, json, hashlib, time, argparse, warnings, gc
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'analysis/domain_shift'))
from run_analysis import BoundedCache
import numpy as np
import pandas as pd
import cv2, joblib
from threadpoolctl import threadpool_limits
import dalvin_core as b
import data_protocol as p

DATA = Path.home()/'Downloads/reid_student_data_main'
OLD = ROOT/'Models/Abhay/Model12/local_runs/large_2052'
ORIGINAL = ROOT/'Models/Abhay/Model7/local_runs/paired_2052'
SHIFT = ROOT/'analysis/domain_shift/results'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(obj, path):
    path=Path(path); temp=path.with_suffix('.tmp')
    joblib.dump(obj,temp,compress=3); os.replace(temp,path)

def rr(pred, truth):
    return np.array([next((1/(j+1) for j,v in enumerate(row) if v==t),0.) for row,t in zip(pred,truth)])

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def configure(out, seed, workers):
    out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    warnings.filterwarnings('ignore',message='Only one sample available.*')
    b.PARAMS.update(n_estimators=400,n_jobs=workers,random_state=seed)
    train,test,all_df=p.load_data(DATA); b.NX=b.X
    roles=p.allocate_roles(train,3,seed)
    rows=pd.DataFrame([dict(id=b.IDS[i],identity=b.Y[i],species=b.SP[i],role=name)
                       for name,ix in roles.items() for i in ix])
    assert rows.groupby('identity').role.nunique().eq(1).all()
    rows.to_csv(out/'identity_roles.csv',index=False)
    cache=BoundedCache(DATA,out,workers,150)
    for base in [OLD, ORIGINAL, SHIFT]:
        files=list(base.glob('sift_mutual_*.npz'))
        if files:
            with np.load(files[0]) as z:
                cache.done.update({tuple(k):tuple(v) for k,v in zip(z['pairs'],z['values'])})
    print('CONFIGURED',seed,'cached pairs',len(cache.done),flush=True)
    return train,test,roles,cache

def from_manifest(manifest, ep):
    z=manifest[manifest.episode==ep]; lookup=pd.Index(b.IDS)
    g=lookup.get_indexer(z.loc[z.assignment=='gallery','id'])
    qrows=z[z.assignment!='gallery']; q=lookup.get_indexer(qrows.id)
    assert (g>=0).all() and (q>=0).all()
    if 'truth' in qrows and qrows.truth.notna().all(): t=qrows.truth.to_numpy()
    else: t=np.where(qrows.assignment.to_numpy()=='unknown_query',p.NEW,b.Y[q])
    return g,q,t

def validate_episode(g,q,t,roles,stage=None):
    assert len(np.unique(g))==len(g) and len(np.unique(q))==len(q)
    assert not set(g)&set(q)
    assert not set(b.Y[q[t==p.NEW]])&set(b.Y[g])
    assert set(b.Y[q[t!=p.NEW]])<=set(b.Y[g])
    if stage:
        assert set(q)<=set(roles[f'stage{stage}'])
        forbidden=np.concatenate([roles['validation']]+[roles[f'stage{j}'] for j in range(stage+1,4)])
        assert not set(b.Y[np.r_[g,q]])&set(b.Y[forbidden])
    else: assert set(q)<=set(roles['validation'])

def manifest_rows(ep,stage,g,q,t):
    return ([dict(episode=ep,stage=stage,id=b.IDS[i],assignment='gallery',truth='') for i in g]+
            [dict(episode=ep,stage=stage,id=b.IDS[i],assignment='query',truth=tt) for i,tt in zip(q,t)])

def prepare_development(out, workers):
    _,_,roles,cache=configure(out,2052,workers)
    models=joblib.load(OLD/'large_models.joblib')
    source=pd.read_csv(OLD/'training_episode_splits.csv')
    context=np.concatenate([roles[k] for k in ['calibration','stage1','stage2','stage3']])
    assert not set(b.Y[context])&set(b.Y[roles['validation']])
    files=[out/f'train_{ep}.joblib' for ep in range(5052,5064)]
    if not all(f.exists() for f in files):
        cache.active_queries=set(roles['stage3']); cache.activate(context)
    for ep,file in zip(range(5052,5064),files):
        if file.exists():continue
        g,q,t=from_manifest(source,ep);validate_episode(g,q,t,roles,3)
        s=p.frame(g,q,t)
        for model in models[:2]:s=b.augment(s,b.predict(model,s))
        dump(s,file); print('PREPARED training',ep,flush=True)
    vm=pd.read_csv(SHIFT/'expanded_manifests.csv')
    for ep in [22052,22053,22054]:
        file=out/f'eval_{ep}.joblib'
        if file.exists():continue
        g,q,t=from_manifest(vm,ep);validate_episode(g,q,t,roles)
        cache.active_queries=set(q);cache.activate(np.r_[g,q])
        s=p.frame(g,q,t)
        for model in models[:2]:s=b.augment(s,b.predict(model,s))
        r=b.predict(models[2],s);pred=b.rank(s,r)
        ref=pd.read_csv(OLD/'validation_predictions.csv')
        ref=ref[ref.episode==ep].set_index('id')
        assert [' '.join(x) for x in pred]==ref.loc[b.IDS[s['order']],'large_predictions'].tolist()
        dump(dict(frame=s,result=r,gallery=g,episode=ep),file)
        print('BASELINE EXACT',ep,float(rr(pred,s['truth']).mean()),flush=True)
    write_json(out/'preparation.json',dict(seed=2052,baseline_sha256=sha(OLD/'large_models.joblib'),baseline_parity=True,
        interpretation='Previously inspected development identities; never claim an untouched holdout.'))

def prepare_fresh(out, seed, workers):
    train,_,roles,cache=configure(out,seed,workers)
    model_file=out/'baseline_models.joblib'
    models=joblib.load(model_file) if model_file.exists() else []
    all_splits=[]; contexts=[]
    for stage in range(1,4):
        own=roles[f'stage{stage}']
        bg=np.concatenate([roles['calibration']]+[roles[f'stage{j}'] for j in range(1,stage)])
        context=np.r_[own,bg]
        contexts.extend(dict(stage=stage,id=b.IDS[i]) for i in context)
        epfile=out/f'stage{stage}_episodes.joblib'
        if epfile.exists():episodes=joblib.load(epfile)
        else:
            # Split encounter components with the original role-local graph.
            cache.active_queries=None; cache.activate(own)
            episodes=[]
            for j in range(12):
                ep=seed+1000*stage+j
                g,q,t=p.split_episode(own,ep,link=b.REGIMES[j%4])
                g=np.r_[g,bg];validate_episode(g,q,t,roles,stage)
                episodes.append((ep,g,q,t))
            dump(episodes,epfile)
        for ep,g,q,t in episodes:all_splits.extend(manifest_rows(ep,stage,g,q,t))
        files=[out/f'train_{ep}.joblib' for ep,_,_,_ in episodes]
        need_frames=len(models)<stage or stage==3
        if need_frames and not all(f.exists() for f in files):
            cache.active_queries=set(own);cache.activate(context)
        frames=[]
        if need_frames:
            for (ep,g,q,t),file in zip(episodes,files):
                if file.exists():s=joblib.load(file)
                else:
                    s=p.frame(g,q,t)
                    for model in models[:stage-1]:s=b.augment(s,b.predict(model,s))
                    dump(s,file)
                frames.append(s);print('FRESH stage',stage,'episode',ep,flush=True)
        if len(models)<stage:
            models.append(b.fit(frames));dump(models,model_file)
            print('FITTED baseline stage',stage,flush=True)
        pd.DataFrame(all_splits).to_csv(out/'training_episode_splits.csv',index=False)
        pd.DataFrame(contexts).to_csv(out/'training_contexts.csv',index=False)
        del frames;gc.collect()
    write_json(out/'training_complete.json',dict(seed=seed,baseline_sha256=sha(model_file),
        stages=3,episodes_per_stage=12,source_sha256=sha(__file__)))
    # Evaluation is deliberately a separate command after candidate lock.

def prepare_evaluation(out,seed,workers):
    lock=HERE/'recipe_lock.json'
    if not lock.exists():raise RuntimeError('Freeze recipe before fresh evaluation')
    train,_,roles,cache=configure(out,seed,workers)
    models=joblib.load(out/'baseline_models.joblib')
    assert len(models)==3
    epfile=out/'evaluation_episodes.joblib'
    if epfile.exists():episodes=joblib.load(epfile)
    else:
        cache.active_queries=None;cache.activate(roles['validation'])
        extra=np.setdiff1d(np.arange(len(train)),roles['validation'])
        episodes=[]
        for j in range(3):
            ep=seed+20000+j;g,q,t=p.split_episode(roles['validation'],ep,link=(.95,111))
            g=np.r_[g,extra];validate_episode(g,q,t,roles)
            episodes.append((ep,g,q,t))
        dump(episodes,epfile)
    rows=[]
    for ep,g,q,t in episodes:
        rows.extend(manifest_rows(ep,'validation',g,q,t))
        file=out/f'eval_{ep}.joblib'
        if file.exists():continue
        cache.active_queries=set(q);cache.activate(np.r_[g,q])
        s=p.frame(g,q,t)
        for model in models[:2]:s=b.augment(s,b.predict(model,s))
        r=b.predict(models[2],s)
        dump(dict(frame=s,result=r,gallery=g,episode=ep),file)
        print('PREPARED fresh evaluation',ep,flush=True)
    pd.DataFrame(rows).to_csv(out/'validation_episode_splits.csv',index=False)
    write_json(out/'evaluation_preparation.json',dict(recipe_lock_sha256=sha(lock),seed=seed))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['development','train','evaluation'])
    ap.add_argument('--seed',type=int,default=27613);ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    out=HERE/'local_runs'/('development_2052' if args.mode=='development' else f'fresh_{args.seed}')
    if args.mode=='development':prepare_development(out,args.workers)
    elif args.mode=='train':prepare_fresh(out,args.seed,args.workers)
    else:prepare_evaluation(out,args.seed,args.workers)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
