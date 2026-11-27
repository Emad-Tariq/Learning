"""Larger-gallery retraining with strict forward identity separation.

Keep original 36 training query episodes and all Dalvin settings. Background
identities are never queries. Completed earlier query roles may be references;
future-stage and validation identities are excluded from every fitting context.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,json,hashlib,warnings,time,platform,argparse,gc
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'analysis/domain_shift'))
from run_analysis import BoundedCache
import numpy as np
import pandas as pd
import cv2,joblib
from threadpoolctl import threadpool_limits
import dalvin_core as b
import data_protocol as p

SOURCE=ROOT/'Models/Abhay/Model7/local_runs/paired_2052'
SHIFT=ROOT/'analysis/domain_shift/results'
DATA=Path.home()/'Downloads/reid_student_data_main'
VAL_SEEDS=[22052,22053,22054]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def saved_episode(manifest,ep):
    z=manifest[manifest.episode==ep]
    lookup=pd.Index(b.IDS)
    g=lookup.get_indexer(z.loc[z.assignment=='gallery','id'])
    qrows=z[z.assignment!='gallery']
    q=lookup.get_indexer(qrows.id)
    assert (g>=0).all() and (q>=0).all()
    t=np.where(qrows.assignment.to_numpy()=='unknown_query',p.NEW,b.Y[q])
    return g,q,t

def backgrounds(roles,stage):
    return np.concatenate([roles['calibration']]+[roles[f'stage{j}'] for j in range(1,stage)])

def safety(g,q,t,roles,stage=None):
    assert len(np.unique(g))==len(g) and len(np.unique(q))==len(q)
    assert not set(g)&set(q)
    assert not set(b.Y[q[t==p.NEW]])&set(b.Y[g])
    assert set(b.Y[q[t!=p.NEW]])<=set(b.Y[g])
    if stage is not None:
        assert set(q)<=set(roles[f'stage{stage}'])
        forbidden=np.concatenate([roles['validation']]+[roles[f'stage{j}'] for j in range(stage+1,4)])
        assert not set(b.Y[np.r_[g,q]])&set(b.Y[forbidden])

def rr(pred,truth):
    return np.array([next((1/(j+1) for j,v in enumerate(row) if v==t),0.) for row,t in zip(pred,truth)])

def slice_rows(g,order,truth):
    rows=[]
    for i,t in zip(order,truth):
        known=t!=p.NEW;own=g[b.Y[g]==b.Y[i]] if known else np.array([],dtype=int)
        v=b.VIEW[i];side=v in (0,1,4,5,6,7)
        same=side and np.any((b.VIEW[own]>=0)&(b.FLANK[b.VIEW[own].clip(0)]==b.FLANK[v]))
        rows.append(dict(id=b.IDS[i],identity=b.Y[i],species=b.SP[i],truth=t,unknown=not known,
            side_query=side,missing_side=bool(known and side and not same),true_gallery_images=len(own)))
    return pd.DataFrame(rows)

def metrics(d):
    rows=[]
    masks={'overall':np.ones(len(d),bool),'known':~d.unknown,'unknown':d.unknown,
        'hyena':d.species=='hyena','leopard':d.species=='leopard',
        'known_missing_side':~d.unknown & d.side_query & d.missing_side,
        'known_small_gallery':~d.unknown & d.true_gallery_images.between(1,3)}
    masks.update({f'episode_{ep}':d.episode==ep for ep in VAL_SEEDS})
    for name,mask in masks.items():
        z=d[mask];diff=z.large_rr5-z.baseline_rr5
        rows.append(dict(slice=name,appearances=len(z),unique_photos=z.id.nunique(),
            baseline_map5=float(z.baseline_rr5.mean()),large_map5=float(z.large_rr5.mean()),
            delta=float(diff.mean()),improved=int((diff>0).sum()),worsened=int((diff<0).sum()),
            baseline_top1=float((z.baseline_rr5==1).mean()),large_top1=float((z.large_rr5==1).mean())))
    agg=d.assign(delta=d.large_rr5-d.baseline_rr5).groupby(['species','identity']).delta.agg(['sum','size'])
    rng=np.random.default_rng(12052);num=np.zeros(3000);den=np.zeros(3000)
    for _,part in agg.groupby(level=0):
        a=part.to_numpy();ix=rng.integers(len(a),size=(3000,len(a)))
        num+=a[ix,0].sum(1);den+=a[ix,1].sum(1)
    interval=np.quantile(num/den,[.025,.975]).tolist()
    r={v['slice']:v for v in rows}
    gate=(r['overall']['delta']>0 and interval[0]>0 and
          all(r[f'episode_{ep}']['delta']>0 for ep in VAL_SEEDS) and
          all(r[k]['delta']>=0 for k in ['known','unknown']) and
          r['unknown']['large_top1']>=r['unknown']['baseline_top1'])
    return {'slices':rows,'overall_delta_cluster_interval':interval,'promotion_gate_passed':bool(gate)}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'local_runs/large_2052')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    cfg=dict(seed=2052,stages=3,episodes_per_stage=12,candidate_trees=400,detector_trees=300,
        neighbours=150,candidates=15,workers=8,validation_seeds=VAL_SEEDS,
        background_rule='Dedicated background (legacy calibration role) plus completed earlier stage roles; no future-stage or validation identities',
        query_episodes='Exact preserved baseline manifests; same queries, truths, own-identity gallery photos and regimes',
        validation='Exact domain_shift expanded manifests for both frozen baseline and retrained model',
        gate='Positive overall and all episodes; positive cluster lower bound; no known/unknown MAP or unknown top1 decline',
        selection='One predeclared arm; no tuning or model selection on validation',
        source_model_hash=sha(SOURCE/'paired_models.joblib'),
        source_hashes={str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/'Models/Abhay/Model7/dalvin_core.py',
            ROOT/'Models/Abhay/Model7/data_protocol.py',ROOT/'analysis/domain_shift/run_analysis.py',ROOT/'analysis/domain_shift/matcher.py']})
    config=out/'config.json'
    if config.exists():assert json.loads(config.read_text())==cfg,'Configuration changed: use a new output directory'
    config.write_text(json.dumps(cfg,indent=2))
    cv2.setNumThreads(1);warnings.filterwarnings('ignore',message='Only one sample available.*')
    b.PARAMS.update(n_estimators=400,n_jobs=8,random_state=2052)
    train,_,_=p.load_data(DATA);b.NX=b.X
    roles=p.allocate_roles(train,3,2052)
    role_rows=pd.DataFrame([dict(id=b.IDS[i],identity=b.Y[i],species=b.SP[i],role=('background' if name=='calibration' else name))
        for name,ix in roles.items() for i in ix])
    assert role_rows.groupby('identity').role.nunique().eq(1).all()
    role_rows.to_csv(out/'identity_roles.csv',index=False)
    manifest=pd.read_csv(SOURCE/'episode_splits.csv')
    cache=BoundedCache(DATA,out,8,150)
    # Compatible cached measurements only; activation masks edges to the current
    # allowed role context, irrespective of where a measurement was cached.
    for f in [next(SHIFT.glob('sift_mutual_*.npz')),next(SOURCE.glob('sift_mutual_*.npz'))]:
        with np.load(f) as z:cache.done.update({tuple(k):tuple(v) for k,v in zip(z['pairs'],z['values'])})
    print('CACHED',len(cache.done),flush=True)
    checkpoint=out/'large_models.joblib'
    models=joblib.load(checkpoint) if checkpoint.exists() else []
    diagnostics=[];splits=[];contexts=[]
    for stage in range(1,4):
        own=roles[f'stage{stage}'];bg=backgrounds(roles,stage);context=np.r_[own,bg]
        forbidden=set(b.Y[roles['validation']])
        for j in range(stage+1,4):forbidden.update(b.Y[roles[f'stage{j}']])
        assert not forbidden&set(b.Y[context])
        contexts.extend(dict(stage=stage,id=b.IDS[i]) for i in context)
        resumed=len(models)>=stage
        cache.active_queries=set(own)
        if not resumed:cache.activate(context)
        frames=[]
        for j in range(12):
            ep=2052+1000*stage+j
            g,q,t=saved_episode(manifest,ep);large_g=np.r_[g,bg]
            safety(large_g,q,t,roles,stage)
            # Additional identities cannot add any true-query reference photo.
            assert not set(b.Y[q])&set(b.Y[bg])
            splits.extend(dict(stage=stage,episode=ep,id=b.IDS[i],assignment='gallery') for i in large_g)
            splits.extend(dict(stage=stage,episode=ep,id=b.IDS[i],assignment='unknown_query' if tt==p.NEW else 'known_query') for i,tt in zip(q,t))
            for sp in ['hyena','leopard']:
                gg=large_g[b.SP[large_g]==sp];qq=q[b.SP[q]==sp]
                diagnostics.append(dict(stage=stage,episode=ep,species=sp,original_gallery=int((b.SP[g]==sp).sum()),
                    large_gallery=len(gg),gallery_identities=len(np.unique(b.Y[gg])),queries=len(qq),
                    unknown_queries=int(((b.SP[q]==sp)&(t==p.NEW)).sum())))
            if not resumed:
                frame=p.frame(large_g,q,t)
                for model in models:frame=b.augment(frame,b.predict(model,frame))
                frames.append(frame)
            print('TRAIN',stage,j+1,'/12','resumed' if resumed else 'prepared',flush=True)
        if not resumed:
            print('FITTING STAGE',stage,flush=True)
            models.append(b.fit(frames))
            tmp=checkpoint.with_suffix('.tmp');joblib.dump(models,tmp);os.replace(tmp,checkpoint)
        pd.DataFrame(splits).to_csv(out/'training_episode_splits.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(out/'training_episode_diagnostics.csv',index=False)
        pd.DataFrame(contexts).to_csv(out/'training_contexts.csv',index=False)
        (out/'progress.json').write_text(json.dumps(dict(completed_stages=stage,elapsed_seconds=time.monotonic()-started),indent=2))
        del frames;gc.collect()
    baseline=joblib.load(SOURCE/'paired_models.joblib')['baseline']
    val_manifest=pd.read_csv(SHIFT/'expanded_manifests.csv')
    predictions=[];candidate_rows=[]
    for ep in VAL_SEEDS:
        g,q,t=saved_episode(val_manifest,ep)
        # Expanded manifests use assignment=query and a separate truth column.
        targets=val_manifest[(val_manifest.episode==ep)&(val_manifest.assignment=='query')].set_index('id').truth
        t=targets.loc[b.IDS[q]].to_numpy()
        safety(g,q,t,roles)
        assert set(q)<=set(roles['validation'])
        cache.active_queries=set(q);cache.activate(np.r_[g,q])
        frame=p.frame(g,q,t)
        d=slice_rows(g,frame['order'],frame['truth']);d['episode']=ep
        reference=pd.read_csv(SHIFT/f'expanded_{ep}.csv').set_index('id')
        for name,stack in [('baseline',baseline),('large',models)]:
            s,r=p.through_stack(frame,stack);pred=b.rank(s,r)
            d[name+'_predictions']=[' '.join(row) for row in pred]
            d[name+'_rr5']=rr(pred,s['truth'])
            d[name+'_pnew']=r['pb'];d[name+'_detector']=r['pn']
            if name=='baseline':
                assert d[name+'_predictions'].tolist()==reference.loc[d.id,'predictions'].tolist()
                np.testing.assert_allclose(d[name+'_rr5'],reference.loc[d.id,'rr5'],atol=1e-12)
            # Save frozen outputs for independently reconstructing every rank.
            c=s['c'][['q','ident']].copy();c['episode']=ep;c['method']=name
            c['id']=np.repeat(b.IDS[s['order']],b.K)
            c['pc']=r['pc'].ravel();c['raw']=r['raw'].ravel()
            candidate_rows.append(c)
        predictions.append(d)
        print('VALIDATION',ep,'baseline',d.baseline_rr5.mean(),'large',d.large_rr5.mean(),flush=True)
    d=pd.concat(predictions,ignore_index=True)
    d.to_csv(out/'validation_predictions.csv',index=False)
    pd.concat(candidate_rows,ignore_index=True).to_csv(out/'validation_candidate_scores.csv',index=False)
    val_manifest.to_csv(out/'validation_episode_splits.csv',index=False)
    report=metrics(d)
    report.update(appearances=len(d),unique_photos=d.id.nunique(),identities=d.identity.nunique(),
        baseline_exact_parity=True,elapsed_seconds=time.monotonic()-started,
        limitations=['One previously inspected outer identity allocation; not an untouched test or leaderboard result.',
        'Episodes reuse query photos and identities; intervals condition on fixed fitted models.',
        'Background gallery labels are extra fitting exposure relative to the saved baseline.',
        'Gallery growth is progressive to prevent earlier stages seeing future-stage query identities.',
        'Query-pool composition is unchanged and remains smaller than real inference.'])
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    (out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,opencv=cv2.__version__),indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
