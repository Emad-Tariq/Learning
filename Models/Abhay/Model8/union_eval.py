"""Predeclared paired evaluation of baseline and baseline+raw union shortlists."""
import os
os.environ['OMP_NUM_THREADS']='8'
os.environ['OPENBLAS_NUM_THREADS']='4'
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tmp'/'abhay_deps'))
sys.path.append(str(HERE.parent/'Model7'))
import argparse, hashlib, json, time, shutil, warnings, platform
import importlib.metadata
import numpy as np
import pandas as pd
import cv2, joblib
from threadpoolctl import threadpool_limits
import dalvin_core as b
import data_protocol as protocol

METHODS=('baseline','union')
def select(method):
    b.SELECTOR=method; b.K=15 if method=='baseline' else 30

def paired_frames(gi,qi,truth):
    select('union')
    union=protocol.frame(gi,qi,truth)
    base=dict(union)
    base['q']=union['q'].copy()
    base['c']=union['c'].loc[union['c']['rank']<15].drop(columns='_valid').reset_index(drop=True)
    assert len(base['c'])==15*len(base['q'])
    assert union['c'].groupby('q')._valid.sum().between(15,30).all()
    return {'baseline':base,'union':union}

def metric(d):
    result={'query_appearances':len(d),'unique_queries':int(d.id.nunique()),'unique_identities':int(d.identity.nunique())}
    for m in METHODS:
        c=m+'_rr5'; known=d[~d.unknown]; unknown=d[d.unknown]
        result[m]={'map5':float(d[c].mean()) if len(d) else None,
                   'known_map5':float(known[c].mean()) if len(known) else None,
                   'unknown_map5':float(unknown[c].mean()) if len(unknown) else None,
                   'top1':float((d[c]==1).mean()) if len(d) else None,
                   'candidate_recall_known':float(known[m+'_candidate_present'].mean()) if len(known) else None}
    result['delta']=float((d.union_rr5-d.baseline_rr5).mean()) if len(d) else None
    result['improved']=int((d.union_rr5>d.baseline_rr5).sum()); result['worsened']=int((d.union_rr5<d.baseline_rr5).sum())
    return result

def slice_masks(d):
    known=~d.unknown
    return {'all':np.ones(len(d),bool),'known':known,'unknown':d.unknown,
            'missing_side':known&d.side_query&d.missing_side,
            'same_side_available':known&d.side_query&~d.missing_side,
            'small_gallery_1to3':known&(d.true_gallery_images<=3),
            'gallery_4plus':known&(d.true_gallery_images>=4),
            'missing_side_small_gallery':known&d.side_query&d.missing_side&(d.true_gallery_images<=3),
            'viewpoint_missing':d.viewpoint_code<0}

def bootstrap(d,seed=2052,repeats=3000):
    # Cluster repeated photos, appearances and episodes by true identity, stratify species.
    clusters=d.assign(diff=d.union_rr5-d.baseline_rr5).groupby(['species','identity']).agg(total=('diff','sum'),n=('diff','size'))
    if not len(clusters): return None
    rng=np.random.default_rng(seed); sums=np.zeros(repeats); counts=np.zeros(repeats)
    for _,part in clusters.groupby(level=0):
        a=part[['total','n']].to_numpy(); sampled=rng.integers(0,len(a),size=(repeats,len(a)))
        sums+=a[sampled,0].sum(1); counts+=a[sampled,1].sum(1)
    lo,hi=np.quantile(sums/counts,[.025,.975])
    return {'identity_cluster_bootstrap_95pct':[float(lo),float(hi)],'repeats':repeats,'clusters':len(clusters)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,default=Path.home()/'Downloads/reid_student_data_main')
    ap.add_argument('--output',type=Path,default=HERE/'local_runs/union_2052')
    ap.add_argument('--cache-source',type=Path,default=HERE.parent/'Model7/local_runs/paired_2052')
    ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    started=time.monotonic(); seed=2052; sims=12
    config={'seed':seed,'sims_per_stage':sims,'stages':3,'trees':400,'detector_trees':300,'neighbours':150,
            'workers':args.workers,'data':str(args.data),'evaluation_episodes':[22052,22053,22054],
            'shortlists':{'baseline':'top 15 by LDA + 0.05*tanh(SIFT_inliers/15)',
                          'union':'baseline 15 followed by distinct raw top15; 15 to30 valid candidates'},
            'small_gallery':'1 to 3 true-identity gallery images (known queries only)',
            'missing_side':'left/right query with zero true-identity gallery images on the same flank',
            'gate':'positive pooled and each episode delta; no known or unknown MAP5 decrease',
            'protocol':'same as paired_2052: 20% identities validation,20% reserved,3 disjoint20% training stages',
            'limitations':['EDA previously examined the whole labelled data; not a pristine holdout',
                          'Three episodes reuse validation identities and photos',
                          'Smaller galleries than final competition inference; not leaderboard reproduction']}
    cfgpath=out/'config.json'
    if cfgpath.exists(): assert json.loads(cfgpath.read_text())==config,'Refusing to mix run configurations'
    cfgpath.write_text(json.dumps(config,indent=2))
    cv2.setNumThreads(1)
    warnings.filterwarnings('ignore',message='Only one sample available.*',category=UserWarning)
    b.PARAMS.update(n_estimators=400,n_jobs=args.workers,random_state=seed)
    train,test,all_df=protocol.load_data(args.data); b.NX=b.X
    roles=protocol.allocate_roles(train,3,seed)
    manifest=pd.DataFrame([{'id':b.IDS[i],'species':b.SP[i],'identity':b.Y[i],'role':role} for role,ix in roles.items() for i in ix])
    assert manifest.groupby('identity').role.nunique().max()==1
    manifest.to_csv(out/'identity_roles.csv',index=False)
    # Copy the cache so this experiment cannot overwrite the earlier experiment's artifacts.
    source_files=list(args.cache_source.glob('sift_mutual_*.npz'))
    for source in source_files:
        dest=out/source.name
        if not dest.exists(): shutil.copy2(source,dest)
    cache=protocol.SiftCache(args.data,out,args.workers,150)
    paths=[HERE/'dalvin_core.py',HERE/'union_eval.py',HERE.parent/'Model7/data_protocol.py']
    provenance={'python':platform.python_version(),'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','lightgbm','opencv-python']}}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    checkpoint=out/'paired_models.joblib'
    models=joblib.load(checkpoint) if checkpoint.exists() else {m:[] for m in METHODS}
    assert len(models['baseline'])==len(models['union'])
    splits=[]
    def episode(ix,episode_seed,role,link):
        select('baseline')
        g,q,truth=protocol.split_episode(ix,episode_seed,link=link)
        lookup=dict(zip(q,truth))
        splits.extend({'episode':episode_seed,'role':role,'id':b.IDS[i],'assignment':'gallery'} for i in g)
        splits.extend({'episode':episode_seed,'role':role,'id':b.IDS[i],'assignment':'unknown_query' if lookup[i]==protocol.NEW else 'known_query'} for i in q)
        return g,q,truth
    def save_checkpoint():
        tmp=checkpoint.with_suffix('.tmp'); joblib.dump(models,tmp); os.replace(tmp,checkpoint)
    for stage in range(3):
        role=f'stage{stage+1}'
        print(f'Preparing {role}',flush=True)
        cache.activate(roles[role])
        if stage<len(models['baseline']):
            print(f'Resuming: {role} models already fitted',flush=True)
            for j in range(sims): episode(roles[role],seed+1000*(stage+1)+j,role,b.REGIMES[j%len(b.REGIMES)])
            continue
        training={m:[] for m in METHODS}
        for j in range(sims):
            g,q,truth=episode(roles[role],seed+1000*(stage+1)+j,role,b.REGIMES[j%len(b.REGIMES)])
            paired=paired_frames(g,q,truth)
            for method,s in paired.items():
                select(method)
                for model in models[method]: s=b.augment(s,b.predict(model,s))
                training[method].append(s)
            print(f'{role} simulation {j+1}/{sims}',flush=True)
        for method in METHODS:
            select(method); print(f'Fitting {method} {role}',flush=True)
            models[method].append(b.fit(training[method]))
        save_checkpoint(); del training
        (out/'progress.json').write_text(json.dumps({'completed_stages':stage+1,'elapsed_seconds':time.monotonic()-started},indent=2))
    print('Models frozen; preparing validation SIFT',flush=True)
    cache.activate(roles['validation'])
    results=[]
    for episode_seed in config['evaluation_episodes']:
        g,q,truth=episode(roles['validation'],episode_seed,'validation',(.95,111))
        paired=paired_frames(g,q,truth); order=paired['baseline']['order']
        d=pd.DataFrame({'episode':episode_seed,'id':b.IDS[order],'identity':b.Y[order],'species':b.SP[order],
                        'truth':paired['baseline']['truth'],'viewpoint_code':b.VIEW[order]})
        d['unknown']=d.truth==protocol.NEW
        d['side_query']=(d.viewpoint_code>=0)&np.isin(b.FLANK[d.viewpoint_code.clip(lower=0)], [0,1])
        d['true_gallery_images']=[int((b.Y[g]==b.Y[i]).sum()) for i in order]
        d['missing_side']=[bool(v>=0 and b.FLANK[v] in (0,1) and not np.any((b.Y[g]==b.Y[i])&(b.VIEW[g]>=0)&(b.FLANK[b.VIEW[g].clip(0)]==b.FLANK[v]))) for i,v in zip(order,b.VIEW[order])]
        for method,s in paired.items():
            select(method); c=s['c']; valid=b.valid_rows(s)
            present=c.loc[valid].groupby('q').ident.apply(set)
            d[method+'_candidate_present']=[target in present[k] for k,target in enumerate(d.truth)]
            d[method+'_candidate_count']=c.loc[valid].groupby('q').size().to_numpy()
            s,r=protocol.through_stack(s,models[method]); pred=b.rank(s,r)
            assert all(len(set(row))==5 for row in pred)
            d[method+'_rr5']=[next((1/(j+1) for j,label in enumerate(row) if label==target),0.) for row,target in zip(pred,d.truth)]
            d[method+'_predictions']=[' '.join(row) for row in pred]
            d[method+'_pnew']=r['pb']; d[method+'_detector']=r['pn']; d[method+'_rawsum']=r['raw'].sum(1)
        results.append(d)
        d.to_csv(out/f'validation_{episode_seed}.csv',index=False)
        print('Validation',episode_seed,json.dumps(metric(d)),flush=True)
    result=pd.concat(results,ignore_index=True)
    result.to_csv(out/'validation_predictions.csv',index=False)
    pd.DataFrame(splits).to_csv(out/'episode_splits.csv',index=False)
    report=metric(result)
    report['slices']={name:dict(metric(result[mask]),uncertainty=bootstrap(result[mask])) for name,mask in slice_masks(result).items()}
    report['episodes']={str(k):metric(d) for k,d in result.groupby('episode')}
    report['species']={str(k):metric(d) for k,d in result.groupby('species')}
    report['species_slices']={str(k):{name:metric(d[mask]) for name,mask in slice_masks(d).items()} for k,d in result.groupby('species')}
    report['gate_passed']=bool(report['delta']>0 and all(v['delta']>0 for v in report['episodes'].values()) and all(report['union'][k]>=report['baseline'][k] for k in ['known_map5','unknown_map5']))
    report['elapsed_seconds_this_invocation']=time.monotonic()-started
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('FINISHED',json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4,user_api='blas'): main()
