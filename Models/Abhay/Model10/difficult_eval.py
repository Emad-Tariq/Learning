"""Full three-stage retraining: ordinary, 25%, and 50% difficult episodes."""
import os
os.environ['OMP_NUM_THREADS']='8'; os.environ['OPENBLAS_NUM_THREADS']='4'
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tmp/abhay_deps')); sys.path.append(str(HERE.parent/'Model7'))
import argparse,json,time,hashlib,shutil,warnings,platform
import importlib.metadata
import numpy as np
import pandas as pd
import cv2,joblib
from threadpoolctl import threadpool_limits
import dalvin_core as b
import data_protocol as protocol
from difficult_protocol import hard_positions,difficult_episode,slices

METHODS={'baseline':0.,'difficult25':.25,'difficult50':.5}
DEV_SEEDS=list(range(52052,52058)); VAL_SEEDS=[22052,22053,22054]

def masks(d):
    k=~d.unknown
    return {'all':np.ones(len(d),bool),'known':k,'unknown':d.unknown,
            'missing_side':k&d.side_query&d.missing_side,
            'small_gallery_1to3':k&d.true_gallery_images.between(1,3),
            'missing_side_small_gallery':k&d.side_query&d.missing_side&d.true_gallery_images.between(1,3),
            'same_side_available':k&d.side_query&~d.missing_side,'gallery_4plus':k&(d.true_gallery_images>=4)}

def stats(d):
    result={}
    for method in METHODS:
        diff=d[method+'_rr5']-d.baseline_rr5
        result[method]={'map5':float(d[method+'_rr5'].mean()) if len(d) else None,
                        'delta':float(diff.mean()) if len(d) else None,
                        'improved':int((diff>0).sum()),'worsened':int((diff<0).sum())}
    return result

def uncertainty(d,method):
    if not len(d): return None
    a=d[['species','identity']].assign(delta=d[method+'_rr5']-d.baseline_rr5).groupby(['species','identity']).delta.agg(['sum','size'])
    rng=np.random.default_rng(10052); total=np.zeros(3000); counts=np.zeros(3000)
    for _,part in a.groupby(level=0):
        v=part.to_numpy(); idx=rng.integers(0,len(v),size=(3000,len(v)))
        total+=v[idx,0].sum(1); counts+=v[idx,1].sum(1)
    return np.quantile(total/counts,[.025,.975]).tolist()

def rr(pred,truth):
    return [next((1/(j+1) for j,label in enumerate(row) if label==t),0.) for row,t in zip(pred,truth)]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=HERE/'local_runs/difficult_2052')
    args=ap.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic(); seed=2052
    cfg={'seed':seed,'stages':3,'episodes_per_stage':12,'mixtures':METHODS,'workers':8,
         'candidate_trees':400,'detector_trees':300,'neighbours':150,'candidates':15,
         'hard_schedules':{m:[sorted(hard_positions(stage,f)) for stage in range(3)] for m,f in METHODS.items()},
         'hard_episode':'thin half of gallery identities to1-3 photos, remove one query flank if feasible; same unknown count with nearest unknown photos',
         'development_episodes':DEV_SEEDS,'validation_episodes':VAL_SEEDS,
         'selection':'highest development MAP5 among variants with no overall loss, unknown delta>=-0.001, no missing-side/small-gallery loss; baseline is fallback',
         'gate':'positive pooled and every validation episode delta; no known/unknown/missing-side/small-gallery MAP5 decline',
         'limits':['Previously inspected dataset and validation identities','One outer identity allocation, correlated episodes; no leaderboard claim']}
    cfgpath=out/'config.json'
    if cfgpath.exists(): assert json.loads(cfgpath.read_text())==cfg
    cfgpath.write_text(json.dumps(cfg,indent=2))
    paths=[HERE/'difficult_protocol.py',HERE/'difficult_eval.py',HERE.parent/'Model7/dalvin_core.py',HERE.parent/'Model7/data_protocol.py']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if (out/'source_hashes.json').exists(): assert json.loads((out/'source_hashes.json').read_text())==hashes
    (out/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
    (out/'environment.json').write_text(json.dumps({'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','lightgbm','opencv-python']}},indent=2))
    cv2.setNumThreads(1); warnings.filterwarnings('ignore',message='Only one sample available.*',category=UserWarning)
    b.PARAMS.update(n_estimators=400,n_jobs=8,random_state=seed); b.K=15
    data=Path.home()/'Downloads/reid_student_data_main'
    train,_,_=protocol.load_data(data); b.NX=b.X
    roles=protocol.allocate_roles(train,3,seed)
    manifest=pd.DataFrame([{'id':b.IDS[i],'identity':b.Y[i],'species':b.SP[i],'role':name} for name,ix in roles.items() for i in ix])
    assert manifest.groupby('identity').role.nunique().max()==1
    manifest.to_csv(out/'identity_roles.csv',index=False)
    for source in (HERE.parent/'Model9/local_runs/coverage_2052').glob('sift_mutual_*.npz'):
        if not (out/source.name).exists(): shutil.copy2(source,out/source.name)
    cache=protocol.SiftCache(data,out,8,150)
    checkpoint=out/'models.joblib'
    models=joblib.load(checkpoint) if checkpoint.exists() else {m:[] for m in METHODS}
    assert len(set(map(len,models.values())))==1
    episode_manifests=[]; summary=[]; slice_summary=[]
    for stage in range(3):
        role=f'stage{stage+1}'; print('Preparing',role,flush=True); cache.activate(roles[role])
        resumed=stage<len(models['baseline'])
        training={m:[] for m in METHODS}
        for j in range(12):
            eseed=seed+1000*(stage+1)+j
            g,q,t=protocol.split_episode(roles[role],eseed,link=b.REGIMES[j%len(b.REGIMES)])
            need_hard=any(j in hard_positions(stage,f) for f in METHODS.values())
            if need_hard:
                gh,qh,th,diag=difficult_episode(roles[role],g,q,t,eseed)
            else: gh=qh=th=diag=None
            pairs={'ordinary':(g,q,t)}
            if need_hard: pairs['difficult']=(gh,qh,th)
            frames={}
            for kind,(gg,qq,tt) in pairs.items():
                episode_manifests.extend({'stage':stage+1,'episode':eseed,'kind':kind,'id':b.IDS[i],'assignment':'gallery'} for i in gg)
                episode_manifests.extend({'stage':stage+1,'episode':eseed,'kind':kind,'id':b.IDS[i],'assignment':'unknown_query' if target==protocol.NEW else 'known_query'} for i,target in zip(qq,tt))
                sr=pd.DataFrame(slices(gg,qq,tt)); sm=masks(sr)
                summary.append({'stage':stage+1,'episode':eseed,'kind':kind,'gallery':len(gg),'queries':len(qq),
                                **{name:int(mask.sum()) for name,mask in sm.items()},**(diag if kind=='difficult' else {})})
                if not resumed: frames[kind]=protocol.frame(gg,qq,tt)
            for method,fraction in METHODS.items():
                hard=j in hard_positions(stage,fraction)
                slice_summary.append({'stage':stage+1,'episode':eseed,'method':method,'kind':'difficult' if hard else 'ordinary'})
                if not resumed:
                    frame=frames['difficult' if hard else 'ordinary']
                    for model in models[method]: frame=b.augment(frame,b.predict(model,frame))
                    training[method].append(frame)
            print(f'{role} episode {j+1}/12'+(' resumed' if resumed else ''),flush=True)
        if not resumed:
            for method in METHODS:
                print('Fitting',method,role,flush=True); models[method].append(b.fit(training[method]))
            temp=checkpoint.with_suffix('.tmp'); joblib.dump(models,temp); os.replace(temp,checkpoint)
        pd.DataFrame(summary).to_csv(out/'training_episode_diagnostics.csv',index=False)
        pd.DataFrame(slice_summary).to_csv(out/'training_method_schedule.csv',index=False)
        (out/'progress.json').write_text(json.dumps({'completed_stages':stage+1,'elapsed_seconds':time.monotonic()-start},indent=2))
        del training
    pd.DataFrame(episode_manifests).to_csv(out/'training_episode_splits.csv',index=False)

    def evaluate(indices,seeds,role):
        path=out/f'{role}_predictions.csv'
        if path.exists(): return pd.read_csv(path)
        cache.activate(indices); frames=[]; splitrows=[]
        for eseed in seeds:
            g,q,t=protocol.split_episode(indices,eseed,link=(.95,111))
            raw=protocol.frame(g,q,t); order=raw['order']
            d=pd.DataFrame(slices(g,order,raw['truth'])); d['episode']=eseed
            present=raw['c'].groupby('q').ident.apply(set)
            d['candidate_present']=[target in present[k] for k,target in enumerate(d.truth)]
            for method in METHODS:
                s,r=protocol.through_stack(raw,models[method]); pred=b.rank(s,r)
                assert all(len(set(row))==5 for row in pred)
                d[method+'_rr5']=rr(pred,d.truth)
                d[method+'_predictions']=[' '.join(row) for row in pred]
                d[method+'_pnew']=r['pb']
                conditional=np.take_along_axis(s['c'].ident.to_numpy().reshape(len(d),15),np.argsort(-r['pc'],axis=1)[:,:5],axis=1)
                d[method+'_conditional_rr5']=rr(conditional,d.truth)
            frames.append(d)
            splitrows.extend({'episode':eseed,'id':b.IDS[i],'assignment':'gallery'} for i in g)
            splitrows.extend({'episode':eseed,'id':b.IDS[i],'assignment':'unknown_query' if target==protocol.NEW else 'known_query'} for i,target in zip(q,t))
            print(role,eseed,json.dumps(stats(d)),flush=True)
        result=pd.concat(frames,ignore_index=True); result.to_csv(path,index=False)
        pd.DataFrame(splitrows).to_csv(out/f'{role}_splits.csv',index=False)
        return result

    dev=evaluate(roles['calibration'],DEV_SEEDS,'development')
    dev_stats={name:stats(dev[mask]) for name,mask in masks(dev).items()}
    eligible=['baseline']
    for method in ('difficult25','difficult50'):
        if (dev_stats['all'][method]['delta']>=0 and dev_stats['unknown'][method]['delta']>=-.001
                and all(dev_stats[name][method]['delta']>=0 for name in ['missing_side','small_gallery_1to3'])):
            eligible.append(method)
    selected=max(eligible,key=lambda m:(dev_stats['all'][m]['map5'],-METHODS[m]))
    selection={'method':selected,'eligible':eligible,'development':dev_stats}
    (out/'selection.json').write_text(json.dumps(selection,indent=2)); print('FROZEN SELECTION',selected,flush=True)
    val=evaluate(roles['validation'],VAL_SEEDS,'validation')
    # Refit baseline must be identical to the preserved full pipeline.
    ref=pd.read_csv(HERE.parent/'Model8/local_runs/union_2052/validation_predictions.csv')
    merged=val.merge(ref[['episode','id','baseline_predictions']],on=['episode','id'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(merged)==len(val) and (merged.baseline_predictions_new==merged.baseline_predictions_old).all()
    report={'selection':{'method':selected,'eligible':eligible},'methods':stats(val),'slices':{},
            'episodes':{str(k):stats(v) for k,v in val.groupby('episode')},
            'species':{str(k):stats(v) for k,v in val.groupby('species')},
            'baseline_parity':{'all_top5_identical':True,'query_appearances':len(val)},
            'elapsed_seconds':time.monotonic()-start}
    for name,mask in masks(val).items():
        part=val[mask]
        report['slices'][name]={'appearances':len(part),'unique_queries':int(part.id.nunique()),'identities':int(part.identity.nunique()),
                               'methods':stats(part),'ci95':{m:uncertainty(part,m) for m in ['difficult25','difficult50']}}
    report['gates']={m:bool(report['methods'][m]['delta']>0 and all(r[m]['delta']>0 for r in report['episodes'].values())
                           and all(report['slices'][name]['methods'][m]['delta']>=0 for name in ['known','unknown','missing_side','small_gallery_1to3'])) for m in ['difficult25','difficult50']}
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('FINISHED',json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4,user_api='blas'): main()
