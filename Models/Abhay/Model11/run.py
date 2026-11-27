"""Paired open-set evaluation: full-vector MLP evidence for P(new)."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4'); os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'tmp/abhay_deps'))
import argparse,json,hashlib,warnings
import numpy as np
import pandas as pd
import cv2,joblib
from threadpoolctl import threadpool_limits
import dalvin_core as b
import data_protocol as p
import verifier as v


def metrics(d,column):
    return dict(map5=float(d[column].mean()),known_map5=float(d.loc[~d.unknown,column].mean()),
        unknown_map5=float(d.loc[d.unknown,column].mean()),
        unknown_top1=float((d.loc[d.unknown,column]==1).mean()),
        known_top1=float((d.loc[~d.unknown,column]==1).mean()))


def bootstrap(d,column):
    z=d.assign(delta=d[column]-d.baseline_rr5).groupby(['species','identity']).delta.agg(['sum','size'])
    rng=np.random.default_rng(2052); sums=np.zeros(2000); counts=np.zeros(2000)
    for _,part in z.groupby(level=0):
        a=part.to_numpy(); ix=rng.integers(0,len(a),(2000,len(a)))
        sums+=a[ix,0].sum(1); counts+=a[ix,1].sum(1)
    return np.quantile(sums/counts,[.025,.975]).tolist()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=HERE/'local_runs/main')
    ap.add_argument('--seed',type=int,default=2052); ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--epochs',type=int,default=40); ap.add_argument('--max-pairs',type=int,default=24000)
    ap.add_argument('--baseline-sims',type=int,default=12); ap.add_argument('--calibration-episodes',type=int,default=6)
    ap.add_argument('--trees',type=int,default=400); ap.add_argument('--reuse-local',action='store_true')
    args=ap.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    if min(args.epochs,args.max_pairs,args.baseline_sims,args.calibration_episodes,args.trees,args.workers)<1:raise ValueError('Counts must be positive')
    config={k:str(a) if isinstance(a,Path) else a for k,a in vars(args).items()}
    config.update(primary='mlp_blend',blend_weight=.5,roles='60% baseline/verifier,20% calibration,20% validation',
        pair_stop='20% of verifier training identities, disjoint from pair fitting',
        calibrated_label='query animal absent from episode gallery',
        sources={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ['run.py','verifier.py','dalvin_core.py','data_protocol.py']})
    path=out/'config.json'
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('Configuration/source changed: choose a NEW output directory')
    path.write_text(json.dumps(config,indent=2))
    cv2.setNumThreads(1);warnings.filterwarnings('ignore',message='Only one sample available.*',category=UserWarning)
    b.PARAMS.update(n_estimators=args.trees,n_jobs=args.workers,random_state=args.seed)
    train,_,_=p.load_data(args.data); b.NX=b.X
    roles=p.allocate_roles(train,3,args.seed); supervised=np.concatenate([roles[f'stage{i}'] for i in (1,2,3)])
    assert not set(b.Y[supervised])&set(b.Y[roles['calibration']])
    assert not set(b.Y[roles['validation']])&set(b.Y[np.r_[supervised,roles['calibration']]])
    pd.DataFrame([dict(id=b.IDS[i],identity=b.Y[i],species=b.SP[i],role=role) for role,ix in roles.items() for i in ix]).to_csv(out/'identity_roles.csv',index=False)
    cache=p.SiftCache(args.data,out,args.workers,150)
    source=HERE.parent/'Model7/local_runs/paired_2052'
    if args.reuse_local:
        if args.seed!=2052 or args.trees!=400 or args.baseline_sims!=12:raise ValueError('Local baseline reuse requires original settings')
        for directory in [source,HERE.parent/'Model9/local_runs/coverage_2052']:
            for file in directory.glob('sift_mutual_*.npz'):
                with np.load(file) as z:cache.done.update({tuple(k):tuple(val) for k,val in zip(z['pairs'],z['values'])})
    baseline_path=out/'baseline_models.joblib'
    if baseline_path.exists():models=joblib.load(baseline_path)
    elif args.reuse_local:
        models=joblib.load(source/'paired_models.joblib')['baseline'];joblib.dump(models,baseline_path)
    else:models=[]
    splitrows=[]
    def episode(indices,seed,role,link):
        gi,qi,truth=p.split_episode(indices,seed,link=link)
        splitrows.extend(dict(episode=seed,role=role,id=b.IDS[i],assignment='gallery') for i in gi)
        splitrows.extend(dict(episode=seed,role=role,id=b.IDS[i],assignment='unknown_query' if t==p.NEW else 'known_query') for i,t in zip(qi,truth))
        return gi,qi,truth
    for stage in range(3):
        if stage<len(models):continue
        ix=roles[f'stage{stage+1}'];cache.activate(ix);frames=[]
        for j in range(args.baseline_sims):
            gi,qi,truth=episode(ix,args.seed+1000*(stage+1)+j,f'stage{stage+1}',b.REGIMES[j%4])
            s=p.frame(gi,qi,truth)
            for model in models:s=b.augment(s,b.predict(model,s))
            frames.append(s);print('Baseline stage',stage+1,'episode',j+1,flush=True)
        models.append(b.fit(frames));joblib.dump(models,baseline_path)
    network_path=out/'verifier.joblib'
    if network_path.exists():network=joblib.load(network_path)
    else:
        network=v.fit_verifier(b.X,b.Y,b.SP,supervised,args.seed,args.epochs,args.max_pairs)
        joblib.dump(network,network_path)
    pairroles={i:'pair_fit' for i in network['fit_indices']}|{i:'pair_early_stop' for i in network['stop_indices']}
    pd.DataFrame([dict(id=b.IDS[i],identity=b.Y[i],role=role) for i,role in pairroles.items()]).to_csv(out/'pair_identity_roles.csv',index=False)
    pd.DataFrame(network['history']).to_csv(out/'pair_training_history.csv',index=False)
    def bundle(role,seeds):
        path=out/f'{role}_bundle.joblib'
        if path.exists():return joblib.load(path)
        cache.activate(roles[role]);rows=[];contexts=[];rich=[];pcs=[];pbs=[];candidates=[]
        for seed in seeds:
            gi,qi,truth=episode(roles[role],seed,role,(.95,111));s=p.frame(gi,qi,truth);s,r=p.through_stack(s,models)
            a,z=v.query_features(network,b.X,b.Y,gi,s,r)
            contexts.append(a);rich.append(z);pcs.append(r['pc']);pbs.append(r['pb'])
            ids=s['c'].ident.to_numpy().reshape(-1,15);candidates.append(ids)
            for j,i in enumerate(s['order']):
                rows.append(dict(episode=seed,id=b.IDS[i],identity=b.Y[i],species=b.SP[i],truth=s['truth'][j],
                    unknown=s['truth'][j]==p.NEW,candidate_present=s['truth'][j] in ids[j],
                    true_gallery_images=int((b.Y[gi]==b.Y[i]).sum()) if s['truth'][j]!=p.NEW else 0))
            print('Built',role,seed,'queries',len(s['order']),flush=True)
        result=dict(rows=pd.DataFrame(rows),context=np.concatenate(contexts),rich=np.concatenate(rich),pc=np.concatenate(pcs),pb=np.concatenate(pbs),candidates=np.concatenate(candidates))
        joblib.dump(result,path);return result
    cal=bundle('calibration',[args.seed+40000+j for j in range(args.calibration_episodes)])
    # Both calibration arms have identical labels, queries and linear architecture.
    simple=v.calibrator(cal['context'],cal['rows'].unknown.astype(int))
    rich=v.calibrator(cal['rich'],cal['rows'].unknown.astype(int))
    joblib.dump(dict(summary=simple,mlp=rich),out/'calibrators.joblib')
    # Calibration and network fitting finish before validation labels are evaluated.
    val=bundle('validation',[args.seed+20000+j for j in range(3)])
    d=val['rows'].copy();p_simple=simple.predict_proba(val['context'])[:,1];p_mlp=rich.predict_proba(val['rich'])[:,1]
    settings={'baseline':val['pb'],'summary_blend':.5*(val['pb']+p_simple),
        'mlp_blend':.5*(val['pb']+p_mlp),'mlp_replacement':p_mlp}
    report={'pair_network_parameters':network['parameters'],'pair_fit_count':network['fit_pairs'],'pair_stop_count':network['stop_pairs'],
        'primary':'mlp_blend','methods':{},'episodes':{},'species':{}}
    for name,pnew in settings.items():
        pred=v.rank(val['candidates'],val['pc'],pnew);d[name+'_rr5']=v.rr5(pred,d.truth)
        d[name+'_predictions']=[' '.join(row) for row in pred];d[name+'_pnew']=pnew
        assert all(len(set(row))==5 for row in pred)
        report['methods'][name]=metrics(d,name+'_rr5')
    if args.reuse_local:
        saved=pd.read_csv(source/'validation_predictions.csv').set_index(['episode','id'])
        assert d.baseline_predictions.tolist()==saved.loc[list(zip(d.episode,d.id)),'baseline_predictions'].tolist()
        report['baseline_saved_prediction_parity']=True
    for key in ('episodes','species'):
        field='episode' if key=='episodes' else 'species'
        report[key]={str(k):{name:metrics(z,name+'_rr5') for name in settings} for k,z in d.groupby(field)}
    for name in settings:
        report['methods'][name]['delta_map5']=report['methods'][name]['map5']-report['methods']['baseline']['map5']
        if name!='baseline':report['methods'][name]['delta_identity_bootstrap95']=bootstrap(d,name+'_rr5')
    main=report['methods']['mlp_blend'];base=report['methods']['baseline']
    report['promotion_gate_passed']=bool(main['delta_map5']>0 and main['known_map5']>=base['known_map5'] and
        main['unknown_map5']>=base['unknown_map5'] and main['unknown_top1']>=base['unknown_top1'] and
        all(z['mlp_blend']['map5']>z['baseline']['map5'] for z in report['episodes'].values()) and
        main['delta_identity_bootstrap95'][0]>0)
    report['mlp_minus_summary_map5']=main['map5']-report['methods']['summary_blend']['map5']
    report['limitations']=['Existing seed2052 evaluation has been inspected before; development evidence only.',
        'Repeated episodes reuse photos/identities; bootstrap clusters true identities, not retraining variance.',
        'Smaller role galleries than full competition inference. Fixed settings, no validation tuning.',
        'Pair probabilities are not P(known). Only separately calibrated query outputs enter final ranking.']
    d.to_csv(out/'validation_predictions.csv',index=False)
    pd.DataFrame(splitrows).to_csv(out/'episode_splits_this_invocation.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('FINISHED',json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4,user_api='blas'):main()
