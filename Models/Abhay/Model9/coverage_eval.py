"""Fit bounded coverage residuals on disjoint development identities; freeze then compare."""
import os
os.environ['OMP_NUM_THREADS']='8'; os.environ['OPENBLAS_NUM_THREADS']='4'
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tmp/abhay_deps')); sys.path.append(str(HERE.parent/'Model7'))
import json, time, hashlib, shutil, warnings, importlib.metadata
import numpy as np
import pandas as pd
import cv2, joblib
from threadpoolctl import threadpool_limits
from sklearn.model_selection import StratifiedKFold
import dalvin_core as b
import data_protocol as protocol
from coverage_heads import evidence,train_heads,adjust,predictions,rr5

OUT=HERE/'local_runs/coverage_2052'
METHODS=('baseline','rank_only','unknown_only','coverage')
DEVELOPMENT_SEEDS=list(range(42052,42064))
VALIDATION_SEEDS=[22052,22053,22054]
STRENGTHS=[0.,.25,.5,1.]

def masks(d):
    known=~d.unknown
    return {'all':np.ones(len(d),bool),'known':known,'unknown':d.unknown,
            'missing_side':known&d.side_query&d.missing_side,
            'small_gallery_1to3':known&d.true_gallery_images.between(1,3),
            'missing_side_small_gallery':known&d.side_query&d.missing_side&d.true_gallery_images.between(1,3),
            'same_side_available':known&d.side_query&~d.missing_side,
            'gallery_4plus':known&(d.true_gallery_images>=4)}

def fold_assignments(rows):
    animals=rows[['species','identity']].drop_duplicates().sort_values(['species','identity']).reset_index(drop=True)
    assignments={}
    for fold,(_,test) in enumerate(StratifiedKFold(n_splits=4,shuffle=True,random_state=9052).split(animals,animals.species)):
        assignments.update({name:fold for name in animals.identity.iloc[test]})
    return rows.identity.map(assignments).to_numpy()

def build_bundle(indices,seeds,role,cache,models):
    path=OUT/f'{role}_bundle.joblib'
    if path.exists(): return joblib.load(path)
    cache.activate(indices)
    parts=[]; splitrows=[]
    for seed in seeds:
        g,q,truth=protocol.split_episode(indices,seed,link=(.95,111))
        s=protocol.frame(g,q,truth); s,r=protocol.through_stack(s,models)
        order=s['order']; target=s['truth']; n=len(order)
        candidate=s['c'].ident.to_numpy().reshape(n,15)
        rankf,uf=evidence(s,r)
        d=pd.DataFrame({'episode':seed,'id':b.IDS[order],'identity':b.Y[order],'species':b.SP[order],
                        'truth':target,'unknown':target==protocol.NEW,'viewpoint_code':b.VIEW[order]})
        d['side_query']=np.isin(d.viewpoint_code,[0,1,4,5,6,7])
        d['true_gallery_images']=[int((b.Y[g]==b.Y[i]).sum()) for i in order]
        d['missing_side']=[bool(v>=0 and b.FLANK[v] in (0,1) and not np.any((b.Y[g]==b.Y[i])&(b.VIEW[g]>=0)&(b.FLANK[b.VIEW[g].clip(0)]==b.FLANK[v]))) for i,v in zip(order,b.VIEW[order])]
        ti=np.array([int(np.flatnonzero(row==label)[0]) if label in row else -1 for row,label in zip(candidate,target)])
        parts.append({'rows':d,'rank_features':rankf,'unknown_features':uf,'pc':r['pc'],'pb':r['pb'],'candidates':candidate,'target':ti})
        for i in g: splitrows.append({'episode':seed,'id':b.IDS[i],'assignment':'gallery'})
        for i,label in zip(q,truth): splitrows.append({'episode':seed,'id':b.IDS[i],'assignment':'unknown_query' if label==protocol.NEW else 'known_query'})
        print(f'{role} episode {seed}: {n} queries',flush=True)
    combined={k:pd.concat([a[k] for a in parts],ignore_index=True) if k=='rows' else np.concatenate([a[k] for a in parts],axis=0) for k in parts[0]}
    joblib.dump(combined,path)
    pd.DataFrame(splitrows).to_csv(OUT/f'{role}_splits.csv',index=False)
    return combined

def evaluate(bundle,rh,uh,rs,us):
    pc,pb=adjust(bundle,rh,uh,rs,us)
    pred=predictions(bundle,pc,pb)
    return rr5(pred,bundle['rows'].truth),pred,pb

def select_strengths(bundle):
    rows=bundle['rows']; folds=fold_assignments(rows)
    assert pd.DataFrame({'id':rows.identity,'fold':folds}).groupby('id').fold.nunique().max()==1
    rank_delta=np.zeros_like(bundle['pc']); unknown_delta=np.zeros_like(bundle['pb'])
    for fold in range(4):
        fit=np.flatnonzero(folds!=fold); test=np.flatnonzero(folds==fold)
        rh,uh=train_heads(bundle,fit)
        rank_delta[test]=rh.correction(bundle['rank_features'][test]); unknown_delta[test]=uh.correction(bundle['unknown_features'][test])
        print('Development identity fold',fold,'fitted',flush=True)
    # Corrections are out of fold. Select strengths using only these predictions.
    from scipy.special import softmax,expit,logit
    candidates=[]; allscores={}; mm=masks(rows)
    baseline=rr5(predictions(bundle,bundle['pc'],bundle['pb']),rows.truth)
    for rs in STRENGTHS:
      for us in STRENGTHS:
        pc=softmax(np.log(np.maximum(bundle['pc'],1e-300))+rs*rank_delta,axis=1) if rs else bundle['pc']
        pb=expit(logit(np.clip(bundle['pb'],1e-9,1-1e-9))+us*unknown_delta) if us else bundle['pb']
        score=rr5(predictions(bundle,pc,pb),rows.truth); allscores[(rs,us)]=score
        delta={name:float((score-baseline)[mask].mean()) for name,mask in mm.items()}
        # Overall performance must not fall; limit unknown decline to 0.001 on development.
        eligible=delta['all']>=-1e-12 and delta['unknown']>=-.001
        objective=.7*delta['all']+.15*delta['missing_side']+.15*delta['small_gallery_1to3']
        candidates.append({'rank_strength':rs,'unknown_strength':us,'eligible':bool(eligible),'objective':objective,**{name+'_delta':v for name,v in delta.items()}})
    table=pd.DataFrame(candidates)
    winner=table[table.eligible].sort_values(['objective','rank_strength','unknown_strength'],ascending=[False,True,True]).iloc[0]
    selected={'rank_strength':float(winner.rank_strength),'unknown_strength':float(winner.unknown_strength)}
    table.to_csv(OUT/'development_strength_search.csv',index=False)
    rows.assign(identity_fold=folds).to_csv(OUT/'development_folds.csv',index=False)
    if not selected['rank_strength'] and not selected['unknown_strength']:
        # Still evaluate a fixed full-strength experimental arm as a negative-result diagnostic.
        selected.update(primary_is_baseline=True,diagnostic_rank_strength=1.,diagnostic_unknown_strength=1.)
    else: selected['primary_is_baseline']=False
    # Rank-only / unknown-only ablations use the strengths chosen above (or fixed diagnostic strength).
    (OUT/'selection.json').write_text(json.dumps(selected,indent=2))
    print('Frozen development selection',json.dumps(selected),flush=True)
    return selected

def uncertainty(rows,values,seed=9052):
    if not len(rows): return None
    d=rows[['species','identity']].assign(delta=values)
    a=d.groupby(['species','identity']).delta.agg(['sum','size'])
    rng=np.random.default_rng(seed); sums=np.zeros(3000); counts=np.zeros(3000)
    for _,part in a.groupby(level=0):
        v=part.to_numpy(); idx=rng.integers(0,len(v),size=(3000,len(v)))
        sums+=v[idx,0].sum(1); counts+=v[idx,1].sum(1)
    return np.quantile(sums/counts,[.025,.975]).tolist()

def main():
    started=time.monotonic(); OUT.mkdir(parents=True,exist_ok=True)
    config={'seed':2052,'development_episodes':DEVELOPMENT_SEEDS,'validation_episodes':VALIDATION_SEEDS,
            'folds':4,'fold_seed':9052,'penalty':.1,'rank_correction_cap':1.,'unknown_logodds_cap':float(np.log(2)),
            'strengths':STRENGTHS,'selection_objective':'.7 overall + .15 missing-side + .15 small-gallery delta',
            'selection_constraints':'overall delta >=0; unknown delta >= -0.001; ties prefer weaker corrections',
            'features':'coverage, SIFT support, similarity and baseline-neighbour evidence; no truth features',
            'training':'frozen original Dalvin 3d stack, calibration-role identities fit residual heads only',
            'primary':'OOF-selected strengths; if baseline selected, full-strength arm is diagnostic only',
            'gate':'positive pooled and each episode; no known, unknown, missing-side or small-gallery MAP5 decline',
            'limitations':['Validation identities and EDA have been inspected before; not untouched',
                          'Repeated episodes/photos are clustered by identity; no retraining-variance claim']}
    path=OUT/'config.json'
    if path.exists(): assert json.loads(path.read_text())==config
    path.write_text(json.dumps(config,indent=2))
    provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'coverage_heads.py',HERE/'coverage_eval.py',HERE.parent/'Model7/dalvin_core.py',HERE.parent/'Model7/data_protocol.py']}
    old=OUT/'source_hashes.json'
    if old.exists(): assert json.loads(old.read_text())==provenance,'Source changed; use a new run directory'
    old.write_text(json.dumps(provenance,indent=2))
    cv2.setNumThreads(1); warnings.filterwarnings('ignore',message='Only one sample available.*',category=UserWarning)
    data=Path.home()/'Downloads/reid_student_data_main'
    train,_,_=protocol.load_data(data); b.NX=b.X; b.K=15
    roles=protocol.allocate_roles(train,3,2052)
    manifest=pd.DataFrame([{'id':b.IDS[i],'identity':b.Y[i],'role':name} for name,ix in roles.items() for i in ix])
    assert manifest.groupby('identity').role.nunique().max()==1
    manifest.to_csv(OUT/'identity_roles.csv',index=False)
    source=HERE.parent/'Model8/local_runs/union_2052'
    for p in source.glob('sift_mutual_*.npz'):
        if not (OUT/p.name).exists(): shutil.copy2(p,OUT/p.name)
    cache=protocol.SiftCache(data,OUT,8,150)
    models=joblib.load(HERE.parent/'Model7/local_runs/paired_2052/paired_models.joblib')['baseline']
    development=build_bundle(roles['calibration'],DEVELOPMENT_SEEDS,'development',cache,models)
    selection=select_strengths(development)
    rh,uh=train_heads(development,np.arange(len(development['rows'])))
    joblib.dump({'rank':rh,'unknown':uh,'selection':selection},OUT/'coverage_heads.joblib')
    # All fitting, cross-validation and strength selection finish before opening validation frames.
    validation=build_bundle(roles['validation'],VALIDATION_SEEDS,'validation',cache,models)
    rs=selection.get('diagnostic_rank_strength',selection['rank_strength'])
    us=selection.get('diagnostic_unknown_strength',selection['unknown_strength'])
    settings={'baseline':(0.,0.),'rank_only':(rs,0.),'unknown_only':(0.,us),'coverage':(rs,us)}
    d=validation['rows'].copy(); report={'selection':selection,'methods':{},'slices':{},'episodes':{},'species':{}}
    for method,(a,z) in settings.items():
        score,pred,pnew=evaluate(validation,rh,uh,a,z)
        d[method+'_rr5']=score; d[method+'_predictions']=[' '.join(row) for row in pred]; d[method+'_pnew']=pnew
        assert all(len(set(row))==5 for row in pred)
    # Audit the frozen baseline against the preceding experiment on all three episodes.
    reference=pd.read_csv(source/'validation_predictions.csv')
    parity=d.merge(reference[['episode','id','baseline_predictions']],on=['episode','id'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(parity)==len(d) and (parity.baseline_predictions_new==parity.baseline_predictions_old).all()
    report['baseline_parity']={'queries':len(parity),'all_top5_identical':True}
    def stats(part):
        base=part.baseline_rr5
        return {m:{'map5':float(part[m+'_rr5'].mean()) if len(part) else None,
                   'delta':float((part[m+'_rr5']-base).mean()) if len(part) else None,
                   'improved':int((part[m+'_rr5']>base).sum()),'worsened':int((part[m+'_rr5']<base).sum())} for m in METHODS}
    report['methods']=stats(d)
    for name,mask in masks(d).items():
        part=d[mask]
        report['slices'][name]={'appearances':len(part),'unique_queries':int(part.id.nunique()),'identities':int(part.identity.nunique()),
                              'methods':stats(part),'coverage_delta_ci95':uncertainty(part,(part.coverage_rr5-part.baseline_rr5).to_numpy())}
    report['episodes']={str(seed):stats(part) for seed,part in d.groupby('episode')}
    report['species']={str(sp):stats(part) for sp,part in d.groupby('species')}
    report['gate_passed']=bool(not selection['primary_is_baseline'] and report['methods']['coverage']['delta']>0 and all(ep['coverage']['delta']>0 for ep in report['episodes'].values()) and all(report['slices'][k]['methods']['coverage']['delta']>=0 for k in ['known','unknown','missing_side','small_gallery_1to3']))
    report['elapsed_seconds']=time.monotonic()-started
    d.to_csv(OUT/'validation_predictions.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('FINISHED',json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4,user_api='blas'): main()
