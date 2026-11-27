"""Bounded development search, then frozen candidate evaluation.

Development searches are recorded, including losers. The recipe lock must be
written before a fresh evaluation is prepared or scored.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.special import softmax
from threadpoolctl import threadpool_limits
import experiment as e
import joint_ranker as j
b,p=e.b,e.p
HERE=e.HERE

CONFIGS={
    'joint_small':dict(n_estimators=200,num_leaves=7,max_depth=3,min_child_samples=80,reg_lambda=20.),
    'joint_regular':dict(n_estimators=400,num_leaves=15,max_depth=5,min_child_samples=50,reg_lambda=10.),
    'joint_flexible':dict(n_estimators=400,num_leaves=31,max_depth=6,min_child_samples=50,reg_lambda=10.),
}
HEAD_CONFIGS={
    'head_weighted':dict(random_state=13052),
    'head_regular':dict(num_leaves=15,min_child_samples=60,reg_lambda=5.,reg_alpha=.1,random_state=13052),
    'head_strong':dict(num_leaves=15,max_depth=5,min_child_samples=80,reg_lambda=15.,reg_alpha=.5,random_state=13052),
}

def original_scores(frame,r,all_candidate=False):
    pc=r['raw']/np.maximum(r['raw'].sum(1,keepdims=True),1e-12) if all_candidate else r['pc']
    return np.c_[(1-r['pb'][:,None])*pc,r['pb']]

def labels(frame):
    k=b.K
    return np.c_[frame['c'].ident.to_numpy().reshape(-1,k),np.full(len(frame['q']),p.NEW)]

def rank(lab,scores):
    return np.take_along_axis(lab,np.argsort(-scores,axis=1,kind='stable')[:,:5],axis=1)

def row_frame(bundle):
    f=bundle['frame'];order=f['order'];truth=f['truth'];g=bundle['gallery']
    out=pd.DataFrame(dict(episode=bundle['episode'],id=b.IDS[order],identity=b.Y[order],species=b.SP[order],truth=truth,unknown=truth==p.NEW))
    out['true_gallery_images']=[int(np.sum(b.Y[g]==b.Y[i])) if t!=p.NEW else 0 for i,t in zip(order,truth)]
    return out

def summarize(df, method, boot=5000):
    delta=df[method+'_rr5']-df.baseline_rr5
    groups=df.assign(delta=delta).groupby(['species','identity']).delta.agg(['sum','size'])
    rng=np.random.default_rng(130726);num=np.zeros(boot);den=np.zeros(boot)
    for _,part in groups.groupby(level=0):
        a=part.to_numpy();ix=rng.integers(len(a),size=(boot,len(a)))
        num+=a[ix,0].sum(1);den+=a[ix,1].sum(1)
    slices={}
    masks={'overall':np.ones(len(df),bool),'known':~df.unknown,'unknown':df.unknown,
           'hyena':df.species=='hyena','leopard':df.species=='leopard',
           'known_hyena':(~df.unknown)&(df.species=='hyena'),
           'known_leopard':(~df.unknown)&(df.species=='leopard'),
           'small_gallery':(~df.unknown)&df.true_gallery_images.between(1,3)}
    masks.update({str(ep):df.episode==ep for ep in df.episode.unique()})
    for name,mask in masks.items():
        z=df[mask];a=z.baseline_rr5;v=z[method+'_rr5']
        slices[name]=dict(n=len(z),baseline=float(a.mean()),candidate=float(v.mean()),delta=float((v-a).mean()),
            baseline_top1=float((a==1).mean()),candidate_top1=float((v==1).mean()))
    return dict(slices=slices,delta_ci95=np.quantile(num/den,[.025,.975]).tolist(),
        appearances=len(df),unique_photos=df.id.nunique(),identities=df.identity.nunique(),
        improved=int((delta>0).sum()),worsened=int((delta<0).sum()))

def evaluate(out, methods):
    p.load_data(e.DATA);b.NX=b.X
    rows=[];score_outputs=[]
    for path in sorted(out.glob('eval_*.joblib')):
        bundle=joblib.load(path);f,r=bundle['frame'],bundle['result']
        lab=labels(f);base=original_scores(f,r);df=row_frame(bundle)
        scores={'baseline':base}
        for name,model,recipe in methods:
            candidate_frame=f
            if recipe.get('feature_set')=='plda_v1':
                candidate_frame=joblib.load(path.with_name('plda_'+path.name))['frame']
            if recipe['kind']=='all_candidate':s=original_scores(f,r,True)
            elif recipe['kind']=='head':
                head_result=b.predict(model,candidate_frame)
                s=original_scores(f,head_result,recipe.get('all_candidate',False))
                s=(1-recipe['blend'])*base+recipe['blend']*s
            else:
                _,raw=j.predict(model,f)
                if recipe['blend']==1.:s=raw
                else:s=(1-recipe['blend'])*base+recipe['blend']*softmax(raw,axis=1)
            scores[name]=s
        for name,score in scores.items():
            pred=rank(lab,score)
            df[name+'_predictions']=[' '.join(x) for x in pred]
            df[name+'_rr5']=e.rr(pred,f['truth'])
        rows.append(df)
        score_outputs.append(dict(episode=bundle['episode'],id=df.id.to_numpy(),labels=lab,scores=scores))
    if not rows:raise RuntimeError('No prepared evaluation frames')
    result=pd.concat(rows,ignore_index=True)
    return result,score_outputs

def development(out):
    frames=[joblib.load(out/f'train_{ep}.joblib') for ep in range(5052,5064)]
    methods=[];recipes={}
    for name,params in CONFIGS.items():
        path=out/(name+'.joblib')
        if path.exists():model=joblib.load(path)
        else:
            model=j.fit(frames,params);e.dump(model,path)
        print('TRAINED',name,json.dumps(model.metadata),flush=True)
        for weight in [.25,.5,1.]:
            arm=f'{name}_blend{weight:g}'
            recipe=dict(kind='joint',params=params,blend=weight,weight_by_photo=True)
            methods.append((arm,model,recipe));recipes[arm]=recipe
    recipes['all_candidate']=dict(kind='all_candidate')
    methods.append(('all_candidate',None,recipes['all_candidate']))
    result,scores=evaluate(out,methods)
    result.to_csv(out/'development_predictions.csv',index=False)
    e.dump(scores,out/'development_scores.joblib')
    metrics={name:summarize(result,name) for name,_,_ in methods}
    e.write_json(out/'development_metrics.json',metrics)
    e.write_json(out/'development_recipes.json',recipes)
    table=pd.DataFrame([dict(method=name,map5=v['slices']['overall']['candidate'],delta=v['slices']['overall']['delta'],
        known=v['slices']['known']['delta'],unknown=v['slices']['unknown']['delta'],ci_low=v['delta_ci95'][0],
        worst_episode=min(v['slices'][str(ep)]['delta'] for ep in result.episode.unique())) for name,v in metrics.items()])
    table.sort_values('map5',ascending=False).to_csv(out/'development_comparison.csv',index=False)
    print(table.sort_values('map5',ascending=False).to_string(index=False),flush=True)

def heads(out, feature_set=None):
    import regularized_head as h
    prefix='plda_' if feature_set else ''
    frames=[joblib.load(out/f'{prefix}train_{ep}.joblib') for ep in range(5052,5064)]
    b.PARAMS.update(n_estimators=400,n_jobs=4,random_state=2052)
    methods=[];recipes={}
    configs={name:params for name,params in HEAD_CONFIGS.items() if not feature_set or name!='head_strong'}
    for name,params in configs.items():
        name=prefix+name
        path=out/(name+'.joblib')
        if path.exists():model=joblib.load(path)
        else:
            model=h.fit(frames,params,weight_by_photo=True);e.dump(model,path)
        print('TRAINED',name,flush=True)
        for weight in [.5,1.]:
            for all_candidate in [False,True]:
                arm=f'{name}_blend{weight:g}_all{int(all_candidate)}'
                recipe=dict(kind='head',params=params,blend=weight,weight_by_photo=True,all_candidate=all_candidate)
                if feature_set:recipe['feature_set']=feature_set
                methods.append((arm,model,recipe));recipes[arm]=recipe
    result,scores=evaluate(out,methods)
    result.to_csv(out/f'{prefix}head_development_predictions.csv',index=False)
    e.dump(scores,out/f'{prefix}head_development_scores.joblib')
    metrics={name:summarize(result,name) for name,_,_ in methods}
    e.write_json(out/f'{prefix}head_development_metrics.json',metrics)
    e.write_json(out/f'{prefix}head_development_recipes.json',recipes)
    table=pd.DataFrame([dict(method=name,map5=v['slices']['overall']['candidate'],delta=v['slices']['overall']['delta'],
        known=v['slices']['known']['delta'],unknown=v['slices']['unknown']['delta'],ci_low=v['delta_ci95'][0],
        worst_episode=min(v['slices'][str(ep)]['delta'] for ep in result.episode.unique())) for name,v in metrics.items()])
    table.sort_values('map5',ascending=False).to_csv(out/f'{prefix}head_development_comparison.csv',index=False)
    print(table.sort_values('map5',ascending=False).to_string(index=False),flush=True)

def frozen(out,seed):
    lock=json.loads((HERE/'recipe_lock.json').read_text());recipe=lock['recipe']
    assert seed in lock['outer_seeds']
    model=None
    if recipe['kind']=='joint':
        files=[out/f'train_{seed+3000+j}.joblib' for j in range(12)]
        model=j.fit([joblib.load(f) for f in files],recipe['params'],weight_by_photo=recipe['weight_by_photo'])
        e.dump(model,out/'candidate.joblib')
    elif recipe['kind']=='head':
        import regularized_head as h
        b.PARAMS.update(n_estimators=400,n_jobs=4,random_state=seed)
        prefix='plda_' if recipe.get('feature_set')=='plda_v1' else ''
        files=[out/f'{prefix}train_{seed+3000+k}.joblib' for k in range(12)]
        model=h.fit([joblib.load(f) for f in files],recipe['params'],weight_by_photo=recipe['weight_by_photo'])
        e.dump(model,out/'candidate.joblib')
    result,scores=evaluate(out,[('candidate',model,recipe)])
    result.to_csv(out/'validation_predictions.csv',index=False);e.dump(scores,out/'validation_scores.joblib')
    report=summarize(result,'candidate')
    report.update(seed=seed,recipe_lock_sha256=e.sha(HERE/'recipe_lock.json'),
        baseline_model_sha256=e.sha(out/'baseline_models.joblib'),
        limitations=['Previously studied dataset; new split is independent of this candidate selection, not a new external dataset.',
                     'Episodes share identities and images; interval conditions on these fitted models.',
                     'PCA and query-neighbor features use unlabelled query batches as in Model12.'])
    e.write_json(out/'metrics.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['development','heads','plda_heads','frozen']);ap.add_argument('--seed',type=int,default=27613)
    args=ap.parse_args();out=HERE/'local_runs'/('development_2052' if args.mode in ['development','heads','plda_heads'] else f'fresh_{args.seed}')
    with threadpool_limits(limits=4):
        if args.mode=='development':development(out)
        elif args.mode=='heads':heads(out)
        elif args.mode=='plda_heads':heads(out,'plda_v1')
        else:frozen(out,args.seed)
