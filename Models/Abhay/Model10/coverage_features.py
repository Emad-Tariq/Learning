"""Follow-up: make observable candidate coverage available to the tree detector."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import dalvin_core as b
import data_protocol as protocol

PREFIX='cov_'

def add_coverage(s):
    s=dict(s); q=s['q'].copy(); c=s['c']; n=len(q)
    matrix=lambda key:c[key].to_numpy(float).reshape(n,15)
    raw=matrix('mxr'); sift=matrix('sx')
    leaders={'lda':np.zeros(n,dtype=int),'raw':raw.argmax(1),'sift':sift.argmax(1)}
    side=np.isin(q['view'],[0,1,4,5,6,7])
    for kind,choice in leaders.items():
        cnt=matrix('cnt')[np.arange(n),choice]
        flank=matrix('n_flank')[np.arange(n),choice]
        view=matrix('n_view')[np.arange(n),choice]
        sx=sift[np.arange(n),choice]
        support=matrix('support')[np.arange(n),choice]
        q[f'cov_{kind}_log_count']=np.log1p(cnt)
        q[f'cov_{kind}_small']=(cnt<=3).astype(float)
        q[f'cov_{kind}_log_flank']=np.log1p(flank)
        q[f'cov_{kind}_missing_side']=np.where(side,(flank==0).astype(float),np.nan)
        q[f'cov_{kind}_log_view']=np.log1p(view)
        q[f'cov_{kind}_sift_fraction']=support/np.maximum(cnt,1)
        q[f'cov_{kind}_missing_weak_sift']=np.where(side&(flank==0),1/(1+sx),0.)
    q['cov_top3_small_fraction']=(matrix('cnt')[:,:3]<=3).mean(1)
    q['cov_top3_missing_side_fraction']=np.where(side,((matrix('n_flank')[:,:3]==0).mean(1)),np.nan)
    s['q']=q
    return s

def frame(g,q,truth=None):
    return add_coverage(protocol.frame(g,q,truth))

def strip_coverage(s):
    return dict(s,q=s['q'].drop(columns=[c for c in s['q'].columns if c.startswith(PREFIX)]))

def fit(sims,method):
    clean=[strip_coverage(s) for s in sims]
    if method=='baseline': return b.fit(clean)
    qf,cf=b.columns(clean[0])
    df=qf+[c for c in sims[0]['q'].columns if c.startswith(PREFIX)]
    cx=pd.concat([b.candidate_matrix(s,qf,cf) for s in sims])
    cy=np.concatenate([s['c'].y.to_numpy() for s in sims])
    known=np.concatenate([s['q'].unk.to_numpy()[s['c'].q.to_numpy()]==0 for s in sims])
    detector=lgb.LGBMClassifier(**{**b.PARAMS,'n_estimators':300,'num_leaves':15}).fit(
        pd.concat([s['q'][df] for s in sims]),np.concatenate([s['q'].unk.to_numpy() for s in sims]))
    closed=lgb.LGBMClassifier(**b.PARAMS).fit(cx[known],cy[known])
    anyq=lgb.LGBMClassifier(**b.PARAMS).fit(cx,cy)
    return {'detector':detector,'closed':closed,'anyq':anyq,'query_features':qf,'candidate_features':cf,'detector_features':df}

def predict(model,s):
    if not isinstance(model,dict): return b.predict(model,s)
    n=len(s['q']); cx=b.candidate_matrix(s,model['query_features'],model['candidate_features'])
    pn=model['detector'].predict_proba(s['q'][model['detector_features']])[:,1]
    raw=model['anyq'].predict_proba(cx)[:,1].reshape(n,15)
    pc=model['closed'].predict_proba(cx)[:,1].reshape(n,15)
    pc/=np.maximum(pc.sum(1,keepdims=True),1e-9)
    return {'pn':pn,'raw':raw,'pc':pc,'pb':(pn+np.clip(1-raw.sum(1),0,1))/2}

def through_stack(s,models):
    for i,model in enumerate(models):
        r=predict(model,s)
        if i<len(models)-1: s=b.augment(s,r)
    return s,r
