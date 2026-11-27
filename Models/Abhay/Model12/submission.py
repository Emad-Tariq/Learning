"""Portable Model 12 inference and strict competition CSV validation."""
import hashlib,json,time,os
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import sparse
import dalvin_core as b
import data_protocol as p

NEW='new_individual'

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def canonical(frame,columns):
    rows=frame[columns].fillna('').astype(str).values.tolist()
    return hashlib.sha256(json.dumps(rows,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()

def fingerprint(root):
    root=Path(root)
    return dict(train=canonical(pd.read_csv(root/'train.csv'),['id','identity','species','viewpoint']),
        test=canonical(pd.read_csv(root/'test.csv'),['id','species','viewpoint']),
        feature_ids=canonical(pd.read_csv(root/'miewid_ids.csv'),['id']),
        feature_array=sha(root/'miewid_features.npy'))

class NativeHead:
    def __init__(self,path,workers):
        self.model=lgb.Booster(model_file=str(path));self.workers=workers
    def predict_proba(self,frame):
        prob=np.asarray(self.model.predict(frame,num_threads=self.workers,validate_features=True))
        if prob.ndim!=1 or not np.isfinite(prob).all():raise ValueError('Invalid binary model output')
        return np.column_stack([1-prob,prob])

def load_models(assets,workers=4):
    assets=Path(assets);schema=json.loads((assets/'model_schema.json').read_text())
    models=[]
    for stage in schema['stages']:
        heads=[NativeHead(assets/name,workers) for name in stage['heads']]
        models.append(tuple(heads)+((stage['query_features'],stage['candidate_features']),))
    if len(models)!=3:raise ValueError('Expected the complete three-stage Model 12')
    return models

def prepare(root,assets,sample_path=None):
    root=Path(root);assets=Path(assets)
    expected=json.loads((assets/'input_fingerprint.json').read_text())
    actual=fingerprint(root)
    if actual!=expected:
        changed=[k for k in expected if actual.get(k)!=expected[k]]
        raise ValueError(f'Competition input differs from the trained-model/SIFT bundle: {changed}. Attach the original competition data.')
    train,test,all_df=p.load_data(root);b.NX=b.X
    sample=pd.read_csv(sample_path or root/'sample_submission.csv',dtype=str,keep_default_na=False)
    if list(sample.columns)!=['id','predictions']:raise ValueError('Expected sample_submission columns: id,predictions')
    if not sample.id.is_unique or set(sample.id)!=set(test.id):raise ValueError('Sample IDs do not exactly match test IDs')
    with np.load(assets/'full_test_sift_graph.npz',allow_pickle=False) as z:
        pairs=z['pairs'];good=z['good'];inliers=z['inliers'];n=int(z['image_count'])
    if n!=len(all_df) or pairs.ndim!=2 or pairs.shape[1]!=2:raise ValueError('Incorrect SIFT graph dimensions')
    if not (len(pairs)==len(good)==len(inliers)):raise ValueError('Incomplete SIFT graph')
    if not ((pairs[:,0]>=0)&(pairs[:,0]<pairs[:,1])&(pairs[:,1]<n)).all():raise ValueError('Invalid SIFT graph indices')
    if not (pairs[:,1]>=len(train)).all():raise ValueError('Graph contains unrelated training-only edges')
    a,c=pairs.T
    def matrix(v):return sparse.coo_matrix((np.r_[v,v],(np.r_[a,c],np.r_[c,a])),shape=(n,n)).tocsr()
    b.GOOD=matrix(good);b.INL=matrix(inliers)
    gi=np.arange(len(train));qi=np.arange(len(train),len(all_df))
    print('Preparing full-gallery features:',len(train),'reference photos,',len(test),'test photos',flush=True)
    frame=p.frame(gi,qi)
    return dict(train=train,test=test,sample=sample,frame=frame)

def validate_submission(submission,sample,test,train):
    if list(submission.columns)!=list(sample.columns) or list(sample.columns)!=['id','predictions']:
        raise ValueError('Submission headers must be exactly id,predictions, with no index column')
    if len(submission)!=len(sample) or len(submission)!=len(test):raise ValueError('Incorrect submission row count')
    if submission.isna().any().any():raise ValueError('Submission contains missing values')
    if not submission.id.is_unique:raise ValueError('Duplicate submission IDs')
    if submission.id.tolist()!=sample.id.tolist():raise ValueError('Submission IDs must follow the exact sample order')
    if set(submission.id)!=set(test.id):raise ValueError('Missing or unexpected test IDs')
    species=test.set_index('id').species
    catalog=train.groupby('species').identity.apply(set).to_dict()
    for iid,value in zip(submission.id,submission.predictions):
        if not isinstance(value,str):raise ValueError(f'{iid}: predictions must be a string')
        labels=value.split()
        if len(labels)!=5 or len(set(labels))!=5:raise ValueError(f'{iid}: expected exactly five distinct labels')
        if value!=' '.join(labels):raise ValueError(f'{iid}: use single spaces between prediction labels')
        allowed=catalog[species.loc[iid]]|{NEW}
        invalid=set(labels)-allowed
        if invalid:raise ValueError(f'{iid}: unknown or wrong-species identity labels: {invalid}')
    return dict(format_valid=True,rows=len(submission),columns=list(submission.columns),
        unique_ids=True,exact_sample_order=True,labels_per_row=5,distinct_labels=True,
        labels_valid_for_species=True,index_column_absent=True)

def generate(context,models,output):
    start=time.monotonic();frame=context['frame']
    s,r=p.through_stack(frame,models);pred=b.rank(s,r)
    predictions=pd.DataFrame({'id':b.IDS[s['order']],'predictions':[' '.join(row) for row in pred]})
    submission=context['sample'][['id']].merge(predictions,on='id',how='left',sort=False,validate='one_to_one')
    checks=validate_submission(submission,context['sample'],context['test'],context['train'])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.stem+'.checked.tmp.csv')
    submission.to_csv(temporary,index=False,encoding='utf-8',lineterminator='\n')
    reread=pd.read_csv(temporary,dtype=str,keep_default_na=False)
    validate_submission(reread,context['sample'],context['test'],context['train'])
    os.replace(temporary,output)
    checks.update(csv_sha256=sha(output),prediction_seconds=time.monotonic()-start,
        new_individual_first=int(submission.predictions.str.split().str[0].eq(NEW).sum()))
    output.with_suffix('.validation.json').write_text(json.dumps(checks,indent=2))
    print('Saved and re-read validated submission:',output,flush=True)
    print(json.dumps(checks,indent=2),flush=True)
    return submission,checks
