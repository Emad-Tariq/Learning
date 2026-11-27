"""Generate competition predictions from the saved Model 11 checkpoints."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'tmp/abhay_deps'))
import argparse,json,hashlib,warnings
import time
from functools import lru_cache
import numpy as np
import pandas as pd
import joblib,cv2
from threadpoolctl import threadpool_limits
import dalvin_core as b
import data_protocol as p
import verifier as v
from concurrent.futures import ThreadPoolExecutor
from scipy import sparse


class SubmissionSiftCache(p.SiftCache):
    """Keep original neighbour selection; skip edges between two gallery photos."""
    def verify(self,pair):
        a,c=pair;da,xa=self.descriptors(a);dc,xc=self.descriptors(c)
        if min(len(da),len(dc))<4:return 0,0
        bf=cv2.BFMatcher()
        matches=bf.knnMatch(da,dc,k=2)
        ratio=[m[0] for m in matches if len(m)==2 and m[0].distance<.85*m[1].distance]
        if not ratio:return 0,0
        back=bf.match(dc[np.array([m.trainIdx for m in ratio])],da)
        good=[m for m,reverse in zip(ratio,back) if reverse.trainIdx==m.queryIdx]
        if len(good)<4:return len(good),0
        cv2.setRNGSeed(int((int(a)*100003+int(c))%(2**31-1)))
        _,mask=cv2.estimateAffinePartial2D(np.float32([xa[m.queryIdx] for m in good]),
            np.float32([xc[m.trainIdx] for m in good]),method=cv2.RANSAC,ransacReprojThreshold=15.)
        return len(good),int(mask.sum()) if mask is not None else 0
    def activate(self,indices):
        activation_start=time.monotonic()
        wanted=set()
        for species in ('hyena','leopard'):
            ix=indices[b.SP[indices]==species]
            for start in range(0,len(ix),256):
                queries=ix[start:start+256]
                sim=b.X[queries]@b.X[ix].T
                sim[np.arange(len(queries)),np.arange(start,start+len(queries))]=-np.inf
                nb=np.argsort(-sim,axis=1)[:,:min(self.neighbours,len(ix)-1)]
                for a,row in zip(queries,nb):
                    for c in ix[row]:
                        if int(a) in self.query_indices or int(c) in self.query_indices:
                            wanted.add(tuple(sorted((int(a),int(c)))))
        missing=sorted(wanted-self.done.keys())
        print('Submission SIFT:',len(wanted),'required pairs;',len(missing),'uncached; gallery/gallery checks skipped',flush=True)
        started=time.monotonic();last_save=started
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            # Executor.map eagerly queues its entire iterable on Python <=3.13.
            # Bound that queue to avoid hundreds of thousands of live Futures.
            for start in range(0,len(missing),1024):
                chunk=missing[start:start+1024]
                for key,value in zip(chunk,pool.map(self.verify,chunk)):
                    self.done[key]=value
                done=start+len(chunk);elapsed=time.monotonic()-started
                print('Verified',done,'/',len(missing),'rate',round(done/max(elapsed,1),1),'pairs/s',flush=True)
                if time.monotonic()-last_save>=60:
                    self.save();last_save=time.monotonic()
        self.save()
        keys=np.array(sorted(wanted),dtype=int).reshape(-1,2)
        values=np.array([self.done[tuple(k)] for k in keys]).reshape(-1,2)
        def matrix(column):
            a,c=keys.T;v=values[:,column]
            return sparse.coo_matrix((np.r_[v,v],(np.r_[a,c],np.r_[c,a])),shape=(len(b.X),len(b.X))).tocsr()
        b.GOOD,b.INL=matrix(0),matrix(1)
        self.performance=dict(required_pairs=len(wanted),new_pairs=len(missing),
            reused_pairs=len(wanted)-len(missing),seconds=time.monotonic()-activation_start)


def submission_frame(sample,test,train,order_ids,pred):
    if not sample.id.is_unique or set(sample.id)!=set(test.id):
        raise ValueError('Sample submission IDs do not match test.csv')
    frame=pd.DataFrame({'id':order_ids,'predictions':[' '.join(row) for row in pred]})
    if not frame.id.is_unique or set(frame.id)!=set(test.id):
        raise ValueError('Predictions do not cover every test ID exactly once')
    result=sample[['id']].merge(frame,on='id',how='left',validate='one_to_one')
    species=test.set_index('id').species
    allowed=train.groupby('species').identity.apply(set).to_dict()
    for row in result.itertuples(index=False):
        labels=row.predictions.split()
        if len(labels)!=5 or len(set(labels))!=5:raise ValueError('Need five distinct labels')
        if any(label!='new_individual' and label not in allowed[species[row.id]] for label in labels):
            raise ValueError('Invalid identity or cross-species prediction')
    assert result.id.tolist()==sample.id.tolist()
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--sample',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--descriptor-cache',type=int,default=384)
    args=ap.parse_args();run=args.run
    config=json.loads((run/'config.json').read_text())
    for name in ('verifier.py','dalvin_core.py','data_protocol.py'):
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=config['sources'][name]:
            raise ValueError(f'{name} changed since training; restore the training source snapshot')
    models=joblib.load(run/'baseline_models.joblib')
    if len(models)!=3:raise ValueError('Baseline training has not completed all three stages')
    network=joblib.load(run/'verifier.joblib')
    calibrators=joblib.load(run/'calibrators.joblib')
    cv2.setNumThreads(1)
    warnings.filterwarnings('ignore',message='Only one sample available.*',category=UserWarning)
    train,test,meta=p.load_data(args.data);b.NX=b.X
    sample=pd.read_csv(args.sample)
    if not sample.id.is_unique or set(sample.id)!=set(test.id):raise ValueError('Wrong sample submission file')
    # All labelled photos are now gallery; every test photo is an unlabelled query.
    # Role-local caches are reselected for this full population by activate().
    cache=SubmissionSiftCache(args.data,run,args.workers,150)
    original_descriptors=cache.descriptors.__wrapped__
    cache.descriptors=lru_cache(maxsize=args.descriptor_cache)(lambda i:original_descriptors(cache,i))
    cache.query_indices=set(range(len(train),len(meta)))
    print('Preparing full-gallery SIFT. This can take substantially longer than validation.',flush=True)
    cache.activate(np.arange(len(meta)))
    sift_performance=cache.performance
    # The verified matrices now hold all inference evidence. Descriptor arrays
    # and the Python pair dictionary are not needed for PCA/LDA or MLP inference.
    cache.descriptors.cache_clear()
    del cache
    import gc
    gc.collect()
    gi=np.arange(len(train));qi=np.arange(len(train),len(meta))
    score_start=time.monotonic()
    s=p.frame(gi,qi);s,r=p.through_stack(s,models)
    baseline_seconds=time.monotonic()-score_start
    print('Computing verifier evidence against full labelled gallery',flush=True)
    score_start=time.monotonic()
    _,features=v.query_features(network,b.X,b.Y,gi,s,r)
    pnew=.5*(r['pb']+calibrators['mlp'].predict_proba(features)[:,1])
    verifier_seconds=time.monotonic()-score_start
    candidates=s['c'].ident.to_numpy().reshape(-1,15)
    for name,score in [('mlp_blend',pnew),('baseline',r['pb'])]:
        pred=v.rank(candidates,r['pc'],score)
        result=submission_frame(sample,test,train,b.IDS[s['order']],pred)
        dest=run/f'submission_{name}.csv'
        result.to_csv(dest,index=False)
        print('WROTE',dest,'rows',len(result),flush=True)
    (run/'submission_info.json').write_text(json.dumps(dict(
        gallery_photos=len(train),test_queries=len(test),method='fixed 50/50 MLP blend',
        checkpoints='Frozen baseline, verifier and calibrators from validation experiment; no refitting',
        sift_performance=sift_performance,baseline_seconds=baseline_seconds,verifier_seconds=verifier_seconds,
        workers=args.workers,descriptor_cache=args.descriptor_cache,
        caveat='Full-gallery distribution differs from calibration/validation. No leaderboard improvement is established.'),indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=4,user_api='blas'):main()
