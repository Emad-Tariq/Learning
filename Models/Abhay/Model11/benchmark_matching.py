"""Benchmark a distance-reuse matcher; do not change production automatically."""
import sys,time,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'tmp/abhay_deps'))
import numpy as np,pandas as pd,cv2
from threadpoolctl import threadpool_limits
from functools import lru_cache
from threading import Lock
import data_protocol as p
import dalvin_core as b


def reuse_verify(cache,pair):
    a,c=pair;da,xa=cache.descriptors(a);dc,xc=cache.descriptors(c)
    if min(len(da),len(dc))<4:return 0,0
    dist=np.maximum((da*da).sum(1)[:,None]+(dc*dc).sum(1)[None,:]-2*(da@dc.T),0)
    back=dist.argmin(0);best=dist.argmin(1);rows=np.arange(len(da))
    closest=dist[rows,best].copy();dist[rows,best]=np.inf
    second=dist.min(1)
    # Preserve OpenCV behavior near floating-point ties/ratio boundaries.
    if np.any(np.abs(closest-(.85**2)*second)<1e-4):
        return p.SiftCache.verify(cache,pair)
    good=rows[(closest<(.85**2)*second)&(back[best]==rows)]
    if len(good):
        dist[rows,best]=closest
        reverse_two=np.partition(dist[:,best[good]],1,axis=0)[:2]
        if np.any(reverse_two[1]-reverse_two[0]<1e-4):return p.SiftCache.verify(cache,pair)
    if len(good)<4:return len(good),0
    cv2.setRNGSeed(int((int(a)*100003+int(c))%(2**31-1)))
    _,mask=cv2.estimateAffinePartial2D(xa[good],xc[best[good]],method=cv2.RANSAC,ransacReprojThreshold=15.)
    return len(good),int(mask.sum()) if mask is not None else 0


def subset_verify(cache,pair):
    """Same OpenCV distances; reverse-check only matches passing the ratio test."""
    a,c=pair;da,xa=cache.descriptors(a);dc,xc=cache.descriptors(c)
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


def main():
    cv2.setNumThreads(1);data=Path.home()/'Downloads/reid_student_data_main'
    train=pd.read_csv(data/'train.csv');test=pd.read_csv(data/'test.csv');b.IDS=pd.concat([train,test]).id.to_numpy()
    cache=p.SiftCache.__new__(p.SiftCache);cache.z=[p.SiftSource(p.find_sift(data,s)) for s in ('train','test')];cache.read_lock=Lock()
    raw=p.SiftCache.descriptors.__wrapped__;cache.descriptors=lru_cache(maxsize=200)(lambda i:raw(cache,i))
    source=HERE/'local_runs/pc_retrain_20261126/sift_mutual_665bbfe949f3eed7.npz'
    with np.load(source) as z:keys=z['pairs'];values=z['values']
    eligible=np.flatnonzero(keys[:,1]>=len(train));rng=np.random.default_rng(42)
    picked=rng.choice(eligible,min(5000,len(eligible)),replace=False)
    keys=keys[picked];values=values[picked]
    with threadpool_limits(limits=1,user_api='blas'):
        start=time.monotonic();computed=[subset_verify(cache,tuple(k)) for k in keys];elapsed=time.monotonic()-start
    mismatches=[dict(pair=k.tolist(),baseline=v.tolist(),reuse=list(c)) for k,v,c in zip(keys,values,computed) if tuple(v)!=tuple(c)]
    result=dict(pairs=len(keys),seconds=elapsed,pairs_per_second=len(keys)/elapsed,mismatches=mismatches)
    (HERE/'local_runs/pc_retrain_20261126/matching_benchmark.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
