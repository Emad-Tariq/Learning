"""Check newly computed cross-role fitting matches with preserved OpenCV."""
from run import *
from threading import Lock

def main():
    train,_,_=p.load_data(DATA);cv2.setNumThreads(1)
    out=HERE/'local_runs/large_2052'
    roles=pd.read_csv(out/'identity_roles.csv').set_index('id').role
    role=np.array([roles.get(i,'test') for i in b.IDS])
    with np.load(next(out.glob('sift_mutual_*.npz'))) as z:pairs=z['pairs'];vals=z['values']
    a,c=pairs.T
    good=(role[a]!=role[c])&~np.isin(role[a],['validation','test'])&~np.isin(role[c],['validation','test'])
    eligible=np.flatnonzero(good);assert len(eligible)>0
    rng=np.random.default_rng(12052)
    pick=rng.choice(eligible,min(200,len(eligible)),replace=False)
    strong=eligible[vals[eligible,1]>=20]
    if len(strong):pick=np.unique(np.r_[pick,rng.choice(strong,min(50,len(strong)),replace=False)])
    cache=p.SiftCache.__new__(p.SiftCache)
    cache.z=[p.SiftSource(p.find_sift(DATA,s)) for s in ['train','test']]
    cache.read_lock=Lock();errors=[]
    for i in pick:
        pair=tuple(map(int,pairs[i]));actual=tuple(map(int,vals[i]));expected=p.SiftCache.verify(cache,pair)
        if actual!=expected:errors.append(dict(pair=pair,cached=actual,opencv=expected))
    report=dict(cross_role_training_pairs_checked=len(pick),mismatches=errors,seed=12052)
    (out/'independent_matching_checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));assert not errors

if __name__=='__main__':
    with threadpool_limits(limits=1):main()
