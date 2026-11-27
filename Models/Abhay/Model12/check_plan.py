"""Audit real-data role boundaries and deliberately reject leaking episodes."""
from run import *

def main():
    train,_,_=p.load_data(DATA)
    roles=p.allocate_roles(train,3,2052)
    source=pd.read_csv(SOURCE/'episode_splits.csv')
    previous=set();counts=[]
    for stage in range(1,4):
        bg=backgrounds(roles,stage)
        context=set(b.Y[np.r_[roles[f'stage{stage}'],bg]])
        all_queries=set()
        for j in range(12):
            ep=2052+1000*stage+j;g,q,t=saved_episode(source,ep)
            gg=np.r_[g,bg];safety(gg,q,t,roles,stage)
            assert not set(b.Y[q])&previous
            all_queries.update(b.Y[q])
            assert set(gg)-set(g)==set(bg)
            for name in np.unique(b.Y[q]):
                assert np.array_equal(g[b.Y[g]==name],gg[b.Y[gg]==name])
            # A genuine unknown image in the gallery must be rejected.
            bad=np.r_[gg,q[t==p.NEW][0]]
            try:safety(bad,q,t,roles,stage)
            except AssertionError:pass
            else:raise AssertionError('Unknown/query contamination was accepted')
            # A validation reference must also be rejected during fitting.
            bad=np.r_[gg,roles['validation'][0]]
            try:safety(bad,q,t,roles,stage)
            except AssertionError:pass
            else:raise AssertionError('Validation contamination was accepted')
        assert not all_queries&previous
        previous|=context
        counts.append(dict(stage=stage,context_animals=len(context),
            background_animals=len(set(b.Y[bg])),query_animals=len(all_queries)))
    print(json.dumps(dict(all_checks_passed=True,episodes=36,stages=counts),indent=2))

if __name__=='__main__':main()
