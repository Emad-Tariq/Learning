import unittest
from unittest.mock import patch
import numpy as np
import predict_submission as m


class SubmissionOptimizationTests(unittest.TestCase):
    def test_reverse_subset_matches_original_including_ambiguous_descriptors(self):
        rng=np.random.default_rng(12)
        raw=rng.random((30,128),dtype=np.float32)
        da=np.sqrt(raw/raw.sum(1,keepdims=True));xy=rng.random((30,2),dtype=np.float32)*400
        order=rng.permutation(30);dc=da[order].copy();yc=(xy*1.1+np.array([8,12],np.float32))[order]
        cache=m.SubmissionSiftCache.__new__(m.SubmissionSiftCache)
        for duplicates in (False,True):
            if duplicates:dc[1]=dc[0]
            cache.descriptors=lambda i:(da,xy) if i==0 else (dc,yc)
            self.assertEqual(m.p.SiftCache.verify(cache,(0,1)),cache.verify((0,1)))

    def test_query_evidence_and_bounded_work_queue(self):
        rng=np.random.default_rng(12);m.b.X=m.b.unit(rng.normal(size=(80,8)))
        m.b.SP=np.array(['hyena']*80);batch_sizes=[]
        class Pool:
            def __init__(self,*a,**k):pass
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def map(self,fn,keys):batch_sizes.append(len(keys));return map(fn,keys)
        def create(cls):
            cache=cls.__new__(cls);cache.done={};cache.neighbours=30;cache.workers=2
            cache.save=lambda:None;cache.verify=lambda key:(sum(key)+3,sum(key)+1)
            return cache
        original=create(m.p.SiftCache);original.activate(np.arange(80));good=m.b.GOOD.copy();inl=m.b.INL.copy()
        optimized=create(m.SubmissionSiftCache);optimized.query_indices=set(range(60))
        with patch.object(m,'ThreadPoolExecutor',Pool):optimized.activate(np.arange(80))
        np.testing.assert_array_equal(good[:60].toarray(),m.b.GOOD[:60].toarray())
        np.testing.assert_array_equal(inl[:60].toarray(),m.b.INL[:60].toarray())
        self.assertGreater(len(batch_sizes),1);self.assertLessEqual(max(batch_sizes),1024)
        self.assertLess(len(optimized.done),len(original.done))


if __name__=='__main__':unittest.main()
