"""Check residual bounds, baseline identity and truth-free inference features."""
import unittest
import numpy as np
import pandas as pd
from scipy.special import softmax
from coverage_heads import BoundedResidual,evidence,adjust,predictions

class CoverageTests(unittest.TestCase):
    def test_residuals_learn_and_remain_bounded(self):
        rng=np.random.default_rng(17)
        f=rng.normal(size=(100,15,3)); y=f[:,:,0].argmax(1)
        head=BoundedResidual('rank').fit(f,np.zeros((100,15)),y,np.ones(100))
        self.assertLess(head.final_loss,head.initial_loss)
        self.assertLessEqual(np.abs(head.correction(f*1000)).max(),1.)
        self.assertGreater((head.correction(f).argmax(1)==y).mean(),.8)
        u=rng.normal(size=(100,4)); truth=(u[:,0]>0).astype(int)
        uh=BoundedResidual('unknown').fit(u,np.zeros(100),truth,np.ones(100))
        self.assertLess(uh.final_loss,uh.initial_loss)
        self.assertLessEqual(np.abs(uh.correction(u*1000)).max(),np.log(2))

    def test_zero_strength_preserves_exact_baseline(self):
        rng=np.random.default_rng(18)
        pc=softmax(rng.normal(size=(8,15)),axis=1); pb=rng.uniform(size=8)
        bundle={'pc':pc,'pb':pb,'candidates':np.tile([f'id{x}' for x in range(15)],(8,1))}
        a,z=adjust(bundle)
        self.assertIs(a,pc); self.assertIs(z,pb)
        self.assertTrue(all(len(set(row))==5 for row in predictions(bundle,a,z)))

    def test_feature_extraction_ignores_training_truth_columns(self):
        rng=np.random.default_rng(19); n=8
        names=['cnt','n_flank','n_view','sx','support','mxr','mxl','cl','nbev','r_nbev','r_sbev']
        c=pd.DataFrame({name:rng.uniform(.1,3,size=n*15) for name in names})
        q=pd.DataFrame({'view':[0,1,2,3,4,5,-1,8],'qq_minus_g':rng.normal(size=n),'qs_minus_s':rng.normal(size=n),'sp':np.arange(n)%2})
        r={'pc':softmax(rng.normal(size=(n,15)),axis=1),'pn':rng.uniform(size=n),'raw':rng.uniform(size=(n,15))}
        a,u=evidence({'q':q,'c':c},r)
        c['y']=rng.integers(0,2,size=n*15); q['unk']=rng.integers(0,2,size=n)
        b,v=evidence({'q':q,'c':c},r)
        np.testing.assert_array_equal(a,b); np.testing.assert_array_equal(u,v)

if __name__=='__main__': unittest.main()
