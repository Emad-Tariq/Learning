"""Candidate coverage must be observable and must not reach candidate scorers."""
from test_difficult import *
import pandas as pd
import coverage_features as coverage

class CoverageFeatureTests(unittest.TestCase):
    def test_feature_truth_invariance_and_original_feature_parity(self):
        rng=np.random.default_rng(73); n=8
        q=pd.DataFrame({'view':[0,1,2,3,4,5,6,-1],'unk':np.arange(n)%2,'sp':np.arange(n)%2})
        c=pd.DataFrame({k:rng.uniform(.1,4,n*15) for k in ['cnt','n_flank','n_view','mxr','sx','support']})
        c['y']=rng.integers(0,2,len(c)); s={'q':q,'c':c}
        a=coverage.add_coverage(s)
        q['unk']=1-q.unk; c['y']=1-c.y
        z=coverage.add_coverage(s)
        names=[name for name in a['q'] if name.startswith('cov_')]
        self.assertEqual(len(names),23)
        pd.testing.assert_frame_equal(a['q'][names],z['q'][names])
        pd.testing.assert_frame_equal(coverage.strip_coverage(a)['q'],s['q'].assign(unk=1-s['q'].unk))
        self.assertFalse(any(name.startswith('cov_') for name in coverage.strip_coverage(a)['q']))

if __name__=='__main__': unittest.main()
