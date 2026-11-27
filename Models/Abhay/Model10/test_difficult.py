"""Episode proportions, label conservation, side thinning and unknown hardness."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tmp/abhay_deps')); sys.path.append(str(HERE.parent/'Model7'))
import unittest
import numpy as np
import dalvin_core as b
import data_protocol as protocol
from difficult_protocol import difficult_episode,hard_positions,slices

class DifficultTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(51)
        b.X=b.unit(rng.normal(size=(72,16)).astype(np.float32))
        b.IDS=np.array([f'image{i}' for i in range(72)])
        b.Y=np.repeat([f'animal{i}' for i in range(12)],6)
        b.SP=np.repeat(['hyena','leopard'],36)
        b.VIEW=np.tile([0,1,2,3,0,1],12)
        self.g=np.concatenate([np.arange(i*6,i*6+4) for i in [0,1,2,3,6,7,8,9]])
        self.q=np.r_[np.concatenate([np.arange(i*6+4,i*6+6) for i in [0,1,2,3,6,7,8,9]]),
                     np.concatenate([np.arange(i*6,i*6+2) for i in [4,5,10,11]])]
        self.truth=np.where(np.isin(b.Y[self.q],b.Y[self.g]),b.Y[self.q],protocol.NEW)

    def test_proportions_and_nested_schedules(self):
        for stage in range(3):
            a=hard_positions(stage,.25); c=hard_positions(stage,.5)
            self.assertEqual(len(a),3); self.assertEqual(len(c),6); self.assertTrue(a<c)
            self.assertTrue(all(0<=j<12 for j in c))

    def test_preserves_labels_and_no_new_gallery_query_overlap(self):
        g,q,t,diag=difficult_episode(np.arange(72),self.g,self.q,self.truth,42)
        self.assertEqual(set(b.Y[g]),set(b.Y[self.g]))
        self.assertTrue(set(g)<=set(self.g)); self.assertFalse(set(g)&set(q))
        self.assertEqual(len(q),len(self.q))
        self.assertEqual(int((t==protocol.NEW).sum()),int((self.truth==protocol.NEW).sum()))
        self.assertEqual(set(q[t!=protocol.NEW]),set(self.q[self.truth!=protocol.NEW]))
        self.assertFalse(set(b.Y[q[t==protocol.NEW]])&set(b.Y[g]))
        self.assertGreater(diag['identities_side_thinned'],0)
        before=slices(self.g,self.q,self.truth); after=slices(g,q,t)
        self.assertGreater(sum(r['missing_side'] for r in after),sum(r['missing_side'] for r in before))
        self.assertTrue(any(0<r['true_gallery_images']<=3 for r in after if not r['unknown']))
        # Every thinned known gallery is a subset, so encounter groups cannot newly cross.
        for ident in np.unique(b.Y[self.g]):
            self.assertTrue(set(g[b.Y[g]==ident])<=set(self.g[b.Y[self.g]==ident]))

    def test_unknown_queries_are_at_least_as_convincing_and_deterministic(self):
        g,q,t,_=difficult_episode(np.arange(72),self.g,self.q,self.truth,42)
        again=difficult_episode(np.arange(72),self.g,self.q,self.truth,42)
        for a,c in zip((g,q,t),again[:3]): np.testing.assert_array_equal(a,c)
        for name in np.unique(b.Y[self.q[self.truth==protocol.NEW]]):
            old=self.q[(self.truth==protocol.NEW)&(b.Y[self.q]==name)]
            new=q[(t==protocol.NEW)&(b.Y[q]==name)]
            gallery=g[b.SP[g]==b.SP[old[0]]]
            self.assertEqual(len(old),len(new))
            self.assertGreaterEqual(float((b.X[new]@b.X[gallery].T).max(1).mean())+1e-7,
                                    float((b.X[old]@b.X[gallery].T).max(1).mean()))

if __name__=='__main__': unittest.main()
