"""Meaningful checks for sparse-zero provenance and label-free supplementation."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import check_coverage as e


class CoverageTests(unittest.TestCase):
    def test_checked_zero_is_not_an_unchecked_pair(self):
        matrix = e.sparse.coo_matrix(([0., 9.], ([0, 0], [1, 2])), shape=(3, 3)).tocsr()
        checked = e.checked_matrix(matrix)
        self.assertEqual(checked[0, 1], 1)
        self.assertEqual(checked[0, 2], 1)
        self.assertEqual(checked[1, 2], 0)

    def test_candidates_and_non_sift_evidence_stay_fixed_and_truth_cannot_select_pairs(self):
        b=e.b; rng=e.np.random.default_rng(11)
        n=72
        b.X=b.unit(rng.normal(size=(n, 8))); b.NX=b.X.copy()
        b.IDS=e.np.array([f'photo{i}' for i in range(n)])
        b.SP=e.np.array(['hyena']*36+['leopard']*36)
        b.Y=e.np.array([f'{sp}{j//2}' for sp in ('h','l') for j in range(30)] + ['unused']*12, dtype=object)
        # Gallery occupies first 30 rows per species; six queries follow it.
        b.Y=e.np.array([f'{sp}{j//2}' if j<30 else f'{sp}{j-30}' for sp in ('h','l') for j in range(36)],dtype=object)
        b.VIEW=e.np.zeros(n,dtype=int)
        gi=e.np.r_[e.np.arange(30),e.np.arange(36,66)]
        qi=e.np.r_[e.np.arange(30,36),e.np.arange(66,72)]
        # One attempted but zero-inlier pair per query must be counted as checked.
        cols=e.np.where(qi<36,0,36)
        matrix=e.sparse.coo_matrix((e.np.zeros(24),(e.np.r_[qi,cols],e.np.r_[cols,qi])),shape=(n,n)).tocsr()
        ex=e.Experiment.__new__(e.Experiment)
        ex.workers=2; ex.roles={'validation':e.np.arange(n)}; ex.active_role='validation'
        ex.inl=matrix; ex.good=matrix.copy(); ex.checked=e.checked_matrix(matrix)
        ex.original=b.species_features; ex.original_embed=b.embed
        ex.audit=[]; ex.pair_audit=[]
        ex.cache=SimpleNamespace(done={},verify=lambda key:(12,8),save=lambda:None)
        e.fixed_function()
        def split(truth):
            with patch.object(e.p,'split_episode',return_value=(gi,qi,truth)):
                return ex.frames(1,None)
        with e.threadpool_limits(limits=1,user_api='blas'):
            base, changed, stats=split(b.Y[qi])
            base2, changed2, stats2=split(e.np.full(len(qi),e.p.NEW))
        self.assertGreater(stats['additional_pairs'],0)
        self.assertEqual(stats['additional_pairs'],stats2['additional_pairs'])
        self.assertEqual(stats2['newly_computed_pairs'],0)
        self.assertTrue(base['c'].ident.equals(changed['c'].ident))
        for col in changed['c']:
            if col != 'y':
                e.pd.testing.assert_series_equal(changed['c'][col],changed2['c'][col])
        for col in changed['q']:
            if col != 'unk':
                e.pd.testing.assert_series_equal(changed['q'][col],changed2['q'][col])
        checked_candidate=changed['c'][changed['c'].ident.isin(['h0','l0'])]
        self.assertTrue((checked_candidate.verified_photos==1).all())
        other=changed['c'][~changed['c'].ident.isin(['h0','l0'])]
        self.assertTrue((other.verified_photos==2).all())


if __name__=='__main__':
    unittest.main()
