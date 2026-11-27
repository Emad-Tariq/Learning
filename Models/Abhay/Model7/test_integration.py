"""Verify baseline feature parity and supervised identity separation."""
import ast
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import dalvin_core as b
import paired_eval
import data_protocol


class IntegrationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        b.X = b.unit(rng.normal(size=(100, 24)).astype(np.float32))
        b.NX = b.X
        b.Y = np.array([f'id{i // 4}' for i in range(100)])
        b.SP = np.array(['hyena'] * 100, dtype=object)
        b.VIEW = np.arange(100) % 10
        b.INL = sparse.csr_matrix((100, 100))
        b.GOOD = b.INL.copy()
        b.K = 15

    def test_baseline_species_features_match_original(self):
        reference = Path(__file__).with_name('baseline_species_features.py')
        namespace = dict(vars(b))
        exec(compile(reference.read_text(), str(reference), 'exec'), namespace)
        g = np.r_[np.arange(0, 100, 4), np.arange(1, 100, 4)]
        q = np.arange(2, 100, 4)
        actual = b.species_features(g, q, 'hyena')
        expected = namespace['species_features'](g, q, 'hyena')
        for a, e in zip(actual, expected):
            pd.testing.assert_frame_equal(a, e)

    def test_nuisance_fit_never_reads_other_identity_labels(self):
        b.SP[50:] = 'leopard'
        fit = np.r_[np.arange(0, 40), np.arange(60, 100)]
        first, directions = paired_eval.nuisance_features(fit, seed=11)
        b.Y[40:60] = 'unused-label'
        second, again = paired_eval.nuisance_features(fit, seed=11)
        np.testing.assert_allclose(first, second)
        for species in directions:
            np.testing.assert_allclose(directions[species], again[species])
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1, atol=1e-6)

    def test_sift_checkpoint_preserves_completed_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache = object.__new__(data_protocol.SiftCache)
            cache.path = Path(temporary) / 'pairs.npz'
            cache.done = {(1, 3): (6, 4), (2, 5): (12, 10)}
            cache.save()
            with np.load(cache.path) as saved:
                recovered = {tuple(p): tuple(v) for p, v in zip(saved['pairs'], saved['values'])}
            self.assertEqual(recovered, cache.done)
            self.assertFalse(cache.path.with_suffix('.tmp.npz').exists())


if __name__ == '__main__':
    unittest.main()
