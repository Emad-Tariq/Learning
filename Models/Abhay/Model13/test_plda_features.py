"""Check exact PLDA likelihood algebra against direct Gaussian covariances."""
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

import plda_features as p


class PLDATest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(67)
        labels = np.repeat(["a", "b", "c", "d", "e"], [3, 4, 6, 5, 4])
        means = rng.normal(size=(5, 4)) * 3.
        _, codes = np.unique(labels, return_inverse=True)
        self.gallery = means[codes] + rng.normal(size=(len(labels), 4)) * .6
        self.labels = labels
        self.queries = rng.normal(size=(3, 4))
        self.model = p.fit(self.gallery, labels, components=4)

    def test_pair_scores_equal_full_joint_gaussian_llr(self):
        m = self.model
        w, b = m.within_covariance, m.between_covariance
        total = w + b
        joint = np.block([[total, b], [b, total]])
        q = m.pca.transform(self.queries) - m.mean
        g = m.pca.transform(self.gallery[:5]) - m.mean
        expected = np.empty((len(q), len(g)))
        for i, qi in enumerate(q):
            for j, gj in enumerate(g):
                expected[i, j] = (multivariate_normal.logpdf(np.r_[qi, gj], cov=joint)
                                  - multivariate_normal.logpdf(qi, cov=total)
                                  - multivariate_normal.logpdf(gj, cov=total))
        assert_allclose(m.pair_scores(self.queries, self.gallery[:5]), expected, atol=1e-8)
        assert_allclose(m.projection.T @ w @ m.projection, np.eye(4), atol=1e-10)

    def test_predictive_scores_equal_full_conditional_gaussian(self):
        m = self.model
        w, b = m.within_covariance, m.between_covariance
        total = w + b
        q = m.pca.transform(self.queries) - m.mean
        z = m.pca.transform(self.gallery) - m.mean
        expected = np.empty((len(q), len(m.identities)))
        for j, identity in enumerate(m.identities):
            block = z[self.labels == identity]
            # The sample centroid is sufficient and has noise W / n.
            cross = b @ np.linalg.inv(b + w / len(block))
            posterior_mean = cross @ block.mean(0)
            posterior_cov = total - cross @ b
            for i, qi in enumerate(q):
                expected[i, j] = (multivariate_normal.logpdf(qi, mean=posterior_mean, cov=posterior_cov)
                                  - multivariate_normal.logpdf(qi, cov=total))
        assert_allclose(m.predictive_scores(self.queries), expected, atol=1e-8)

    def test_candidate_alignment_and_gallery_only_fit(self):
        candidates = np.array([["c", "a"], ["b", "d"], ["e", "c"]])
        cf, qf = p.candidate_features(self.model, self.queries, candidates)
        self.assertEqual(qf.shape, (3, 10))
        self.assertEqual(cf.shape, (6, 9))
        scores = self.model.identity_scores(self.queries)
        for i, row in enumerate(candidates):
            for k, identity in enumerate(row):
                j = np.where(self.model.identities == identity)[0][0]
                self.assertEqual(cf.iloc[i * 2 + k].pl_predictive, scores["predictive"][i, j])
        before = self.model.projection.copy()
        self.model.pair_scores(self.queries * 100)
        assert_allclose(before, self.model.projection, atol=0)
        with self.assertRaises(ValueError):
            p.candidate_features(self.model, self.queries, np.full((3, 2), "absent"))

    def test_sample_limited_and_degenerate_between_covariance(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(10, 20))
        model = p.fit(x, np.repeat(["a", "b"], 5), components=256)
        self.assertEqual(model.metadata["pca_dimensions"], 9)
        self.assertTrue(np.isfinite(model.pair_scores(x[:2])).all())
        self.assertTrue(np.isfinite(model.predictive_scores(x[:2])).all())
        with self.assertRaises(ValueError):
            p.fit(x, np.arange(10))

    def test_randomized_pca_stored_gallery_matches_explicit_transform(self):
        rng = np.random.default_rng(73)
        # This shape selects sklearn's randomized PCA under the auto policy:
        # both dimensions exceed 500 and retained components are much smaller.
        gallery = rng.normal(size=(540, 520))
        labels = np.repeat(np.arange(30), 18)
        model = p.fit(gallery, labels, components=12)
        self.assertEqual(model.pca._fit_svd_solver, 'randomized')
        assert_allclose(model.gallery, model.transform(gallery), atol=1e-12)
        query = rng.normal(size=(3, 520))
        assert_allclose(model.pair_scores(query),
                        model.pair_scores(query, gallery), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
