"""Validate baseline interface and photo weighting for regularized heads."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

# The experiment bootstrap exposes the preserved Model7 modules and bundled
# LightGBM dependency for direct invocation from the project root.
import experiment
import regularized_head as rh
from test_joint_ranker import example


class RegularizedHeadTest(unittest.TestCase):
    def frame(self, n=30, seed=3):
        frame = example(n, seed)
        frame["c"]["y"] = (frame["c"].ident.to_numpy() == frame["truth"][frame["c"].q.to_numpy()]).astype(int)
        return frame

    def test_training_matches_predict_interface_without_mutating_defaults(self):
        frame = self.frame()
        before = copy.deepcopy(rh.b.PARAMS)
        model = rh.fit([frame], dict(n_estimators=30, num_leaves=7,
                       min_child_samples=2, learning_rate=.15, n_jobs=1))
        self.assertEqual(before, rh.b.PARAMS)
        result = rh.b.predict(model, self.frame(24, 50))
        self.assertEqual(result["pc"].shape, (24, 15))
        np.testing.assert_allclose(result["pc"].sum(1), 1.)
        pred = rh.b.rank(self.frame(24, 50), result)
        target = np.where(np.arange(24) % 3 == 0, 'new_individual', 'animal_0')
        self.assertGreater((pred[:, 0] == target).mean(), .95)

    def test_repeated_photo_weights_propagate_to_candidate_rows(self):
        frame = self.frame(6)
        second = self.frame(6, 9)
        second["order"] = np.array([0, 1, 2, 20, 21, 22])
        captures = []
        class Recorder:
            def __init__(self, **params): self.params = params
            def fit(self, x, y, sample_weight):
                captures.append((len(x), np.asarray(y), np.asarray(sample_weight)))
                return self
        with patch.object(rh.lgb, 'LGBMClassifier', Recorder):
            model = rh.fit([frame, second])
        expected = np.array([.5, .5, .5, 1, 1, 1, .5, .5, .5, 1, 1, 1])
        expected /= expected.mean()
        np.testing.assert_allclose(captures[0][2], expected)
        np.testing.assert_allclose(captures[2][2], np.repeat(expected, 15))
        self.assertEqual(model[0].params['n_estimators'], 300)
        self.assertEqual(model[0].params['num_leaves'], 15)
        self.assertEqual(model[0].photo_weighting_metadata_['unique_photos'], 9)


if __name__ == '__main__':
    unittest.main()
