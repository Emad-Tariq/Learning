"""Safety and learning checks for the 16-choice open-set ranker."""
import copy
import unittest

import numpy as np
import pandas as pd

import joint_ranker as jr


def example(n=60, seed=3):
    rng = np.random.default_rng(seed)
    unknown = np.arange(n) % 3 == 0
    truth = np.where(unknown, jr.NEW, "animal_0")
    evidence = rng.normal(0, .03, (n, 15))
    evidence[~unknown, 0] += 1
    return dict(
        q=pd.DataFrame({"r_pn": unknown.astype(float), "view": np.zeros(n),
                        "nb0": np.arange(n), "nw0": np.ones(n), "unk": unknown.astype(int)}),
        c=pd.DataFrame({"q": np.repeat(np.arange(n), 15),
                        "c": np.tile(np.arange(15), n),
                        "ident": np.tile([f"animal_{i}" for i in range(15)], n),
                        "evidence": evidence.ravel(),
                        "y": (np.tile(np.arange(15), n) == 0).astype(int)}),
        truth=truth, order=np.arange(n),
    )


class JointRankerTest(unittest.TestCase):
    def test_truth_and_index_metadata_cannot_change_features(self):
        frame = example(6)
        first, labels = jr.design(frame)
        changed = copy.deepcopy(frame)
        changed["truth"][:] = "poison"
        changed["q"]["unk"] = 98
        changed["q"]["nb0"] = 95
        changed["q"]["nw0"] = 96
        changed["c"]["y"] = 97
        changed["c"]["c"] = 94
        second, again = jr.design(changed)
        pd.testing.assert_frame_equal(first, second)
        np.testing.assert_array_equal(labels, again)
        self.assertEqual(first.shape, (96, 4))
        self.assertTrue(first.iloc[15::16]["c__evidence"].isna().all())
        self.assertTrue((first.iloc[15::16]["is_new"] == 1).all())

    def test_malformed_groups_and_schema_fail_loudly(self):
        frame = example(6)
        bad = copy.deepcopy(frame)
        bad["c"] = bad["c"].iloc[1:]
        with self.assertRaises(ValueError):
            jr.design(bad)
        bad = copy.deepcopy(frame)
        bad["c"].loc[1, "ident"] = "animal_0"
        with self.assertRaises(ValueError):
            jr.design(bad)
        with self.assertRaises(ValueError):
            jr.design(frame, ["view"], ["evidence"], 15)

    def test_known_and_unknown_learning_and_missing_candidate_accounting(self):
        frame = example()
        frame["truth"][1] = "not_shortlisted"
        model = jr.fit([frame, copy.deepcopy(frame)], dict(
            n_estimators=45, min_child_samples=2, num_leaves=7,
            learning_rate=.15, reg_lambda=.1, reg_alpha=0., n_jobs=1,
            colsample_bytree=1.))
        self.assertEqual(model.metadata["omitted_missing_true_candidate"], 2)
        self.assertEqual(model.metadata["used_groups"], 118)
        self.assertEqual(model.metadata["unique_training_photos"], 59)
        unseen = example(24, seed=55)
        unseen.pop("truth")
        unseen["q"] = unseen["q"].drop(columns="unk")
        unseen["c"] = unseen["c"].drop(columns="y")
        pred, scores = jr.predict(model, unseen)
        target = np.where(np.arange(24) % 3 == 0, jr.NEW, "animal_0")
        self.assertGreater(np.mean(pred[:, 0] == target), .95)
        self.assertEqual(scores.shape, (24, 16))
        self.assertTrue(all(len(set(row)) == 5 for row in pred))
        shuffled = copy.deepcopy(unseen)
        shuffled["c"] = shuffled["c"].sample(frac=1, random_state=12)
        other, _ = jr.predict(model, shuffled)
        np.testing.assert_array_equal(other[:, 0], pred[:, 0])


if __name__ == "__main__":
    unittest.main()
