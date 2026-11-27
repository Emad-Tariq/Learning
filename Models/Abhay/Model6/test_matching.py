"""Small end-to-end and split-integrity checks for Model 6."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import matching


class MatchingTest(unittest.TestCase):
    def test_pair_order_and_identity_isolation(self):
        rng = np.random.default_rng(8)
        meta = pd.DataFrame({"id": range(6), "species": ["hyena"] * 6,
                             "identity": ["a", "a", "b", "b", "c", "c"],
                             "viewpoint": ["left", "right", "front", "left", "right", "front"]})
        x = matching.unit(rng.normal(size=(6, 12)).astype(np.float32))
        z = matching.unit(rng.normal(size=(6, 5)).astype(np.float32))
        a = np.array([0, 1, 2]); b = np.array([1, 3, 5])
        np.testing.assert_array_equal(matching.pair_features(meta, x, z, a, b),
                                      matching.pair_features(meta, x, z, b, a))

    def test_end_to_end(self):
        rng = np.random.default_rng(2026)
        rows, vectors = [], []
        for species in ("hyena", "leopard"):
            for identity_number in range(18):
                center = matching.unit(rng.normal(size=(1, 16)).astype(np.float32))[0]
                for sighting in range(4):
                    image_id = f"{species}_{identity_number}_{sighting}"
                    rows.append((image_id, species, f"{species}_{identity_number}",
                                 "left" if sighting % 2 else "right"))
                    vectors.append(center + rng.normal(0, .12, 16))
        meta = pd.DataFrame(rows, columns=["id", "species", "identity", "viewpoint"])
        vectors = np.asarray(vectors, dtype=np.float32)
        permutation = rng.permutation(len(meta))
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data"; output = Path(temporary) / "output"
            data.mkdir()
            meta.to_csv(data / "train.csv", index=False)
            pd.DataFrame({"id": meta.id.iloc[permutation]}).to_csv(data / "miewid_ids.csv", index=False)
            np.save(data / "miewid_features.npy", vectors[permutation])
            subprocess.run([sys.executable, str(Path(matching.__file__)), "--data", str(data),
                            "--output", str(output), "--dimensions", "8", "--max-iter", "4"],
                           check=True, capture_output=True, text=True)
            roles = pd.read_csv(output / "identity_roles.csv")
            self.assertEqual(roles.groupby("identity").role.nunique().max(), 1)
            metrics = json.loads((output / "metrics.json").read_text())
            for role in ("development", "evaluation"):
                split = pd.read_csv(output / f"{role}_split.csv")
                self.assertEqual(split.groupby("duplicate_group").assignment.nunique().max(), 1)
                self.assertGreater(metrics[role]["queries"], 0)
                self.assertEqual(metrics[role]["queries"],
                                 len(pd.read_csv(output / f"{role}_predictions.csv")))
                for method in ("cosine", "lda", "pair_mlp"):
                    self.assertTrue(0 <= metrics[role]["scores"][method]["map5"] <= 1)


if __name__ == "__main__":
    unittest.main()
