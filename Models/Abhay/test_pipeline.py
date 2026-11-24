"""Synthetic integration and leakage checks; no competition data required."""
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import train_eval as p


def fixture(root):
    rng = np.random.default_rng(50)
    train, test, vectors, sift = [], [], [], [{}, {}]
    for sp in ("hyena", "leopard"):
        for ident in range(40):
            centre = rng.normal(size=16)
            for j in range(6):
                iid = f"tr_{len(train)+1:05}"
                train.append(dict(id=iid, species=sp, viewpoint=["left", "right"][j % 2], identity=f"{sp}_{ident}"))
                vectors.append(centre + rng.normal(size=16))
                sift[0][iid+"_desc"] = rng.integers(0, 255, (8, 128), dtype=np.uint8)
                sift[0][iid+"_xy"] = rng.uniform(0, 512, (8, 2)).astype(np.float32)
    for sp in ("hyena", "leopard"):
        for j in range(8):
            iid = f"te_{len(test)+1:05}"
            test.append(dict(id=iid, species=sp, viewpoint="left"))
            vectors.append(rng.normal(size=16))
            sift[1][iid+"_desc"] = np.empty((0, 128), dtype=np.uint8)
            sift[1][iid+"_xy"] = np.empty((0, 2), dtype=np.float32)
    pd.DataFrame(train).to_csv(root/"train.csv", index=False)
    pd.DataFrame(test).to_csv(root/"test.csv", index=False)
    ids = [r["id"] for r in train+test]
    # Deliberately permute feature rows to exercise ID alignment.
    perm = rng.permutation(len(ids))
    pd.DataFrame({"id": np.array(ids)[perm]}).to_csv(root/"miewid_ids.csv", index=False)
    np.save(root/"miewid_features.npy", np.float32(vectors)[perm])
    for i, split in enumerate(("train", "test")):
        np.savez(root/f"sift_features_{split}.npz", **sift[i])
    pd.DataFrame({"id": [r["id"] for r in test], "predictions": p.NEW}).to_csv(root/"sample_submission.csv", index=False)


class PipelineTests(unittest.TestCase):
    def test_end_to_end_three_rounds(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            # Match the actual Kaggle mount: core files in a child directory,
            # SIFT siblings exposed as directories named *.npz.
            data = root / "reid_student_data_main"
            data.mkdir()
            for split in ("train", "test"):
                archive = root / f"sift_features_{split}.npz"
                with zipfile.ZipFile(archive) as z:
                    arrays = {name: z.read(name) for name in z.namelist()}
                archive.unlink()
                archive.mkdir()
                for name, blob in arrays.items():
                    (archive / name).write_bytes(blob)
            for file in list(root.iterdir()):
                if file.is_file():
                    file.rename(data / file.name)
            out = root/"output"
            proc = subprocess.run([sys.executable, str(Path(p.__file__)), "--data", str(data), "--output", str(out),
                            "--rounds", "3", "--sims", "1", "--calibration-sims", "2", "--neighbours", "3",
                            "--candidates", "5", "--trees", "8", "--workers", "2", "--predict-test"],
                           check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            roles = pd.read_csv(out/"identity_roles.csv")
            self.assertEqual(roles.groupby("identity").role.nunique().max(), 1)
            splits = pd.read_csv(out/"episode_splits.csv").merge(roles[["id", "identity"]], on="id")
            for _, group in splits.groupby(["episode", "role"]):
                gallery = set(group.loc[group.assignment == "gallery", "identity"])
                self.assertFalse(gallery & set(group.loc[group.assignment == "unknown_query", "identity"]))
                self.assertTrue(set(group.loc[group.assignment == "known_query", "identity"]) <= gallery)
            metrics = json.loads((out/"metrics.json").read_text())
            for m in metrics["methods"].values():
                self.assertTrue(0 <= m["map5"] <= 1)
            val = pd.read_csv(out/"validation_predictions.csv")
            # All three methods preserve the exact same relative known-identity ranking.
            for row in val.itertuples():
                a = [v for v in row.dalvin_rule_predictions.split() if v != p.NEW]
                c = [v for v in row.abhay_fusion_predictions.split() if v != p.NEW]
                self.assertEqual(a[:min(len(a), len(c))], c[:min(len(a), len(c))])
            for method in metrics["methods"]:
                sub = pd.read_csv(out/f"submission_{method}.csv")
                self.assertEqual(len(sub), 16)
                self.assertTrue(sub.id.is_unique)
                self.assertTrue(sub.predictions.str.split().map(lambda x: len(x) == len(set(x)) == 5).all())
            self.assertTrue((out/"models.joblib").exists())

    def test_compressed_sift_and_sibling_discovery(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            archive = root / "sift_features_train.npz"
            source = p.SiftSource(archive)
            self.assertEqual(source["tr_00001_desc"].shape, (8, 128))
            source.archive.close()
            child = root / "metadata"
            child.mkdir()
            self.assertEqual(p.find_sift(child, "train"), archive)

    def test_id_mapping_and_no_label_features(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture(root)
            train, test, all_df = p.load_data(root)
            raw = np.load(root/"miewid_features.npy")
            ids = pd.read_csv(root/"miewid_ids.csv")
            row = int(np.flatnonzero(ids.id == train.id.iloc[0])[0])
            np.testing.assert_allclose(p.b.X[0], p.b.unit(raw[[row]])[0], atol=1e-6)
            self.assertTrue(pd.isna(p.b.Y[len(train):]).all())


if __name__ == "__main__":
    unittest.main()
