"""Model 5: Open-Set Score Fusion (OSSF), based on Dalvin Model 3b.

Run --help. Validation is identity-disjoint from every supervised training stage.
"""
import argparse
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from threading import Lock

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import dalvin_core as b

MODEL_NAME = "Model 5: Open-Set Score Fusion (OSSF)"
MODEL_VERSION = "1.0"
NEW = "new_individual"


def find_sift(root, split):
    """Kaggle mounts SIFT beside the metadata folder, sometimes as extracted NPZ directories."""
    name = f"sift_features_{split}.npz"
    for parent in (root, root.parent, root.parent.parent):
        candidate = parent / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find {name} in {root} or its two parent directories. "
                            "Attach the full projectwild3 competition input, including both SIFT folders.")


class SiftSource:
    def __init__(self, path):
        self.archive = None
        self.files = {}
        if path.is_file():
            self.archive = np.load(path)
        else:
            for file in path.rglob("*.npy"):
                if file.stem in self.files:
                    raise ValueError(f"Duplicate SIFT key {file.stem} in {path}")
                self.files[file.stem] = file
            if not self.files:
                raise ValueError(f"No extracted .npy SIFT arrays in {path}")

    def __getitem__(self, key):
        return self.archive[key] if self.archive is not None else np.load(self.files[key], allow_pickle=False)


def load_data(root):
    train = pd.read_csv(root / "train.csv")
    test = pd.read_csv(root / "test.csv")
    ids = pd.read_csv(root / "miewid_ids.csv")
    all_df = pd.concat([train, test.assign(identity=None)], ignore_index=True)
    if not all_df.id.is_unique or not ids.id.is_unique:
        raise ValueError("Duplicate image IDs")
    for col in ("species", "viewpoint", "identity"):
        if col not in all_df:
            raise ValueError(f"Missing column: {col}")
    if train.identity.isna().any() or not set(all_df.species) <= {"hyena", "leopard"}:
        raise ValueError("Invalid identity/species metadata")
    raw = np.load(root / "miewid_features.npy", mmap_mode="r")
    if len(raw) != len(ids) or set(ids.id) != set(all_df.id):
        raise ValueError("Feature ID mapping does not match metadata")
    ix = pd.Index(ids.id).get_indexer(all_df.id)
    X = np.asarray(raw[ix], dtype=np.float32)
    if not np.isfinite(X).all() or np.any(np.linalg.norm(X, axis=1) == 0):
        raise ValueError("Features contain nonfinite values or zero vectors")
    b.X = b.unit(X)
    b.IDS = all_df.id.to_numpy()
    b.Y = all_df.identity.to_numpy(dtype=object)
    b.SP = all_df.species.to_numpy()
    b.VIEW = all_df.viewpoint.map({v: i for i, v in enumerate(b.VIEWS)}).fillna(-1).to_numpy(int)
    return train, test, all_df


class SiftCache:
    """Role-local neighbour selection; thread workers share lazily loaded descriptors."""
    def __init__(self, root, out, workers, neighbours):
        paths = [find_sift(root, s) for s in ("train", "test")]
        print("SIFT sources:", *paths, sep="\n", flush=True)
        self.z = [SiftSource(path) for path in paths]
        self.read_lock = Lock()
        self.workers, self.neighbours = workers, neighbours
        fingerprint = hashlib.sha256(b.IDS.tobytes() if b.IDS.dtype != object else "\n".join(b.IDS).encode())
        fingerprint.update(np.ascontiguousarray(b.X).tobytes())
        self.path = out / ("sift_" + fingerprint.hexdigest()[:16] + ".npz")
        self.done = {}
        if self.path.exists():
            with np.load(self.path) as z:
                self.done = {tuple(p): tuple(v) for p, v in zip(z["pairs"], z["values"])}

    @lru_cache(maxsize=1200)
    def descriptors(self, i):
        iid = b.IDS[i]
        z = self.z[0 if iid.startswith("tr_") else 1]
        # NpzFile's shared ZIP handle is not safe for concurrent reads.
        with self.read_lock:
            d = np.asarray(z[f"{iid}_desc"], dtype=np.float32)
            xy = np.asarray(z[f"{iid}_xy"], dtype=np.float32)
        if d.ndim != 2 or xy.shape != (len(d), 2):
            raise ValueError(f"Malformed SIFT arrays for {iid}")
        d = np.sqrt(d / np.maximum(d.sum(1, keepdims=True), 1))
        return d, xy

    def verify(self, pair):
        a, c = pair
        da, xa = self.descriptors(a)
        dc, xc = self.descriptors(c)
        if min(len(da), len(dc)) < 4:
            return 0, 0
        matches = cv2.BFMatcher().knnMatch(da, dc, k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < .8*m[1].distance]
        if len(good) < 4:
            return len(good), 0
        # Local RNG seed per task. OpenCV RNG state is thread-local.
        cv2.setRNGSeed(int((int(a)*100003 + int(c)) % (2**31-1)))
        _, mask = cv2.estimateAffinePartial2D(
            np.float32([xa[m.queryIdx] for m in good]),
            np.float32([xc[m.trainIdx] for m in good]),
            method=cv2.RANSAC, ransacReprojThreshold=10.)
        return len(good), int(mask.sum()) if mask is not None else 0

    def activate(self, indices):
        wanted = set()
        for sp in ("hyena", "leopard"):
            ix = indices[b.SP[indices] == sp]
            # Chunked similarities avoid allocating the full role x role matrix.
            for start in range(0, len(ix), 256):
                q = ix[start:start+256]
                s = b.X[q] @ b.X[ix].T
                s[np.arange(len(q)), np.arange(start, start+len(q))] = -np.inf
                k = min(self.neighbours, len(ix)-1)
                nb = np.argsort(-s, axis=1)[:, :k]
                for a, row in zip(q, nb):
                    wanted.update((min(int(a), int(c)), max(int(a), int(c))) for c in ix[row])
        missing = sorted(wanted - self.done.keys())
        print(f"SIFT: {len(wanted)} role-local pairs; {len(missing)} uncached", flush=True)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for j, (pair, value) in enumerate(zip(missing, pool.map(self.verify, missing)), 1):
                self.done[pair] = value
                if j % 10000 == 0:
                    print(f"  verified {j}/{len(missing)}", flush=True)
        pairs = sorted(self.done)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, pairs=np.array(pairs, dtype=np.int64).reshape(-1, 2),
                            values=np.array([self.done[p] for p in pairs]).reshape(-1, 2))
        os.replace(tmp, self.path)
        # Only expose requested pairs, even when cache contains other roles/test.
        pp = np.array(sorted(wanted), dtype=int).reshape(-1, 2)
        vv = np.array([self.done[tuple(p)] for p in pp]).reshape(-1, 2)
        n = len(b.X)
        def matrix(v):
            a, c = pp.T
            return sparse.coo_matrix((np.r_[v, v], (np.r_[a, c], np.r_[c, a])), shape=(n, n)).tocsr()
        b.GOOD, b.INL = matrix(vv[:, 0]), matrix(vv[:, 1])


def allocate_roles(train, rounds, seed):
    rng = np.random.default_rng(seed)
    roles = {f"stage{i+1}": [] for i in range(rounds)} | {"calibration": [], "validation": []}
    for sp in ("hyena", "leopard"):
        names = np.array(sorted(train.loc[train.species == sp, "identity"].unique()))
        rng.shuffle(names)
        n = max(1, int(round(.2*len(names))))
        roles["validation"].extend(names[:n])
        roles["calibration"].extend(names[n:2*n])
        for i, block in enumerate(np.array_split(names[2*n:], rounds)):
            roles[f"stage{i+1}"].extend(block)
    sets = [set(v) for v in roles.values()]
    assert all(not a & c for i, a in enumerate(sets) for c in sets[i+1:])
    return {k: np.flatnonzero(train.identity.isin(v).to_numpy()) for k, v in roles.items()}


def split_episode(indices, seed, cap=32, link=(.95, 100)):
    """Unknown identities absent from gallery; known-query duplicate groups kept whole."""
    rng = np.random.default_rng(seed)
    gallery, known, unknown = [], [], []
    for sp in ("hyena", "leopard"):
        ix = indices[b.SP[indices] == sp]
        names = np.unique(b.Y[ix])
        rng.shuffle(names)
        nu = max(1, int(round(rng.uniform(.2, .35)*len(names))))
        unseen = set(names[:nu])
        for name in names:
            rows = ix[b.Y[ix] == name].copy()
            rng.shuffle(rows)
            if name in unseen:
                unknown.extend(rows[:cap])
                continue
            held = []
            groups = b.duplicate_groups(rows, link)
            target = min(max(1, int(round(.35*len(rows)))), cap)
            for c in rng.permutation(np.unique(groups)):
                members = rows[groups == c].tolist()
                if len(held) < target and len(held)+len(members) < len(rows):
                    held.extend(members)
            # Never truncate a duplicate group after selecting it.
            known.extend(held)
            gallery.extend(i for i in rows if i not in set(held))
    gi, qi = np.array(gallery, dtype=int), np.array(known+unknown, dtype=int)
    if not len(known) or not len(unknown):
        raise ValueError("Episode lacks known/unknown queries; use a larger role or different seed")
    for sp in ("hyena", "leopard"):
        if len(np.unique(b.Y[gi[b.SP[gi] == sp]])) < b.K or np.sum(b.SP[qi] == sp) < 6:
            raise ValueError("Too few identities/queries per species; reduce --rounds or --candidates")
    assert not set(b.Y[unknown]) & set(b.Y[gi])
    assert set(b.Y[known]) <= set(b.Y[gi])
    return gi, qi, np.where(np.isin(qi, unknown), NEW, b.Y[qi])


def frame(gi, qi, truth=None):
    s = b.build_frames(gi, qi)
    if truth is not None:
        lookup = dict(zip(qi, truth))
        s["truth"] = np.array([lookup[i] for i in s["order"]])
        s["q"]["unk"] = (s["truth"] == NEW).astype(int)
        s["c"]["y"] = (s["c"].ident.to_numpy() == s["truth"][s["c"].q.to_numpy()]).astype(int)
    return s


def through_stack(s, models):
    for i, m in enumerate(models):
        r = b.predict(m, s)
        if i < len(models)-1:
            s = b.augment(s, r)
    return s, r


def fusion_features(s, r):
    """Only inference-available evidence; no identity strings, labels or image indices."""
    q = s["q"]
    pc = r["pc"]
    top = pc.argmax(1)
    rows = np.arange(len(q))
    c = s["c"].iloc[rows*b.K+top]
    ordered = -np.sort(-pc, axis=1)
    return pd.DataFrame({
        "detector": r["pn"], "legacy_new": r["pb"],
        "raw_sum": r["raw"].sum(1), "raw_max": r["raw"].max(1),
        "pc_max": ordered[:, 0], "pc_gap": ordered[:, 0]-ordered[:, 1],
        "pc_entropy": -(pc*np.log(np.maximum(pc, 1e-12))).sum(1),
        "raw_similarity": c.mxr.to_numpy(), "lda_similarity": c.mxl.to_numpy(),
        "inliers": np.log1p(c.sx.to_numpy()), "sift_support": np.log1p(c.support.to_numpy()),
        "same_view_similarity": c.same.to_numpy(), "same_view_available": c.same.notna().to_numpy().astype(int),
        "gallery_support": np.log1p(c.cnt.to_numpy()), "hyena": q.sp.to_numpy(),
        "raw_margin": q.rm12.to_numpy(), "lda_margin": q.lm12.to_numpy(),
    })


def probabilities(s, r, fusion):
    return {"dalvin_rule": r["pb"], "detector_only": r["pn"],
            "abhay_fusion": fusion.predict_proba(fusion_features(s, r))[:, 1]}


def evaluate(s, r, probs, all_df, output):
    meta = all_df.iloc[s["order"]].reset_index(drop=True)
    unknown = s["truth"] == NEW
    result = pd.DataFrame({"id": meta.id, "identity": meta.identity, "species": meta.species,
                           "viewpoint": meta.viewpoint, "truth": s["truth"], "unknown": unknown})
    labels = s["c"].ident.to_numpy().reshape(-1, b.K)
    result["candidate_hit"] = (labels == s["truth"][:, None]).any(1)
    reports = {}
    for name, p in probs.items():
        preds = b.rank(s, dict(r, pb=p))
        score = np.array([next((1/(j+1) for j, v in enumerate(row) if v == t), 0.) for row, t in zip(preds, s["truth"])])
        result[f"{name}_p_new"] = p
        result[f"{name}_ap5"] = score
        result[f"{name}_predictions"] = [" ".join(row) for row in preds]
        reports[name] = {"map5": float(score.mean()), "known_map5": float(score[~unknown].mean()),
                         "unknown_map5": float(score[unknown].mean()),
                         "known_new_at1": float(np.mean(preds[~unknown, 0] == NEW)),
                         "unknown_new_at1": float(np.mean(preds[unknown, 0] == NEW))}
    result.to_csv(output / "validation_predictions.csv", index=False)
    slices = []
    for fields in (["species", "unknown"], ["viewpoint", "unknown"]):
        for key, group in result.groupby(fields, dropna=False):
            slices.append({"slice": str(dict(zip(fields, key))), "n": len(group),
                           **{name: float(group[f"{name}_ap5"].mean()) for name in probs}})
    pd.DataFrame(slices).to_csv(output / "validation_slices.csv", index=False)
    return {"n_queries": len(result), "unknown_fraction": float(unknown.mean()),
            "known_candidate_recall": float(result.loc[~unknown, "candidate_hit"].mean()),
            "methods": reports, "fusion_delta_map5": reports["abhay_fusion"]["map5"]-reports["dalvin_rule"]["map5"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("abhay_output"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--rounds", type=int, choices=[1, 2, 3], default=3)
    ap.add_argument("--sims", type=int, default=12, help="simulations per identity stage")
    ap.add_argument("--calibration-sims", type=int, default=4)
    ap.add_argument("--neighbours", type=int, default=150)
    ap.add_argument("--candidates", type=int, default=15)
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--predict-test", action="store_true", help="after validation, also write all three submissions")
    args = ap.parse_args()
    if min(args.sims, args.calibration_sims, args.neighbours, args.workers, args.trees) < 1 or args.candidates < 5:
        ap.error("counts must be positive and --candidates >= 5")
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["model"] = MODEL_NAME
    config["model_version"] = MODEL_VERSION
    config["python"] = platform.python_version()
    config["packages"] = {n: version(n) for n in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "joblib")}
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    np.random.seed(args.seed)
    cv2.setNumThreads(1)
    b.K = args.candidates
    b.PARAMS.update(n_estimators=args.trees, n_jobs=args.workers, random_state=args.seed)
    train, test, all_df = load_data(args.data)
    roles = allocate_roles(train, args.rounds, args.seed)
    manifest = []
    for name, ix in roles.items():
        manifest.extend({"id": b.IDS[i], "identity": b.Y[i], "species": b.SP[i], "role": name} for i in ix)
    pd.DataFrame(manifest).to_csv(args.output / "identity_roles.csv", index=False)
    cache = SiftCache(args.data, args.output, args.workers, args.neighbours)
    models = []
    split_rows = []
    def episode(ix, seed, role, link):
        gi, qi, truth = split_episode(ix, seed, link=link)
        truth_map = dict(zip(qi, truth))
        gallery_set = set(gi)
        split_rows.extend({"episode": seed, "role": role, "id": b.IDS[i],
                           "assignment": "gallery" if i in gallery_set else "unknown_query" if truth_map[i] == NEW else "known_query"}
                          for i in np.r_[gi, qi])
        return frame(gi, qi, truth)
    for stage in range(args.rounds):
        name = f"stage{stage+1}"
        ix = roles[name]
        cache.activate(ix)
        sims = []
        for j in range(args.sims):
            s = episode(ix, args.seed+1000*(stage+1)+j, name, b.REGIMES[j % len(b.REGIMES)])
            for model in models:
                r = b.predict(model, s)
                s = b.augment(s, r)
            sims.append(s)
        print(f"Fitting {name}: {len(ix)} images, {len(sims)} simulations", flush=True)
        models.append(b.fit(sims))
    cache.activate(roles["calibration"])
    cx, cy, cw = [], [], []
    for j in range(args.calibration_sims):
        s = episode(roles["calibration"], args.seed+10000+j, "calibration", (.95, 100))
        s, r = through_stack(s, models)
        cx.append(fusion_features(s, r))
        cy.append(s["truth"] == NEW)
        cw.extend(b.IDS[s["order"]])
    Xc, yc = pd.concat(cx, ignore_index=True), np.concatenate(cy).astype(int)
    # An image can recur in calibration episodes: total weight per image is one.
    counts = pd.Series(cw).value_counts()
    weights = np.array([1/counts[i] for i in cw])
    fusion = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                           LogisticRegression(C=1., max_iter=3000, random_state=args.seed))
    fusion.fit(Xc, yc, logisticregression__sample_weight=weights)
    feature_names = fusion[0].get_feature_names_out(Xc.columns)
    pd.DataFrame({"feature": feature_names, "standardized_coefficient": fusion[-1].coef_[0]}).to_csv(
        args.output / "fusion_coefficients.csv", index=False)
    pd.DataFrame({"id": cw, "unknown": yc, "sample_weight": weights}).to_csv(args.output / "calibration_examples.csv", index=False)
    # Save frozen models before final validation. Validation cannot affect fitting.
    joblib.dump({"models": models, "fusion": fusion, "config": config, "fusion_columns": list(Xc)}, args.output / "models.joblib")
    cache.activate(roles["validation"])
    s = episode(roles["validation"], args.seed+20000, "validation", (.95, 100))
    s, r = through_stack(s, models)
    report = evaluate(s, r, probabilities(s, r, fusion), all_df, args.output)
    report["config"] = config
    report["protocol"] = "identity-disjoint sequential stacking stages, calibration, outer validation"
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(split_rows).to_csv(args.output / "episode_splits.csv", index=False)
    print(json.dumps(report, indent=2), flush=True)
    if args.predict_test:
        # Evaluation is complete; only now expose competition test queries.
        cache.activate(np.arange(len(all_df)))
        s, r = through_stack(frame(np.arange(len(train)), np.arange(len(train), len(all_df))), models)
        sample = pd.read_csv(args.data / "sample_submission.csv")
        for name, p in probabilities(s, r, fusion).items():
            preds = b.rank(s, dict(r, pb=p))
            sub = pd.DataFrame({"id": b.IDS[s["order"]], "predictions": [" ".join(v) for v in preds]})
            sub = sample[["id"]].merge(sub, on="id", how="left", validate="one_to_one")
            if len(sub) != len(test) or set(sub.id) != set(test.id) or sub.predictions.isna().any():
                raise ValueError("Submission IDs missing or invalid")
            known = train.groupby("species").identity.apply(set).to_dict()
            species = test.set_index("id").species
            for row in sub.itertuples():
                tokens = row.predictions.split()
                assert len(tokens) == len(set(tokens)) == 5
                assert all(t == NEW or t in known[species[row.id]] for t in tokens)
            sub.to_csv(args.output / f"submission_{name}.csv", index=False)


if __name__ == "__main__":
    main()
