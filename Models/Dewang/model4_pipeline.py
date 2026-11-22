#!/usr/bin/env python3
"""
MODEL 4 (Dewang): whitened MiewID retrieval + viewpoint-aware scoring
                  + learned open-set (new_individual) detector.
                  Optional: SIFT/RANSAC re-scoring (--use-sift) and
                  transductive test-neighbour feature (--transductive).

Pipeline
  1. Unsupervised PCA-whitening of the MiewID vectors (no labels used).
  2. Open-set validation splits (same logic/seeds as Emad's Model 2, so numbers
     are comparable). The splits are saved to Processed_Data/splits/.
  3. Tune (PCA size, viewpoint bonus a, viewpoint penalty b) on known queries
     with closed-set MAP@5.
  4. Identity score = best gallery-image cosine (+ viewpoint adjustment, + optional SIFT bonus).
  5. A per-species logistic-regression detector predicts P(new_individual) from
     similarity features. Candidate probabilities are (1-p)*softmax(score/T), and
     new_individual gets p, so new_individual is placed at the rank its probability
     implies (this is what MAP@5 rewards).
  6. Ablation table, then test inference + submission validation.

Run from anywhere:  python model4_pipeline.py            (defaults: project root = ../..)
                    python model4_pipeline.py --use-sift --transductive --tag full
"""
import argparse
import json
import os
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NEW = "new_individual"
VP_CODES = {"left": 0, "right": 1, "back": 2, "front": 3, "down": 4, "up": 5,
            "frontleft": 6, "frontright": 7, "backleft": 8, "backright": 9}
VP_SIDE = {"left": -1, "frontleft": -1, "backleft": -1,
           "right": 1, "frontright": 1, "backright": 1}


# ----------------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------------
def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.abspath(os.path.join(here, "..", "..")),
                    help="project root containing Processed_Data/ (default: two levels above this script)")
    ap.add_argument("--sample-submission", default=None,
                    help="path to sample_submission.csv (default: Processed_Data/sample_submission.csv if it exists)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 100, 2026])
    ap.add_argument("--unknown-prop", type=float, default=0.25)
    ap.add_argument("--k-pool", type=int, default=20, help="candidate identities per query")
    ap.add_argument("--k-sift", type=int, default=5, help="candidates verified with SIFT")
    ap.add_argument("--use-sift", action="store_true", help="add SIFT/RANSAC inlier bonus + detector features")
    ap.add_argument("--transductive", action="store_true",
                    help="final model uses query-to-query similarity feature (ablation is always printed)")
    ap.add_argument("--target-unknown", type=float, default=0.30,
                    help="assumed share of new_individual queries used to weight the detector and to pick "
                         "T / thresholds (the raw validation split has ~70%% because a few huge identities are held out)")
    ap.add_argument("--new-bias", type=float, default=0.0,
                    help="logit offset added to P(new) at test time (>0 predicts more new_individual)")
    ap.add_argument("--tag", default="base", help="suffix for output file names")
    ap.add_argument("--sift-mem", type=int, default=2000, help="number of images kept in RAM for SIFT")
    ap.add_argument("--no-test", action="store_true", help="validation only, skip test inference")
    return ap.parse_args()


# ----------------------------------------------------------------------------
# Open-set split  (copied from Emad's run_model2_pipeline.py so splits match)
# ----------------------------------------------------------------------------
def make_openset_split(df, unknown_prop=0.25, seed=42):
    rng = np.random.default_rng(seed)
    gallery_idx, known_val_idx, unknown_val_idx = [], [], []
    for sp in sorted(df["species"].unique()):
        sp_df = df[df["species"] == sp]
        unique_ids = np.sort(sp_df["identity"].unique())
        rng.shuffle(unique_ids)

        n_unknown = int(round(len(unique_ids) * unknown_prop))
        unknown_ids = set(unique_ids[:n_unknown])
        known_ids = set(unique_ids[n_unknown:])

        id_groups = sp_df.groupby("identity").groups
        for u_id in sorted(unknown_ids):
            unknown_val_idx.extend(id_groups[u_id].tolist())

        for k_id in sorted(known_ids):
            indices = np.sort(id_groups[k_id].to_numpy())
            rng.shuffle(indices)
            if len(indices) >= 2:
                n_val = max(1, int(round(len(indices) * 0.2)))
                n_val = min(n_val, len(indices) - 1)
                known_val_idx.extend(indices[:n_val])
                gallery_idx.extend(indices[n_val:])
            else:
                gallery_idx.extend(indices)

    return (
        df.loc[gallery_idx].reset_index(drop=True),
        df.loc[known_val_idx].reset_index(drop=True),
        df.loc[unknown_val_idx].reset_index(drop=True),
    )


# ----------------------------------------------------------------------------
# Features / spaces
# ----------------------------------------------------------------------------
def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def build_spaces(raw, pca_max=512, ks=(64, 128, 256, 512)):
    """Label-free embedding spaces: 'raw' (centred + L2) and PCA-whitened prefixes."""
    X = l2n(np.asarray(raw, dtype=np.float32))
    X = X - X.mean(0, keepdims=True)
    spaces = {"raw": l2n(X)}
    nc = min(pca_max, X.shape[0] - 1, X.shape[1] - 1)
    pca = PCA(n_components=nc, whiten=True, svd_solver="randomized", random_state=0).fit(X)
    Zw = pca.transform(X).astype(np.float32)
    for k in ks:
        if k <= nc:
            spaces[k] = l2n(Zw[:, :k])
    return spaces


class Gallery:
    """Gallery sorted by identity so per-identity max is a single reduceat."""

    def __init__(self, g_df):
        names, codes = np.unique(g_df["identity"].to_numpy(dtype=object), return_inverse=True)
        order = np.argsort(codes, kind="stable")
        sc = codes[order]
        self.names = names
        self.df = g_df.iloc[order].reset_index(drop=True)
        self.starts = np.flatnonzero(np.r_[True, sc[1:] != sc[:-1]])
        self.ends = np.r_[self.starts[1:], len(sc)]
        self.counts = self.ends - self.starts
        self.mi = self.df["miewid_index"].to_numpy()
        self.sp = (self.df["species"].to_numpy(dtype=object) == "leopard").astype(np.int8)
        self.vp = vp_codes(self.df)
        self.side = vp_side(self.df)
        self.ids = self.df["id"].to_numpy(dtype=object)


def vp_codes(df):
    return np.array([VP_CODES.get(v, -1) if isinstance(v, str) else -1 for v in df["viewpoint"]], dtype=np.int8)


def vp_side(df):
    return np.array([VP_SIDE.get(v, 0) if isinstance(v, str) else 0 for v in df["viewpoint"]], dtype=np.int8)


# ----------------------------------------------------------------------------
# Retrieval: candidate pools
# ----------------------------------------------------------------------------
def vp_matrices(q_vp, q_side, g_vp, g_side):
    same = ((q_vp[:, None] == g_vp[None, :]) & (q_vp[:, None] >= 0)).astype(np.float32)
    opp = ((q_side[:, None] * g_side[None, :]) == -1).astype(np.float32)
    return same, opp


def base_pool(q_df, gal, space, a, b, k_pool, k_sift, chunk=700):
    """Top-k_pool identities per query (same species only), viewpoint-adjusted."""
    n = len(q_df)
    Zq_all = space[q_df["miewid_index"].to_numpy()]
    Zg = space[gal.mi]
    q_sp = (q_df["species"].to_numpy(dtype=object) == "leopard").astype(np.int8)
    q_vp, q_side = vp_codes(q_df), vp_side(q_df)

    cand_idx = np.zeros((n, k_pool), dtype=np.int64)
    score = np.zeros((n, k_pool), dtype=np.float32)
    raw_c = np.zeros((n, k_pool), dtype=np.float32)
    raw_best = np.zeros(n, dtype=np.float32)
    best_img = np.zeros((n, k_sift), dtype=np.int64)

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        S = Zq_all[s:e] @ Zg.T
        same, opp = vp_matrices(q_vp[s:e], q_side[s:e], gal.vp, gal.side)
        A = S + a * same - b * opp
        bad = q_sp[s:e, None] != gal.sp[None, :]
        S[bad] = -9.0
        A[bad] = -9.0
        raw_id = np.maximum.reduceat(S, gal.starts, axis=1)
        adj_id = np.maximum.reduceat(A, gal.starts, axis=1)
        idx = np.argpartition(-adj_id, k_pool - 1, axis=1)[:, :k_pool]
        sc = np.take_along_axis(adj_id, idx, 1)
        o = np.argsort(-sc, axis=1)
        idx = np.take_along_axis(idx, o, 1)
        cand_idx[s:e] = idx
        score[s:e] = np.take_along_axis(sc, o, 1)
        raw_c[s:e] = np.take_along_axis(raw_id, idx, 1)
        raw_best[s:e] = raw_id.max(1)
        for i in range(e - s):
            for j in range(k_sift):
                c = idx[i, j]
                st, en = gal.starts[c], gal.ends[c]
                best_img[s + i, j] = st + int(A[i, st:en].argmax())

    # transductive feature: nearest *other* query of the same species
    qq_max = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        Q = Zq_all[s:e] @ Zq_all.T
        Q[q_sp[s:e, None] != q_sp[None, :]] = -9.0
        Q[np.arange(e - s), np.arange(s, e)] = -9.0
        qq_max[s:e] = Q.max(1)

    return dict(cand_idx=cand_idx, score=score, raw=raw_c, raw_best=raw_best,
                best_img=best_img, qq_max=qq_max, inl=None)


def finalize(pool, beta=0.0, cap=20.0):
    """Apply SIFT bonus (if inliers present) and re-sort candidates."""
    score = pool["score"].copy()
    inl_full = np.zeros_like(score)
    if pool["inl"] is not None:
        ks = pool["inl"].shape[1]
        inl_full[:, :ks] = pool["inl"]
        score[:, :ks] += beta * np.minimum(pool["inl"] / cap, 1.0)
    o = np.argsort(-score, axis=1, kind="stable")
    take = lambda A: np.take_along_axis(A, o, 1)
    return dict(cand_idx=take(pool["cand_idx"]), score=take(score),
                raw=take(pool["raw"]), inl=take(inl_full))


# ----------------------------------------------------------------------------
# SIFT / RANSAC (optional)
# ----------------------------------------------------------------------------
class SiftMatcher:
    def __init__(self, npz_path, cache_file, mem_items=2000):
        import cv2  # imported lazily so the script runs without OpenCV when --use-sift is off
        self.cv2 = cv2
        # sift_features.npz may be a real .npz file OR a folder of <key>.npy files (an unzipped npz)
        self.is_dir = os.path.isdir(npz_path)
        if self.is_dir:
            self.dir = npz_path
            self.data, self.keys = None, None
        else:
            self.data = np.load(npz_path)
            self.keys = set(self.data.files)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.cache_file = cache_file
        self.pairs = {}
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                self.pairs = json.load(f)
        self._load = lru_cache(maxsize=mem_items)(self._load_raw)
        self.dirty = 0

    def check(self, probe_id):
        """Fail early with a readable message if the SIFT layout is not what we expect."""
        if self.is_dir:
            if not os.path.exists(os.path.join(self.dir, f"{probe_id}_desc.npy")):
                raise RuntimeError(
                    f"{self.dir} is a folder but has no '{probe_id}_desc.npy'. First entries: "
                    f"{sorted(os.listdir(self.dir))[:8]}. Tell me what these look like and adapt SiftMatcher._array().")
        elif f"{probe_id}_desc" not in self.keys:
            raise RuntimeError(f"key '{probe_id}_desc' not in npz. First keys: {sorted(self.keys)[:8]}")

    def _array(self, key):
        """Return array for key like 'tr_00001_desc', or None if the image has no entry."""
        if self.is_dir:
            path = os.path.join(self.dir, key + ".npy")
            return np.load(path) if os.path.exists(path) else None
        return self.data[key] if key in self.keys else None

    def _load_raw(self, iid):
        d = self._array(f"{iid}_desc")
        xy = self._array(f"{iid}_xy")
        if d is None or xy is None:
            return np.zeros((0, 128), np.float32), np.zeros((0, 2), np.float32)
        d = np.asarray(d, dtype=np.float32).reshape(-1, 128)
        xy = np.asarray(xy, dtype=np.float32).reshape(len(d), -1)[:, :2]
        if len(d) and d.min() >= 0:  # RootSIFT
            d = np.sqrt(d / np.maximum(d.sum(1, keepdims=True), 1e-12))
        return np.ascontiguousarray(d), xy

    def inliers(self, qid, gid, ratio=0.75, thr=12.0):
        key = f"{qid}|{gid}"
        if key in self.pairs:
            return self.pairs[key]
        qd, qxy = self._load(qid)
        gd, gxy = self._load(gid)
        n_in = 0
        if len(qd) >= 4 and len(gd) >= 4:
            matches = self.bf.knnMatch(qd, gd, k=2)
            qp, gp = [], []
            for m in matches:
                if len(m) == 2 and m[0].distance < ratio * m[1].distance:
                    qp.append(qxy[m[0].queryIdx])
                    gp.append(gxy[m[0].trainIdx])
            if len(qp) >= 4:
                _, mask = self.cv2.estimateAffinePartial2D(
                    np.float32(qp), np.float32(gp), method=self.cv2.RANSAC, ransacReprojThreshold=thr)
                n_in = int(mask.sum()) if mask is not None else 0
        self.pairs[key] = n_in
        self.dirty += 1
        return n_in

    def save(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self.pairs, f)
        self.dirty = 0


def add_inliers(pool, q_df, gal, sm, label=""):
    n, ks = pool["best_img"].shape
    qids = q_df["id"].to_numpy(dtype=object)
    inl = np.zeros((n, ks), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        for j in range(ks):
            inl[i, j] = sm.inliers(qids[i], gal.ids[pool["best_img"][i, j]])
        if (i + 1) % 1000 == 0:
            print(f"    SIFT {label}: {i + 1}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    pool["inl"] = inl
    sm.save()


# ----------------------------------------------------------------------------
# Ranking + metric
# ----------------------------------------------------------------------------
def rank_with_prob(names, scores, p_new, T):
    s = np.asarray(scores, dtype=np.float64)
    w = np.exp((s - s.max()) / T)
    w /= w.sum()
    items = list(zip(names, (1.0 - p_new) * w)) + [(NEW, p_new)]
    items.sort(key=lambda x: -x[1])
    return [x[0] for x in items[:5]]


def rank_with_threshold(names, scores, th):
    items = list(zip(names, scores)) + [(NEW, th)]
    items.sort(key=lambda x: -x[1])
    return [x[0] for x in items[:5]]


def ap5(pred, gt):
    for r, x in enumerate(pred[:5], 1):
        if x == gt:
            return 1.0 / r
    return 0.0


def summarize(preds, d, target=0.30):
    sc = np.array([ap5(p, g) for p, g in zip(preds, d["truth"])])
    k, u = ~d["unk"], d["unk"]
    top_new = np.array([p[0] == NEW for p in preds])
    out = dict(target_map=(1 - target) * sc[k].mean() + target * sc[u].mean(),
               overall=sc.mean(), known=sc[k].mean(), unknown=sc[u].mean(),
               hyena=sc[d["sp"] == "hyena"].mean(), leopard=sc[d["sp"] == "leopard"].mean(),
               known_new_at1_pct=100 * top_new[k].mean(), unk_new_at1_pct=100 * top_new[u].mean())
    return out


def avg_over_seeds(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


# ----------------------------------------------------------------------------
# Open-set detector
# ----------------------------------------------------------------------------
F_BASE = ["s1", "s2", "margin", "mean5", "raw1", "raw_best", "log_n"]
F_SIFT = ["log_inl1", "log_inl_max"]
F_TRANS = ["qq_max", "qq_gap"]


def make_features(fin, pool, gal, use_sift):
    s = fin["score"]
    f = dict(s1=s[:, 0], s2=s[:, 1], margin=s[:, 0] - s[:, 1], mean5=s[:, :5].mean(1),
             raw1=fin["raw"][:, 0], raw_best=pool["raw_best"],
             log_n=np.log(gal.counts[fin["cand_idx"][:, 0]]),
             qq_max=pool["qq_max"], qq_gap=pool["qq_max"] - pool["raw_best"])
    if use_sift:
        f["log_inl1"] = np.log1p(fin["inl"][:, 0])
        f["log_inl_max"] = np.log1p(fin["inl"].max(1))
    return pd.DataFrame(f)


def new_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def sample_weights(y, target):
    """Re-weight so that unseen-identity queries make up `target` of the training mass."""
    n1, n0 = max(int(y.sum()), 1), max(int(len(y) - y.sum()), 1)
    return np.where(y == 1, target / n1, (1 - target) / n0) * len(y)


def fit_one(X, y, target):
    return new_model().fit(X, y, logisticregression__sample_weight=sample_weights(y, target))


def detector_oof(X, y, sp, groups, cols, target, n_splits=5):
    """Out-of-fold P(new), per species, grouped by true identity (no identity in both folds)."""
    p = np.zeros(len(y))
    for s in ("hyena", "leopard"):
        idx = np.flatnonzero(sp == s)
        g = groups[idx]
        k = min(n_splits, len(np.unique(g)))
        for tr, te in GroupKFold(n_splits=k).split(idx, groups=g):
            m = fit_one(X.iloc[idx[tr]][cols], y[idx[tr]], target)
            p[idx[te]] = m.predict_proba(X.iloc[idx[te]][cols])[:, 1]
    return p


def detector_fit(X, y, sp, cols, target):
    return {s: fit_one(X[sp == s][cols], y[sp == s], target) for s in ("hyena", "leopard")}


def detector_predict(models, X, sp, cols, bias=0.0):
    p = np.zeros(len(X))
    for s, m in models.items():
        idx = np.flatnonzero(sp == s)
        if len(idx):
            z = m.decision_function(X.iloc[idx][cols]) + bias
            p[idx] = 1.0 / (1.0 + np.exp(-z))
    return p


# ----------------------------------------------------------------------------
# Submission checks
# ----------------------------------------------------------------------------
def validate_submission(sub, test_df, train_df, sample_path):
    assert len(sub) == len(test_df), "row count mismatch"
    assert sub["id"].is_unique, "duplicate ids"
    assert set(sub["id"]) == set(test_df["id"]), "ids differ from test ids"
    if sample_path and os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        assert list(sample.columns) == list(sub.columns), f"columns {list(sub.columns)} != sample {list(sample.columns)}"
        assert (sub["id"].values == sample["id"].values).all(), "id order differs from sample_submission"
    else:
        print("  (sample_submission.csv not found: only internal checks done)")
    known = {sp: set(train_df[train_df["species"] == sp]["identity"]) for sp in ("hyena", "leopard")}
    sp_of = dict(zip(test_df["id"], test_df["species"]))
    for i, pr in zip(sub["id"], sub["predictions"]):
        t = pr.split()
        assert len(t) == 5 and len(set(t)) == 5, f"bad row {i}"
        for x in t:
            assert x == NEW or x in known[sp_of[i]], f"wrong-species/unknown label {x} in {i}"
    print("  submission checks passed")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    args = parse_args()
    t_start = time.time()
    TGT = args.target_unknown
    proc = os.path.join(args.root, "Processed_Data")
    here = os.path.dirname(os.path.abspath(__file__))
    print("=" * 78)
    print("MODEL 4 (Dewang): whitened MiewID + viewpoint-aware retrieval + open-set detector")
    print(f"  root={args.root}  sift={args.use_sift}  transductive(final)={args.transductive}  tag={args.tag}")
    print("=" * 78)

    master = pd.read_csv(os.path.join(proc, "master_dataset.csv"))
    raw = np.load(os.path.join(proc, "miewid_features.npy"))
    train_df = master[master["split"] == "train"].reset_index(drop=True)
    test_df = master[master["split"] == "test"].reset_index(drop=True)
    print(f"train={len(train_df)} test={len(test_df)}  features={raw.shape}")

    spaces = build_spaces(raw)
    print(f"spaces: {list(spaces)}")

    sm = None
    if args.use_sift:
        sm = SiftMatcher(os.path.join(proc, "sift_features.npz"),
                         os.path.join(here, "cache", "sift_pairs.json"), args.sift_mem)
        sm.check(train_df["id"].iloc[0])

    # ---- 1. splits ---------------------------------------------------------
    split_dir = os.path.join(proc, "splits")
    os.makedirs(split_dir, exist_ok=True)
    seeds_data = []
    for seed in args.seeds:
        gal_df, kq, uq = make_openset_split(train_df, args.unknown_prop, seed)
        assert not set(gal_df["identity"]) & set(uq["identity"]), "identity leakage"
        assert set(kq["identity"]) <= set(gal_df["identity"]), "known query without gallery identity"
        q = pd.concat([kq, uq], ignore_index=True)
        unk = np.r_[np.zeros(len(kq), bool), np.ones(len(uq), bool)]
        truth = np.where(unk, NEW, q["identity"].to_numpy(dtype=object))
        out = pd.concat([gal_df.assign(role="gallery"), kq.assign(role="known_query"),
                         uq.assign(role="unknown_query")])[["id", "species", "viewpoint", "identity", "role"]]
        out.to_csv(os.path.join(split_dir, f"val_split_seed{seed}.csv"), index=False)
        seeds_data.append(dict(seed=seed, gal=Gallery(gal_df), q=q, unk=unk, truth=truth,
                               sp=q["species"].to_numpy(dtype=object),
                               true_id=q["identity"].to_numpy(dtype=object), n_known=len(kq)))
        print(f"  seed {seed}: gallery={len(gal_df)} ({gal_df['identity'].nunique()} ids)  "
              f"known_q={len(kq)}  unknown_q={len(uq)}")
    print(f"splits saved to {split_dir}")

    k_min = min(min((d["gal"].df.groupby("species")["identity"].nunique()).min() for d in seeds_data),
                train_df.groupby("species")["identity"].nunique().min())
    K = int(min(args.k_pool, k_min))
    KS = min(args.k_sift, K)

    # ---- 2. tune space + viewpoint on KNOWN queries (closed-set) -----------
    print("\n[1/4] Tuning PCA size and viewpoint weights (closed-set MAP@5 on known queries)")
    a_grid, b_grid = [0.0, 0.02, 0.05], [0.0, 0.03, 0.08]
    rows = []
    for sk, space in spaces.items():
        per = {(a, b): [] for a in a_grid for b in b_grid}
        for d in seeds_data:
            kq = d["q"].iloc[:d["n_known"]]
            gal = d["gal"]
            Zq = space[kq["miewid_index"].to_numpy()]
            Zg = space[gal.mi]
            S = Zq @ Zg.T
            same, opp = vp_matrices(vp_codes(kq), vp_side(kq), gal.vp, gal.side)
            bad = ((kq["species"].to_numpy(dtype=object) == "leopard").astype(np.int8)[:, None] != gal.sp[None, :])
            truth = kq["identity"].to_numpy(dtype=object)
            for a in a_grid:
                for b in b_grid:
                    A = S + a * same - b * opp
                    A[bad] = -9.0
                    ID = np.maximum.reduceat(A, gal.starts, axis=1)
                    top5 = np.argsort(-ID, axis=1)[:, :5]
                    names = gal.names[top5]
                    per[(a, b)].append(np.mean([ap5(list(r), t) for r, t in zip(names, truth)]))
        for (a, b), v in per.items():
            rows.append(dict(space=str(sk), a=a, b=b, known_map=float(np.mean(v))))
    tune = pd.DataFrame(rows).sort_values("known_map", ascending=False)
    print(tune.head(6).to_string(index=False))
    base = tune[(tune["a"] == 0) & (tune["b"] == 0)].sort_values("known_map", ascending=False).iloc[0]
    print(f"  best without viewpoint: space={base['space']}  known MAP={base['known_map']:.4f}")
    best = tune.iloc[0]
    space_key = best["space"] if best["space"] == "raw" else int(best["space"])
    A_VP, B_VP = float(best["a"]), float(best["b"])
    space = spaces[space_key]
    print(f"  -> using space={space_key}, a={A_VP}, b={B_VP}  (known MAP={best['known_map']:.4f})")

    # ---- 3. pools for all validation queries -------------------------------
    print("\n[2/4] Building candidate pools for validation queries")
    for d in seeds_data:
        d["pool"] = base_pool(d["q"], d["gal"], space, A_VP, B_VP, K, KS)
        if sm is not None:
            add_inliers(d["pool"], d["q"], d["gal"], sm, label=f"seed{d['seed']}")

    beta = 0.0
    if sm is not None:
        res = []
        for bt in [0.0, 0.03, 0.05, 0.1, 0.2, 0.4]:
            v = []
            for d in seeds_data:
                fin = finalize(d["pool"], bt)
                names = d["gal"].names[fin["cand_idx"][:d["n_known"], :5]]
                v.append(np.mean([ap5(list(r), t) for r, t in zip(names, d["truth"][:d["n_known"]])]))
            res.append((bt, float(np.mean(v))))
        print("  SIFT bonus beta vs known MAP:", ", ".join(f"{b}:{m:.4f}" for b, m in res))
        beta = max(res, key=lambda x: x[1])[0]
        print(f"  -> beta={beta}")
    for d in seeds_data:
        d["fin"] = finalize(d["pool"], beta)
        d["X"] = make_features(d["fin"], d["pool"], d["gal"], args.use_sift)
        d["cand_names"] = d["gal"].names[d["fin"]["cand_idx"]]

    # ---- 4. ablations -------------------------------------------------------
    print("\n[3/4] Open-set ablations (all numbers: mean over seeds)")
    table = {}

    # (a) closed-set baseline = Model-1 style
    rows_ = [summarize([list(r[:5]) for r in d["cand_names"]], d, TGT) for d in seeds_data]
    table["closed-set (no new_individual)"] = avg_over_seeds(rows_)

    # (b) single threshold per species on the top score (Emad's Config-A style)
    thr = {}
    for sp in ("hyena", "leopard"):
        best_th, best_m = None, -1
        s_all = np.concatenate([d["fin"]["score"][d["sp"] == sp, 0] for d in seeds_data])
        for th in np.linspace(np.quantile(s_all, 0.02), np.quantile(s_all, 0.98), 40):
            m = []
            for d in seeds_data:
                idx = np.flatnonzero(d["sp"] == sp)
                pr = [rank_with_threshold(d["cand_names"][i], d["fin"]["score"][i], th) for i in idx]
                sc_ = np.array([ap5(p, d["truth"][i]) for p, i in zip(pr, idx)])
                u_ = d["unk"][idx]
                m.append((1 - TGT) * sc_[~u_].mean() + TGT * sc_[u_].mean())
            if np.mean(m) > best_m:
                best_th, best_m = th, float(np.mean(m))
        thr[sp] = best_th
    rows_ = []
    for d in seeds_data:
        pr = [rank_with_threshold(d["cand_names"][i], d["fin"]["score"][i], thr[d["sp"][i]])
              for i in range(len(d["q"]))]
        rows_.append(summarize(pr, d, TGT))
    table["threshold per species"] = avg_over_seeds(rows_)

    # (c)/(d) learned detector
    X_all = pd.concat([d["X"] for d in seeds_data], ignore_index=True)
    y_all = np.concatenate([d["unk"] for d in seeds_data]).astype(int)
    sp_all = np.concatenate([d["sp"] for d in seeds_data])
    grp_all = np.concatenate([d["true_id"] for d in seeds_data])
    off = np.cumsum([0] + [len(d["q"]) for d in seeds_data])
    print(f"  pooled detector data: {len(y_all)} queries, {y_all.mean() * 100:.1f}% from unseen identities "
          f"(re-weighted to {TGT:.0%} for training and for choosing T/thresholds)")

    variants = {"detector (similarity feats)": F_BASE + (F_SIFT if args.use_sift else []),
                "detector + transductive": F_BASE + (F_SIFT if args.use_sift else []) + F_TRANS}
    T_grid = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
    chosen = {}
    for name, cols in variants.items():
        p_oof = detector_oof(X_all, y_all, sp_all, grp_all, cols, TGT)
        best_T, best_row = None, None
        for T in T_grid:
            rows_ = []
            for si, d in enumerate(seeds_data):
                p = p_oof[off[si]:off[si + 1]]
                pr = [rank_with_prob(d["cand_names"][i], d["fin"]["score"][i], p[i], T)
                      for i in range(len(d["q"]))]
                rows_.append(summarize(pr, d, TGT))
            r = avg_over_seeds(rows_)
            if best_row is None or r["target_map"] > best_row["target_map"]:
                best_T, best_row = T, r
        table[name + f"  [T={best_T}]"] = best_row
        chosen[name] = (cols, best_T)

    df_tab = pd.DataFrame(table).T
    print(df_tab.round(4).to_string())
    print(f"  target_map = (1-{TGT})*known + {TGT}*unknown, i.e. MAP if {TGT:.0%} of queries are new animals; "
          f"'overall' is the raw split mix")
    print("  known_new_at1_pct = known-animal queries wrongly sent to new_individual at rank 1")
    print("  (Detector numbers are out-of-fold; T, PCA size and viewpoint weights were tuned on these same splits,\n"
          "   so treat them as slightly optimistic. Compare against the leaderboard.)")

    final_name = "detector + transductive" if args.transductive else "detector (similarity feats)"
    cols, T = chosen[final_name]
    print(f"\n  FINAL CONFIG for test: {final_name}, T={T}, beta={beta}, bias={args.new_bias}")

    summary = dict(config=dict(target_unknown=TGT, space=str(space_key), a=A_VP, b=B_VP, beta=beta, T=T, k_pool=K, k_sift=KS,
                               final=final_name, use_sift=args.use_sift, new_bias=args.new_bias,
                               thresholds={k: float(v) for k, v in thr.items()}),
                   viewpoint_tuning=tune.head(15).to_dict("records"),
                   ablation=df_tab.round(5).reset_index().rename(columns={"index": "model"}).to_dict("records"))
    with open(os.path.join(proc, f"model4_validation_summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if args.no_test:
        print("--no-test given: stopping after validation.")
        return

    # ---- 5. test inference --------------------------------------------------
    print("\n[4/4] Test inference (gallery = full train)")
    gal_full = Gallery(train_df)
    pool = base_pool(test_df, gal_full, space, A_VP, B_VP, K, KS)
    if sm is not None:
        add_inliers(pool, test_df, gal_full, sm, label="test")
    fin = finalize(pool, beta)
    X_te = make_features(fin, pool, gal_full, args.use_sift)
    sp_te = test_df["species"].to_numpy(dtype=object)
    models = detector_fit(X_all, y_all, sp_all, cols, TGT)
    p_te = detector_predict(models, X_te, sp_te, cols, bias=args.new_bias)
    names_te = gal_full.names[fin["cand_idx"]]
    preds = [rank_with_prob(names_te[i], fin["score"][i], p_te[i], T) for i in range(len(test_df))]
    sub = pd.DataFrame({"id": test_df["id"], "predictions": [" ".join(p) for p in preds]})

    sample_path = args.sample_submission or os.path.join(proc, "sample_submission.csv")
    validate_submission(sub, test_df, train_df, sample_path)
    out_path = os.path.join(proc, f"submission_model4_{args.tag}.csv")
    sub.to_csv(out_path, index=False)
    top_new = np.array([p[0] == NEW for p in preds])
    print(f"  saved {out_path}")
    print(f"  mean P(new)={p_te.mean():.3f} | new_individual at rank 1: {top_new.sum()} "
          f"({100 * top_new.mean():.1f}%) | in top 5: {sum(NEW in p for p in preds)}")
    print(f"  (detector assumes {TGT:.0%} new animals; if predicted share or the leaderboard disagrees, "
          f"change --target-unknown or use --new-bias +/-0.5)")
    print(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
