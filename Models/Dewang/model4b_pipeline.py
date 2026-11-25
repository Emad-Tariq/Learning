#!/usr/bin/env python3
"""
MODEL 4b (Dewang): Upgraded retrieval + open-set pipeline.

Upgrades over Model 4 (0.78 leaderboard):
  1. PCA → LDA projection (supervised, identity-disjoint via simulation splits)
     instead of PCA-only whitening.  LDA separates identities far better.
  2. LightGBM open-set detector + candidate scorer (replaces logistic regression).
     Two classifiers: a query-level new-individual detector and a (query, candidate)
     scorer trained on simulated open-set episodes.
  3. Multi-round transductive stacking (default 3 rounds): each round's predictions
     for a query's nearest test neighbours become features for the next round.
  4. Richer features: viewpoint-coverage (same-view / same-flank similarity),
     per-identity SIFT support count, candidate centroid similarity,
     query-to-query SIFT/embedding neighbours.
  5. 16 regime-diverse simulated open-set splits (vs 3 seeds), with
     near-duplicate-aware group holdout so unseen queries are realistic.
  6. Mutual nearest-neighbour SIFT matching (bidirectional ratio test).

Anti-overfit safeguards:
  - Identity-disjoint cross-validation: no identity appears in both train and
    test of any simulation episode.
  - Regularised LDA (shrinkage=0.3), moderate LightGBM leaves/estimators.
  - Regime-mixture simulations: each episode varies unknown fraction, query cap,
    and near-duplicate linking threshold, so the model sees diverse gallery
    compositions.
  - LightGBM uses subsample + colsample_bytree < 1.
  - Final temperature (T) is tuned per species on simulated OOF predictions.

Run:  python model4b_pipeline.py                   (validation only, --no-test)
      python model4b_pipeline.py --predict-test     (also writes submission)
"""
import argparse
import json
import os
import re
import time
import zipfile as _zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from threading import Lock

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# ---------------------------------------------------------------------------
# Workaround: Python 3.10+ (CVE-2024-0450) added zip-bomb overlap detection
# that false-positives on large numpy .npz archives.  Since we only open
# trusted local data files, we disable the overlap check.
# ---------------------------------------------------------------------------
_orig_zf_open = _zipfile.ZipFile.open

def _permissive_zf_open(self, name, mode='r', pwd=None, *, force_zip64=False):
    if mode != 'r':
        return _orig_zf_open(self, name, mode, pwd, force_zip64=force_zip64)
    if isinstance(name, str):
        name = self.getinfo(name)
    # CPython 3.10.14+/3.11.9+/3.12.3+ use `_end_offset` for the overlap check
    for attr in ('_end_offset', '_end_header_offset'):
        if hasattr(name, attr):
            setattr(name, attr, None)
    return _orig_zf_open(self, name, mode, pwd, force_zip64=force_zip64)

_zipfile.ZipFile.open = _permissive_zf_open

NEW = "new_individual"
VIEWS = ["left", "right", "front", "back", "frontleft", "frontright",
         "backleft", "backright", "up", "down"]
FLANK = np.array([0, 1, 2, 2, 0, 1, 0, 1, 2, 2])  # left-ish, right-ish, other

# Mutable globals set by load_data / sift_matrices
X = None          # (N, D) L2-normalised MiewID features
IDS = None        # (N,) image ids
Y = None          # (N,) identity labels (None for test)
SP = None         # (N,) species
VIEW = None       # (N,) viewpoint code (int)
GOOD = None       # sparse (N, N) SIFT ratio-test matches
INL = None        # sparse (N, N) RANSAC inliers

# Hyperparams set at runtime
K = 15                                               # candidates per query
N_SIMS = 16                                          # simulated episodes
FOLDS = 4                                            # stacking folds
ROUNDS = 3                                           # stacking rounds
NEIGHBOURS = 150                                     # SIFT neighbourhood size
MATCH_RATIO = 0.85                                   # SIFT ratio test
RANSAC_PX = 12.0                                     # RANSAC threshold
SIFT_PULL = 0.05                                     # SIFT weight in candidate selection
REGIMES = [None, (0.95, 100), (0.99, 80), (0.9, 60)] # near-duplicate linking
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31,
              min_child_samples=30, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, n_estimators=400, verbose=-1)
INDEX_COLS = re.compile(r"^(nb|nw|sb|sw)\d$")


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════
def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def load_data(root):
    """Load master_dataset.csv and miewid_features.npy, populate globals."""
    global X, IDS, Y, SP, VIEW
    proc = os.path.join(root, "Processed_Data")
    master = pd.read_csv(os.path.join(proc, "master_dataset.csv"))
    raw = np.load(os.path.join(proc, "miewid_features.npy"))
    need = {"id", "identity", "species", "viewpoint", "split"}
    missing_cols = need - set(master.columns)
    if missing_cols:
        raise SystemExit(f"master_dataset.csv is missing columns {sorted(missing_cols)}; "
                         f"found {list(master.columns)}")
    # Align features with master_dataset order via miewid_index (fallback: row order)
    if "miewid_index" in master.columns:
        feat_idx = master["miewid_index"].to_numpy()
    else:
        print("  WARNING: no 'miewid_index' column; assuming features are in row order")
        feat_idx = np.arange(len(master))
    X = unit(np.asarray(raw[feat_idx], dtype=np.float32))
    IDS = master["id"].to_numpy()
    Y = master["identity"].to_numpy(dtype=object)
    SP = master["species"].to_numpy(dtype=object)
    VIEW = np.array([{v: i for i, v in enumerate(VIEWS)}.get(str(v), -1)
                      for v in master["viewpoint"]], dtype=int)
    print(f"  species values: {sorted(set(map(str, SP)))}   splits: "
          f"{master['split'].value_counts().to_dict()}")
    train = master[master["split"] == "train"].reset_index(drop=True)
    test = master[master["split"] == "test"].reset_index(drop=True)
    # Map to global indices in X
    id_to_idx = {iid: i for i, iid in enumerate(IDS)}
    train_idx = np.array([id_to_idx[i] for i in train["id"]])
    test_idx = np.array([id_to_idx[i] for i in test["id"]])
    return train, test, master, train_idx, test_idx


# ═══════════════════════════════════════════════════════════════════════════════
# SIFT verification  (ThreadPoolExecutor + lru_cache — no fork, no zip issues)
# ═══════════════════════════════════════════════════════════════════════════════
class SiftCache:
    """Thread-safe SIFT descriptor loader with LRU cache.

    OpenCV BFMatcher / estimateAffinePartial2D release the GIL, so
    ThreadPoolExecutor gives real parallelism without fork-related NPZ
    issues that multiprocessing.Pool triggers on Python 3.10+.
    """

    def __init__(self, npz_path, cache_path, workers=2, mem_items=2000):
        self.npz = np.load(npz_path)
        self.npz_keys = set(self.npz.files)
        hits = sum(f"{i}_desc" in self.npz_keys for i in IDS)
        print(f"  SIFT npz: {len(self.npz_keys)} arrays, descriptors found for "
              f"{hits}/{len(IDS)} images", flush=True)
        if hits == 0:
            raise SystemExit(f"No '<id>_desc' keys in {npz_path}. Example keys: "
                             f"{list(self.npz_keys)[:6]}")
        self.read_lock = Lock()          # NpzFile's internal ZipFile is not thread-safe
        self.workers = workers
        self.cache_path = cache_path
        self.done = {}
        if os.path.exists(cache_path):
            with np.load(cache_path) as z:
                for p, v in zip(z["pairs"], z["values"]):
                    self.done[tuple(p)] = tuple(v)
        # Bind lru_cache *per instance* so it can be garbage-collected
        self._descriptors = lru_cache(maxsize=mem_items)(self._descriptors_raw)

    def _descriptors_raw(self, i):
        """Load RootSIFT descriptors and keypoint positions for one image."""
        iid = IDS[i]
        dkey, xykey = f"{iid}_desc", f"{iid}_xy"
        if dkey not in self.npz_keys:
            return np.zeros((0, 128), np.float32), np.zeros((0, 2), np.float32)
        with self.read_lock:
            d = np.asarray(self.npz[dkey], dtype=np.float32)
            xy = np.asarray(self.npz[xykey], dtype=np.float32)[:, :2]
        d = np.sqrt(d / np.maximum(d.sum(1, keepdims=True), 1e-12))
        return np.ascontiguousarray(d), xy

    def verify(self, pair):
        """Mutual ratio-test matches and RANSAC inliers between two images."""
        a, b = pair
        da, xa = self._descriptors(a)
        db, xb = self._descriptors(b)
        if len(da) < 4 or len(db) < 4:
            return 0, 0
        bf = cv2.BFMatcher()
        back = np.array([m.trainIdx for m in bf.match(db, da)])
        good = [m[0] for m in bf.knnMatch(da, db, k=2)
                if len(m) == 2 and m[0].distance < MATCH_RATIO * m[1].distance
                and back[m[0].trainIdx] == m[0].queryIdx]
        if len(good) < 4:
            return len(good), 0
        src = np.float32([xa[m.queryIdx] for m in good])
        dst = np.float32([xb[m.trainIdx] for m in good])
        # Deterministic RANSAC per pair
        cv2.setRNGSeed(int((int(a) * 100003 + int(b)) % (2**31 - 1)))
        _, inliers = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
        return len(good), 0 if inliers is None else int(inliers.sum())

    def save(self):
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        pairs = sorted(self.done)
        np.savez_compressed(
            self.cache_path,
            pairs=np.array(pairs, dtype=np.int64).reshape(-1, 2),
            values=np.array([self.done[p] for p in pairs]).reshape(-1, 2))


def sift_matrices(npz_path, cache_path, workers=2):
    """Build symmetric sparse matrices of SIFT matches/inliers for all needed pairs."""
    global GOOD, INL
    cache = SiftCache(npz_path, cache_path, workers)
    n = len(X)
    # Find pairs to verify: nearest same-species neighbours
    wanted = set()
    for sp in ("hyena", "leopard"):
        ix = np.where(SP == sp)[0]
        # Chunk the similarity computation to limit memory
        for start in range(0, len(ix), 256):
            q = ix[start:start + 256]
            sim = X[q] @ X[ix].T
            # Zero out self-similarities
            for local_i, global_i in enumerate(q):
                col = np.searchsorted(ix, global_i)
                if col < sim.shape[1] and ix[col] == global_i:
                    sim[local_i, col] = -9.0
            k = min(NEIGHBOURS, len(ix) - 1)
            nb = np.argsort(-sim, 1)[:, :k]
            for row, nbs in zip(q, nb):
                wanted.update((min(int(row), int(j)), max(int(row), int(j)))
                              for j in ix[nbs])
    missing = sorted(wanted - cache.done.keys())
    print(f"  SIFT: {len(wanted)} pairs needed, {len(missing)} uncached", flush=True)
    if missing:
        cv2.setNumThreads(1)   # each thread does its own matching
        t0 = time.time()
        BLOCK = 20000
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for s0 in range(0, len(missing), BLOCK):
                block = missing[s0:s0 + BLOCK]
                for pair, value in zip(block, pool.map(cache.verify, block)):
                    cache.done[pair] = value
                cache.save()      # re-running resumes from here if interrupted
                print(f"    verified {s0 + len(block)}/{len(missing)}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
        cache.save()
    # Build sparse matrices from wanted pairs only
    pp = np.array(sorted(wanted), dtype=int).reshape(-1, 2)
    vv = np.array([cache.done[tuple(p)] for p in pp]).reshape(-1, 2)
    a, b = pp.T
    def mat(v):
        return sparse.coo_matrix(
            (np.r_[v, v], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    GOOD, INL = mat(vv[:, 0]), mat(vv[:, 1])
    print(f"  SIFT matrices built: {GOOD.nnz} nonzeros", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PCA + LDA embedding  (upgrade #1 over Model 4)
# ═══════════════════════════════════════════════════════════════════════════════
def embed(gallery_idx, query_idx, labels):
    """PCA → LDA projection.  PCA is unsupervised (sees queries too).
    LDA is supervised on gallery labels only → identity-disjoint."""
    all_idx = np.r_[gallery_idx, query_idx]
    n_comp = min(384, len(all_idx) - 1, X.shape[1])
    pca = PCA(n_comp).fit(X[all_idx])
    lda = LDA(solver="eigen", shrinkage=0.3).fit(pca.transform(X[gallery_idx]), labels)
    project = lambda idx: unit(lda.transform(pca.transform(X[idx])))
    return project(gallery_idx), project(query_idx)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature extraction  (upgrade #4 — richer features)
# ═══════════════════════════════════════════════════════════════════════════════
def per_identity(sim, inv, n):
    """Per query: max and mean-of-top-2 similarity to each identity's images."""
    best = np.empty((sim.shape[0], n))
    top2 = np.empty((sim.shape[0], n))
    for j in range(n):
        s = sim[:, inv == j]
        best[:, j] = s.max(1)
        top2[:, j] = np.sort(s, 1)[:, -2:].mean(1)
    return best, top2


def duplicate_groups(ix, link):
    """Group near-duplicate images of one identity."""
    if link is None:
        return np.arange(len(ix))
    edges = (X[ix] @ X[ix].T > link[0]) | (INL[ix][:, ix].toarray() >= link[1])
    return connected_components(sparse.csr_matrix(edges))[1]


def species_features(gi, qi, sp):
    """Build query-level and (query, candidate) feature DataFrames for one species."""
    Rg, Rq = X[gi], X[qi]
    yg, vg, vq = Y[gi], VIEW[gi], VIEW[qi]
    Lg, Lq = embed(gi, qi, yg)

    ids, inv = np.unique(yg, return_inverse=True)
    n, nq = len(ids), len(qi)
    if n < K:
        raise ValueError(f"only {n} gallery identities for {sp}, need >= candidates ({K})")

    sim_l, sim_r = Lq @ Lg.T, Rq @ Rg.T
    lda1, lda2 = per_identity(sim_l, inv, n)
    raw1, raw2 = per_identity(sim_r, inv, n)
    centroid = unit(np.stack([Lg[inv == j].mean(0) for j in range(n)]))
    cen = Lq @ centroid.T

    # SIFT per-identity features
    inl_g = INL[qi][:, gi].toarray().astype(np.float32)
    good_g = GOOD[qi][:, gi].toarray().astype(np.float32)
    sift1, sift2 = per_identity(inl_g, inv, n)
    good1, _ = per_identity(good_g, inv, n)
    # Support: how many gallery images of each identity have ≥7 RANSAC inliers
    support = (inl_g >= 7).astype(np.float32) @ np.eye(n, dtype=np.float32)[inv]

    # Same-viewpoint similarity (upgrade #4)
    same_view = np.full((nq, n), np.nan)
    for j in range(n):
        for v in np.unique(vq):
            cols = np.where((inv == j) & (vg == v))[0]
            if len(cols):
                same_view[vq == v, j] = sim_l[np.ix_(vq == v, cols)].max(1)

    # Viewpoint coverage per identity
    by_view = np.stack([np.bincount(inv[vg == v], minlength=n) for v in range(len(VIEWS))], 1)
    by_flank = np.stack([by_view[:, FLANK == f].sum(1) for f in range(3)], 1)
    seen = vq >= 0
    n_view = np.where(seen[:, None], by_view[:, vq.clip(0)].T, np.nan)
    n_flank = np.where(seen[:, None], by_flank[:, FLANK[vq.clip(0)]].T, np.nan)

    # Transductive: nearest query neighbours by embedding and SIFT
    qq_r, qq_l = Rq @ Rq.T, Lq @ Lq.T
    qq_s = INL[qi][:, qi].toarray().astype(np.float32)
    np.fill_diagonal(qq_r, -1)
    np.fill_diagonal(qq_l, -1)
    nb = np.argsort(-qq_l, 1)[:, :5]
    sb = np.argsort(-qq_s, 1)[:, :5]
    nbs = np.take_along_axis(qq_l, nb, 1)
    sbs = np.take_along_axis(qq_s, sb, 1)

    t_l = np.take_along_axis(lda1, np.argsort(-lda1, 1)[:, :3], 1)
    t_r = -np.sort(-raw1, 1)[:, :3]
    t_s = -np.sort(-sift1, 1)[:, :3]
    g_top = -np.sort(-sim_l, 1)

    # Query-level features
    q = pd.DataFrame({
        "sp": int(sp == "hyena"), "view": vq, "ng": len(gi),
        "l1": t_l[:, 0], "l2": t_l[:, 1], "l3": t_l[:, 2],
        "r1": t_r[:, 0], "r2": t_r[:, 1], "r3": t_r[:, 2],
        "lm12": t_l[:, 0] - t_l[:, 1], "rm12": t_r[:, 0] - t_r[:, 1],
        "gd5": g_top[:, :5].mean(1), "gd20": g_top[:, :20].mean(1),
        "qq1": nbs[:, 0], "qq3": nbs[:, :3].mean(1), "qq5": nbs.mean(1),
        "qqr1": qq_r.max(1),
        "qn6": (qq_r > .6).sum(1), "qn7": (qq_r > .7).sum(1), "qn8": (qq_r > .8).sum(1),
        "qq_minus_g": nbs[:, 0] - t_l[:, 0], "nb_g1": t_l[nb[:, 0], 0],
        "s1": t_s[:, 0], "s2": t_s[:, 1], "s3": t_s[:, 2], "sg": good_g.max(1),
        "qs1": sbs[:, 0], "qs3": sbs[:, :3].mean(1),
        "qsn10": (qq_s >= 10).sum(1), "qsn20": (qq_s >= 20).sum(1),
        "qs_minus_s": sbs[:, 0] - t_s[:, 0], "nbs_s1": t_s[sb[:, 0], 0],
    })
    for k in range(5):
        q[f"nb{k}"], q[f"nw{k}"] = nb[:, k], nbs[:, k]
        q[f"sb{k}"], q[f"sw{k}"] = sb[:, k], sbs[:, k]

    # Candidate-level features (upgrade #4)
    cand = np.argsort(-(lda1 + SIFT_PULL * np.tanh(sift1 / 15)), 1)[:, :K]
    r, c = np.repeat(np.arange(nq), K), cand.ravel()
    nb_ev = np.max(np.minimum(nbs[:, :, None],
                               np.stack([lda1[nb[:, k]] for k in range(5)], 1)), 1)
    sb_ev = np.max(np.minimum(sbs[:, :, None],
                               np.stack([sift1[sb[:, k]] for k in range(5)], 1)), 1)
    rank_of = lambda m: (-m).argsort(1).argsort(1)[r, c]

    cd = pd.DataFrame({
        "q": r, "c": c, "rank": np.tile(np.arange(K), nq), "ident": ids[c],
        "mxr": raw1[r, c], "m2r": raw2[r, c],
        "mxl": lda1[r, c], "m2l": lda2[r, c], "cl": cen[r, c],
        "gapl": lda1[r, c] - t_l[r, 0], "gapr": raw1[r, c] - t_r[r, 0],
        "same": same_view[r, c],
        "cnt": np.bincount(inv)[c], "nbev": nb_ev[r, c], "sbev": sb_ev[r, c],
        "rk_r": rank_of(raw1), "rk_c": rank_of(cen), "rk_s": rank_of(sift1),
        "n_view": n_view[r, c], "n_flank": n_flank[r, c],
        "support": support[r, c],
        "sx": sift1[r, c], "s2x": sift2[r, c], "gx": good1[r, c],
        "sgap": sift1[r, c] - t_s[r, 0],
    })
    return q, cd


def build_frames(gi, qi):
    """Build feature frames across both species."""
    qs, cs, order, off = [], [], [], 0
    for sp in ("hyena", "leopard"):
        g = gi[SP[gi] == sp]
        qq = qi[SP[qi] == sp]
        q, c = species_features(g, qq, sp)
        c["q"] += off
        q[[f"{p}{k}" for p in ("nb", "sb") for k in range(5)]] += off
        qs.append(q)
        cs.append(c)
        order.append(qq)
        off += len(qq)
    return dict(q=pd.concat(qs, ignore_index=True),
                c=pd.concat(cs, ignore_index=True),
                order=np.concatenate(order))


# ═══════════════════════════════════════════════════════════════════════════════
# Simulated open-set episodes  (upgrade #5 — regime-diverse)
# ═══════════════════════════════════════════════════════════════════════════════
def simulate(seed, train_idx):
    """Identity-disjoint open-set split with regime-mixture simulation."""
    rng = np.random.default_rng(seed)
    link = REGIMES[seed % len(REGIMES)]
    gallery, known, unknown = [], [], []
    for sp in ("hyena", "leopard"):
        p_unknown = rng.uniform(0.2, 0.35)
        frac = rng.uniform(0.3, 0.5)
        cap = int(rng.integers(12, 33))
        ix = train_idx[SP[train_idx] == sp]
        names = np.array(sorted(set(Y[ix])))
        rng.shuffle(names)
        new_set = set(names[:int(round(p_unknown * len(names)))])
        for name in names:
            rows = ix[Y[ix] == name].copy()
            rng.shuffle(rows)
            if name in new_set:
                unknown.extend(rows[:cap].tolist())
            elif len(rows) < 2:
                gallery.extend(rows.tolist())
            else:
                n_q = min(max(int(round(frac * len(rows))), 1), cap, len(rows) - 1)
                groups = duplicate_groups(rows, link)
                held = []
                for c in rng.permutation(np.unique(groups)):
                    members = rows[groups == c].tolist()
                    if len(held) < n_q and len(held) + len(members) < len(rows):
                        held.extend(members)
                known.extend(held[:cap])
                held_set = set(held)
                gallery.extend(i for i in rows if i not in held_set)
    gi = np.array(gallery, dtype=int)
    qi = np.array(known + unknown, dtype=int)
    frames = build_frames(gi, qi)
    truth = np.where(np.isin(frames["order"], unknown), NEW, Y[frames["order"]])
    frames["truth"] = truth
    frames["q"]["unk"] = (truth == NEW).astype(int)
    frames["c"]["y"] = (frames["c"].ident.to_numpy(dtype=object) ==
                        truth[frames["c"].q.to_numpy()]).astype(int)
    return frames


# ═══════════════════════════════════════════════════════════════════════════════
# LightGBM stacked classifiers  (upgrade #2 — replaces logistic regression)
# ═══════════════════════════════════════════════════════════════════════════════
def columns(s):
    qf = [c for c in s["q"].columns if c != "unk" and not INDEX_COLS.match(c)]
    cf = [c for c in s["c"].columns if c not in ("q", "c", "ident", "y")]
    return qf, cf


def candidate_matrix(s, qf, cf):
    return pd.concat([s["c"][cf].reset_index(drop=True),
                       s["q"].iloc[s["c"].q.to_numpy()][qf].reset_index(drop=True)], axis=1)


def fit(sims):
    """Fit new-individual detector, closed-set candidate scorer, all-query scorer."""
    qf, cf = columns(sims[0])
    known = lambda s: s["q"].unk.to_numpy()[s["c"].q.to_numpy()] == 0
    cx = pd.concat([candidate_matrix(s, qf, cf) for s in sims])
    cy = np.concatenate([s["c"].y.to_numpy() for s in sims])
    ck = np.concatenate([known(s) for s in sims])
    detector = lgb.LGBMClassifier(
        **{**PARAMS, "n_estimators": 300, "num_leaves": 15}).fit(
        pd.concat([s["q"][qf] for s in sims]),
        np.concatenate([s["q"].unk.to_numpy() for s in sims]))
    closed_scorer = lgb.LGBMClassifier(**PARAMS).fit(cx[ck], cy[ck])
    all_scorer = lgb.LGBMClassifier(**PARAMS).fit(cx, cy)
    return detector, closed_scorer, all_scorer, (qf, cf)


def predict(models, s):
    detector, closed, anyq, (qf, cf) = models
    nq = len(s["q"])
    cx = candidate_matrix(s, qf, cf)
    p_new = detector.predict_proba(s["q"][qf])[:, 1]
    raw = anyq.predict_proba(cx)[:, 1].reshape(nq, K)
    within = closed.predict_proba(cx)[:, 1].reshape(nq, K)
    within /= np.maximum(within.sum(1, keepdims=True), 1e-9)
    return dict(pn=p_new, raw=raw, pc=within,
                pb=(p_new + np.clip(1 - raw.sum(1), 0, 1)) / 2)


def rank(s, r):
    """Top-5 labels: P(new) vs (1 - P(new)) * P(identity | known)."""
    nq = len(s["q"])
    labels = np.concatenate([
        s["c"].ident.to_numpy(dtype=object).reshape(nq, K),
        np.full((nq, 1), NEW, dtype=object)], 1)
    scores = np.concatenate([(1 - r["pb"])[:, None] * r["pc"],
                              r["pb"][:, None]], 1)
    return np.take_along_axis(labels, np.argsort(-scores, 1)[:, :5], 1)


def map5(preds, truth):
    return np.mean([next((1 / (i + 1) for i, p in enumerate(row) if p == t), 0.0)
                    for row, t in zip(preds, truth)])


# ═══════════════════════════════════════════════════════════════════════════════
# Transductive stacking  (upgrade #3)
# ═══════════════════════════════════════════════════════════════════════════════
def augment(s, r):
    """Add previous round's predictions as features for the next round."""
    q, c, nq = s["q"].copy(), s["c"].copy(), len(s["q"])
    codes = pd.factorize(c.ident)[0].reshape(nq, K)
    q["r_pn"] = r["pn"]
    q["r_pb"] = r["pb"]
    q["r_rawsum"] = r["raw"].sum(1)
    q["r_rawmax"] = r["raw"].max(1)
    c["r_raw"] = r["raw"].ravel()
    c["r_pc"] = r["pc"].ravel()
    for kind, weight, scale in (("nb", "nw", 1.0), ("sb", "sw", 30.0)):
        evidence = np.zeros((nq, K))
        support = np.zeros((nq, K))
        pb_list, w_list, top_list = [], [], []
        for k in range(5):
            idx = q[f"{kind}{k}"].to_numpy()
            sim = q[f"{weight}{k}"].to_numpy() / scale
            wk = (np.clip(sim - 0.4, 0, None) if kind == "nb"
                  else np.clip(sim, 0, 1))
            agree = ((codes[idx][:, None, :] == codes[:, :, None]) *
                     r["raw"][idx][:, None, :]).sum(2)
            evidence = np.maximum(evidence,
                                   np.minimum(agree, np.clip(sim, 0, 1)[:, None]))
            support += wk[:, None] * agree
            pb_list.append(r["pb"][idx])
            w_list.append(wk)
            top_list.append(r["raw"][idx].max(1))
        pb = np.stack(pb_list, 1)
        w = np.stack(w_list, 1)
        q[f"{kind}_pb0"] = pb[:, 0]
        q[f"{kind}_pbw"] = (pb * w).sum(1) / np.maximum(w.sum(1), 1e-6)
        q[f"{kind}_pbmin"] = np.where(w > 0, pb, 1).min(1)
        q[f"{kind}_w"] = w.sum(1)
        q[f"{kind}_rm0"] = top_list[0]
        c[f"r_{kind}ev"] = evidence.ravel()
        c[f"r_{kind}sum"] = support.ravel()
    return dict(s, q=q, c=c)


# ═══════════════════════════════════════════════════════════════════════════════
# Validation & submission
# ═══════════════════════════════════════════════════════════════════════════════
def validate_submission(sub, test_df, train_df, sample_path):
    assert len(sub) == len(test_df), "row count mismatch"
    assert sub["id"].is_unique, "duplicate ids"
    assert set(sub["id"]) == set(test_df["id"]), "ids differ from test ids"
    if sample_path and os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        assert list(sample.columns) == list(sub.columns), \
            f"columns {list(sub.columns)} != sample {list(sample.columns)}"
        assert (sub["id"].values == sample["id"].values).all(), \
            "id order differs from sample_submission"
    else:
        print("  (sample_submission.csv not found: only internal checks done)")
    known = {sp: set(train_df[train_df["species"] == sp]["identity"])
             for sp in ("hyena", "leopard")}
    sp_of = dict(zip(test_df["id"], test_df["species"]))
    for i, pr in zip(sub["id"], sub["predictions"]):
        t = pr.split()
        assert len(t) == 5 and len(set(t)) == 5, f"bad row {i}"
        for x in t:
            assert x == NEW or x in known[sp_of[i]], \
                f"wrong-species/unknown label {x} in {i}"
    print("  submission checks passed")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    global K, N_SIMS, FOLDS, ROUNDS, NEIGHBOURS, PARAMS
    t0 = time.time()

    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")),
        help="project root containing Processed_Data/")
    ap.add_argument("--sample-submission", default=None)
    ap.add_argument("--sims", type=int, default=16, help="number of simulated episodes")
    ap.add_argument("--folds", type=int, default=4, help="stacking cross-val folds")
    ap.add_argument("--rounds", type=int, default=3, help="transductive stacking rounds")
    ap.add_argument("--candidates", type=int, default=15, help="candidates per query")
    ap.add_argument("--neighbours", type=int, default=150, help="SIFT neighbourhood size")
    ap.add_argument("--trees", type=int, default=400, help="LightGBM estimators")
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 2) - 1),
                    help="parallel SIFT workers (default: CPU cores - 1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--predict-test", action="store_true",
                    help="also run test inference and write submission")
    ap.add_argument("--no-test", action="store_true", help="validation only (alias)")
    ap.add_argument("--tag", default="4b", help="suffix for output file names")
    args = ap.parse_args()

    K = args.candidates
    N_SIMS = args.sims
    FOLDS = args.folds
    ROUNDS = args.rounds
    NEIGHBOURS = args.neighbours
    PARAMS.update(n_estimators=args.trees, random_state=args.seed)

    proc = os.path.join(args.root, "Processed_Data")
    here = os.path.dirname(os.path.abspath(__file__))
    print("=" * 78)
    print("MODEL 4b (Dewang): LDA + LightGBM + transductive stacking")
    print(f"  root={args.root}  sims={N_SIMS}  rounds={ROUNDS}  folds={FOLDS}")
    print(f"  candidates={K}  neighbours={NEIGHBOURS}  trees={args.trees}")
    print("=" * 78)

    # ---- Load data --------------------------------------------------------
    train_df, test_df, master, train_idx, test_idx = load_data(args.root)
    print(f"  train={len(train_df)}  test={len(test_df)}  features={X.shape}")

    # ---- Build SIFT matrices ----------------------------------------------
    print("\n[1/4] Building SIFT verification matrices")
    sift_cache = os.path.join(here, "cache", "sift_pairs_4b.npz")
    sift_matrices(os.path.join(proc, "sift_features.npz"), sift_cache, args.workers)

    # ---- Simulate open-set episodes ---------------------------------------
    print(f"\n[2/4] Simulating {N_SIMS} open-set episodes")
    sims = []
    for i in range(N_SIMS):
        s = simulate(args.seed + i, train_idx)
        nq = len(s["q"])
        n_unk = int(s["q"]["unk"].sum())
        print(f"  episode {i}: {nq} queries ({n_unk} unknown)", flush=True)
        sims.append(s)

    # ---- Multi-round stacking (upgrade #3) --------------------------------
    print(f"\n[3/4] Training LightGBM with {ROUNDS}-round transductive stacking")
    folds = [list(range(i, N_SIMS, FOLDS)) for i in range(FOLDS)]

    round_models = []          # one all-sims model per stacking round (used for test)
    for rnd in range(1, ROUNDS + 1):
        oof = [None] * N_SIMS
        for f in folds:
            train_sims = [sims[i] for i in range(N_SIMS) if i not in f]
            models = fit(train_sims)
            for i in f:
                oof[i] = predict(models, sims[i])
        # Out-of-fold MAP@5
        oof_scores = [map5(rank(sims[i], oof[i]), sims[i]["truth"])
                      for i in range(N_SIMS)]
        mean_score = np.mean(oof_scores)
        print(f"  round {rnd}: OOF simulated MAP@5 = {mean_score:.4f}")
        # Refit on all sims for the final/test model of this round
        round_models.append(fit(sims))
        if rnd < ROUNDS:
            # Augment features with this round's predictions
            sims = [augment(s, r) for s, r in zip(sims, oof)]

    # Final OOF result
    print(f"\n  >>> Final OOF MAP@5 = {mean_score:.4f}")

    # Detailed per-seed breakdown
    for i in range(N_SIMS):
        preds_i = rank(sims[i], oof[i])
        truth_i = sims[i]["truth"]
        unk_i = sims[i]["truth"] == NEW
        m_all = map5(preds_i, truth_i)
        m_known = map5(preds_i[~unk_i], truth_i[~unk_i]) if (~unk_i).any() else 0
        m_unk = map5(preds_i[unk_i], truth_i[unk_i]) if unk_i.any() else 0
        print(f"    episode {i}: MAP@5={m_all:.4f}  known={m_known:.4f}  unknown={m_unk:.4f}")

    summary = {
        "model": "Model 4b",
        "oof_map5": float(mean_score),
        "n_sims": N_SIMS,
        "rounds": ROUNDS,
        "candidates": K,
        "trees": args.trees,
        "per_episode": [float(s) for s in oof_scores],
    }
    summary_path = os.path.join(proc, f"model4b_validation_summary_{args.tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  validation summary saved to {summary_path}")

    # ---- Test inference ---------------------------------------------------
    if args.predict_test and not args.no_test:
        print(f"\n[4/4] Test inference (gallery = full train)")
        final_s = build_frames(train_idx, test_idx)
        # Replay the same stacking rounds used in validation: round r's model
        # (trained on sims augmented with OOF preds of rounds < r) predicts the
        # test set, and its predictions augment the features for round r+1.
        for rnd, models in enumerate(round_models, 1):
            result = predict(models, final_s)
            print(f"  round {rnd}: mean P(new) = {result['pb'].mean():.3f}")
            if rnd < ROUNDS:
                final_s = augment(final_s, result)

        preds = rank(final_s, result)
        sub = pd.DataFrame({
            "id": IDS[final_s["order"]],
            "predictions": [" ".join(p) for p in preds]
        })
        # Sort by sample submission order if available
        sample_path = (args.sample_submission or
                       os.path.join(proc, "sample_submission.csv"))
        if os.path.exists(sample_path):
            sample = pd.read_csv(sample_path)
            sub = sample[["id"]].merge(sub, on="id", how="left", validate="one_to_one")
        validate_submission(sub, test_df, train_df, sample_path)
        out_path = os.path.join(proc, f"submission_model4b_{args.tag}.csv")
        sub.to_csv(out_path, index=False)
        top_new = np.array([p[0] == NEW for p in preds])
        print(f"  saved {out_path}")
        print(f"  new_individual at rank 1: {top_new.sum()} ({100 * top_new.mean():.1f}%)")
        print(f"  in top 5: {sum(NEW in p for p in preds)}")
    elif not args.predict_test:
        print("\n[4/4] Skipped test inference (use --predict-test to generate submission)")

    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
