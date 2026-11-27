#!/usr/bin/env python3
"""
MODEL 3e: Model 4c + the gaps listed in the team's improvement note (26 Nov).
Keep model4c_pipeline.py in the same folder.

  1. Test mismatch      simulated queries follow the test viewpoint mix; a test-vs-sim
                        classifier measures the remaining gap and gives importance
                        weights, used only if they help weighted validation.
  2. Validation         folds are split by ANIMAL, not by simulated episode.
  3. One ranker         LightGBM lambdarank over K candidates + a new_individual row;
                        compared with 4c's hand-made fusion and a 50/50 blend.
  4. Background SIFT    generic keypoints (similar to descriptors in many images) are
                        down-weighted.
  5. More SIFT          weighted inliers, inlier hull area, rotation/scale
                        consistency, affine-model inliers.
  6. PLDA               two-covariance likelihood ratio per (query, identity).

No leaderboard probing: every choice is made on training-set validation and one
submission file is written.

Rules: provided features only; numpy / scipy / scikit-learn / LightGBM / OpenCV
matching.  Item 1 uses UNLABELLED test features and metadata: confirm with the TA
team, or pass --no-test-adapt.

Validation caveat: neighbour-stacking features still come from first-level models
that saw other folds (ordinary stacking, not nested), so scores stay slightly
optimistic.  They are not comparable with 4c's episode-split numbers.

Run:  python model3e_pipeline.py                  validation only
      python model3e_pipeline.py --predict-test   validation + submission
      add --retune to search LightGBM size/regularisation on the animal split
      add --quick for one seed (fast first look)
The first run recomputes SIFT pairs (new statistics, new cache file).
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import model4c_pipeline as c4

NEW = c4.NEW
SPECIES = ("hyena", "leopard")
EXTRA = {}                  # name -> sparse (N, N) matrix of extra SIFT statistics
EXTRA_NAMES = ["wsift", "hull", "rig", "aff"]
EXTRA_PREFIX = tuple(f"{n}_" for n in EXTRA_NAMES + ["plda"])
_base_species_features = c4.species_features


# ═══ SIFT: keypoint distinctiveness + richer pair statistics ═════════════════
class SiftCache3e(c4.SiftCache):
    """4c's matcher (same good/inlier counts) with extra per-pair statistics."""

    def __init__(self, npz_path, cache_path, workers):
        super().__init__(npz_path, cache_path, workers, mem_items=2000)
        self.weights = {}

    def genericness(self, train_idx, path, n_ref=200, per_ref=100, top=10):
        """For each keypoint: mean similarity to its closest descriptor in each of the
        `top` most similar reference TRAIN images.  High = common texture."""
        g = {}
        if os.path.exists(path):
            with np.load(path) as z:
                g = {int(k): z[k] for k in z.files}
        rng = np.random.default_rng(0)
        for sp in SPECIES:
            todo = [i for i in np.where(c4.SP == sp)[0] if i not in g]
            if not todo:
                continue
            ref_ids, blocks = [], []
            for i in rng.permutation(train_idx[c4.SP[train_idx] == sp]):
                d = c4.SiftCache._descriptors_raw(self, int(i))[0]
                if len(d):
                    ref_ids.append(int(i))
                    blocks.append(d[rng.choice(len(d), per_ref, replace=len(d) < per_ref)])
                if len(ref_ids) == n_ref:
                    break
            ref = np.concatenate(blocks)
            column = {i: j for j, i in enumerate(ref_ids)}
            print(f"  keypoint weights, {sp}: {len(todo)} images vs "
                  f"{len(ref_ids)} reference images", flush=True)
            for n, i in enumerate(todo, 1):
                d = c4.SiftCache._descriptors_raw(self, int(i))[0]
                out = np.zeros(len(d), np.float32)
                for s in range(0, len(d), 512):
                    sim = d[s:s + 512] @ ref.T
                    best = sim.reshape(len(sim), len(ref_ids), per_ref).max(2)
                    if i in column:
                        best[:, column[i]] = -1.0
                    k = min(top, best.shape[1])
                    out[s:s + 512] = np.partition(best, -k, 1)[:, -k:].mean(1)
                g[int(i)] = out
                if n % 500 == 0 or n == len(todo):
                    print(f"    {n}/{len(todo)}", flush=True)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    np.savez_compressed(path, **{str(k): v for k, v in g.items()})
        for sp in SPECIES:
            ix = [i for i in np.where(c4.SP == sp)[0] if len(g.get(i, ()))]
            if not ix:
                continue
            lo, hi = np.quantile(np.concatenate([g[i] for i in ix]), [0.5, 0.95])
            for i in ix:
                self.weights[int(i)] = np.clip(
                    (hi - g[i]) / max(hi - lo, 1e-6), 0.05, 1.0).astype(np.float32)

    def _descriptors_raw(self, i):
        d, xy = super()._descriptors_raw(i)
        w = self.weights.get(int(i), np.ones(len(d), np.float32))
        area = (cv2.contourArea(cv2.convexHull(np.ascontiguousarray(xy)))
                if len(xy) >= 3 else 0.0)
        return d, xy, w, area

    def verify(self, pair):
        """(good, inliers, weighted inliers, hull fraction, rigid score, affine inliers)"""
        a, b = pair
        da, xa, wa, area_a = self._descriptors(a)
        db, xb, wb, area_b = self._descriptors(b)
        if len(da) < 4 or len(db) < 4:
            return 0, 0, 0.0, 0.0, 0.0, 0
        bf = cv2.BFMatcher()
        back = np.array([m.trainIdx for m in bf.match(db, da)])
        good = [m[0] for m in bf.knnMatch(da, db, k=2)
                if len(m) == 2 and m[0].distance < c4.MATCH_RATIO * m[1].distance
                and back[m[0].trainIdx] == m[0].queryIdx]
        if len(good) < 4:
            return len(good), 0, 0.0, 0.0, 0.0, 0
        qi = np.array([m.queryIdx for m in good])
        ti = np.array([m.trainIdx for m in good])
        src, dst = np.float32(xa[qi]), np.float32(xb[ti])
        cv2.setRNGSeed(int((int(a) * 100003 + int(b)) % (2**31 - 1)))
        M, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=c4.RANSAC_PX)
        if M is None or mask is None:
            return len(good), 0, 0.0, 0.0, 0.0, 0
        mask = mask.ravel().astype(bool)
        inl = int(mask.sum())
        winl = float(np.minimum(wa[qi[mask]], wb[ti[mask]]).sum())
        hull = rigid = 0.0
        if inl >= 5:
            fr = [cv2.contourArea(cv2.convexHull(p[mask])) / max(area, 1e-6)
                  for p, area in ((src, area_a), (dst, area_b))]
            hull = float(np.clip(min(fr), 0, 1))
            angle = abs(np.arctan2(M[1, 0], M[0, 0]))
            logscale = abs(np.log(max(np.hypot(M[0, 0], M[1, 0]), 1e-6)))
            rigid = float(np.exp(-angle / 0.5 - logscale / 0.5) * np.log1p(winl))
        _, mask2 = cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=c4.RANSAC_PX)
        return (len(good), inl, winl, hull, rigid,
                0 if mask2 is None else int(mask2.sum()))

    def save(self):
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        pairs = sorted(self.done)
        np.savez_compressed(
            self.cache_path,
            pairs=np.array(pairs, dtype=np.int64).reshape(-1, 2),
            values=np.array([self.done[p] for p in pairs], float).reshape(-1, 6))


def sift_matrices(npz_path, cache_dir, train_idx, workers):
    cache = SiftCache3e(npz_path, os.path.join(cache_dir, "sift_pairs_3e.npz"), workers)
    cv2.setNumThreads(1)
    n = len(c4.X)
    wanted = set()
    for sp in SPECIES:                       # same neighbourhood rule as 4c
        ix = np.where(c4.SP == sp)[0]
        k = min(c4.NEIGHBOURS, len(ix) - 1)
        for start in range(0, len(ix), 256):
            q = ix[start:start + 256]
            sim = c4.X[q] @ c4.X[ix].T
            sim[np.arange(len(q)), np.searchsorted(ix, q)] = -9.0
            for row, nbs in zip(q, np.argsort(-sim, 1)[:, :k]):
                wanted.update((min(int(row), int(j)), max(int(row), int(j)))
                              for j in ix[nbs])
    missing = sorted(wanted - cache.done.keys())
    print(f"  SIFT: {len(wanted)} pairs needed, {len(missing)} uncached", flush=True)
    if missing:
        cache.genericness(train_idx, os.path.join(cache_dir, "keypoint_generic_3e.npz"))
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for s0 in range(0, len(missing), 20000):
                block = missing[s0:s0 + 20000]
                for pair, value in zip(block, pool.map(cache.verify, block)):
                    cache.done[pair] = value
                cache.save()
                print(f"    verified {s0 + len(block)}/{len(missing)}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
    pp = np.array(sorted(wanted), dtype=int).reshape(-1, 2)
    vv = np.array([cache.done[tuple(p)] for p in pp], float).reshape(-1, 6)
    a, b = pp.T

    def mat(v):
        return sparse.coo_matrix((np.r_[v, v], (np.r_[a, b], np.r_[b, a])),
                                 shape=(n, n)).tocsr()

    c4.GOOD, c4.INL = mat(vv[:, 0].astype(int)), mat(vv[:, 1].astype(int))
    for j, name in enumerate(EXTRA_NAMES, 2):
        EXTRA[name] = mat(vv[:, j].astype(np.float32))


# ═══ PLDA and the extra features ═════════════════════════════════════════════
def plda_scores(gi, qi, inv, n, dim=128, shrink=0.1, n_cap=4):
    """Two-covariance PLDA log likelihood ratio: same identity vs unrelated animal."""
    allx = c4.X[np.r_[gi, qi]]
    d = min(dim, len(allx) - 1, allx.shape[1])
    z = PCA(d, svd_solver="randomized", random_state=0).fit(allx).transform(allx)
    z = c4.unit(z - z[:len(gi)].mean(0)) * np.sqrt(d)
    zg, zq = z[:len(gi)], z[len(gi):]
    count = np.bincount(inv, minlength=n).astype(float)
    means = np.zeros((n, d))
    np.add.at(means, inv, zg)
    means /= count[:, None]
    res = zg - means[inv]
    Sw = res.T @ res / max(len(gi) - n, 1)
    Sw = (1 - shrink) * Sw + shrink * (np.trace(Sw) / d + 1e-6) * np.eye(d)
    mu = means - means.mean(0)
    psi, V = eigh(mu.T @ mu / max(n - 1, 1), Sw)          # V' Sw V = I
    psi = np.maximum(psi - np.mean(1 / count), 1e-4)       # noise in the class means
    centre = zg.mean(0)
    uq, ug = (zq - centre) @ V, (means - centre) @ V
    ne = np.minimum(count, n_cap)[:, None]                 # images are not independent
    pm = ne * psi / (ne * psi + 1) * ug
    pv = 1 + psi / (ne * psi + 1)
    same = -0.5 * ((uq ** 2) @ (1 / pv).T - 2 * uq @ (pm / pv).T
                   + (pm ** 2 / pv + np.log(pv)).sum(1))
    diff = -0.5 * ((uq ** 2) @ (1 / (1 + psi)) + np.log(1 + psi).sum())
    return same - diff[:, None]


def species_features(gi, qi, sp):
    q, cd = _base_species_features(gi, qi, sp)
    ids, inv = np.unique(c4.Y[gi], return_inverse=True)
    n = len(ids)
    r, c = cd["q"].to_numpy(), cd["c"].to_numpy()

    def add(name, M):
        top = -np.sort(-M, 1)[:, :2]
        cd[f"{name}_v"] = M[r, c]
        cd[f"{name}_gap"] = M[r, c] - top[r, 0]
        cd[f"{name}_rk"] = (-M).argsort(1).argsort(1)[r, c]
        q[f"{name}_top"] = top[:, 0]
        q[f"{name}_m12"] = top[:, 0] - top[:, 1]

    for name in EXTRA_NAMES:
        S = EXTRA[name][qi][:, gi].toarray().astype(np.float32)
        best, top2 = c4.per_identity(S, inv, n)
        add(name, best)
        cd[f"{name}_top2"] = top2[r, c]
    # Raw likelihood ratios change scale with gallery size (sim vs full train), so
    # they are standardised per query and absolute top values are not used.
    L = plda_scores(gi, qi, inv, n)
    L = (L - L.mean(1, keepdims=True)) / np.maximum(L.std(1, keepdims=True), 1e-9)
    add("plda", L)
    del q["plda_top"]
    return q, cd


c4.species_features = species_features


# ═══ simulation that follows the test viewpoint mix ══════════════════════════
def viewpoint_weights(train_idx, test_idx):
    """Per image: test share / train share of its (species, viewpoint), clipped."""
    w = np.ones(len(c4.X))
    for sp in SPECIES:
        tr = train_idx[c4.SP[train_idx] == sp]
        te = test_idx[c4.SP[test_idx] == sp]
        if not len(tr) or not len(te):
            continue
        for v in np.unique(c4.VIEW[tr]):
            ratio = np.mean(c4.VIEW[te] == v) / np.mean(c4.VIEW[tr] == v)
            w[tr[c4.VIEW[tr] == v]] = np.clip(ratio, 0.35, 3.0)
    return w


def simulate(seed, train_idx, view_w=None):
    """4c's open-set episode; held-out images are drawn in proportion to view_w."""
    if view_w is None:
        return c4.simulate(seed, train_idx)
    rng = np.random.default_rng(seed)
    link = c4.REGIMES[seed % len(c4.REGIMES)]
    gallery, known, unknown = [], [], []
    for sp in SPECIES:
        p_unknown = rng.uniform(0.2, 0.35)
        frac = rng.uniform(*c4.FRAC)
        cap = int(rng.integers(12, 33))
        ix = train_idx[c4.SP[train_idx] == sp]
        names = np.array(sorted(set(c4.Y[ix])))
        rng.shuffle(names)
        new_set = set(names[:int(round(p_unknown * len(names)))])
        for name in names:
            rows = ix[c4.Y[ix] == name].copy()
            rng.shuffle(rows)
            if name in new_set:
                keys = rng.random(len(rows)) ** (1 / view_w[rows])
                unknown.extend(rows[np.argsort(-keys)][:cap].tolist())
            elif len(rows) < 2:
                gallery.extend(rows.tolist())
            else:
                n_q = min(max(int(round(frac * len(rows))), 1), cap, len(rows) - 1)
                groups = c4.duplicate_groups(rows, link)
                uniq = np.unique(groups)
                gw = np.array([view_w[rows[groups == g]].mean() for g in uniq])
                held = []
                for g in uniq[np.argsort(-rng.random(len(uniq)) ** (1 / gw))]:
                    members = rows[groups == g].tolist()
                    if len(held) < n_q and len(held) + len(members) < len(rows):
                        held.extend(members)
                known.extend(held[:cap])
                held_set = set(held)
                gallery.extend(i for i in rows if i not in held_set)
    gi = np.array(gallery, dtype=int)
    qi = np.array(known + unknown, dtype=int)
    frames = c4.build_frames(gi, qi)
    truth = np.where(np.isin(frames["order"], unknown), NEW, c4.Y[frames["order"]])
    frames["truth"] = truth
    frames["q"]["unk"] = (truth == NEW).astype(int)
    frames["c"]["y"] = (frames["c"].ident.to_numpy(dtype=object) ==
                        truth[frames["c"].q.to_numpy()]).astype(int)
    return frames


def adversarial(sims, final):
    """Can a classifier tell real test queries from simulated ones?  Returns the AUC,
    the most telling features and a weight per simulated query."""
    cols = [x for x in c4.columns(sims[0])[0] if x in final["q"].columns]
    A = pd.concat([s["q"][cols] for s in sims], ignore_index=True).astype(float)
    B = final["q"][cols].astype(float).reset_index(drop=True)
    x = pd.concat([A, B], ignore_index=True)
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    group = np.r_[np.concatenate([s["order"] for s in sims]), final["order"]]
    p, imp = np.zeros(len(y)), np.zeros(len(cols))
    for tr, te in GroupKFold(5).split(x, y, group):      # repeated images stay together
        m = lgb.LGBMClassifier(n_estimators=150, num_leaves=7, learning_rate=0.05,
                               min_child_samples=100, reg_lambda=5.0, verbose=-1)
        m.fit(x.iloc[tr], y[tr])
        p[te] = m.predict_proba(x.iloc[te])[:, 1]
        imp += m.booster_.feature_importance("gain")
    ps = np.clip(p[:len(A)], 1e-4, 1 - 1e-4)
    w = np.clip(np.sqrt(ps / (1 - ps) * len(A) / len(B)), 0.25, 4.0)
    top = [(cols[i], imp[i] / max(imp.sum(), 1e-9)) for i in np.argsort(-imp)[:8]]
    return roc_auc_score(y, p), top, w / w.mean()


# ═══ identity-disjoint training ══════════════════════════════════════════════
def identity_folds(train_idx, n_folds, seed):
    """One fold per animal, balanced by image count within each species."""
    rng = np.random.default_rng(seed)
    fold = {}
    for sp in SPECIES:
        names, counts = np.unique(c4.Y[train_idx[c4.SP[train_idx] == sp]],
                                  return_counts=True)
        load = np.zeros(n_folds)
        for i in np.lexsort((rng.random(len(names)), -counts)):
            f = int(load.argmin())
            fold[names[i]] = f
            load[f] += counts[i]
    return fold


def fit_models(Q, C, unk, y, qmask, w):
    """4c's three bagged classifiers, trained on the queries selected by qmask."""
    K = c4.K
    cmask = np.repeat(qmask, K)
    kmask = np.repeat(qmask & (unk == 0), K)
    wc = np.repeat(w, K)

    def bag(params, x, target, weight):
        base = params.get("random_state", 0) or 0
        return c4.Bag([lgb.LGBMClassifier(**{**params, "random_state": base + 101 * k})
                       .fit(x, target, sample_weight=weight) for k in range(c4.SEEDS)])

    det = {**c4.PARAMS, "n_estimators": 300,
           "num_leaves": min(15, c4.PARAMS["num_leaves"])}
    return (bag(det, Q[qmask], unk[qmask], w[qmask]),
            bag(c4.PARAMS, C[kmask], y[kmask], wc[kmask]),
            bag(c4.PARAMS, C[cmask], y[cmask], wc[cmask]))


def stack(sims, qfold, weights, label, rounds=None, keep_models=True):
    """Multi-round stacking with animal-disjoint folds.  weights=None means uniform."""
    rounds = rounds or c4.ROUNDS
    sizes = [len(s["q"]) for s in sims]
    cuts = np.cumsum(sizes)[:-1]
    fold = np.concatenate(qfold)
    w = np.ones(len(fold)) if weights is None else weights
    round_models = []
    for rnd in range(1, rounds + 1):
        qf, cf = c4.columns(sims[0])
        Q = pd.concat([s["q"][qf] for s in sims], ignore_index=True).astype(np.float32)
        C = pd.concat([c4.candidate_matrix(s, qf, cf) for s in sims],
                      ignore_index=True).astype(np.float32)
        unk = np.concatenate([s["q"].unk.to_numpy() for s in sims])
        y = np.concatenate([s["c"].y.to_numpy() for s in sims])
        res = {k: np.zeros((len(fold), c4.K)) for k in ("raw", "pc")}
        res.update(pn=np.zeros(len(fold)), pb=np.zeros(len(fold)))
        for f in np.unique(fold):
            test = fold == f
            ctest = np.repeat(test, c4.K)
            det, closed, anyq = fit_models(Q, C, unk, y, ~test, w)
            raw = anyq.predict_proba(C[ctest])[:, 1].reshape(-1, c4.K)
            within = closed.predict_proba(C[ctest])[:, 1].reshape(-1, c4.K)
            res["raw"][test] = raw
            res["pc"][test] = within / np.maximum(within.sum(1, keepdims=True), 1e-9)
            res["pn"][test] = det.predict_proba(Q[test])[:, 1]
            res["pb"][test] = (res["pn"][test] + np.clip(1 - raw.sum(1), 0, 1)) / 2
        if keep_models:
            round_models.append((*fit_models(Q, C, unk, y, np.ones(len(fold), bool), w),
                                 (qf, cf)))
        oof = [{k: v for k, v in zip(res, vals)} for vals in
               zip(*[np.split(res[k], cuts) for k in res])]
        ap = np.concatenate([ap_of(s, fusion(s, r)) for s, r in zip(sims, oof)])
        print(f"  [{label}] round {rnd}: MAP@5 = {ap.mean():.4f}", flush=True)
        if rnd < rounds:
            sims = [c4.augment(s, r) for s, r in zip(sims, oof)]
    return dict(models=round_models, sims=sims, oof=oof, ap=ap)


# ═══ scores, the single ranker and the comparison ════════════════════════════
def fusion(s, r):
    """4c's scores over K candidates + new_individual (rows sum to 1)."""
    return np.concatenate([(1 - r["pb"])[:, None] * r["pc"], r["pb"][:, None]], 1)


def top5(s, scores):
    nq = len(s["q"])
    labels = np.concatenate([s["c"].ident.to_numpy(dtype=object).reshape(nq, c4.K),
                             np.full((nq, 1), NEW, dtype=object)], 1)
    return np.take_along_axis(labels, np.argsort(-scores, 1, kind="stable")[:, :5], 1)


def ap_of(s, scores):
    hit = top5(s, scores) == s["truth"][:, None]
    return (hit / np.arange(1, 6)).sum(1)


def rank_matrix(s, qf, cf):
    """(queries x (K+1), features): K candidate rows then one new_individual row."""
    nq, K = len(s["q"]), c4.K
    C = s["c"][cf].to_numpy(np.float32).reshape(nq, K, len(cf))
    C = np.concatenate([C, np.full((nq, 1, len(cf)), np.nan, np.float32)], 1)
    Q = np.repeat(s["q"][qf].to_numpy(np.float32)[:, None, :], K + 1, 1)
    flag = np.zeros((nq, K + 1, 1), np.float32)
    flag[:, K] = 1
    return np.concatenate([C, Q, flag], 2).reshape(nq * (K + 1), -1)


def fit_ranker(X, rel, qmask, w):
    """rel: (queries, K+1) relevance.  Queries with no correct row are skipped."""
    use = qmask & (rel.sum(1) > 0)
    rows = np.repeat(use, rel.shape[1])
    params = {k: v for k, v in c4.PARAMS.items() if k != "objective"}
    return [lgb.LGBMRanker(objective="lambdarank", lambdarank_truncation_level=8,
                           **{**params, "random_state": 11 + 101 * k})
            .fit(X[rows], rel[use].ravel(), group=np.full(use.sum(), rel.shape[1]),
                 sample_weight=np.repeat(w[use], rel.shape[1]))
            for k in range(c4.SEEDS)]


def ranker_scores(models, X):
    z = np.mean([m.predict(X, raw_score=True) for m in models], 0).reshape(-1, c4.K + 1)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def final_layer(run, qfold, weights):
    """Out-of-fold ranker on top of the last round; returns scores for three choices."""
    sims = [c4.augment(s, r) for s, r in zip(run["sims"], run["oof"])]
    qf, cf = c4.columns(sims[0])
    X = np.concatenate([rank_matrix(s, qf, cf) for s in sims])
    rel = np.concatenate([np.concatenate(
        [s["c"].y.to_numpy().reshape(-1, c4.K), s["q"].unk.to_numpy()[:, None]], 1)
        for s in sims])
    fold = np.concatenate(qfold)
    w = np.ones(len(fold)) if weights is None else weights
    scores = np.zeros((len(fold), c4.K + 1))
    for f in np.unique(fold):
        test = fold == f
        scores[test] = ranker_scores(fit_ranker(X, rel, ~test, w),
                                     X[np.repeat(test, c4.K + 1)])
    parts = np.split(scores, np.cumsum([len(s["q"]) for s in sims])[:-1])
    options = {"fusion": [fusion(s, r) for s, r in zip(sims, run["oof"])],
               "ranker": parts}
    options["blend"] = [(a + b) / 2 for a, b in zip(options["fusion"], parts)]
    ap = {k: np.concatenate([ap_of(s, m) for s, m in zip(sims, v)])
          for k, v in options.items()}
    full = (fit_ranker(X, rel, np.ones(len(fold), bool), w), qf, cf)
    return ap, full


def bootstrap(diff, animals, weights=None, reps=2000, seed=0):
    """Mean difference and 95% interval, resampling whole animals."""
    w = np.ones(len(diff)) if weights is None else weights
    _, g = np.unique(animals, return_inverse=True)
    num, den = np.bincount(g, diff * w), np.bincount(g, w)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(num), (reps, len(num)))
    boot = num[draws].sum(1) / den[draws].sum(1)
    return num.sum() / den.sum(), np.quantile(boot, [0.025, 0.975])


def prior_em(p, train_prior, iters=200):
    """Saerens et al. 2002: re-estimate the new-animal share from unlabelled scores."""
    pi = train_prior
    for _ in range(iters):
        a = p * pi / train_prior
        post = a / (a + (1 - p) * (1 - pi) / (1 - train_prior))
        pi = float(np.clip(post.mean(), 1e-3, 1 - 1e-3))
    return pi, post


RETUNE_GRID = [(31, 40, 2.0), (15, 40, 2.0), (15, 100, 5.0),
               (31, 100, 5.0), (7, 100, 10.0), (63, 40, 2.0)]


# ═══ main ════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--root", default=os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    ap_.add_argument("--sample-submission", default=None)
    ap_.add_argument("--sims", type=int, default=24)
    ap_.add_argument("--folds", type=int, default=5, help="animal folds")
    ap_.add_argument("--rounds", type=int, default=3)
    ap_.add_argument("--candidates", type=int, default=15)
    ap_.add_argument("--neighbours", type=int, default=150)
    ap_.add_argument("--trees", type=int, default=400)
    ap_.add_argument("--seeds", type=int, default=3)
    ap_.add_argument("--seed", type=int, default=42)
    ap_.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 2) - 1))
    ap_.add_argument("--quick", action="store_true", help="one LightGBM seed")
    ap_.add_argument("--retune", action="store_true")
    ap_.add_argument("--no-compare", action="store_true", help="skip the 4c-feature run")
    ap_.add_argument("--no-test-adapt", action="store_true",
                     help="no viewpoint matching, weighting or prior estimate from test")
    ap_.add_argument("--em-prior", action="store_true",
                     help="apply the estimated new-animal share to the submission")
    ap_.add_argument("--drop", default="",
                     help="comma list of feature columns to exclude (e.g. shifted ones)")
    ap_.add_argument("--predict-test", action="store_true")
    args = ap_.parse_args()
    if args.em_prior and args.no_test_adapt:
        ap_.error("--em-prior uses test predictions; drop --no-test-adapt")

    c4.K, c4.ROUNDS, c4.NEIGHBOURS = args.candidates, args.rounds, args.neighbours
    c4.SEEDS = 1 if args.quick else args.seeds
    c4.PARAMS.update(n_estimators=args.trees, random_state=args.seed)
    adapt = not args.no_test_adapt
    proc = os.path.join(args.root, "Processed_Data")
    here = os.path.dirname(os.path.abspath(__file__))

    print(f"MODEL 3e   sims={args.sims} rounds={c4.ROUNDS} animal folds={args.folds} "
          f"seeds={c4.SEEDS} test-adapt={adapt}")
    train_df, test_df, _, train_idx, test_idx = c4.load_data(args.root)
    print("\n[1] SIFT")
    sift_matrices(os.path.join(proc, "sift_features.npz"),
                  os.path.join(here, "cache"), train_idx, args.workers)

    print("\n[2] Simulated episodes")
    view_w = viewpoint_weights(train_idx, test_idx) if adapt else None
    sims = []
    for i in range(args.sims):
        sims.append(simulate(args.seed + i, train_idx, view_w))
        print(f"  episode {i} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  candidate recall@{c4.K}: "
          f"{np.mean([c4.candidate_recall(s) for s in sims]):.4f}")
    order = np.concatenate([s["order"] for s in sims])
    for sp in SPECIES:
        flank = lambda ix: np.mean(np.isin(c4.VIEW[ix[c4.SP[ix] == sp]], (0, 1)))
        print(f"  left/right share, {sp}: train {flank(train_idx):.2f}  "
              f"sim queries {flank(order):.2f}  test {flank(test_idx):.2f}")
    drop = [x for x in args.drop.split(",") if x]
    protected = {"unk", "q", "c", "y", "ident"} | {
        f"{p}{k}" for p in ("nb", "nw", "sb", "sw") for k in range(5)}
    known_cols = set(sims[0]["q"]) | set(sims[0]["c"])
    if set(drop) - (known_cols - protected):
        raise SystemExit(f"cannot drop {sorted(set(drop) - (known_cols - protected))}")

    def without(s):
        return dict(s, q=s["q"].drop(columns=drop, errors="ignore"),
                    c=s["c"].drop(columns=drop, errors="ignore"))

    fold_of = identity_folds(train_idx, args.folds, args.seed)
    qfold = [np.array([fold_of[c4.Y[i]] for i in s["order"]]) for s in sims]
    animals = c4.Y[order].astype(str)

    final = c4.build_frames(train_idx, test_idx) if adapt or args.predict_test else None
    # Apply --drop to the simulations and the test frame alike.
    sims = [without(s) for s in sims]
    final = None if final is None else without(final)
    adv_w = None
    if adapt:
        print("\n[3] Test-vs-simulation classifier")
        auc, top, adv_w = adversarial(sims, final)
        print(f"  AUC = {auc:.3f}  (0.5 = indistinguishable)")
        for name, share in top:
            print(f"    {name:12s} {share:.0%} of gain")
        print(f"  effective sample size under weights: "
              f"{adv_w.sum() ** 2 / (adv_w ** 2).sum() / len(adv_w):.0%}")
    metric_w = adv_w                       # used to score; None = plain mean
    score = lambda a: float(np.average(a, weights=metric_w))
    summary = {"model": "3e", "test_adapt": adapt}
    if adapt:
        summary["adversarial_auc"] = float(auc)

    print("\n[4] Animal-disjoint validation")
    base = None
    if not args.no_compare:
        stripped = [dict(s, q=s["q"].loc[:, ~s["q"].columns.str.startswith(EXTRA_PREFIX)],
                         c=s["c"].loc[:, ~s["c"].columns.str.startswith(EXTRA_PREFIX)])
                    for s in sims]
        base = stack(stripped, qfold, None, "4c features", keep_models=False)["ap"]
    if args.retune:
        results = []
        for leaves, child, lam in RETUNE_GRID:
            c4.PARAMS.update(num_leaves=leaves, min_child_samples=child, reg_lambda=lam)
            a = stack(sims, qfold, None, f"leaves={leaves} min={child} l2={lam}",
                      rounds=1, keep_models=False)["ap"]
            results.append(score(a))
        leaves, child, lam = RETUNE_GRID[int(np.argmax(results))]
        c4.PARAMS.update(num_leaves=leaves, min_child_samples=child, reg_lambda=lam)
        print(f"  retune chose leaves={leaves} min_child={child} l2={lam}")
        summary["retuned"] = [leaves, child, lam]

    run = stack(sims, qfold, None, "3e features")
    train_w = None
    if adapt:
        wrun = stack(sims, qfold, adv_w, "3e weighted")
        better = score(wrun["ap"]) > score(run["ap"])
        print(f"  weighted-validation score: plain {score(run['ap']):.4f}  "
              f"weighted {score(wrun['ap']):.4f} -> {'weighted' if better else 'plain'}")
        if better:
            run, train_w = wrun, adv_w
    layer_ap, ranker = final_layer(run, qfold, train_w)
    choice = max(layer_ap, key=lambda k: score(layer_ap[k]))
    print("\n[5] Final layer" + (" (scores weighted towards test-like queries)"
                                 if adapt else ""))
    for k, a in layer_ap.items():
        print(f"  {k:7s} MAP@5 = {score(a):.4f}  (unweighted {a.mean():.4f})")
    print(f"  chosen: {choice}")
    chosen = layer_ap[choice]
    summary.update(choice=choice, weighted_training=train_w is not None,
                   map5=score(chosen), map5_unweighted=float(chosen.mean()))
    unk = np.concatenate([s["q"].unk.to_numpy() for s in sims])
    for name, m in (("known", unk == 0), ("new", unk == 1)):
        print(f"  {name:5s} queries: MAP@5 = {chosen[m].mean():.4f}")
    if base is not None:
        gain, (lo, hi) = bootstrap(chosen - base, animals, metric_w)
        summary.update(map5_4c=score(base), gain=float(gain), gain_95=[float(lo), float(hi)])
        print(f"\n  >>> 3e - 4c features = {gain:+.4f}   95% interval over animals "
              f"[{lo:+.4f}, {hi:+.4f}]")
        print("      " + ("interval is above zero: a real gain on this validation"
                          if lo > 0 else "interval includes zero or less: not shown to help"))
    with open(os.path.join(proc, "model3e_validation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if args.predict_test:
        print("\n[6] Test inference")
        for rnd, models in enumerate(run["models"], 1):
            result = c4.predict(models, final)
            final = c4.augment(final, result)
        models, qf, cf = ranker
        rs = ranker_scores(models, rank_matrix(final, qf, cf))
        fs = fusion(final, result)
        scores = {"fusion": fs, "ranker": rs, "blend": (fs + rs) / 2}[choice]
        if adapt:
            # reference share and calibration both come from the simulations
            prior = float(np.average(unk, weights=train_w))
            pi, post = prior_em(np.clip(result["pb"], 1e-5, 1 - 1e-5), prior)
            print(f"  new-animal share: simulations {prior:.3f}, test estimate {pi:.3f}"
                  f" ({'applied' if args.em_prior else 'not applied'})")
            if args.em_prior:
                p = np.clip(result["pb"], 1e-5, 1 - 1e-5)
                odds = (post / (1 - post)) / (p / (1 - p))
                scores = scores.copy()
                scores[:, -1] *= odds
        preds = top5(final, scores)
        sub = pd.DataFrame({"id": c4.IDS[final["order"]],
                            "predictions": [" ".join(p) for p in preds]})
        sample = args.sample_submission or os.path.join(proc, "sample_submission.csv")
        if os.path.exists(sample):
            sub = pd.read_csv(sample)[["id"]].merge(
                sub, on="id", how="left", validate="one_to_one")
        c4.validate_submission(sub, test_df, train_df, sample)
        path = os.path.join(proc, "submission_model3e.csv")
        sub.to_csv(path, index=False)
        print(f"  saved {path}  (new at rank 1: {100 * np.mean(preds[:, 0] == NEW):.1f}%)")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
