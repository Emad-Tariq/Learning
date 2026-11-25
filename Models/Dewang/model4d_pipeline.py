#!/usr/bin/env python3
"""
MODEL 4d: Model 4c + new evidence sources.  Keep model4c_pipeline.py in the same folder.

New feature groups (column prefix):
  ml_  linear metric learning (supervised contrastive loss, plain numpy, one matrix)
  df_  label diffusion on a kNN graph of embedding + SIFT edges (transductive)
  kr_  k-reciprocal neighbourhood (Jaccard) re-ranking similarity
  qe_  alpha query expansion
  cu_  average-linkage clustering of queries, evidence pooled per cluster
  sf_  extra SIFT statistics (inlier ratio, per-identity inlier mass)
Plus: LambdaRank candidate ranker and tuning of the new_individual score.

Rules: provided features only.  df_ (graph label propagation) is not named
explicitly in the permitted list: confirm with the TA team, or run with --off df.

Run:  python model4d_pipeline.py                      (4c vs 4d on identical episodes)
      python model4d_pipeline.py --ablate             (also drop one group at a time)
      python model4d_pipeline.py --predict-test       (write submissions)
      python model4d_pipeline.py --off df,cu          (disable groups)
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

import model4c_pipeline as c4

GROUPS = ["ml", "df", "kr", "qe", "cu", "sf"]
OFF = set()
_base_species_features = c4.species_features


# ─── similarity signals ──────────────────────────────────────────────────────
def metric_embed(gi, qi, labels, dim=128, steps=300, batch_ids=64, temp=0.1,
                 lr=1e-3, reg=1e-2, seed=0):
    """Linear projection trained with a supervised contrastive loss on GALLERY labels."""
    rng = np.random.default_rng(seed)
    allx = c4.X[np.r_[gi, qi]]
    d = min(256, len(allx) - 1, allx.shape[1])
    pca = PCA(d, svd_solver="randomized", random_state=seed).fit(allx)
    scale = (pca.explained_variance_ + 1e-9) ** -0.25          # partial whitening
    xg = c4.unit(pca.transform(c4.X[gi]) * scale)
    xq = c4.unit(pca.transform(c4.X[qi]) * scale)
    dim = min(dim, d)
    W0 = np.eye(d, dim)
    W = W0.copy()
    m, v = np.zeros_like(W), np.zeros_like(W)
    members = [np.where(labels == n)[0] for n in np.unique(labels)]
    members = [r for r in members if len(r) >= 2]
    if len(members) >= 2:
        for t in range(1, steps + 1):
            pick = rng.choice(len(members), min(batch_ids, len(members)), replace=False)
            rows = [rng.choice(members[p], min(4, len(members[p])), replace=False)
                    for p in pick]
            y = np.repeat(np.arange(len(rows)), [len(r) for r in rows])
            x = xg[np.concatenate(rows)]
            z = x @ W
            nz = np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
            u = z / nz
            s = u @ u.T / temp
            np.fill_diagonal(s, -1e9)
            p = np.exp(s - s.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            pos = (y[:, None] == y[None, :]).astype(float)
            np.fill_diagonal(pos, 0)
            pos /= pos.sum(1, keepdims=True)
            G = (p - pos) / len(y)
            du = (G + G.T) @ u / temp
            dz = (du - u * (u * du).sum(1, keepdims=True)) / nz
            grad = x.T @ dz + reg * (W - W0)
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad ** 2
            W -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return c4.unit(xg @ W), c4.unit(xq @ W)


def knn_matrix(sim, k):
    """Sparse 0/1 matrix marking each row's k largest entries."""
    n = len(sim)
    k = min(k, n - 1)
    nb = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    return sparse.csr_matrix(
        (np.ones(n * k), (np.repeat(np.arange(n), k), nb.ravel())), shape=(n, n))


def diffuse(S_cos, inl, inv, n_ids, k=8, alpha=0.8, iters=25):
    """Spread gallery identity labels over the image graph.  Nodes: gallery, then query."""
    N, ng = len(S_cos), len(inv)
    A = knn_matrix(S_cos, k).multiply(np.clip(S_cos, 0, None) ** 3)
    sift = np.where(inl >= 5, np.tanh((inl - 4) / 15.0), 0.0)
    np.fill_diagonal(sift, 0)
    A = sparse.csr_matrix(A) + sparse.csr_matrix(sift)
    A = (A + A.T) * 0.5
    d = 1.0 / np.sqrt(np.maximum(np.asarray(A.sum(1)).ravel(), 1e-12))
    A = sparse.diags(d) @ A @ sparse.diags(d)
    Y0 = np.zeros((N, n_ids))
    Y0[np.arange(ng), inv] = 1.0 / np.sqrt(np.bincount(inv, minlength=n_ids))[inv]
    F = Y0.copy()
    for _ in range(iters):
        F = alpha * (A @ F) + (1 - alpha) * Y0
    return F[ng:]


def k_reciprocal(S, ng, k=12):
    """Jaccard overlap of k-reciprocal neighbour sets (query rows, gallery columns)."""
    near = knn_matrix(S, k)
    R = near.multiply(near.T).tocsr() + sparse.identity(len(S), format="csr")
    size = np.asarray(R.sum(1)).ravel()
    inter = (R[ng:] @ R[:ng].T).toarray()
    return inter / np.maximum(size[ng:, None] + size[None, :ng] - inter, 1e-12)


def query_expand(gi, qi, S, top=3):
    """Replace each query by a similarity^3-weighted mean of itself and its neighbours."""
    allx = c4.X[np.r_[gi, qi]]
    ng = len(gi)
    rows = S[ng:]
    nb = np.argsort(-rows, 1)[:, :top]
    w = np.clip(np.take_along_axis(rows, nb, 1), 0, None) ** 3
    q = c4.unit(c4.X[qi] + (w[:, :, None] * allx[nb]).sum(1))
    return q @ c4.X[gi].T


def cluster_queries(Sqq, inl_qq, cut=0.35):
    """Average-linkage clusters of query images (embedding or SIFT agreement)."""
    n = len(Sqq)
    if n < 2:
        return np.zeros(n, int)
    sim = np.maximum(Sqq, np.tanh(inl_qq / 20.0))
    dist = np.clip(1 - (sim + sim.T) / 2, 0, 2)
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform(dist, checks=False), method="average")
    return np.unique(fcluster(Z, cut, criterion="distance"), return_inverse=True)[1]


# ─── extra features, appended to the 4c frames ───────────────────────────────
def species_features(gi, qi, sp):
    q, cd = _base_species_features(gi, qi, sp)
    ids, inv = np.unique(c4.Y[gi], return_inverse=True)
    n, nq, ng = len(ids), len(qi), len(gi)
    r, c = cd["q"].to_numpy(), cd["c"].to_numpy()
    both = np.r_[gi, qi]
    S = c4.X[both] @ c4.X[both].T
    np.fill_diagonal(S, -1.0)
    inl = c4.INL[both][:, both].toarray().astype(np.float32)
    good_qg = c4.GOOD[qi][:, gi].toarray().astype(np.float32)
    inl_qg = inl[ng:, :ng]

    def add(name, M, summary=True):
        """Candidate value, gap to the query's best, rank; plus query top-1 and margin."""
        top = -np.sort(-M, 1)[:, :2]
        cd[name] = M[r, c]
        cd[f"{name}_gap"] = M[r, c] - top[r, 0]
        cd[f"{name}_rk"] = (-M).argsort(1).argsort(1)[r, c]
        if summary:
            q[f"{name}_top"] = top[:, 0]
            q[f"{name}_m12"] = top[:, 0] - top[:, 1]

    if "ml" not in OFF:
        Mg, Mq = metric_embed(gi, qi, c4.Y[gi])
        sim = Mq @ Mg.T
        best, top2 = c4.per_identity(sim, inv, n)
        add("ml_max", best)
        add("ml_top2", top2, summary=False)
        post = c4.posterior(sim, inv, n)
        add("ml_post", post)
    if "df" not in OFF:
        F = diffuse(S, inl, inv, n)
        mass = F.sum(1)
        P = F / np.maximum(mass[:, None], 1e-12)
        add("df_p", P)
        add("df_abs", F)
        q["df_mass"] = mass
        q["df_ent"] = -(P * np.log(np.maximum(P, 1e-12))).sum(1) / np.log(max(n, 2))
    if "kr" not in OFF:
        J = k_reciprocal(S, ng)
        add("kr_max", c4.per_identity(J, inv, n)[0])
        add("kr_mix", c4.per_identity(0.7 * S[ng:, :ng] + 0.3 * J, inv, n)[0])
    if "qe" not in OFF:
        add("qe_max", c4.per_identity(query_expand(gi, qi, S), inv, n)[0])
    if "cu" not in OFF:
        lab = cluster_queries(S[ng:, ng:].clip(-1, 1), inl[ng:, ng:])
        size = np.bincount(lab)
        member = sparse.csr_matrix((1.0 / size[lab], (lab, np.arange(nq))),
                                   shape=(len(size), nq))
        raw = c4.per_identity(c4.X[qi] @ c4.X[gi].T, inv, n)[0]
        sift = c4.per_identity(inl_qg, inv, n)[0]
        add("cu_raw", np.asarray(member @ raw)[lab])
        add("cu_sift", np.asarray(member @ sift)[lab])
        q["cu_size"] = size[lab]
    if "sf" not in OFF:
        ratio = np.where(inl_qg >= 5, inl_qg / np.maximum(good_qg, 1), 0.0)
        add("sf_ratio", c4.per_identity(ratio, inv, n)[0])
        mass = np.where(inl_qg >= 6, np.log1p(inl_qg), 0.0)
        mass = np.stack([mass[:, inv == j].sum(1) for j in range(n)], 1)
        add("sf_mass", mass)
        add("sf_share", mass / np.maximum(mass.sum(1, keepdims=True), 1e-9))
    return q, cd


c4.species_features = species_features


# ─── training helpers ────────────────────────────────────────────────────────
def strip(s, groups):
    """Copy of a frame without the columns of the given feature groups."""
    pre = tuple(f"{g}_" for g in groups)
    keep = lambda df: df[[col for col in df.columns if not col.startswith(pre or ("\0",))]]
    return dict(s, q=keep(s["q"]), c=keep(s["c"]))


def stack(sims, label):
    """4c's multi-round stacking.  Returns round models, final-round frames and OOF."""
    n = len(sims)
    folds = [list(range(i, n, c4.FOLDS)) for i in range(c4.FOLDS)]
    round_models = []
    for rnd in range(1, c4.ROUNDS + 1):
        oof = [None] * n
        for f in folds:
            models = c4.fit([sims[i] for i in range(n) if i not in f])
            for i in f:
                oof[i] = c4.predict(models, sims[i])
        scores = np.array([c4.map5(c4.rank(s, r), s["truth"]) for s, r in zip(sims, oof)])
        print(f"  [{label}] round {rnd}: OOF MAP@5 = {scores.mean():.4f}", flush=True)
        if label != "ablation":
            round_models.append(c4.fit(sims))
        if rnd < c4.ROUNDS:
            sims = [c4.augment(s, r) for s, r in zip(sims, oof)]
    return round_models, sims, oof, scores


def fit_ranker(sims):
    qf, cf = c4.columns(sims[0])
    xs, ys = [], []
    for s in sims:
        y = s["c"].y.to_numpy().reshape(-1, c4.K)
        has = y.sum(1) > 0                       # queries whose true identity is a candidate
        mask = np.repeat(has, c4.K)
        xs.append(c4.candidate_matrix(s, qf, cf)[mask])
        ys.append(y[has].ravel())
    x, y = pd.concat(xs), np.concatenate(ys)
    params = {k: v for k, v in c4.PARAMS.items() if k != "objective"}
    models = [lgb.LGBMRanker(objective="lambdarank",
                             **{**params, "random_state": 7 + 101 * k})
              .fit(x, y, group=[c4.K] * (len(y) // c4.K)) for k in range(c4.SEEDS)]
    return models, (qf, cf)


def ranker_probs(ranker, s):
    models, (qf, cf) = ranker
    x = c4.candidate_matrix(s, qf, cf)
    z = np.mean([m.predict(x) for m in models], 0).reshape(-1, c4.K)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def adjust(r, pr, w, alpha, gamma):
    """w: ranker weight.  alpha: detector weight in P(new).  gamma: odds multiplier on new."""
    pb = alpha * r["pn"] + (1 - alpha) * np.clip(1 - r["raw"].sum(1), 0, 1)
    pb = gamma * pb / (gamma * pb + 1 - pb)
    return dict(r, pc=(1 - w) * r["pc"] + w * pr, pb=pb)


def tune(sims, oof, prs):
    grid = [(w, a, g) for w in (0.0, 0.25, 0.5) for a in (0.25, 0.5, 0.75)
            for g in (0.7, 0.85, 1.0, 1.2, 1.4)]
    table = np.array([[c4.map5(c4.rank(s, adjust(r, p, *g)), s["truth"])
                       for s, r, p in zip(sims, oof, prs)] for g in grid])
    default = grid.index((0.0, 0.5, 1.0))
    even = np.arange(len(sims)) % 2 == 0
    gains = []                      # choose on one half of the episodes, score on the other
    for pick, test in ((even, ~even), (~even, even)):
        if pick.any() and test.any():
            best = table[:, pick].mean(1).argmax()
            gains.append(table[best, test].mean() - table[default, test].mean())
    gain = float(np.mean(gains)) if gains else 0.0
    best = int(table.mean(1).argmax()) if gain > 0 else default
    print(f"  tuning: held-out gain {gain:+.4f}; using ranker_w={grid[best][0]} "
          f"alpha={grid[best][1]} gamma={grid[best][2]}")
    return grid[best], float(table[default].mean() + max(gain, 0.0))


def main():
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
    ap.add_argument("--sample-submission", default=None)
    ap.add_argument("--sims", type=int, default=24)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--candidates", type=int, default=15)
    ap.add_argument("--neighbours", type=int, default=150)
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 2) - 1))
    ap.add_argument("--off", default="", help=f"comma list of groups to disable: {GROUPS}")
    ap.add_argument("--no-compare", action="store_true", help="skip the 4c baseline run")
    ap.add_argument("--ablate", action="store_true", help="leave-one-group-out runs (slow)")
    ap.add_argument("--predict-test", action="store_true")
    ap.add_argument("--tag", default="4d")
    args = ap.parse_args()

    OFF.update(g for g in args.off.split(",") if g)
    if OFF - set(GROUPS):
        raise SystemExit(f"unknown groups {sorted(OFF - set(GROUPS))}; choose from {GROUPS}")
    on = [g for g in GROUPS if g not in OFF]
    c4.K, c4.FOLDS, c4.ROUNDS = args.candidates, args.folds, args.rounds
    c4.NEIGHBOURS, c4.SEEDS = args.neighbours, args.seeds
    c4.PARAMS.update(n_estimators=args.trees, random_state=args.seed)

    proc = os.path.join(args.root, "Processed_Data")
    here = os.path.dirname(os.path.abspath(__file__))
    print(f"MODEL 4d   groups on: {on}   sims={args.sims} rounds={args.rounds}")
    train_df, test_df, _, train_idx, test_idx = c4.load_data(args.root)
    print(f"  train={len(train_df)} test={len(test_df)}")
    c4.sift_matrices(os.path.join(proc, "sift_features.npz"),
                     os.path.join(here, "cache", "sift_pairs_4b.npz"), args.workers)

    print(f"\nSimulating {args.sims} episodes")
    sims = []
    for i in range(args.sims):
        sims.append(c4.simulate(args.seed + i, train_idx))
        print(f"  episode {i} done ({time.time() - t0:.0f}s)", flush=True)
    rec = np.mean([c4.candidate_recall(s) for s in sims])
    print(f"  candidate recall@{c4.K}: {rec:.4f}")

    summary = {"model": "4d", "groups": on, "candidate_recall": float(rec)}
    if not args.no_compare:
        print("\nBaseline (4c features only)")
        base = stack([strip(s, GROUPS) for s in sims], "4c")[3]
        summary["oof_4c"] = float(base.mean())

    print("\nModel 4d")
    round_models, final_sims, oof, scores = stack(sims, "4d")
    summary["oof_4d_untuned"] = float(scores.mean())
    if not args.no_compare:
        diff = scores - base
        summary["delta_vs_4c"] = float(diff.mean())
        print(f"  >>> 4d - 4c = {diff.mean():+.4f}  "
              f"(better in {(diff > 0).sum()}/{len(diff)} episodes)")

    print("\nRanker + new_individual tuning")
    n = len(final_sims)
    prs = [None] * n
    for f in range(c4.FOLDS):
        held = range(f, n, c4.FOLDS)
        rk = fit_ranker([final_sims[i] for i in range(n) if i not in held])
        for i in held:
            prs[i] = ranker_probs(rk, final_sims[i])
    best, tuned = tune(final_sims, oof, prs)
    summary.update(tuned=list(best), oof_4d_tuned=tuned)
    print(f"  >>> estimated OOF MAP@5 after tuning = {tuned:.4f}")

    if args.ablate:
        print("\nAblation: positive 'cost of removing' means the group helps")
        for g in on:
            sc = stack([strip(s, [g]) for s in sims], "ablation")[3]
            summary[f"cost_without_{g}"] = float((scores - sc).mean())
            print(f"  without {g}: cost of removing = {(scores - sc).mean():+.4f}", flush=True)

    with open(os.path.join(proc, f"model4d_validation_summary_{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if args.predict_test:
        print("\nTest inference")
        final = c4.build_frames(train_idx, test_idx)
        c4.shift_report(sims, final)
        for rnd, models in enumerate(round_models, 1):
            result = c4.predict(models, final)
            if rnd < c4.ROUNDS:
                final = c4.augment(final, result)
        pr = ranker_probs(fit_ranker(final_sims), final)
        sample = args.sample_submission or os.path.join(proc, "sample_submission.csv")
        w, alpha, gamma = best
        for name, mult in (("main", 1.0), ("lessnew", 0.75), ("morenew", 1.33)):
            preds = c4.rank(final, adjust(result, pr, w, alpha, gamma * mult))
            sub = pd.DataFrame({"id": c4.IDS[final["order"]],
                                "predictions": [" ".join(p) for p in preds]})
            if os.path.exists(sample):
                sub = pd.read_csv(sample)[["id"]].merge(
                    sub, on="id", how="left", validate="one_to_one")
            c4.validate_submission(sub, test_df, train_df, sample)
            path = os.path.join(proc, f"submission_model4d_{args.tag}_{name}.csv")
            sub.to_csv(path, index=False)
            print(f"  saved {path}  (new at rank 1: "
                  f"{100 * np.mean(preds[:, 0] == c4.NEW):.1f}%)")
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
