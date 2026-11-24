"""Wildlife re-identification (MAP@5, open set).

1. Retrieval: PCA + LDA projection of the MIEW-ID features, per-identity max similarity.
2. Verification: RootSIFT matching + RANSAC between each image and its nearest neighbours.
3. Open-set ranking: LightGBM scorers over (query, candidate) features, trained on simulated
   open-set splits of the training set. Each round also sees the previous round's predictions
   of the query's nearest test neighbours (transductive stacking).
"""
import os
import re
from multiprocessing import Pool

import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "DATA", "reid_student_data_main")
OUT = os.path.join(ROOT, "Processed_Data")
N_SIMS, FOLDS, ROUNDS = 32, 8, 3        # simulated splits, stacking folds, stacking rounds
K, NEIGHBOURS = 15, 150                 # candidates per query, SIFT pairs verified per image
SIFT_PULL = 0.05                        # weight of SIFT evidence when selecting candidates
FLANK = np.array([0, 1, 2, 2, 0, 1, 0, 1, 2, 2])    # left-ish, right-ish, other (per VIEWS order)
VIEWS = ["left", "right", "front", "back", "frontleft", "frontright", "backleft", "backright", "up", "down"]

# ----------------------------------------------------------------------------- data
train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
NTR = len(train)
X = np.load(f"{DATA}/miewid_features.npy")
X /= np.linalg.norm(X, axis=1, keepdims=True)
IDS = np.r_[train.id, test.id]
Y = np.r_[train.identity.to_numpy(dtype=object), np.full(len(test), None, dtype=object)]
SP = np.r_[train.species.to_numpy(dtype=object), test.species.to_numpy(dtype=object)]
VIEW = pd.concat([train.viewpoint, test.viewpoint]).map({v: i for i, v in enumerate(VIEWS)}).fillna(-1).to_numpy(int)


# ----------------------------------------------------------------------------- SIFT verification
def load_sift():
    z = {**np.load(f"{DATA}/sift_features_train.npz"), **np.load(f"{DATA}/sift_features_test.npz")}
    xy = [z[f"{i}_xy"] for i in IDS]
    desc = [np.sqrt(d / np.maximum(d.sum(1, keepdims=True), 1)).astype(np.float32) for d in (z[f"{i}_desc"] for i in IDS)]
    return xy, desc


def verify(pair):
    """(ratio-test matches, RANSAC inliers) between two images."""
    a, b = pair
    if len(DESC[a]) < 4 or len(DESC[b]) < 4:
        return 0, 0
    good = [m[0] for m in cv2.BFMatcher().knnMatch(DESC[a], DESC[b], k=2) if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    if len(good) < 4:
        return len(good), 0
    src = np.float32([XY[a][m.queryIdx] for m in good])
    dst = np.float32([XY[b][m.trainIdx] for m in good])
    _, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=10.0)
    return len(good), 0 if inliers is None else int(inliers.sum())


def sift_matrices():
    """Symmetric sparse matrices of matches / inliers for every image and its nearest same-species neighbours.
    Pairs missing from the cache are verified in parallel and appended."""
    path, n = f"{OUT}/sift_pairs.npz", len(X)
    wanted = set()
    for sp in ("hyena", "leopard"):
        ix = np.where(SP == sp)[0]
        sim = X[ix] @ X[ix].T
        np.fill_diagonal(sim, -9)
        for row, nb in zip(ix, np.argsort(-sim, 1)[:, :NEIGHBOURS]):
            wanted.update(min(row, j) * n + max(row, j) for j in ix[nb])
    done = np.load(path) if os.path.exists(path) else dict(pairs=np.empty((0, 2), int), good=np.empty(0, int), inl=np.empty(0, int))
    missing = np.setdiff1d(np.fromiter(wanted, np.int64), done["pairs"][:, 0] * n + done["pairs"][:, 1])
    if len(missing):
        global XY, DESC
        XY, DESC = load_sift()
        new = np.stack(divmod(missing, n), 1)
        with Pool() as pool:
            res = np.array(pool.map(verify, map(tuple, new), chunksize=256))
        done = dict(pairs=np.r_[done["pairs"], new], good=np.r_[done["good"], res[:, 0]], inl=np.r_[done["inl"], res[:, 1]])
        os.makedirs(OUT, exist_ok=True)
        np.savez(path, **done)
    a, b = done["pairs"].T
    mat = lambda v: sparse.coo_matrix((np.r_[v, v], (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    return mat(done["good"]), mat(done["inl"])


# ----------------------------------------------------------------------------- features
def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def embed(gallery, labels, queries):
    pca = PCA(min(384, len(gallery) + len(queries) - 1)).fit(np.r_[gallery, queries])      # unsupervised, sees the queries too
    lda = LDA(solver="eigen", shrinkage=0.3).fit(pca.transform(gallery), labels)
    project = lambda a: unit(lda.transform(pca.transform(a)))
    return project(gallery), project(queries)


def per_identity(sim, inv, n):
    """Per query: max and mean-of-top-2 similarity to each identity's gallery images."""
    best, top2 = np.empty((sim.shape[0], n)), np.empty((sim.shape[0], n))
    for j in range(n):
        s = sim[:, inv == j]
        best[:, j], top2[:, j] = s.max(1), np.sort(s, 1)[:, -2:].mean(1)
    return best, top2


def species_features(gi, qi, sp):
    """Query frame and (query, candidate) frame for one species; gi/qi are global image indices."""
    Rg, Rq, yg, vg, vq = X[gi], X[qi], Y[gi], VIEW[gi], VIEW[qi]
    Lg, Lq = embed(Rg, yg, Rq)
    ids, inv = np.unique(yg, return_inverse=True)
    n, nq = len(ids), len(qi)
    sim_l, sim_r = Lq @ Lg.T, Rq @ Rg.T
    lda1, lda2 = per_identity(sim_l, inv, n)
    raw1, raw2 = per_identity(sim_r, inv, n)
    centroid = unit(np.stack([Lg[inv == j].mean(0) for j in range(n)]))
    cen = Lq @ centroid.T
    inl_g, good_g = INL[qi][:, gi].toarray().astype(np.float32), GOOD[qi][:, gi].toarray().astype(np.float32)
    sift1, sift2 = per_identity(inl_g, inv, n)
    good1, _ = per_identity(good_g, inv, n)
    support = (inl_g >= 8).astype(np.float32) @ np.eye(n, dtype=np.float32)[inv]     # gallery images confirmed per identity
    same_view = np.full((nq, n), np.nan)
    for j in range(n):
        for v in np.unique(vq):
            cols = np.where((inv == j) & (vg == v))[0]
            if len(cols):
                same_view[vq == v, j] = sim_l[np.ix_(vq == v, cols)].max(1)
    # viewpoint coverage: flanks of one animal look different, so unseen sides cannot be matched
    by_view = np.stack([np.bincount(inv[vg == v], minlength=n) for v in range(len(VIEWS))], 1)
    by_flank = np.stack([by_view[:, FLANK == f].sum(1) for f in range(3)], 1)
    seen = vq >= 0
    n_view = np.where(seen[:, None], by_view[:, vq.clip(0)].T, np.nan)
    n_flank = np.where(seen[:, None], by_flank[:, FLANK[vq.clip(0)]].T, np.nan)
    # transductive: neighbours among the queries themselves
    qq_r, qq_l, qq_s = Rq @ Rq.T, Lq @ Lq.T, INL[qi][:, qi].toarray().astype(np.float32)
    np.fill_diagonal(qq_r, -1), np.fill_diagonal(qq_l, -1)
    nb, sb = np.argsort(-qq_l, 1)[:, :5], np.argsort(-qq_s, 1)[:, :5]
    nbs, sbs = np.take_along_axis(qq_l, nb, 1), np.take_along_axis(qq_s, sb, 1)
    t_l = np.take_along_axis(lda1, np.argsort(-lda1, 1)[:, :3], 1)
    t_r = -np.sort(-raw1, 1)[:, :3]
    t_s = -np.sort(-sift1, 1)[:, :3]
    g_top = -np.sort(-sim_l, 1)
    q = pd.DataFrame({
        "sp": int(sp == "hyena"), "view": vq, "ng": len(gi),
        "l1": t_l[:, 0], "l2": t_l[:, 1], "l3": t_l[:, 2], "r1": t_r[:, 0], "r2": t_r[:, 1], "r3": t_r[:, 2],
        "lm12": t_l[:, 0] - t_l[:, 1], "rm12": t_r[:, 0] - t_r[:, 1],
        "gd5": g_top[:, :5].mean(1), "gd20": g_top[:, :20].mean(1),
        "qq1": nbs[:, 0], "qq3": nbs[:, :3].mean(1), "qq5": nbs.mean(1), "qqr1": qq_r.max(1),
        "qn6": (qq_r > .6).sum(1), "qn7": (qq_r > .7).sum(1), "qn8": (qq_r > .8).sum(1),
        "qq_minus_g": nbs[:, 0] - t_l[:, 0], "nb_g1": t_l[nb[:, 0], 0],
        "s1": t_s[:, 0], "s2": t_s[:, 1], "s3": t_s[:, 2], "sg": good_g.max(1),
        "qs1": sbs[:, 0], "qs3": sbs[:, :3].mean(1), "qsn10": (qq_s >= 10).sum(1), "qsn20": (qq_s >= 20).sum(1),
        "qs_minus_s": sbs[:, 0] - t_s[:, 0], "nbs_s1": t_s[sb[:, 0], 0],
    })
    for k in range(5):
        q[f"nb{k}"], q[f"nw{k}"], q[f"sb{k}"], q[f"sw{k}"] = nb[:, k], nbs[:, k], sb[:, k], sbs[:, k]
    # candidates: best identities by embedding similarity, lightly boosted by SIFT evidence
    cand = np.argsort(-(lda1 + SIFT_PULL * np.tanh(sift1 / 15)), 1)[:, :K]
    r, c = np.repeat(np.arange(nq), K), cand.ravel()
    nb_ev = np.max(np.minimum(nbs[:, :, None], np.stack([lda1[nb[:, k]] for k in range(5)], 1)), 1)
    sb_ev = np.max(np.minimum(sbs[:, :, None], np.stack([sift1[sb[:, k]] for k in range(5)], 1)), 1)
    rank_of = lambda m: (-m).argsort(1).argsort(1)[r, c]
    cd = pd.DataFrame({
        "q": r, "c": c, "rank": np.tile(np.arange(K), nq), "ident": ids[c],
        "mxr": raw1[r, c], "m2r": raw2[r, c], "mxl": lda1[r, c], "m2l": lda2[r, c], "cl": cen[r, c],
        "gapl": lda1[r, c] - t_l[r, 0], "gapr": raw1[r, c] - t_r[r, 0], "same": same_view[r, c],
        "cnt": np.bincount(inv)[c], "nbev": nb_ev[r, c], "sbev": sb_ev[r, c],
        "rk_r": rank_of(raw1), "rk_c": rank_of(cen), "rk_s": rank_of(sift1),
        "n_view": n_view[r, c], "n_flank": n_flank[r, c], "support": support[r, c], "sx": sift1[r, c], "s2x": sift2[r, c], "gx": good1[r, c], "sgap": sift1[r, c] - t_s[r, 0],
    })
    return q, cd


def build_frames(gi, qi):
    qs, cs, order, off = [], [], [], 0
    for sp in ("hyena", "leopard"):
        g, qq = gi[SP[gi] == sp], qi[SP[qi] == sp]
        q, c = species_features(g, qq, sp)
        c["q"] += off
        q[[f"{p}{k}" for p in ("nb", "sb") for k in range(5)]] += off
        qs.append(q), cs.append(c), order.append(qq)
        off += len(qq)
    return dict(q=pd.concat(qs, ignore_index=True), c=pd.concat(cs, ignore_index=True), order=np.concatenate(order))


# ----------------------------------------------------------------------------- open-set simulation
# Near-duplicate linking (raw similarity, SIFT inliers) per regime; None = random image holdout.
REGIMES = [None, (0.95, 100), (0.99, 80), (0.9, 60)]


def duplicate_groups(ix, link):
    """Group images of one identity that are near-duplicates (same encounter); each image alone if link is None."""
    if link is None:
        return np.arange(len(ix))
    edges = (X[ix] @ X[ix].T > link[0]) | (INL[ix][:, ix].toarray() >= link[1])
    return connected_components(sparse.csr_matrix(edges))[1]


def simulate(seed):
    """Identity-disjoint open-set split mimicking the test set: unknown identities, capped queries per
    identity, and (in most regimes) whole near-duplicate groups held out together."""
    rng, link = np.random.default_rng(seed), REGIMES[seed % len(REGIMES)]
    gallery, known, unknown = [], [], []
    for sp in ("hyena", "leopard"):
        p_unknown, frac, cap = rng.uniform(.2, .35), rng.uniform(.3, .5), int(rng.integers(12, 33))
        d = train[train.species == sp]
        names = np.array(sorted(d.identity.unique()))
        rng.shuffle(names)
        new = set(names[:int(round(p_unknown * len(names)))])
        for name, g in d.groupby("identity"):
            ix = g.index.to_numpy().copy()
            rng.shuffle(ix)
            if name in new:
                unknown += list(ix[:cap])
            elif len(ix) < 2:
                gallery += list(ix)
            else:
                n_q, held = min(max(int(round(frac * len(ix))), 1), cap, len(ix) - 1), []
                groups = duplicate_groups(ix, link)
                for c in rng.permutation(np.unique(groups)):
                    members = list(ix[groups == c])
                    if len(held) < n_q and len(held) + len(members) < len(ix):
                        held += members
                known += held[:cap]
                gallery += [i for i in ix if i not in set(held)]
    frames = build_frames(np.array(gallery), np.array(known + unknown))
    truth = np.where(np.isin(frames["order"], unknown), "new_individual", Y[frames["order"]])
    frames["truth"] = truth
    frames["q"]["unk"] = (truth == "new_individual").astype(int)
    frames["c"]["y"] = (frames["c"].ident.to_numpy(dtype=object) == truth[frames["c"].q.to_numpy()]).astype(int)
    return frames


# ----------------------------------------------------------------------------- stacked LightGBM
PARAMS = dict(objective="binary", learning_rate=.05, num_leaves=31, min_child_samples=30, subsample=.8,
              subsample_freq=1, colsample_bytree=.8, n_estimators=400, verbose=-1)
INDEX_COLS = re.compile(r"^(nb|nw|sb|sw)\d$")


def columns(s):
    qf = [c for c in s["q"].columns if c != "unk" and not INDEX_COLS.match(c)]
    cf = [c for c in s["c"].columns if c not in ("q", "c", "ident", "y")]
    return qf, cf


def candidate_matrix(s, qf, cf):
    return pd.concat([s["c"][cf].reset_index(drop=True), s["q"].iloc[s["c"].q.to_numpy()][qf].reset_index(drop=True)], axis=1)


def fit(sims):
    """new-individual detector, closed-set candidate scorer, and all-query candidate scorer."""
    qf, cf = columns(sims[0])
    known = lambda s: s["q"].unk.to_numpy()[s["c"].q.to_numpy()] == 0
    cx = pd.concat([candidate_matrix(s, qf, cf) for s in sims])
    cy = np.concatenate([s["c"].y.to_numpy() for s in sims])
    ck = np.concatenate([known(s) for s in sims])
    detector = lgb.LGBMClassifier(**{**PARAMS, "n_estimators": 300, "num_leaves": 15}).fit(
        pd.concat([s["q"][qf] for s in sims]), np.concatenate([s["q"].unk.to_numpy() for s in sims]))
    return detector, lgb.LGBMClassifier(**PARAMS).fit(cx[ck], cy[ck]), lgb.LGBMClassifier(**PARAMS).fit(cx, cy), (qf, cf)


def predict(models, s):
    detector, closed, anyq, (qf, cf) = models
    nq, cx = len(s["q"]), candidate_matrix(s, qf, cf)
    p_new = detector.predict_proba(s["q"][qf])[:, 1]
    raw = anyq.predict_proba(cx)[:, 1].reshape(nq, K)
    within = closed.predict_proba(cx)[:, 1].reshape(nq, K)
    within /= np.maximum(within.sum(1, keepdims=True), 1e-9)
    return dict(pn=p_new, raw=raw, pc=within, pb=(p_new + np.clip(1 - raw.sum(1), 0, 1)) / 2)


def rank(s, r):
    """Top-5 labels: P(new) vs (1 - P(new)) * P(identity | known)."""
    nq = len(s["q"])
    labels = np.concatenate([s["c"].ident.to_numpy(dtype=object).reshape(nq, K), np.full((nq, 1), "new_individual", dtype=object)], 1)
    scores = np.concatenate([(1 - r["pb"])[:, None] * r["pc"], r["pb"][:, None]], 1)
    return np.take_along_axis(labels, np.argsort(-scores, 1)[:, :5], 1)


def map5(preds, truth):
    return np.mean([next((1 / (i + 1) for i, p in enumerate(row) if p == t), 0.0) for row, t in zip(preds, truth)])


def augment(s, r):
    """Add the previous round's predictions for each query and for its nearest query neighbours."""
    q, c, nq = s["q"].copy(), s["c"].copy(), len(s["q"])
    codes = pd.factorize(c.ident)[0].reshape(nq, K)
    q["r_pn"], q["r_pb"], q["r_rawsum"], q["r_rawmax"] = r["pn"], r["pb"], r["raw"].sum(1), r["raw"].max(1)
    c["r_raw"], c["r_pc"] = r["raw"].ravel(), r["pc"].ravel()
    for kind, weight, scale in (("nb", "nw", 1.0), ("sb", "sw", 30.0)):     # embedding / SIFT neighbours
        evidence, support, pb, w, top = np.zeros((nq, K)), np.zeros((nq, K)), [], [], []
        for k in range(5):
            idx, sim = q[f"{kind}{k}"].to_numpy(), q[f"{weight}{k}"].to_numpy() / scale
            wk = np.clip(sim - 0.4, 0, None) if kind == "nb" else np.clip(sim, 0, 1)
            agree = ((codes[idx][:, None, :] == codes[:, :, None]) * r["raw"][idx][:, None, :]).sum(2)
            evidence = np.maximum(evidence, np.minimum(agree, np.clip(sim, 0, 1)[:, None]))
            support += wk[:, None] * agree
            pb.append(r["pb"][idx]), w.append(wk), top.append(r["raw"][idx].max(1))
        pb, w = np.stack(pb, 1), np.stack(w, 1)
        q[f"{kind}_pb0"], q[f"{kind}_pbw"] = pb[:, 0], (pb * w).sum(1) / np.maximum(w.sum(1), 1e-6)
        q[f"{kind}_pbmin"], q[f"{kind}_w"], q[f"{kind}_rm0"] = np.where(w > 0, pb, 1).min(1), w.sum(1), top[0]
        c[f"r_{kind}ev"], c[f"r_{kind}sum"] = evidence.ravel(), support.ravel()
    return dict(s, q=q, c=c)


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    GOOD, INL = sift_matrices()
    print(f"simulating {N_SIMS} open-set splits...")
    sims = [simulate(seed) for seed in range(N_SIMS)]
    final = build_frames(np.arange(NTR), np.arange(NTR, len(X)))
    folds = [list(range(i, N_SIMS, FOLDS)) for i in range(FOLDS)]
    for rnd in range(1, ROUNDS + 1):
        oof = [None] * N_SIMS
        for f in folds:
            models = fit([sims[i] for i in range(N_SIMS) if i not in f])
            for i in f:
                oof[i] = predict(models, sims[i])
        score = np.mean([map5(rank(sims[i], oof[i]), sims[i]["truth"]) for i in range(N_SIMS)])
        print(f"round {rnd}: out-of-fold simulated MAP@5 = {score:.4f}")
        result = predict(fit(sims), final)
        if rnd < ROUNDS:
            sims, final = [augment(s, r) for s, r in zip(sims, oof)], augment(final, result)
    preds = rank(final, result)
    sub = pd.DataFrame({"id": IDS[final["order"]], "predictions": [" ".join(p) for p in preds]}).sort_values("id")
    sample = pd.read_csv(f"{DATA}/sample_submission.csv")
    assert set(sub.id) == set(sample.id) and sub.id.is_unique and (sub.predictions.str.split().map(len) == 5).all()
    os.makedirs(OUT, exist_ok=True)
    sub.to_csv(f"{OUT}/submission_model3c_viewpoint.csv", index=False)
    print(f"saved {len(sub)} rows; new_individual ranked first for {np.mean(preds[:, 0] == 'new_individual'):.1%}")
