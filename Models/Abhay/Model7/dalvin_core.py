"""Dalvin Model 3d functions; raw evidence can use a frozen nuisance projection."""
import os
import re
import cv2
import lightgbm as lgb
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
K=15
NEIGHBOURS=150
MATCH_RATIO=.85
RANSAC_PX=15.
SIFT_PULL=.05
VIEWS=['left','right','front','back','frontleft','frontright','backleft','backright','up','down']
FLANK=np.array([0,1,2,2,0,1,0,1,2,2])
REGIMES=[None,(.95,111),(.99,89),(.9,67)]
PARAMS=dict(objective='binary',learning_rate=.05,num_leaves=31,min_child_samples=30,subsample=.8,subsample_freq=1,colsample_bytree=.8,n_estimators=400,verbose=-1,random_state=42,n_jobs=4)
INDEX_COLS=re.compile(r'^(nb|nw|sb|sw)\d$')







def unit(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def embed(gallery, labels, queries):
    pca = PCA(min(384, gallery.shape[1], len(gallery) + len(queries) - 1), random_state=0).fit(np.r_[gallery, queries])      # unsupervised, sees the queries too
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
    Rg, Rq = NX[gi], NX[qi]  # Frozen nuisance representation for raw evidence only.
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
    support = (inl_g >= 7).astype(np.float32) @ np.eye(n, dtype=np.float32)[inv]     # gallery images confirmed per identity
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


def duplicate_groups(ix, link):
    """Group images of one identity that are near-duplicates (same encounter); each image alone if link is None."""
    if link is None:
        return np.arange(len(ix))
    edges = (X[ix] @ X[ix].T > link[0]) | (INL[ix][:, ix].toarray() >= link[1])
    return connected_components(sparse.csr_matrix(edges))[1]


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
