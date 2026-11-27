"""Unmodified Dalvin 3d feature function, retained for parity verification."""
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
