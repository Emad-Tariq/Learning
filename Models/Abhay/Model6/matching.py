"""Model 6: pairwise identity verification on supplied MIEW-ID descriptors."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

VIEWS = {"left": -1, "frontleft": -1, "backleft": -1,
         "right": 1, "frontright": 1, "backright": 1}


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)


def load(data):
    meta = pd.read_csv(data / "train.csv")
    index = pd.read_csv(data / "miewid_ids.csv")
    if not meta.id.is_unique or not index.id.is_unique:
        raise ValueError("Image IDs must be unique")
    ix = pd.Index(index.id).get_indexer(meta.id)
    if np.any(ix < 0):
        raise ValueError("Training image missing from descriptor ID mapping")
    raw = np.load(data / "miewid_features.npy", mmap_mode="r")
    if len(raw) != len(index):
        raise ValueError("Descriptor and ID table lengths differ")
    x = np.asarray(raw[ix], dtype=np.float32)
    if not np.isfinite(x).all() or (np.linalg.norm(x, axis=1) == 0).any():
        raise ValueError("Invalid descriptor")
    if meta.identity.isna().any() or not set(meta.species) <= {"hyena", "leopard"}:
        raise ValueError("Invalid training identity or species")
    return meta.reset_index(drop=True), unit(x)


def split_identities(meta, seed):
    rng = np.random.default_rng(seed)
    roles = np.empty(len(meta), dtype=object)
    for species in ("hyena", "leopard"):
        names = np.array(sorted(meta.loc[meta.species == species, "identity"].unique()))
        rng.shuffle(names)
        a, b = int(.6 * len(names)), int(.8 * len(names))
        for role, block in (("fit", names[:a]), ("development", names[a:b]),
                            ("evaluation", names[b:])):
            roles[(meta.species == species) & meta.identity.isin(block)] = role
    assert pd.DataFrame({"identity": meta.identity, "role": roles}).groupby("identity").role.nunique().max() == 1
    return roles


def embedding(x, fit_idx, seed, dimensions):
    n = min(dimensions, len(fit_idx) - 1, x.shape[1])
    if n < 2:
        raise ValueError("Too few fitting images")
    pca = PCA(n_components=n, svd_solver="randomized", random_state=seed).fit(x[fit_idx])
    z = unit(pca.transform(x).astype(np.float32))
    return z, pca


def groups(meta, x, indices, threshold):
    """Near-duplicate groups within an identity, using supplied global features only."""
    labels = np.full(len(meta), -1, dtype=int)
    group_id = 0
    for _, d in meta.iloc[indices].groupby("identity", sort=True):
        ix = d.index.to_numpy()
        links = x[ix] @ x[ix].T >= threshold
        local = connected_components(csr_matrix(links), directed=False)[1]
        for g in np.unique(local):
            labels[ix[local == g]] = group_id
            group_id += 1
    return labels


def gallery_query(meta, indices, group_ids, seed):
    rng = np.random.default_rng(seed)
    gallery, query = [], []
    for _, d in meta.iloc[indices].groupby("identity", sort=True):
        ix = d.index.to_numpy()
        gs = np.unique(group_ids[ix])
        if len(gs) < 2:
            gallery.extend(ix)
            continue
        n_query_groups = min(max(1, int(round(.3 * len(gs)))), 8, len(gs) - 1)
        chosen = rng.choice(gs, size=n_query_groups, replace=False)
        query.extend(ix[np.isin(group_ids[ix], chosen)])
        gallery.extend(ix[~np.isin(group_ids[ix], chosen)])
    gallery, query = np.array(gallery, dtype=int), np.array(query, dtype=int)
    if not len(query):
        raise ValueError("No query groups available")
    assert set(meta.identity.iloc[query]) <= set(meta.identity.iloc[gallery])
    assert not set(group_ids[query]) & set(group_ids[gallery])
    return gallery, query


def train_pairs(meta, x, z, fit_idx, group_ids, seed, hard_k=30, per_image=2):
    """One positive, one hard negative and one random negative per anchor/repeat."""
    rng = np.random.default_rng(seed)
    y = meta.identity.to_numpy()
    sp = meta.species.to_numpy()
    left, right, labels = [], [], []
    for species in ("hyena", "leopard"):
        ix = fit_idx[sp[fit_idx] == species]
        sim = x[ix] @ x[ix].T
        for j, anchor in enumerate(ix):
            positives = ix[(y[ix] == y[anchor]) & (group_ids[ix] != group_ids[anchor])]
            negatives = ix[y[ix] != y[anchor]]
            if not len(positives) or not len(negatives):
                continue
            candidates = np.argsort(-sim[j])
            hard = ix[candidates[y[ix[candidates]] != y[anchor]][:hard_k]]
            if not len(hard):
                raise ValueError("No hard negatives")
            for _ in range(per_image):
                # Positive is biased toward a different flank when available.
                opposite = positives[np.array([VIEWS.get(meta.viewpoint.iloc[p], 0) == -VIEWS.get(meta.viewpoint.iloc[anchor], 0)
                                               and VIEWS.get(meta.viewpoint.iloc[anchor], 0) != 0 for p in positives])]
                pos = rng.choice(opposite if len(opposite) else positives)
                for candidate, match in ((pos, 1), (rng.choice(hard), 0), (rng.choice(negatives), 0)):
                    left.append(anchor); right.append(candidate); labels.append(match)
    a, b = np.array(left), np.array(right)
    if len(np.unique(labels)) != 2:
        raise ValueError("Pair set lacks both classes")
    return pair_features(meta, x, z, a, b), np.array(labels)


def pair_features(meta, x, z, a, b):
    """Symmetric pair representation; invariant to pair order."""
    va = np.array([VIEWS.get(v, 0) for v in meta.viewpoint.iloc[a]], dtype=np.int8)
    vb = np.array([VIEWS.get(v, 0) for v in meta.viewpoint.iloc[b]], dtype=np.int8)
    side = np.column_stack(((va != 0) & (vb != 0) & (va == vb), va * vb == -1, (va == 0) | (vb == 0)))
    return np.column_stack((np.abs(z[a] - z[b]), z[a] * z[b],
                            np.sum(x[a] * x[b], axis=1), np.sum(z[a] * z[b], axis=1), side)).astype(np.float32)


def fit_lda(meta, z, fit_idx):
    models = {}
    for species in ("hyena", "leopard"):
        ix = fit_idx[meta.species.iloc[fit_idx].to_numpy() == species]
        model = LinearDiscriminantAnalysis(solver="eigen", shrinkage=.3).fit(z[ix], meta.identity.iloc[ix])
        models[species] = model
    return models


def lda_transform(meta, z, models):
    out = {}
    for species, model in models.items():
        ix = np.flatnonzero(meta.species.to_numpy() == species)
        out[species] = (ix, unit(model.transform(z[ix]).astype(np.float32)))
    return out


def evaluate(meta, x, z, lda, network, scaler, gallery, queries, batch=4096):
    results = []
    truth = meta.identity.to_numpy()
    species = meta.species.to_numpy()
    for sp in ("hyena", "leopard"):
        gi = gallery[species[gallery] == sp]
        qi = queries[species[queries] == sp]
        if not len(qi):
            continue
        names, inv = np.unique(truth[gi], return_inverse=True)
        lid, lz = lda[sp]
        loc = np.full(len(meta), -1, dtype=int)
        loc[lid] = np.arange(len(lid))
        raw = x[qi] @ x[gi].T
        projected = lz[loc[qi]] @ lz[loc[gi]].T
        for row, q in enumerate(qi):
            pair = np.empty(len(gi), dtype=np.float32)
            for start in range(0, len(gi), batch):
                g = gi[start:start + batch]
                features = pair_features(meta, x, z, np.full(len(g), q, dtype=int), g)
                pair[start:start + len(g)] = network.predict_proba(scaler.transform(features))[:, 1]
            row_scores = {"cosine": raw[row], "lda": projected[row], "pair_mlp": pair}
            record = {"id": meta.id.iloc[q], "species": sp, "identity": truth[q],
                      "viewpoint": meta.viewpoint.iloc[q]}
            for method, scores in row_scores.items():
                per_identity = np.full(len(names), -np.inf)
                np.maximum.at(per_identity, inv, scores)
                ranked = names[np.argsort(-per_identity, kind="stable")[:5]]
                rank = np.flatnonzero(ranked == truth[q])
                record[f"{method}_rr5"] = 1 / (rank[0] + 1) if len(rank) else 0.
                record[f"{method}_rank1"] = int(bool(len(rank) and rank[0] == 0))
                record[f"{method}_top5"] = " ".join(ranked)
                record[f"{method}_best_wrong"] = float(np.max(per_identity[names != truth[q]]))
                record[f"{method}_correct"] = float(per_identity[names == truth[q]][0])
            results.append(record)
    return pd.DataFrame(results)


def summary(frame):
    return {m: {"map5": float(frame[f"{m}_rr5"].mean()),
                "top1": float(frame[f"{m}_rank1"].mean()),
                "top5": float((frame[f"{m}_rr5"] > 0).mean()),
                "wrong_identity_beats_true": float((frame[f"{m}_best_wrong"] >= frame[f"{m}_correct"]).mean())}
            for m in ("cosine", "lda", "pair_mlp")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dimensions", type=int, default=128)
    ap.add_argument("--duplicate-cosine", type=float, default=.97)
    ap.add_argument("--hard-k", type=int, default=30)
    ap.add_argument("--pairs-per-image", type=int, default=2)
    ap.add_argument("--max-iter", type=int, default=80)
    args = ap.parse_args()
    if not 0 < args.duplicate_cosine < 1 or min(args.dimensions, args.hard_k, args.pairs_per_image, args.max_iter) < 1:
        ap.error("Invalid positive counts or duplicate threshold")
    args.output.mkdir(parents=True, exist_ok=True)
    meta, x = load(args.data)
    roles = split_identities(meta, args.seed)
    pd.DataFrame({"id": meta.id, "species": meta.species, "identity": meta.identity,
                  "role": roles}).to_csv(args.output / "identity_roles.csv", index=False)
    fit = np.flatnonzero(roles == "fit")
    z, pca = embedding(x, fit, args.seed, args.dimensions)
    groups_fit = groups(meta, x, fit, args.duplicate_cosine)
    features, targets = train_pairs(meta, x, z, fit, groups_fit, args.seed,
                                    args.hard_k, args.pairs_per_image)
    print(f"Fitting pair verifier on {len(features)} pairs", flush=True)
    scaler = StandardScaler().fit(features)
    network = MLPClassifier(hidden_layer_sizes=(64, 32), alpha=.01, max_iter=args.max_iter,
                            early_stopping=False, random_state=args.seed, n_iter_no_change=12,
                            batch_size=256, learning_rate_init=.001)
    network.fit(scaler.transform(features), targets)
    lda = lda_transform(meta, z, fit_lda(meta, z, fit))
    metrics = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "pair_count": len(features), "fit_identities": int(meta.identity.iloc[fit].nunique()),
               "mlp_epochs": int(network.n_iter_)}
    for role in ("development", "evaluation"):
        ix = np.flatnonzero(roles == role)
        group_ids = groups(meta, x, ix, args.duplicate_cosine)
        gi, qi = gallery_query(meta, ix, group_ids, args.seed + (1 if role == "development" else 2))
        used = np.r_[gi, qi]
        pd.DataFrame({"id": meta.id.iloc[used].to_numpy(),
                      "species": meta.species.iloc[used].to_numpy(),
                      "identity": meta.identity.iloc[used].to_numpy(),
                      "duplicate_group": group_ids[used],
                      "assignment": ["gallery"] * len(gi) + ["query"] * len(qi)}).to_csv(
                          args.output / f"{role}_split.csv", index=False)
        frame = evaluate(meta, x, z, lda, network, scaler, gi, qi)
        frame.to_csv(args.output / f"{role}_predictions.csv", index=False)
        metrics[role] = {"queries": len(frame), "identities": int(meta.identity.iloc[ix].nunique()),
                         "query_identities": int(meta.identity.iloc[qi].nunique()),
                         "gallery_only_identities": int(meta.identity.iloc[gi].nunique() - meta.identity.iloc[qi].nunique()),
                         "scores": summary(frame),
                         "by_species": {sp: summary(part) for sp, part in frame.groupby("species")}}
        print(role, json.dumps(metrics[role]), flush=True)
    joblib.dump({"pca": pca, "lda": lda, "scaler": scaler, "network": network}, args.output / "model.joblib")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
