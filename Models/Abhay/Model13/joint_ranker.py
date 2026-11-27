"""Joint ranking of gallery identities and ``new_individual``.

The model learns one score scale for all 16 choices. LambdaRank uses NDCG's
pairwise surrogate, not an exact MAP objective; final comparisons use MAP@5.
Query truth is used only to create labels, never to construct feature values.
"""
from collections import Counter
from dataclasses import dataclass
import re

import lightgbm as lgb
import numpy as np
import pandas as pd

NEW = "new_individual"
INDEX_COLS = re.compile(r"^(nb|nw|sb|sw)\d+$")
RESERVED = {"unk", "y", "truth", "target", "identity", "id", "label", "order"}
DEFAULT_PARAMS = dict(
    objective="lambdarank", metric="map", label_gain=[0, 1],
    lambdarank_truncation_level=5, lambdarank_norm=True,
    n_estimators=300, learning_rate=.03, num_leaves=15, max_depth=5,
    min_child_samples=50, reg_lambda=10., reg_alpha=.5,
    colsample_bytree=.85, n_jobs=4, random_state=13052, verbosity=-1,
    deterministic=True, force_col_wise=True,
)


@dataclass
class JointRanker:
    estimator: object
    query_features: tuple
    candidate_features: tuple
    candidates: int
    metadata: dict


def feature_schema(frame):
    """Derive a deterministic numeric schema, excluding labels and pointers."""
    qf = tuple(c for c in frame["q"].columns
               if c not in RESERVED and not INDEX_COLS.match(c))
    cf = tuple(c for c in frame["c"].columns
               if c not in RESERVED | {"q", "c", "ident"})
    return qf, cf


def design(frame, query_features=None, candidate_features=None, candidates=None):
    """Return interleaved [K known, NEW] features and labels without reading truth.

    Candidate rows may be supplied in any order. Their within-query order is
    retained, giving deterministic tie behavior consistent with the shortlist.
    """
    q, c = frame["q"], frame["c"]
    nq = len(q)
    if nq == 0:
        raise ValueError("A frame must contain queries")
    actual_q, actual_c = feature_schema(frame)
    qf = actual_q if query_features is None else tuple(query_features)
    cf = actual_c if candidate_features is None else tuple(candidate_features)
    if set(qf) != set(actual_q) or set(cf) != set(actual_c):
        raise ValueError("Feature schema differs from the fitted joint ranker")
    if not {"q", "ident"} <= set(c.columns):
        raise ValueError("Candidate rows require q and ident")
    query_index = c.q.to_numpy()
    if (not np.issubdtype(query_index.dtype, np.integer)
            or (query_index < 0).any() or (query_index >= nq).any()):
        raise ValueError("Candidate q must contain valid integer query indices")
    counts = np.bincount(query_index, minlength=nq)
    k = int(counts[0]) if candidates is None else int(candidates)
    if k < 4 or not np.all(counts == k):
        raise ValueError("Each query must have the same K >= 4 known candidates")
    sort = np.argsort(query_index, kind="stable")
    ordered = c.iloc[sort]
    labels = np.empty((nq, k + 1), dtype=object)
    labels[:, :k] = ordered.ident.to_numpy().reshape(nq, k)
    labels[:, k] = NEW
    if any(len(set(row)) != k + 1 for row in labels):
        raise ValueError("Candidate identities must be unique and exclude NEW")
    qvalues = q.loc[:, qf].to_numpy(dtype=np.float32)
    cvalues = ordered.loc[:, cf].to_numpy(dtype=np.float32)
    if np.isinf(qvalues).any() or np.isinf(cvalues).any():
        raise ValueError("Feature values contain infinity")
    rows = nq * (k + 1)
    values = np.full((rows, len(cf) + len(qf) + 1), np.nan, dtype=np.float32)
    known_rows = (np.arange(nq)[:, None] * (k + 1) + np.arange(k)).ravel()
    values[known_rows, :len(cf)] = cvalues
    values[:, len(cf):-1] = np.repeat(qvalues, k + 1, axis=0)
    values[:, -1] = np.tile(np.r_[np.zeros(k), 1], nq)
    columns = ["c__" + f for f in cf] + ["q__" + f for f in qf] + ["is_new"]
    return pd.DataFrame(values, columns=columns), labels


def fit(frames, params=None, *, weight_by_photo=True):
    """Fit a ranker using frozen upstream predictions in episode frames.

    A known query with no true candidate supplies no ranking supervision and is
    omitted, rather than being mislabeled NEW. Its omission is recorded. Frames
    must contain global image indices in ``order`` for repeated-photo weighting.
    Identity separation of upstream fitting is the caller's responsibility.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("At least one training frame is required")
    qf, cf = feature_schema(frames[0])
    k = int(frames[0]["c"].groupby("q").size().iloc[0])
    matrices, targets, photo_ids, omitted, unknown, total = [], [], [], 0, 0, 0
    for frame in frames:
        matrix, labels = design(frame, qf, cf, k)
        truth = np.asarray(frame["truth"], dtype=object)
        order = np.asarray(frame["order"])
        if truth.shape != (len(labels),) or order.shape != truth.shape:
            raise ValueError("truth and order must align one-to-one with queries")
        target = (labels == truth[:, None]).astype(np.int32)
        positives = target.sum(axis=1)
        if (positives > 1).any():
            raise ValueError("Multiple positive labels in a ranking group")
        keep = positives == 1
        omitted += int((~keep).sum())
        total += len(labels)
        unknown += int((truth[keep] == NEW).sum())
        matrices.append(matrix.loc[np.repeat(keep, k + 1)].reset_index(drop=True))
        targets.append(target[keep].ravel())
        photo_ids.extend(order[keep].tolist())
    if len(photo_ids) < 2:
        raise ValueError("Too few labelled ranking groups")
    frequencies = Counter(photo_ids)
    query_weights = np.array([1. / frequencies[i] for i in photo_ids])
    if not weight_by_photo:
        query_weights[:] = 1.
    # Preserve mean gradient scale while giving each repeated photo equal mass.
    query_weights /= query_weights.mean()
    config = {**DEFAULT_PARAMS, **(params or {})}
    if config.get("objective") != "lambdarank":
        raise ValueError("This module requires the lambdarank objective")
    model = lgb.LGBMRanker(**config)
    model.fit(pd.concat(matrices, ignore_index=True), np.concatenate(targets),
              group=np.full(len(photo_ids), k + 1),
              sample_weight=np.repeat(query_weights, k + 1), eval_at=[5])
    metadata = dict(training_query_appearances=total, used_groups=len(photo_ids),
                    omitted_missing_true_candidate=omitted,
                    unique_training_photos=len(frequencies), unknown_groups=unknown,
                    weight_by_photo=bool(weight_by_photo), parameters=config,
                    objective_note="LambdaRank NDCG surrogate; assessed by MAP@5")
    return JointRanker(model, qf, cf, k, metadata)


def predict(model, frame):
    """Return top-five identity strings and raw Nx(K+1) uncalibrated scores."""
    matrix, labels = design(frame, model.query_features,
                            model.candidate_features, model.candidates)
    scores = np.asarray(model.estimator.predict(matrix)).reshape(labels.shape)
    if not np.isfinite(scores).all():
        raise ValueError("Ranker produced non-finite scores")
    rank = np.argsort(-scores, axis=1, kind="stable")[:, :5]
    return np.take_along_axis(labels, rank, axis=1), scores
