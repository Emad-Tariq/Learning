"""Photo-balanced refitting of the existing three binary scoring heads.

The returned tuple is consumed unchanged by dalvin_core.predict. No new feature
or postprocessing is introduced. Identity-disjoint upstream predictions remain
the runner's responsibility.
"""
from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd

import dalvin_core as b


def _normalized(weights):
    weights = np.asarray(weights, dtype=float)
    if not len(weights) or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Training weights must be finite and strictly positive")
    return weights / weights.mean()


def fit(frames, params=None, *, weight_by_photo=True):
    """Fit detector, known-only candidate head, and all-query candidate head.

    Flat ``params`` apply to all heads. Optional ``detector_params`` and
    ``candidate_params`` dictionaries apply after common parameters. Without
    overrides, detector defaults are the original 300 trees / 15 leaves and
    candidate defaults are an independent copy of ``b.PARAMS``.
    """
    frames = list(frames)
    if not frames:
        raise ValueError("At least one labelled training frame is required")
    params = dict(params or {})
    detector_overrides = dict(params.pop("detector_params", {}))
    candidate_overrides = dict(params.pop("candidate_params", {}))
    detector_params = {**b.PARAMS, "n_estimators": 300, "num_leaves": 15,
                       **params, **detector_overrides}
    candidate_params = {**b.PARAMS, **params, **candidate_overrides}
    if any(p.get("objective") != "binary" for p in [detector_params, candidate_params]):
        raise ValueError("All scoring heads require the binary objective")
    qf, cf = b.columns(frames[0])
    photos = []
    for frame in frames:
        if b.columns(frame) != (qf, cf):
            raise ValueError("Training frames have different feature schemas")
        order = np.asarray(frame["order"])
        if order.shape != (len(frame["q"]),):
            raise ValueError("order must contain one global photo index per query")
        photos.extend(order.tolist())
    counts = Counter(photos)
    query_weights = _normalized([1. / counts[i] for i in photos]) if weight_by_photo else np.ones(len(photos))
    qx, qy, cx, cy, ck, cw = [], [], [], [], [], []
    offset = 0
    for frame in frames:
        q, c = frame["q"], frame["c"]
        local = c.q.to_numpy()
        if (not np.issubdtype(local.dtype, np.integer)
                or (local < 0).any() or (local >= len(q)).any()):
            raise ValueError("Invalid candidate query indices")
        unk = q.unk.to_numpy()
        labels = c.y.to_numpy()
        if not np.isin(unk, [0, 1]).all() or not np.isin(labels, [0, 1]).all():
            raise ValueError("Expected binary detector and candidate labels")
        known = unk[local] == 0
        if (labels[~known] != 0).any():
            raise ValueError("Unknown-query candidates cannot be positive")
        qx.append(q[qf]); qy.append(unk)
        cx.append(b.candidate_matrix(frame, qf, cf)); cy.append(labels); ck.append(known)
        cw.append(query_weights[offset:offset + len(q)][local])
        offset += len(q)
    qx, qy = pd.concat(qx, ignore_index=True), np.concatenate(qy)
    cx, cy = pd.concat(cx, ignore_index=True), np.concatenate(cy)
    ck, cw = np.concatenate(ck), np.concatenate(cw)
    if len(np.unique(qy)) != 2 or len(np.unique(cy[ck])) != 2:
        raise ValueError("Training requires both classes for detector and known-only head")
    detector = lgb.LGBMClassifier(**detector_params).fit(qx, qy, sample_weight=query_weights)
    closed = lgb.LGBMClassifier(**candidate_params).fit(cx.loc[ck], cy[ck], sample_weight=_normalized(cw[ck]))
    anyq = lgb.LGBMClassifier(**candidate_params).fit(cx, cy, sample_weight=_normalized(cw))
    metadata = dict(training_appearances=len(photos), unique_photos=len(counts),
                    weight_by_photo=bool(weight_by_photo),
                    known_queries=int((qy == 0).sum()), unknown_queries=int((qy == 1).sum()))
    for estimator in (detector, closed, anyq):
        estimator.photo_weighting_metadata_ = metadata.copy()
    return detector, closed, anyq, (qf, cf)
