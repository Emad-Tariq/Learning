"""Gallery-conditioned animal verification and viewpoint-aware identity evidence."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse


NEW = 'new_individual'
EXTRA_COLUMNS = [
    'pair_best', 'pair_other_side', 'pair_same_side', 'pair_side_mean', 'pair_side_votes',
    'pair_direct_sift', 'pair_raw', 'pair_cohort', 'album_gap', 'album_peak', 'album_second',
    'direct_inliers', 'gallery_size', 'gallery_views', 'has_query_side',
]


def side_codes(view):
    # Model 12 order: left/right/front/back/frontleft/frontright/backleft/backright/up/down.
    return np.array([0, 1, 2, 3, 3, 1, 2, 1, 2, 3, 3])[view + 1]


def upper_mean(matrix, k):
    k = min(k, matrix.shape[1])
    return np.partition(matrix, -k, axis=1)[:, -k:].mean(1)


def top_pairs(raw, adjusted, inliers, k):
    pieces = []
    width = raw.shape[1]
    for matrix in (raw, adjusted):
        k = min(k, width)
        top = np.argpartition(matrix, -k, axis=1)[:, -k:]
        pieces.append((top + np.arange(len(raw))[:, None] * width).ravel())
    sr, sc = inliers.nonzero()
    strong = np.asarray(inliers[sr, sc]).ravel() >= 5 if len(sr) else np.zeros(0, bool)
    pieces.append(sr[strong] * width + sc[strong])
    return np.divmod(np.unique(np.concatenate(pieces)), width)


def pair_matrix(row, col, raw, adjusted, query_side, gallery_side, good, inliers, density):
    matched = np.asarray(good[row, col]).ravel()
    verified = np.asarray(inliers[row, col]).ravel()
    return pd.DataFrame(dict(cosine=raw[row, col], cohort_excess=adjusted[row, col],
        query_excess=raw[row, col] - density[row], query_side=query_side[row],
        reference_side=gallery_side[col], same_side=(query_side[row] == gallery_side[col]) & (query_side[row] != 0),
        pair_inliers=verified, pair_matches=matched, inlier_ratio=verified / np.maximum(matched, 1)))


def gallery_tails(similarity, group, side):
    """Each reference's top impostor similarities, conditional on query side."""
    impostors = np.where(group[:, None] == group[None, :], -1, similarity)

    def tail(matrix):
        k = min(10, len(matrix))
        values = np.partition(matrix, -k, axis=0)[-k:]
        present = values > -1
        return (np.where(present, values, 0).sum(0) / np.maximum(present.sum(0), 1)).astype(np.float32)

    default = tail(impostors)
    return np.stack([tail(impostors[side == k]) if (side == k).sum() >= 20 else default
                     for k in range(4)])


def verification(g, q, x, identity, side, good, inliers):
    """Learn a same-animal model solely from the episode's labelled gallery."""
    refs, queries = x[g], x[q]
    names, group = np.unique(identity[g], return_inverse=True)
    gg = refs @ refs.T
    tails = gallery_tails(gg, group, side[g])
    adjusted = gg - tails[side[g]]
    np.fill_diagonal(gg, -2)
    np.fill_diagonal(adjusted, -2)
    gi, ii = good[g][:, g], inliers[g][:, g]
    r, c = top_pairs(gg, adjusted, ii, 16)
    keep = r != c
    r, c = r[keep], c[keep]
    examples = pair_matrix(r, c, gg, adjusted, side[g], side[g], gi, ii, upper_mean(gg, 12))
    same = group[r] == group[c]
    if same.sum() < 10 or (~same).sum() < 10:
        raise ValueError('Insufficient same/different gallery pairs for the verifier')
    classifier = lgb.LGBMClassifier(n_estimators=105, num_leaves=15, min_child_samples=70,
        reg_lambda=8, learning_rate=.08, n_jobs=4, verbosity=-1, random_state=1515)
    classifier.fit(examples, same)

    raw = queries @ refs.T
    adjusted = raw - tails[side[q]]
    gi, ii = good[q][:, g], inliers[q][:, g]
    r, c = top_pairs(raw, adjusted, ii, 30)
    evidence = pair_matrix(r, c, raw, adjusted, side[q], side[g], gi, ii, upper_mean(raw, 12))
    probability = classifier.predict_proba(evidence)[:, 1].astype(np.float32)
    return names, group, r, c, evidence, probability


def pad_top(a, k):
    return np.argsort(-a, axis=1, kind='stable')[:, :min(k, a.shape[1])]


def candidate_frame(frame, gallery, x, species, identity, views, good, inliers):
    order = frame['order']
    qf = frame['q']
    cf = frame['c']
    base_labels = cf.ident.to_numpy().reshape(len(order), -1)
    base_scores = cf.r_pc.to_numpy().reshape(base_labels.shape)
    side = side_codes(views)
    parts = []
    for sp in ('hyena', 'leopard'):
        rows = np.flatnonzero(species[order] == sp)
        q, g = order[rows], gallery[species[gallery] == sp]
        names, group, r, c, values, prob = verification(g, q, x, identity, side, good, inliers)
        shape = (len(q), len(names))
        ref_ids = group[c]
        inds = (r, ref_ids)
        by_side = np.zeros((*shape, 4), np.float32)
        np.maximum.at(by_side, (r, ref_ids, side[g[c]]), prob)
        best = by_side.max(2)
        other_side = np.partition(by_side[:, :, 1:], -2, axis=2)[:, :, -2]
        side_votes = (by_side[:, :, 1:] >= .5).sum(2)
        same_side = by_side[np.arange(len(q)), :, side[q]]
        view_available = np.zeros((len(names), 4), bool)
        view_available[group, side[g]] = True

        metrics = {}
        for key in ('cosine', 'cohort_excess', 'pair_inliers'):
            matrix = np.full(shape, -2. if key != 'pair_inliers' else 0., np.float32)
            np.maximum.at(matrix, inds, values[key].to_numpy())
            metrics[key] = matrix
        sift_prob = np.zeros(shape, np.float32)
        strong = values.pair_inliers.to_numpy() >= 7
        np.maximum.at(sift_prob, (r[strong], ref_ids[strong]), prob[strong])

        old = np.searchsorted(names, base_labels[rows])
        assert (names[old] == base_labels[rows]).all()
        prior = np.zeros(shape, np.float32)
        prior[np.arange(len(q))[:, None], old] = base_scores[rows]
        allowed = np.zeros(shape, bool)
        allowed[np.arange(len(q))[:, None], old] = True
        for score, count in ((best, 12), (metrics['cohort_excess'], 6), (sift_prob, 5)):
            top = pad_top(score, count)
            allowed[np.arange(len(q))[:, None], top] |= score[np.arange(len(q))[:, None], top] > 0

        i, j = np.nonzero(allowed)
        peak = best.max(1)
        second = np.partition(best, -2, axis=1)[:, -2]
        common = pd.DataFrame(dict(row=rows, previous_new=qf.r_pb.to_numpy()[rows],
            previous_peak=base_scores[rows].max(1), query_side=side[q], hyena=sp == 'hyena',
            raw_peak=qf.r1.to_numpy()[rows], lda_peak=qf.l1.to_numpy()[rows],
            sift_peak=qf.s1.to_numpy()[rows], query_link=qf.qqr1.to_numpy()[rows],
            album_peak=peak, album_second=second))
        known = common.iloc[i].reset_index(drop=True).assign(name=names[j], is_new=False,
            previous_identity=prior[i, j], pair_best=best[i, j], pair_other_side=other_side[i, j],
            pair_same_side=same_side[i, j], pair_side_mean=by_side[i, j, 1:].mean(1),
            pair_side_votes=side_votes[i, j], pair_direct_sift=sift_prob[i, j],
            pair_raw=metrics['cosine'][i, j], pair_cohort=metrics['cohort_excess'][i, j],
            album_gap=best[i, j]-peak[i], direct_inliers=metrics['pair_inliers'][i, j],
            gallery_size=np.bincount(group)[j], gallery_views=view_available[j, 1:].sum(1),
            has_query_side=view_available[j, side[q[i]]].astype(float))
        novelty = common.assign(name=NEW, is_new=True, previous_identity=common.previous_new)
        parts.extend((known, novelty))
    output = pd.concat(parts, ignore_index=True).sort_values('row', kind='stable').reset_index(drop=True)
    if 'truth' in frame:
        output['positive'] = output.name.to_numpy() == frame['truth'][output.row.to_numpy()]
    return output


def learn_head(frames, photo_ids, prior_only=False):
    df = pd.concat(frames, ignore_index=True)
    cols = [key for key in df if key not in ('row', 'name', 'positive')
            and (not prior_only or key not in EXTRA_COLUMNS)]
    _, inverse, counts = np.unique(np.concatenate(photo_ids), return_inverse=True, return_counts=True)
    weight = 1. / np.sqrt(counts[inverse])
    model = lgb.LGBMClassifier(n_estimators=240, num_leaves=15, max_depth=5, min_child_samples=75,
        reg_lambda=12, learning_rate=.045, n_jobs=4, verbosity=-1, random_state=1516)
    return model.fit(df[cols], df.positive, sample_weight=weight), cols


def head_scores(head, features):
    model, cols = head
    row = features.row.to_numpy()
    raw = model.predict_proba(features[cols])[:, 1]
    return raw / np.maximum(np.bincount(row, weights=raw)[row], 1e-12)


def rank_scores(features, score):
    # Limit the ordered label list to the union of real candidate identities and new_individual.
    row = features.row.to_numpy()
    order = np.lexsort((np.arange(len(row)), -score, row))
    row_sorted = row[order]
    starts = np.r_[0, np.flatnonzero(np.diff(row_sorted)) + 1]
    count = np.diff(np.r_[starts, len(row_sorted)])
    if (count < 5).any():
        raise ValueError('Not enough candidates for top-five output')
    return features.name.to_numpy()[order[starts[:, None] + np.arange(5)]]


def match_matrices(path, size, native=False):
    """Read either the Model 12 native inference graph or the episode pair cache."""
    with np.load(path, allow_pickle=False) as goods:
        a, b = goods['pairs'].T
        values = [goods[key] for key in ('good', 'inliers')] if native else goods['values'].T
    rows, cols = np.r_[a, b], np.r_[b, a]
    return tuple(sparse.csr_matrix((np.r_[value, value], (rows, cols)), shape=(size, size)) for value in values)
