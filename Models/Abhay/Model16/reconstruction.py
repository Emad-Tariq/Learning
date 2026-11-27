"""Identity reconstruction and one conditional softmax over known/new outcomes."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.decomposition import PCA


NEW = 'new_individual'


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def top_two(a):
    if a.shape[1] == 1:
        return a[:, 0], a[:, 0]
    tail = np.partition(a, -2, axis=1)[:, -2:]
    return tail[:, 1], tail[:, 0]


def segment_similarity(sa, sb, a, b):
    """1 - half squared distance to the best convex combination of two unit references."""
    distance = np.maximum(1. - np.einsum('ij,ij->i', a, b), 0.)
    fraction = np.clip((sb - sa + distance) / np.maximum(2. * distance, 1e-7), 0., 1.)
    return sa + fraction * (sb - sa + distance) - fraction**2 * distance


def candidate_matrices(g, q, x, names, views, good, inliers):
    dims = min(192, len(g) - 1, x.shape[1])
    projector = PCA(dims, svd_solver='randomized', random_state=1616).fit(x[g])
    whiten = np.maximum(projector.explained_variance_, .04 * projector.explained_variance_[0]) ** -.35
    zg, zq = (unit(projector.transform(x[rows]) * whiten) for rows in (g, q))
    raw, transformed = x[q] @ x[g].T, zq @ zg.T
    geometry = inliers[q][:, g].toarray().astype(np.float32)
    matches = good[q][:, g].toarray().astype(np.float32)
    catalog, group = np.unique(names[g], return_inverse=True)
    count, n = np.bincount(group), len(q)
    fields = ('raw', 'raw_second', 'projected', 'projected_second', 'centroid', 'segment',
              'same_flank', 'other_flank', 'sift', 'sift_second', 'sift_fraction', 'sift_support',
              'gallery_total', 'same_flank_total')
    evidence = {name: np.zeros((n, len(catalog)), np.float32) for name in fields}
    for c in range(len(catalog)):
        ix = np.flatnonzero(group == c)
        ref = g[ix]
        for key, matrix in [('raw', raw), ('projected', transformed), ('sift', geometry)]:
            best, second = top_two(matrix[:, ix])
            evidence[key][:, c] = best
            evidence[key + '_second'][:, c] = second
        gz = zg[ix]
        similarity = transformed[:, ix]
        evidence['centroid'][:, c] = zq @ unit(gz.mean(0, keepdims=True))[0]
        evidence['segment'][:, c] = evidence['projected'][:, c]
        if len(ix) > 1:
            k = min(4, len(ix))
            proposals = np.argpartition(similarity, -k, axis=1)[:, -k:]
            for a in range(k):
                for b in range(a + 1, k):
                    ia, ib = proposals[:, a], proposals[:, b]
                    sa, sb = similarity[np.arange(n), ia], similarity[np.arange(n), ib]
                    estimate = segment_similarity(sa, sb, gz[ia], gz[ib])
                    evidence['segment'][:, c] = np.maximum(evidence['segment'][:, c], estimate)
        same = (views[q, None] == views[ref][None, :]) & (views[q, None] >= 0)
        evidence['same_flank'][:, c] = np.where(same, similarity, -1.).max(1)
        evidence['other_flank'][:, c] = np.where(~same, similarity, -1.).max(1)
        s = geometry[:, ix]
        selected = s.argmax(1)
        evidence['sift_fraction'][:, c] = s[np.arange(n), selected] / np.maximum(matches[np.arange(n), ix[selected]], 1.)
        evidence['sift_support'][:, c] = (s >= 7).sum(1)
        evidence['gallery_total'][:, c] = np.log1p(count[c])
        evidence['same_flank_total'][:, c] = np.log1p(same.sum(1))
    evidence['segment_gain'] = evidence['segment'] - evidence['projected']

    # Unlabelled queries are a competing reference population for novelty. These
    # features summarize matches numerically and never transfer identity scores.
    qq = x[q] @ x[q].T
    np.fill_diagonal(qq, -1.)
    qq_sift = inliers[q][:, q].toarray().astype(np.float32)
    np.fill_diagonal(qq_sift, 0.)
    cohort_cos, cohort_second = top_two(qq)
    cohort_geom, cohort_geom2 = top_two(qq_sift)
    best_g, second_g = top_two(evidence['raw'])
    best_z, second_z = top_two(evidence['projected'])
    best_sift, second_sift = top_two(evidence['sift'])
    best_segment, second_segment = top_two(evidence['segment'])
    context = dict(raw_peak=best_g, raw_margin=best_g-second_g, projected_peak=best_z,
        projected_margin=best_z-second_z, segment_peak=best_segment,
        segment_margin=best_segment-second_segment, sift_peak=best_sift, sift_margin=best_sift-second_sift,
        cohort_cos=cohort_cos, cohort_cos_second=cohort_second, cohort_geometry=cohort_geom,
        cohort_geometry_second=cohort_geom2, cohort_cos_gap=cohort_cos-best_g,
        cohort_geometry_gap=np.log1p(cohort_geom)-np.log1p(best_sift),
        query_species_size=np.log1p(len(q)), gallery_species_size=np.log1p(len(g)), flank=views[q])
    return catalog, evidence, context


def make_frame(gallery, queries, data, good, inliers, truth=None):
    x, species, names, views = (data[k] for k in ('x', 'species', 'identity', 'flank'))
    parts = []
    for code, sp in enumerate(('hyena', 'leopard')):
        rows = np.flatnonzero(species[queries] == sp)
        g = gallery[species[gallery] == sp]
        q = queries[rows]
        catalog, evidence, context = candidate_matrices(g, q, x, names, views, good, inliers)
        mask = np.zeros_like(evidence['raw'], dtype=bool)
        for key, requested in [('raw', 18), ('segment', 18), ('centroid', 8), ('sift', 10)]:
            scores = evidence[key]
            k = min(requested, len(catalog))
            ix = np.argpartition(scores, -k, axis=1)[:, -k:]
            selected = scores[np.arange(len(q))[:, None], ix]
            mask[np.arange(len(q))[:, None], ix] |= selected >= 4 if key == 'sift' else True
        i, j = np.nonzero(mask)
        feat = {key: val[i, j] for key, val in evidence.items()}
        for key in ('raw', 'projected', 'segment', 'centroid', 'sift'):
            feat[key + '_gap'] = evidence[key][i, j] - evidence[key].max(1)[i]
        common = pd.DataFrame(dict(row=rows, species_hyena=code == 0, **context))
        known = common.iloc[i].reset_index(drop=True).assign(name=catalog[j], is_new=False, **feat)
        unknown = common.assign(name=NEW, is_new=True)
        parts.extend((known, unknown))
    frame = pd.concat(parts, ignore_index=True).sort_values('row', kind='stable').reset_index(drop=True)
    if truth is not None:
        frame['hit'] = frame['name'].to_numpy() == truth[frame.row.to_numpy()]
    return frame


def grouped_probability(raw, sizes):
    begin = np.r_[0, np.cumsum(sizes[:-1])]
    centered = raw - np.repeat(np.maximum.reduceat(raw, begin), sizes)
    weight = np.exp(centered)
    return weight / np.repeat(np.add.reduceat(weight, begin), sizes)


def train(frames, use_reconstruction=True, use_cohort=True):
    sizes, truths, matrix = [], [], []
    for frame in frames:
        has_hit = frame.groupby('row', sort=True).hit.transform('any')
        selected = frame.loc[has_hit]
        sizes.extend(selected.groupby('row', sort=True).size())
        truths.extend(selected.hit.astype(np.float64))
        matrix.append(selected)
    cols = [name for name in frames[0] if name not in ('row', 'name', 'hit')
            and (use_reconstruction or 'segment' not in name)
            and (use_cohort or 'cohort' not in name)]
    sizes, truths = np.asarray(sizes, int), np.asarray(truths, float)
    assert np.array_equal(np.add.reduceat(truths, np.r_[0, np.cumsum(sizes[:-1])]), np.ones(len(sizes)))

    def likelihood(raw, unused):
        prob = grouped_probability(raw, sizes)
        # Diagonal majorizer of the multinomial Hessian for tree fitting.
        return prob - truths, np.maximum(2 * prob * (1 - prob), 1e-5)

    features = pd.concat(matrix, ignore_index=True)[cols].astype(np.float32)
    estimator = lgb.train(dict(objective=likelihood, learning_rate=.07, num_leaves=15,
        min_data_in_leaf=140, lambda_l2=12., verbosity=-1, num_threads=5,
        feature_fraction=.9, seed=1626), lgb.Dataset(features, label=truths), num_boost_round=330)
    return estimator, cols


def logits(model, frame):
    estimator, cols = model
    return estimator.predict(frame[cols].astype(np.float32), num_threads=5)


def adjust_prior(raw, frame, assumed, strength):
    # Conditional posteriors induce the binary novel/known marginal. EM estimates
    # a bounded prior change within species, leaving relative known odds intact.
    adjusted = raw.copy()
    sizes = frame.groupby('row', sort=True).size().to_numpy()
    q = frame.is_new.to_numpy()
    post = grouped_probability(raw, sizes)[q]
    stats = {}
    for sp in (False, True):
        which = q & (frame.species_hyena.to_numpy() == sp)
        prior = assumed[sp]
        ratio = logit(post[frame.species_hyena.to_numpy()[q] == sp].clip(1e-5, 1 - 1e-5)) - logit(prior)
        pi = prior
        for _ in range(80):
            next_pi = np.clip(expit(ratio + logit(pi)).mean(), .10, .75)
            if abs(next_pi - pi) < 1e-6:
                break
            pi = next_pi
        change = np.clip(logit(pi) - logit(prior), -.65, .65)
        adjusted[which] += strength * change
        stats['hyena' if sp else 'leopard'] = dict(training_prior=prior, estimated_prior=float(pi),
            raw_mean=float(expit(ratio + logit(prior)).mean()), applied_log_odds=float(strength * change))
    return adjusted, stats


def top5(raw, frame):
    row = frame.row.to_numpy()
    order = np.lexsort((np.arange(len(row)), -raw, row))
    starts = np.r_[0, np.flatnonzero(np.diff(row[order])) + 1]
    return frame.name.to_numpy()[order[starts[:, None] + np.arange(5)]]
