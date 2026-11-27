"""One-hop score sharing between strongly matched, unlabelled queries."""
import numpy as np
from scipy import sparse


def query_links(vectors, species, inliers, min_cosine, min_inliers, strong_cosine):
    cosine = vectors @ vectors.T
    valid = ((cosine >= min_cosine) & (inliers >= min_inliers)) | (cosine >= strong_cosine)
    valid &= species[:, None] == species[None, :]
    np.fill_diagonal(valid, False)
    return sparse.csr_matrix(np.where(valid, np.clip(cosine, 0, 1), 0))


def score_graph(labels, result, links, blend, all_candidate_mix=0.):
    """Align identity scores across shortlists before combining direct neighbours."""
    catalog, flat_codes = np.unique(labels, return_inverse=True)
    codes = flat_codes.reshape(labels.shape)
    known = result['pc']
    if all_candidate_mix:
        raw = result['raw']
        sums = raw.sum(1, keepdims=True)
        all_known = np.divide(raw, sums, out=known.copy(), where=sums > 0)
        known = (1 - all_candidate_mix) * known + all_candidate_mix * all_known
    unknown = result['pb'][:, None]
    local = np.column_stack([(1 - unknown) * known, unknown])
    original_order = np.argsort(-local, axis=1)[:, :5]
    predictions = np.take_along_axis(labels, original_order, axis=1)
    if not blend or not links.nnz:
        return predictions

    scores = np.zeros((len(labels), len(catalog)))
    scores[np.arange(len(labels))[:, None], codes] = local
    degrees = np.asarray(links.sum(axis=1)).ravel()
    linked = degrees > 0
    shared = (links @ scores)[linked] / degrees[linked, None]
    combined = (1 - blend) * scores[linked] + blend * shared
    available = np.zeros(scores.shape, dtype=bool)
    available[np.arange(len(labels))[:, None], codes] = True
    combined[~(available[linked] | ((links @ available)[linked] > 0))] = -np.inf
    best = np.argsort(-combined, axis=1, kind='stable')[:, :5]
    predictions[linked] = catalog[best]
    return predictions
