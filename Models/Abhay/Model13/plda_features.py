"""Gallery-only Gaussian identity evidence for supplied global descriptors.

In PCA space x = mean + identity_effect + image_noise, with covariances B
and W. Exact same-versus-independent pair likelihood ratios and posterior
predictive identity scores are evaluated after simultaneous diagonalization.
All covariance estimates and PCA use gallery labels/vectors only. Scores are
log likelihood ratios under this fitted model, not calibrated probabilities.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.decomposition import PCA


def _vectors(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty finite 2D descriptor matrix")
    return x


@dataclass
class PLDA:
    pca: object
    mean: np.ndarray
    projection: np.ndarray
    eigenvalues: np.ndarray
    within_covariance: np.ndarray
    between_covariance: np.ndarray
    identities: np.ndarray
    codes: np.ndarray
    counts: np.ndarray
    gallery: np.ndarray
    centroids: np.ndarray
    metadata: dict

    def transform(self, x):
        """Center and map descriptors to W=I, B=diagonal coordinates."""
        x = _vectors(x)
        if x.shape[1] != self.pca.n_features_in_:
            raise ValueError("Descriptor dimensionality differs from fitted gallery")
        return (self.pca.transform(x) - self.mean) @ self.projection

    def pair_scores(self, queries, gallery=None):
        """Exact log p(q,g | same identity) / [p(q)p(g)]."""
        q = self.transform(queries)
        g = self.gallery if gallery is None else self.transform(gallery)
        b = self.eigenvalues
        cross = b / (1. + 2. * b)
        penalty = (b / (1. + 2. * b)) * (b / (1. + b))
        constant = np.sum(np.log1p(b) - .5 * np.log1p(2. * b))
        return ((q * cross) @ g.T
                - .5 * (q * q @ penalty)[:, None]
                - .5 * (g * g @ penalty)[None, :] + constant)

    def predictive_scores(self, queries):
        """Exact log p(query | all gallery photos of identity) / p(query).

        Each image supplies conditionally independent noise under the model.
        Correlated/duplicate photos can violate this assumption; this score is
        auxiliary evidence for the learned heads rather than a replacement
        for the existing calibrated open-set decision.
        """
        q = self.transform(queries)
        b = self.eigenvalues
        nb = self.counts[:, None] * b[None, :]
        latent_variance = b[None, :] / (1. + nb)
        posterior_mean = (nb / (1. + nb)) * self.centroids
        inverse_predictive = 1. / (1. + latent_variance)
        inverse_marginal = 1. / (1. + b)
        constant = .5 * (np.log1p(b)[None, :] - np.log1p(latent_variance)).sum(1)
        constant -= .5 * (posterior_mean * posterior_mean * inverse_predictive).sum(1)
        return (q @ (posterior_mean * inverse_predictive).T
                - .5 * (q * q) @ (inverse_predictive - inverse_marginal).T
                + constant[None, :])

    def identity_scores(self, queries):
        """Aggregate pair evidence and retain the separate Bayesian score."""
        pair = self.pair_scores(queries)
        best = np.empty((len(pair), len(self.identities)))
        top2 = np.empty_like(best)
        for j in range(len(self.identities)):
            values = pair[:, self.codes == j]
            best[:, j] = values.max(1)
            if values.shape[1] == 1:
                top2[:, j] = values[:, 0]
            else:
                top2[:, j] = np.partition(values, -2, axis=1)[:, -2:].mean(1)
        return dict(identities=self.identities.copy(), pair_max=best,
                    pair_top2=top2, predictive=self.predictive_scores(queries))


def fit(gallery, labels, *, components=256, within_shrinkage=.15,
        eigen_floor=1e-5, random_state=13052):
    """Estimate a PLDA mapper from one species' labelled gallery only.

    W is pooled within-identity covariance with isotropic shrinkage. B is the
    equally weighted identity-centroid covariance, corrected by mean(1/n)*W
    for centroid noise, then projected to the positive-semidefinite cone.
    The retained PCA dimension is bounded by samples and input dimension.
    """
    x = _vectors(gallery)
    labels = np.asarray(labels, dtype=object)
    if labels.shape != (len(x),) or pd.isna(labels).any():
        raise ValueError("One nonmissing gallery identity is required per image")
    if components < 1 or not 0 <= within_shrinkage <= 1 or eigen_floor <= 0:
        raise ValueError("Invalid dimensionality or covariance regularization")
    identities, codes, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if len(identities) < 2 or len(x) <= len(identities):
        raise ValueError("At least two identities and some repeated photos are required")
    dim = min(int(components), x.shape[1], len(x) - 1)
    pca = PCA(n_components=dim, svd_solver="auto", random_state=random_state)
    pca.fit(x)
    # Randomized PCA's approximate fit_transform scores can differ from the
    # subsequent transform. Use one coordinate definition for stored gallery
    # points, covariance estimation, and every future query/gallery transform.
    z = pca.transform(x)
    means = np.stack([z[codes == j].mean(0) for j in range(len(identities))])
    center = means.mean(0)
    residual = z - means[codes]
    raw_within = residual.T @ residual / (len(x) - len(identities))
    total_scale = float(np.var(z, axis=0, ddof=1).mean())
    if total_scale <= 0:
        raise ValueError("Gallery descriptors have no variation")
    within_scale = max(float(np.trace(raw_within) / dim), total_scale * eigen_floor)
    within = ((1. - within_shrinkage) * raw_within
              + (within_shrinkage * within_scale + eigen_floor * total_scale) * np.eye(dim))
    centered_means = means - center
    raw_between = centered_means.T @ centered_means / (len(identities) - 1)
    raw_between -= np.mean(1. / counts) * within
    raw_between = (raw_between + raw_between.T) * .5
    values, vectors = eigh(raw_between)
    between = (vectors * np.maximum(values, 0.)) @ vectors.T
    between = (between + between.T) * .5
    eigenvalues, projection = eigh(between, within)
    eigenvalues = np.maximum(eigenvalues, 0.)
    transformed = (z - center) @ projection
    centroids = np.stack([transformed[codes == j].mean(0) for j in range(len(identities))])
    metadata = dict(gallery_images=len(x), gallery_identities=len(identities),
                    input_dimensions=x.shape[1], pca_dimensions=dim,
                    within_shrinkage=float(within_shrinkage), eigen_floor=float(eigen_floor),
                    positive_between_dimensions=int((values > 0).sum()),
                    fit_scope="gallery vectors and gallery identity labels only")
    return PLDA(pca, center, projection, eigenvalues, within, between,
                identities, codes, counts, transformed, centroids, metadata)


def candidate_features(model, queries, candidate_labels):
    """Return (flattened_candidate_frame, query_frame) of numeric additions.

    ``candidate_labels`` has shape [queries, K] and must name gallery identities.
    The returned candidate rows preserve its row-major order. There are no
    labels, truth, or identifiers among the returned feature columns.
    """
    candidate_labels = np.asarray(candidate_labels, dtype=object)
    if candidate_labels.ndim != 2 or candidate_labels.shape[0] != len(queries):
        raise ValueError("Expected one candidate-label row per query")
    indices = pd.Index(model.identities).get_indexer(candidate_labels.ravel()).reshape(candidate_labels.shape)
    if (indices < 0).any():
        raise ValueError("A candidate identity is absent from the fitted gallery")
    scores = model.identity_scores(queries)
    query_features, candidate_values = {}, {}
    for name in ("pair_max", "pair_top2", "predictive"):
        values = scores[name]
        sorted_values = np.sort(values, axis=1)[:, ::-1]
        query_features[f"pl_{name}_best"] = sorted_values[:, 0]
        query_features[f"pl_{name}_second"] = sorted_values[:, 1]
        query_features[f"pl_{name}_margin"] = sorted_values[:, 0] - sorted_values[:, 1]
        selected = np.take_along_axis(values, indices, axis=1)
        ranks = np.argsort(np.argsort(-values, axis=1, kind="stable"), axis=1, kind="stable")
        candidate_values[f"pl_{name}"] = selected.ravel()
        candidate_values[f"pl_{name}_gap"] = (selected - sorted_values[:, :1]).ravel()
        candidate_values[f"pl_{name}_rank"] = np.take_along_axis(ranks, indices, axis=1).ravel()
    query_features["pl_pair_predictive_agree"] = (scores["pair_max"].argmax(1) == scores["predictive"].argmax(1)).astype(int)
    return pd.DataFrame(candidate_values), pd.DataFrame(query_features)
