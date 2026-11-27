"""Gallery-aware accidental-match calibration on identity-disjoint episodes."""
import sys
from pathlib import Path

import numpy as np

source = Path(__file__).parent
candidate = source.parent / "Model6" if source.name == "Research" else source / "Models" / "Abhay" / "Model6"
sys.path.insert(0, str(candidate))
import matching


def view(value):
    value = str(value).lower()
    return "left" if "left" in value else "right" if "right" in value else "front" if "front" in value else "back" if "back" in value else "other"


def size_bin(n):
    return 0 if n == 1 else 1 if n == 2 else 2 if n <= 5 else 3 if n <= 10 else 4


def fit_null(meta, x, fit, seed):
    species = meta.species.to_numpy(); identity = meta.identity.to_numpy()
    views = np.array([view(v) for v in meta.viewpoint])
    rng = np.random.default_rng(seed)
    buckets = {}
    for animal in ("hyena", "leopard"):
        indices = fit[species[fit] == animal]
        sampled = rng.choice(indices, size=min(500, len(indices)), replace=False)
        names = np.unique(identity[indices])
        for name in names:
            gallery = indices[identity[indices] == name]
            queries = sampled[identity[sampled] != name]
            sims = x[queries] @ x[gallery].T
            nbin = size_bin(len(gallery))
            gallery_views = set(views[gallery])
            for query, row in zip(queries, sims):
                key = (animal, nbin, views[query] in gallery_views)
                buckets.setdefault(key, []).append(float(np.max(row)))
    return {k: np.sort(np.asarray(v, dtype=np.float32)) for k, v in buckets.items()}


def episode(meta, x, seed, null):
    roles = matching.split_identities(meta, seed)
    known = np.flatnonzero(roles == "development")
    unknown = np.flatnonzero(roles == "evaluation")
    groups = matching.groups(meta, x, known, .97)
    gallery, known_queries = matching.gallery_query(meta, known, groups, seed + 1)
    rng = np.random.default_rng(seed)
    identity = meta.identity.to_numpy(); species = meta.species.to_numpy()
    views = np.array([view(v) for v in meta.viewpoint])
    unknown_queries = []
    for name in np.unique(identity[unknown]):
        group = unknown[identity[unknown] == name]
        unknown_queries.extend(rng.choice(group, size=min(2, len(group)), replace=False))
    queries = np.r_[known_queries, unknown_queries]
    rows = []
    for animal in ("hyena", "leopard"):
        g = gallery[species[gallery] == animal]
        q = queries[species[queries] == animal]
        names = np.unique(identity[g])
        positions = {name: g[identity[g] == name] for name in names}
        sim = x[q] @ x[g].T
        local = {name: np.flatnonzero(identity[g] == name) for name in names}
        for query, scores in zip(q, sim):
            raw, calibrated = [], []
            for name in names:
                indices = positions[name]
                best = float(np.max(scores[local[name]]))
                key = (animal, size_bin(len(indices)), views[query] in set(views[indices]))
                reference = null.get(key)
                if reference is None or len(reference) < 100:
                    reference = np.concatenate([v for k, v in null.items() if k[0] == animal and k[1] == key[1]])
                percentile = np.searchsorted(reference, best, side="right") / (len(reference) + 1)
                raw.append(best)
                calibrated.append(percentile)
            truth = identity[query] if query in known_queries else "new_individual"
            rows.append((animal, truth, names, np.asarray(raw), np.asarray(calibrated)))
    return rows


def score(rows, method, threshold):
    values = []
    for animal, truth, names, raw, calibrated in rows:
        scores = raw if method == "raw" else calibrated
        order = names[np.argsort(-scores)[:5]].tolist()
        if scores.max() < threshold:
            order = ["new_individual"] + order[:4]
        else:
            order = order[:4] + ["new_individual"]
        rank = order.index(truth) if truth in order else None
        values.append((animal, truth == "new_individual", 1 / (rank + 1) if rank is not None else 0.))
    return values


def main():
    meta, x = matching.load(Path.home() / "Downloads" / "reid_student_data_main")
    seed = 2031
    roles = matching.split_identities(meta, seed)
    null = fit_null(meta, x, np.flatnonzero(roles == "fit"), seed)
    development = episode(meta, x, seed, null)
    thresholds = {"raw": np.linspace(.2, .8, 31), "calibrated": np.linspace(.7, .999, 31)}
    chosen = {}
    for method, grid in thresholds.items():
        chosen[method] = max(grid, key=lambda t: np.mean([r[2] for r in score(development, method, t)]))
        result = score(development, method, chosen[method])
        print("development", method, "threshold", round(chosen[method], 4), "MAP5", round(np.mean([r[2] for r in result]), 4), flush=True)
    for seed in (2036, 2037, 2038):
        roles = matching.split_identities(meta, seed)
        null = fit_null(meta, x, np.flatnonzero(roles == "fit"), seed)
        rows = episode(meta, x, seed, null)
        for method in ("raw", "calibrated"):
            result = score(rows, method, chosen[method])
            print(seed, method, "n", len(result), "MAP5", round(np.mean([r[2] for r in result]), 4),
                  "known", round(np.mean([r[2] for r in result if not r[1]]), 4),
                  "unknown", round(np.mean([r[2] for r in result if r[1]]), 4), flush=True)


if __name__ == "__main__":
    main()
