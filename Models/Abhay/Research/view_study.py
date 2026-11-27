"""Screen viewpoint-aware cosine retrieval on predeclared identity splits."""
import sys
from pathlib import Path

import numpy as np

source = Path(__file__).parent
candidate = source.parent / "Model6" if source.name == "Research" else source / "Models" / "Abhay" / "Model6"
sys.path.insert(0, str(candidate))
import matching


def view_class(value):
    value = str(value).lower()
    if "left" in value:
        return "left"
    if "right" in value:
        return "right"
    if "front" in value:
        return "front"
    if "back" in value:
        return "back"
    return "other"


def episode(meta, x, seed):
    roles = matching.split_identities(meta, seed)
    ix = np.flatnonzero(roles == "evaluation")
    group = matching.groups(meta, x, ix, .97)
    gallery, queries = matching.gallery_query(meta, ix, group, seed + 2)
    identity = meta.identity.to_numpy()
    species = meta.species.to_numpy()
    views = np.array([view_class(v) for v in meta.viewpoint])
    rows = []
    for animal in ("hyena", "leopard"):
        g = gallery[species[gallery] == animal]
        q = queries[species[queries] == animal]
        names, inv = np.unique(identity[g], return_inverse=True)
        similarities = x[q] @ x[g].T
        gallery_similarity = x[g] @ x[g].T
        gallery_similarity[identity[g, None] == identity[g][None, :]] = -np.inf
        density = np.mean(np.sort(gallery_similarity, axis=1)[:, -30:], axis=1)
        for anchor, raw in zip(q, similarities):
            same = (views[g] == views[anchor]).astype(np.float32)
            flank = (views[anchor] in ("left", "right")) & (views[g] == views[anchor])
            candidates = {"cosine": raw,
                          "exact_005": raw + .05 * same,
                          "exact_010": raw + .10 * same,
                          "flank_005": raw + .05 * flank,
                          "flank_010": raw + .10 * flank,
                          "hub_025": raw - .25 * density,
                          "hub_050": raw - .50 * density}
            record = {"seed": seed, "species": animal, "view": views[anchor]}
            for method, scores in candidates.items():
                per_identity = np.full(len(names), -np.inf)
                np.maximum.at(per_identity, inv, scores)
                ranked = names[np.argsort(-per_identity)[:5]]
                rank = np.flatnonzero(ranked == identity[anchor])
                record[method] = 1 / (rank[0] + 1) if len(rank) else 0.
            rows.append(record)
    return rows


def main():
    meta, x = matching.load(Path.home() / "Downloads" / "reid_student_data_main")
    for seed in (2036, 2037, 2038, 2039, 2040):
        rows = episode(meta, x, seed)
        for animal in ("hyena", "leopard", "all"):
            part = [r for r in rows if animal == "all" or r["species"] == animal]
            scores = {m: round(float(np.mean([r[m] for r in part])), 4)
                      for m in ("cosine", "hub_025", "hub_050")}
            print(seed, animal, len(part), scores, flush=True)


if __name__ == "__main__":
    main()
