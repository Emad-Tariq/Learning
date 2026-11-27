"""Fixed nuisance projection against Dalvin's PCA/LDA retrieval component."""
import json
import sys
import warnings
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.extmath import randomized_svd
from nuisance_projection_test import matching


def project(g, q, labels):
    pca = PCA(n_components=min(384, len(g) + len(q) - 1), random_state=2026).fit(np.r_[g, q])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Only one sample available.*', category=UserWarning)
        lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=.3).fit(pca.transform(g), labels)
    return matching.unit(lda.transform(pca.transform(g))), matching.unit(lda.transform(pca.transform(q)))


def main():
    meta, x = matching.load(Path.home() / 'Downloads/reid_student_data_main')
    y = meta.identity.to_numpy(); species = meta.species.to_numpy(); records = []
    fusion_check = '--fusion' in sys.argv
    for seed in ((2049, 2050, 2051) if fusion_check else (2046, 2047, 2048)):
        role = matching.split_identities(meta, seed)
        fit = np.flatnonzero(role == 'fit'); ix = np.flatnonzero(role == 'evaluation')
        gallery, query = matching.gallery_query(meta, ix, matching.groups(meta, x, ix, .97), seed + 2)
        for animal in ('hyena', 'leopard'):
            f = fit[species[fit] == animal]; residuals = []
            for identity in np.unique(y[f]):
                a = x[f[y[f] == identity]]
                if len(a) > 1:
                    residuals.append((a - a.mean(axis=0)) / np.sqrt(len(a) - 1))
            _, _, u = randomized_svd(np.vstack(residuals), n_components=64, n_iter=4, random_state=seed)
            u = u[:16]
            g = gallery[species[gallery] == animal]; q = query[species[query] == animal]
            ng = matching.unit(x[g] - .5 * (x[g] @ u.T) @ u)
            nq = matching.unit(x[q] - .5 * (x[q] @ u.T) @ u)
            lg, lq = project(x[g], x[q], y[g]); nlg, nlq = project(ng, nq, y[g])
            methods = {'cosine': (x[g], x[q]), 'nap': (ng, nq), 'lda': (lg, lq), 'nap_lda': (nlg, nlq)}
            if fusion_check:
                methods = {'lda': (lg, lq),
                           'raw_lda_fusion': (np.column_stack((x[g], lg)) / np.sqrt(2), np.column_stack((x[q], lq)) / np.sqrt(2)),
                           'nap_lda_fusion': (np.column_stack((ng, lg)) / np.sqrt(2), np.column_stack((nq, lq)) / np.sqrt(2))}
            names, inv = np.unique(y[g], return_inverse=True)
            for method, (a, b) in methods.items():
                for anchor, scores in zip(q, b @ a.T):
                    per = np.full(len(names), -np.inf); np.maximum.at(per, inv, scores)
                    rank = np.flatnonzero(names[np.argsort(-per, kind='stable')[:5]] == y[anchor])
                    rr = float(1 / (rank[0] + 1)) if len(rank) else 0.
                    records.append({'seed': seed, 'species': animal, 'id': str(meta.id.iloc[anchor]),
                                    'identity': str(y[anchor]), 'method': method, 'rr5': rr, 'top1': int(rr == 1)})
        print(seed, {m: round(np.mean([r['rr5'] for r in records if r['seed'] == seed and r['method'] == m]), 6)
                     for m in methods}, flush=True)
    Path('nuisance_fusion_results.json' if fusion_check else 'nuisance_lda_results.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
