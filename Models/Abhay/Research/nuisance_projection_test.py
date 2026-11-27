"""Learn identity-balanced nuisance directions; preserve the remaining descriptor."""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.utils.extmath import randomized_svd

source = Path(__file__).parent
sys.path.insert(0, str(source.parent / 'Model6' if source.name == 'Research' else source / 'Models/Abhay/Model6'))
import matching


def episode(meta, x, seed, configurations):
    roles = matching.split_identities(meta, seed)
    fit = np.flatnonzero(roles == 'fit')
    ix = np.flatnonzero(roles == 'evaluation')
    gallery, query = matching.gallery_query(meta, ix, matching.groups(meta, x, ix, .97), seed + 2)
    y = meta.identity.to_numpy(); species = meta.species.to_numpy()
    output = {name: [] for name in ['cosine'] + list(configurations)}
    for animal in ('hyena', 'leopard'):
        f = fit[species[fit] == animal]
        residuals = []
        for identity in np.unique(y[f]):
            a = x[f[y[f] == identity]]
            if len(a) > 1:
                residuals.append((a - a.mean(axis=0)) / np.sqrt(len(a) - 1))
        residuals = np.vstack(residuals)
        _, _, directions = randomized_svd(residuals, n_components=64, n_iter=4, random_state=seed)
        g = gallery[species[gallery] == animal]; q = query[species[query] == animal]
        names, inv = np.unique(y[g], return_inverse=True)
        for name, config in [('cosine', None)] + list(configurations.items()):
            if config is None:
                a, b = x[q], x[g]
            else:
                dim, strength = config
                u = directions[:dim]
                a = matching.unit(x[q] - strength * (x[q] @ u.T) @ u)
                b = matching.unit(x[g] - strength * (x[g] @ u.T) @ u)
            for anchor, scores in zip(q, a @ b.T):
                per = np.full(len(names), -np.inf); np.maximum.at(per, inv, scores)
                rank = np.flatnonzero(names[np.argsort(-per, kind='stable')[:5]] == y[anchor])
                rr = float(1 / (rank[0] + 1)) if len(rank) else 0.
                output[name].append({'species': animal, 'id': str(meta.id.iloc[anchor]),
                                     'identity': str(y[anchor]), 'rr5': rr, 'top1': int(rr == 1)})
    return output


def summarize(output):
    return {name: {group: float(np.mean([r['rr5'] for r in rows if group == 'all' or r['species'] == group]))
                   for group in ('all', 'hyena', 'leopard')} for name, rows in output.items()}


def main():
    meta, x = matching.load(Path.home() / 'Downloads/reid_student_data_main')
    configs = {f'nap_{d}_{int(a*100)}': (d, a) for d in (16, 64) for a in (.25, .5)}
    report = {'screen_seeds': [2041, 2042], 'check_seeds': [2043, 2044, 2045], 'screen': {}, 'check': {}}
    for seed in report['screen_seeds']:
        output = episode(meta, x, seed, configs)
        report['screen'][seed] = summarize(output)
        print(seed, json.dumps(report['screen'][seed]), flush=True)
    winner = max(configs, key=lambda name: np.mean([report['screen'][s][name]['all'] for s in report['screen_seeds']]))
    report['selected'] = winner
    print('Selected', winner, 'before check episodes', flush=True)
    for seed in report['check_seeds']:
        output = episode(meta, x, seed, {winner: configs[winner]})
        report['check'][seed] = {'summary': summarize(output), 'predictions': output}
        print(seed, json.dumps(report['check'][seed]['summary']), flush=True)
    Path('nuisance_projection_results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
