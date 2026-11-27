"""Paired open-set comparison: Dalvin 3d features versus nuisance raw evidence."""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

import dalvin_core as b
import data_protocol as protocol


def nuisance_features(indices, dimensions=16, strength=.5, seed=2052):
    """Every non-singleton fitting identity contributes equal covariance weight."""
    transformed = b.X.copy()
    directions = {}
    for species in ('hyena', 'leopard'):
        fit = indices[b.SP[indices] == species]
        residuals = []
        for identity in np.unique(b.Y[fit]):
            images = b.X[fit[b.Y[fit] == identity]]
            if len(images) > 1:
                residuals.append((images - images.mean(axis=0)) / np.sqrt(len(images) - 1))
        if not residuals:
            raise ValueError('No repeated fitting identities for nuisance estimation')
        residuals = np.vstack(residuals)
        # Match the research implementation: estimate 64 components, use the first 16.
        _, _, u = randomized_svd(residuals, n_components=min(64, min(residuals.shape) - 1),
                                 n_iter=4, random_state=seed)
        u = u[:dimensions]
        directions[species] = u
        ix = np.flatnonzero(b.SP == species)
        transformed[ix] = b.unit(b.X[ix] - strength * (b.X[ix] @ u.T) @ u)
    return transformed, directions


def stats(frame):
    report = {}
    for method in ('baseline', 'nuisance'):
        col = method + '_rr5'
        report[method] = {'map5': float(frame[col].mean()),
                          'known_map5': float(frame.loc[~frame.unknown, col].mean()),
                          'unknown_map5': float(frame.loc[frame.unknown, col].mean()),
                          'top1': float((frame[col] == 1).mean())}
    report['delta'] = report['nuisance']['map5'] - report['baseline']['map5']
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=Path.home() / 'Downloads/reid_student_data_main')
    ap.add_argument('--output', type=Path, default=Path('local_runs/paired_2052'))
    ap.add_argument('--seed', type=int, default=2052)
    ap.add_argument('--sims', type=int, default=12)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--neighbours', type=int, default=150)
    ap.add_argument('--trees', type=int, default=400)
    ap.add_argument('--predict-test', action='store_true')
    args = ap.parse_args()
    if min(args.sims, args.workers, args.neighbours, args.trees) < 1:
        ap.error('Counts must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(rounds=3, nuisance_dimensions=16, nuisance_strength=.5,
                  evaluation_episodes=[args.seed + 20000 + j for j in range(3)],
                  gate='positive pooled and each-episode delta; known and unknown MAP5 do not fall')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    cv2.setNumThreads(1)
    b.PARAMS.update(n_estimators=args.trees, n_jobs=args.workers, random_state=args.seed)
    warnings.filterwarnings('ignore', message='Only one sample available.*', category=UserWarning)
    started = time.monotonic()
    train, test, all_df = protocol.load_data(args.data)
    roles = protocol.allocate_roles(train, 3, args.seed)
    roles['metric_fit'] = roles.pop('calibration')
    manifest = pd.DataFrame([{'id': b.IDS[i], 'species': b.SP[i], 'identity': b.Y[i], 'role': name}
                             for name, ix in roles.items() for i in ix])
    assert manifest.groupby('identity').role.nunique().max() == 1
    manifest.to_csv(args.output / 'identity_roles.csv', index=False)
    nx, directions = nuisance_features(roles['metric_fit'], seed=args.seed)
    joblib.dump(directions, args.output / 'nuisance_directions.joblib')
    cache = protocol.SiftCache(args.data, args.output, args.workers, args.neighbours)
    models = {'baseline': [], 'nuisance': []}
    splits = []

    def split(ix, seed, name, link):
        gi, qi, truth = protocol.split_episode(ix, seed, link=link)
        truth_map = dict(zip(qi, truth))
        for i in gi:
            splits.append({'episode': seed, 'role': name, 'id': b.IDS[i], 'assignment': 'gallery'})
        for i in qi:
            splits.append({'episode': seed, 'role': name, 'id': b.IDS[i],
                           'assignment': 'unknown_query' if truth_map[i] == protocol.NEW else 'known_query'})
        return gi, qi, truth

    def frames(gi, qi, truth):
        result = {}
        for method in models:
            b.NX = b.X if method == 'baseline' else nx
            result[method] = protocol.frame(gi, qi, truth)
        assert np.array_equal(result['baseline']['order'], result['nuisance']['order'])
        assert result['baseline']['c'].ident.equals(result['nuisance']['c'].ident)
        return result

    for stage in range(3):
        role = f'stage{stage + 1}'
        cache.activate(roles[role])
        training = {name: [] for name in models}
        for j in range(args.sims):
            gi, qi, truth = split(roles[role], args.seed + 1000 * (stage + 1) + j,
                                 role, b.REGIMES[j % len(b.REGIMES)])
            paired = frames(gi, qi, truth)
            for method, frame in paired.items():
                for model in models[method]:
                    frame = b.augment(frame, b.predict(model, frame))
                training[method].append(frame)
            print(f'{role} simulation {j + 1}/{args.sims}', flush=True)
        for method in models:
            print(f'Fitting {method} {role}', flush=True)
            models[method].append(b.fit(training[method]))
        joblib.dump(models, args.output / 'paired_models.joblib')
    # Everything supervised is frozen before opening validation identities.
    cache.activate(roles['validation'])
    results = []
    for j in range(3):
        seed = args.seed + 20000 + j
        gi, qi, truth = split(roles['validation'], seed, 'validation', (.95, 111))
        paired = frames(gi, qi, truth)
        frame = pd.DataFrame({'episode': seed, 'id': b.IDS[paired['baseline']['order']],
                              'identity': b.Y[paired['baseline']['order']],
                              'species': b.SP[paired['baseline']['order']],
                              'truth': paired['baseline']['truth']})
        frame['unknown'] = frame.truth == protocol.NEW
        for method, s in paired.items():
            s, r = protocol.through_stack(s, models[method])
            predictions = b.rank(s, r)
            frame[method + '_rr5'] = [next((1 / (k + 1) for k, label in enumerate(row) if label == target), 0.)
                                       for row, target in zip(predictions, frame.truth)]
            frame[method + '_predictions'] = [' '.join(row) for row in predictions]
        results.append(frame)
        print('Validation', seed, json.dumps(stats(frame)), flush=True)
    result = pd.concat(results, ignore_index=True)
    result.to_csv(args.output / 'validation_predictions.csv', index=False)
    pd.DataFrame(splits).to_csv(args.output / 'episode_splits.csv', index=False)
    report = stats(result)
    report['episodes'] = {str(seed): stats(part) for seed, part in result.groupby('episode')}
    report['species'] = {str(sp): stats(part) for sp, part in result.groupby('species')}
    report['query_appearances'] = len(result)
    report['unique_queries'] = result.id.nunique()
    report['gate_passed'] = bool(report['delta'] > 0 and all(v['delta'] > 0 for v in report['episodes'].values())
                                 and all(report['nuisance'][k] >= report['baseline'][k]
                                         for k in ('known_map5', 'unknown_map5')))
    report['elapsed_seconds'] = time.monotonic() - started
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if args.predict_test and report['gate_passed']:
        cache.activate(np.arange(len(all_df)))
        gi, qi = np.arange(len(train)), np.arange(len(train), len(all_df))
        b.NX = nx
        s, r = protocol.through_stack(protocol.frame(gi, qi), models['nuisance'])
        predictions = b.rank(s, r)
        sub = pd.DataFrame({'id': b.IDS[s['order']], 'predictions': [' '.join(row) for row in predictions]})
        sample = pd.read_csv(args.data / 'sample_submission.csv')
        sub = sample[['id']].merge(sub, on='id', validate='one_to_one')
        allowed = train.groupby('species').identity.apply(set).to_dict()
        species = test.set_index('id').species
        assert len(sub) == len(test) and sub.id.is_unique and not sub.predictions.isna().any()
        for row in sub.itertuples():
            labels = row.predictions.split()
            assert len(labels) == len(set(labels)) == 5
            assert all(t == protocol.NEW or t in allowed[species[row.id]] for t in labels)
        sub.to_csv(args.output / 'submission_nuisance.csv', index=False)
        print('Saved submission_nuisance.csv', flush=True)
    elif args.predict_test:
        print('Validation gate failed; submission generation skipped.', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4, user_api='blas'):
        main()
