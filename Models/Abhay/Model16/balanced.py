"""Make query/gallery evidence comparable at a common reference pool size."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import audit
import run as r

m = r.m
SAMPLE = 128
LOCK = r.OUT / 'balanced_recipe.json'


def exposure_weights(population, sample):
    """Probability of each rank being the maximum of a uniform sample without replacement."""
    sample = min(sample, population)
    weights = [sample / population]
    for i in range(1, population - sample + 1):
        value = weights[-1] * (population - sample - i + 1) / (population - i)
        if value < 1e-9:
            break
        weights.append(value)
    return np.asarray(weights, dtype=np.float32)


def expected_peak(matrix, exclude_diagonal=False, sample=SAMPLE):
    n = matrix.shape[1] - int(exclude_diagonal)
    weights = exposure_weights(n, sample)
    if exclude_diagonal:
        np.fill_diagonal(matrix, -np.inf)
    k = len(weights)
    top = np.sort(np.partition(matrix, -k, axis=1)[:, -k:], axis=1)[:, ::-1]
    return top @ weights


def standardized(data, g, q):
    information = {name: np.empty(len(q), dtype=np.float32) for name in (
        'pool_query_cos', 'pool_gallery_cos', 'pool_query_geom', 'pool_gallery_geom',
        'pool_cos_gap', 'pool_geom_gap', 'pool_cos_rate_gap', 'pool_geom_rate_gap')}
    for sp in r.SPECIES:
        mask = data['species'][q] == sp
        qp, gp = q[mask], g[data['species'][g] == sp]
        qx = data['x'][qp]
        qq = qx @ qx.T
        qg = qx @ data['x'][gp].T
        qqg = data['inliers'][qp][:, qp].toarray()
        ggg = data['inliers'][qp][:, gp].toarray()
        qc, gc = expected_peak(qq, True), expected_peak(qg)
        qs, gs = expected_peak(qqg, True), expected_peak(ggg)
        cr = (qq >= .65).mean(1) - (qg >= .65).mean(1)
        sr = (qqg >= 7).mean(1) - (ggg >= 7).mean(1)
        for key, value in zip(information, (qc, gc, np.log1p(qs), np.log1p(gs), qc-gc,
                                           np.log1p(qs)-np.log1p(gs), cr, sr)):
            information[key][mask] = value
    return information


def extend(frame, data, g, q):
    # Retain direct gallery peaks. Replace cohort maxima and batch sizes with
    # features computed at the same query/gallery exposure in every episode.
    drop = [c for c in frame if 'cohort' in c or c in ('query_species_size', 'gallery_species_size')]
    output = frame.drop(columns=drop).copy()
    row = frame.row.to_numpy()
    for name, vector in standardized(data, g, q).items():
        output[name] = vector[row]
    return output


def training_frames(data, outer, count):
    pool = r.roles(data, outer)
    seed = 16016 if outer == 'production' else int(outer) + 16000
    for i, (frame, _) in enumerate(r.frames(data, outer, count)):
        g, q, y = r.simulation(data, pool, seed + i, wide_gallery=i >= 10)
        assert frame.row.nunique() == len(q)
        truth_implied = frame.name.to_numpy() == y[frame.row.to_numpy()]
        assert np.array_equal(frame.hit, truth_implied)
        yield extend(frame, data, g, q)


def fit(data, outer, count, modes):
    frames = list(training_frames(data, outer, count))
    fitted = {}
    for mode in modes:
        model = m.train(frames, use_reconstruction=(mode == 'hull'))
        fitted[mode] = model
        print('Fitted balanced model', outer, mode, flush=True)
    return fitted


def validate(data, outer, models):
    records = []
    pool = set(data['identity'][r.roles(data, outer)])
    for ep, split in r.manifest(outer).groupby('episode'):
        is_g = split.assignment.eq('gallery')
        g = data['index'].get_indexer(split.loc[is_g, 'id'])
        q = data['index'].get_indexer(split.loc[~is_g, 'id'])
        y = split.loc[~is_g, 'truth'].to_numpy(str)
        assert not pool.intersection(data['identity'][q]) and not set(q).intersection(g)
        frame = m.make_frame(g, q, data, data['good'], data['inliers'], y)
        balanced = extend(frame, data, g, q)
        row = pd.DataFrame(dict(id=data['ids'][q], episode=ep, truth=y, animal=data['identity'][q],
            species=data['species'][q], novel=y == m.NEW, found=frame.groupby('row', sort=True).hit.any().to_numpy()))
        for name, model in models.items():
            pred = m.top5(m.logits(model, balanced), balanced)
            row[name] = r.reciprocal(pred, y)
            row[name + '_prediction'] = [' '.join(labels) for labels in pred]
        records.append(row)
        print('Compared balanced model', outer, ep, len(q), flush=True)
    outcomes = pd.concat(records, ignore_index=True)
    return outcomes, {name: r.summary(outcomes, name) for name in models}


def develop(data, settings):
    if LOCK.exists():
        raise FileExistsError('Balanced recipe is already frozen')
    models = fit(data, '2052', settings['training_episodes'], ['hull', 'direct'])
    _, metrics = validate(data, '2052', models)
    winner = max(metrics, key=lambda name: metrics[name]['overall']['map5'])
    r.write_json('balanced_development.json', metrics)
    r.write_json(LOCK.name, dict(variant=winner, fit=settings['training_episodes'],
        sample_size=SAMPLE, selection='Maximum development MAP@5. No competition scores used.',
        source_sha256=r.sha(__file__), shared_recipe_sha256=r.sha(r.OUT / 'recipe.json')))
    print('Balanced development scores', {k: v['overall'] for k, v in metrics.items()}, flush=True)


def confirm(data, settings):
    variant = settings['variant']
    models = fit(data, '27613', settings['fit'], [variant])
    outcomes, metrics = validate(data, '27613', models)
    reference = pd.read_csv(r.ROOT / 'Models/Abhay/Model15/results/specialist_comparison.csv')
    joined = outcomes.merge(reference, on=['episode', 'id', 'truth', 'species'], validate='one_to_one')
    assert len(joined) == len(outcomes) == len(reference)
    delta = joined[variant] - joined['weight_0.3_rr5']
    metrics['against_model15'] = dict(delta=float(delta.mean()), interval=audit.paired_ci(joined, delta),
                                     baseline_map5=float(joined['weight_0.3_rr5'].mean()))
    outcomes.to_csv(r.OUT / 'balanced_confirmation_predictions.csv', index=False)
    r.write_json('balanced_confirmation.json', metrics)
    print(json.dumps(metrics, indent=2), flush=True)


def predict(data, settings):
    variant = settings['variant']
    models = fit(data, 'production', settings['fit'], [variant])
    joblib.dump(models, r.OUT / 'balanced_production.joblib', compress=2)
    count = len(data['train'])
    g, q = np.arange(count), np.arange(count, len(data['ids']))
    frame = extend(m.make_frame(g, q, data, data['good'], data['inliers']), data, g, q)
    pred = m.top5(m.logits(models[variant], frame), frame)
    submission = pd.DataFrame({'id': data['ids'][q], 'predictions': [' '.join(values) for values in pred]})
    sample = pd.read_csv(r.DATA / 'sample_submission.csv', dtype=str)
    submission = sample[['id']].merge(submission, how='left', on='id', validate='one_to_one')
    pending = r.OUT / 'submission_balanced.tmp.csv'
    final = r.OUT / 'submission_balanced.csv'
    submission.to_csv(pending, index=False, lineterminator='\n')
    audit.verify_submission(pending)
    pending.replace(final)
    result = dict(rows=len(submission), valid=True, sha256=r.sha(final),
                  new_first=int((pred[:, 0] == m.NEW).sum()), active_variant=variant)
    r.write_json('balanced_submission_checks.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['develop', 'confirm', 'predict'])
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        data = r.load(r.DATA, r.SOURCE)
        if args.mode == 'develop':
            develop(data, r.recipe(data))
        else:
            locked = json.loads(LOCK.read_text())
            assert locked['source_sha256'] == r.sha(Path(__file__))
            assert locked['shared_recipe_sha256'] == r.sha(r.OUT / 'recipe.json')
            (confirm if args.mode == 'confirm' else predict)(data, locked)


if __name__ == '__main__':
    main()
