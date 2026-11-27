"""Fit and verify a competition model using only provided descriptors and geometric matches."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits

import reconstruction as m

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'results'
SOURCE = Path.home() / 'Documents/Abhay/Semester 5/ML/Project/Learning'
DATA = Path.home() / 'Downloads/reid_student_data_main'
SPECIES = ('hyena', 'leopard')
VARIANTS = {'full': (True, True), 'no_hull': (False, True), 'no_cohort': (True, False)}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(name, result):
    (OUT / name).write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


def load(data, team):
    train, test = (pd.read_csv(data / f'{key}.csv') for key in ('train', 'test'))
    assert train.identity.notna().all() and 'identity' not in test
    population = pd.concat([train, test], ignore_index=True)
    keys = pd.Index(pd.read_csv(data / 'miewid_ids.csv').id)
    locate = keys.get_indexer(population.id)
    assert keys.is_unique and population.id.is_unique and (locate >= 0).all()
    x = m.unit(np.asarray(np.load(data / 'miewid_features.npy', mmap_mode='r')[locate], dtype=np.float32))
    assert np.isfinite(x).all()
    flank = population.viewpoint.map(dict(left=0, frontleft=0, backleft=0, right=1,
        frontright=1, backright=1, front=2, back=2, up=2, down=2)).fillna(-1).to_numpy(int)
    master = pd.read_csv(team / 'Processed_Data/master_dataset.csv')
    if not master.id.equals(population.id) or not master.species.equals(population.species):
        raise ValueError('Geometric-match cache does not align with the competition data')
    path = team / 'Models/Emad/cross_flank_3e3eceb5c8fa/sift_pairs_mutual.npz'
    with np.load(path, allow_pickle=False) as cache:
        a, b = cache['pairs'].T
        assert ((a >= 0) & (a < b) & (b < len(population))).all()
        rc = np.r_[a, b], np.r_[b, a]
        good, inliers = (sparse.csr_matrix((np.tile(cache[k].astype(np.float32), 2), rc),
            shape=(len(population), len(population))) for k in ('good', 'inl'))
    return dict(train=train, test=test, x=x, species=population.species.to_numpy(),
        identity=population.identity.to_numpy(), ids=population.id.to_numpy(), flank=flank,
        index=pd.Index(population.id), good=good, inliers=inliers, cache_sha256=sha(path))


def roles(data, outer):
    if outer == 'production':
        return np.arange(len(data['train']))
    folder = (ROOT / 'Models/Abhay/Model12/local_runs/large_2052' if outer == '2052'
              else ROOT / 'Models/Abhay/Model14/results/fresh_27613')
    table = pd.read_csv(folder / 'identity_roles.csv')
    assert table.groupby('identity').role.nunique().max() == 1
    audit_ids = set(table.loc[table.role.eq('validation'), 'identity'])
    return np.flatnonzero(~data['train'].identity.isin(audit_ids))


def simulation(data, training, seed, wide_gallery=False):
    x, labels, species = data['x'], data['identity'], data['species']
    rng = np.random.default_rng(seed)
    gallery, query, truth = [], [], []
    for sp in SPECIES:
        ix = training[species[training] == sp]
        names = rng.permutation(np.unique(labels[ix]))
        testable = set(names[:int((.30 if wide_gallery else .67) * len(names))])
        n_unknown = int(len(testable) * rng.uniform(.16, .39))
        novel = set(names[:n_unknown])
        fraction = rng.uniform(.25, .48)
        for label in names:
            ref = ix[labels[ix] == label]
            if label in novel:
                query.extend(ref)
                truth.extend([m.NEW] * len(ref))
                continue
            if label not in testable or len(ref) < 2:
                gallery.extend(ref)
                continue
            similarity = x[ref] @ x[ref].T
            geo = data['inliers'][ref][:, ref].toarray()
            _, blocks = connected_components(sparse.csr_matrix((similarity >= .95) | (geo >= 100)))
            selected = np.zeros(len(ref), bool)
            target = max(1, round(len(ref) * fraction))
            for block in rng.permutation(np.unique(blocks)):
                take = blocks == block
                if selected.sum() < target and selected.sum() + take.sum() < len(ref):
                    selected |= take
            gallery.extend(ref[~selected])
            query.extend(ref[selected])
            truth.extend([label] * selected.sum())
    return np.array(gallery), np.array(query), np.array(truth)


def frames(data, outer, count):
    pool = roles(data, outer)
    frame_dir = OUT / 'training' / outer
    frame_dir.mkdir(parents=True, exist_ok=True)
    first_seed = 16016 if outer == 'production' else int(outer) + 16000
    for number, episode in enumerate(range(first_seed, first_seed + count)):
        filename = frame_dir / f'{episode}.joblib'
        if filename.exists():
            frame, priors = joblib.load(filename)
        else:
            g, q, y = simulation(data, pool, episode, wide_gallery=number >= 10)
            assert not set(g) & set(q)
            assert not (set(data['identity'][g]) & set(data['identity'][q[y == m.NEW]]))
            frame = m.make_frame(g, q, data, data['good'], data['inliers'], y)
            priors = {sp: [int(((y == m.NEW) & (data['species'][q] == sp)).sum()),
                int((data['species'][q] == sp).sum())] for sp in SPECIES}
            joblib.dump((frame, priors), filename, compress=1)
            print('Prepared training episode', episode, 'gallery', len(g), 'queries', len(q), flush=True)
        yield frame, priors


def fit(data, outer, count, variants):
    training = list(frames(data, outer, count))
    table, stats = zip(*training)
    totals = {sp: np.asarray([value[sp] for value in stats]).sum(0) for sp in SPECIES}
    priors = {sp == 'hyena': float(totals[sp][0] / totals[sp][1]) for sp in SPECIES}
    output = {}
    for name in variants:
        model = m.train(table, *VARIANTS[name])
        output[name] = model, priors
        print('Fitted', outer, name, 'reference priors', priors, flush=True)
    joblib.dump(output, OUT / f'model_{outer}.joblib', compress=2)
    return output


def manifest(outer):
    if outer == '2052':
        file = ROOT / 'Models/Abhay/Model12/local_runs/large_2052/validation_episode_splits.csv'
    else:
        file = ROOT / 'Models/Abhay/Model15/results/additional_episodes/validation_manifest.csv'
    return pd.read_csv(file)


def reciprocal(pred, truth):
    return ((pred == truth[:, None]) / np.arange(1, 6)).sum(1)


def summary(rows, key):
    subset = dict(overall=rows, known=rows.loc[~rows.novel], novel=rows.loc[rows.novel])
    subset.update({str(index): part for index, part in rows.groupby('episode')})
    subset.update({str(index): part for index, part in rows.groupby('species')})
    results = {name: dict(n=len(part), map5=float(part[key].mean()), top1=float(part[key].eq(1).mean()))
        for name, part in subset.items()}
    results['known_candidate_recall'] = float(rows.loc[~rows.novel, 'found'].mean())
    return results


def evaluate(data, outer, models, methods):
    results = []
    train_pool = roles(data, outer)
    authorized_fit = set(data['identity'][train_pool])
    for episode, split in manifest(outer).groupby('episode'):
        gal, query = split.loc[split.assignment.eq('gallery')], split.loc[~split.assignment.eq('gallery')]
        g, q = data['index'].get_indexer(gal.id), data['index'].get_indexer(query.id)
        y = query.truth.to_numpy(str)
        assert (q >= 0).all() and (g >= 0).all() and not set(g) & set(q)
        assert not authorized_fit.intersection(data['identity'][q])
        assert not (set(data['identity'][g]) & set(data['identity'][q[y == m.NEW]]))
        frame = m.make_frame(g, q, data, data['good'], data['inliers'], y)
        found = frame.groupby('row', sort=True).hit.any().to_numpy()
        out = pd.DataFrame(dict(episode=episode, id=query.id, truth=y, species=data['species'][q],
            animal=data['identity'][q], novel=y == m.NEW, found=found))
        raw = {name: m.logits(models[name][0], frame) for name, _ in methods}
        for variant, strength in methods:
            scores, _ = m.adjust_prior(raw[variant], frame, models[variant][1], strength)
            pred = m.top5(scores, frame)
            name = f'{variant}_{strength:g}'
            out[name] = reciprocal(pred, y)
            if not found.all():
                assert (out.loc[~found, name] == 0).all()
            out[name + '_prediction'] = [' '.join(seq) for seq in pred]
        results.append(out)
        print('Compared episode', episode, len(q), 'queries and', len(g), 'references', flush=True)
    results = pd.concat(results, ignore_index=True)
    return results, {f'{key}_{s:g}': summary(results, f'{key}_{s:g}') for key, s in methods}


def develop(data, count):
    if (OUT / 'recipe.json').exists():
        raise FileExistsError('The competition recipe is already frozen')
    models = fit(data, '2052', count, VARIANTS)
    methods = [(name, strength) for name in VARIANTS for strength in (0., .5, 1.)]
    _, metrics = evaluate(data, '2052', models, methods)
    # Choose tuning strength with the full method. Ablations diagnose the method.
    best = max((s for s in (0., .5, 1.)), key=lambda s: metrics[f'full_{s:g}']['overall']['map5'])
    write_json('development_metrics.json', metrics)
    write_json('recipe.json', dict(variant='full', prior_strength=best, training_episodes=count,
        source_sha256=sha(HERE / 'reconstruction.py'), geometry_sha256=data['cache_sha256'],
        selection='Highest development MAP@5 among three predeclared prior adjustments; no leaderboard tuning.'))
    print('Development MAP@5', {k: round(v['overall']['map5'], 6) for k, v in metrics.items()}, 'locked strength', best, flush=True)


def recipe(data):
    settings = json.loads((OUT / 'recipe.json').read_text())
    assert settings['source_sha256'] == sha(HERE / 'reconstruction.py')
    assert settings['geometry_sha256'] == data['cache_sha256']
    return settings


def confirm(data):
    settings = recipe(data)
    models = load_fitted(data, '27613', settings)
    strength = settings['prior_strength']
    methods = list(dict.fromkeys([('full', strength), ('full', 0.)]))
    predictions, metrics = evaluate(data, '27613', models, methods)
    predictions.to_csv(OUT / 'confirmation_predictions.csv', index=False)
    write_json('confirmation_metrics.json', metrics)
    print('Confirmation metrics', json.dumps(metrics, indent=2), flush=True)


def predict(data, raw_dir):
    settings = recipe(data)
    name, strength = settings['variant'], settings['prior_strength']
    models = load_fitted(data, 'production', settings)
    n = len(data['train'])
    q = np.arange(n, n + len(data['test']))
    frame = m.make_frame(np.arange(n), q, data, data['good'], data['inliers'])
    raw, prior = models[name]
    raw = m.logits(raw, frame)
    adjusted, stats = m.adjust_prior(raw, frame, prior, strength)
    predicted = m.top5(adjusted, frame)
    before = m.top5(raw, frame)
    checked = pd.DataFrame({'id': data['ids'][q], 'predictions': [' '.join(row) for row in predicted]})
    sample = pd.read_csv(raw_dir / 'sample_submission.csv', dtype=str)
    checked = sample[['id']].merge(checked, how='left', on='id', validate='one_to_one')
    temporary = OUT / 'submission.tmp.csv'
    checked.to_csv(temporary, index=False, lineterminator='\n')
    actual = pd.read_csv(temporary, dtype=str, keep_default_na=False)
    assert actual.columns.tolist() == ['id', 'predictions'] and len(actual) == len(q) == 2791
    assert actual.id.is_unique and actual.id.equals(sample.id) and set(actual.id) == set(data['ids'][q])
    known = data['train'].groupby('species').identity.apply(set)
    species = data['test'].set_index('id').species
    for row in actual.itertuples(index=False):
        value = row.predictions.split(' ')
        assert len(value) == len(set(value)) == 5 and set(value) <= known[species[row.id]] | {m.NEW}
    output = OUT / 'submission.csv'
    temporary.replace(output)
    context = frame.loc[frame.is_new]
    weak = context.sift_peak < 7
    cohort_support = (context.cohort_geometry >= 7) & (context.cohort_geometry > context.sift_peak)
    result = dict(rows=len(q), valid=True, sha256=sha(output), prior_adjustment=stats,
        top1_changes_from_prior=int((before[:, 0] != predicted[:, 0]).sum()),
        new_first=int((predicted[:, 0] == m.NEW).sum()),
        test_weak_gallery=int(weak.sum()), weak_gallery_with_stronger_query_geometry=int((weak & cohort_support).sum()),
        by_species={sp: {"queries": int((data['species'][q] == sp).sum()),
            "new_first": int(((data['species'][q] == sp) & (predicted[:, 0] == m.NEW)).sum())} for sp in SPECIES})
    write_json('submission_checks.json', result)
    print('Submission checks', json.dumps(result, indent=2), flush=True)


def profile(data):
    n = len(data['train'])
    q = np.arange(n, n + len(data['test']))
    frame = m.make_frame(np.arange(n), q, data, data['good'], data['inliers'])
    reports = {}
    for sp in SPECIES:
        focus = frame.loc[frame.species_hyena.eq(sp == 'hyena')]
        context = focus.loc[focus.is_new]
        known = focus.loc[~focus.is_new]
        baseline = known.loc[known.groupby('row', sort=True).raw.idxmax()]
        hull = known.loc[known.groupby('row', sort=True).segment.idxmax()]
        peak = known.loc[known.groupby('row', sort=True).projected.idxmax()]
        strong_gallery = context.sift_peak.ge(7)
        strong_query = context.cohort_geometry.ge(7)
        reports[sp] = dict(queries=len(context), references=int(data['train'].species.eq(sp).sum()),
            no_gallery_sift7=int((~strong_gallery).sum()),
            no_gallery_sift7_with_query_sift7=int((~strong_gallery & strong_query).sum()),
            gallery_sift7_with_stronger_query=int((strong_gallery & context.cohort_geometry.gt(context.sift_peak)).sum()),
            median_best_gallery_cosine=float(context.raw_peak.median()),
            median_best_query_cosine=float(context.cohort_cos.median()),
            hull_leader_disagrees_with_raw=int((hull.name.to_numpy() != baseline.name.to_numpy()).sum()),
            hull_leader_disagrees_with_single_projected=int((hull.name.to_numpy() != peak.name.to_numpy()).sum()),
            no_query_viewpoint=int(context.flank.lt(0).sum()))
    write_json('test_profile.json', reports)
    print(json.dumps(reports, indent=2), flush=True)


def fit_for_later(data, outer, count):
    fit(data, outer, count, ['full'])
    write_json(f'checkpoint_{outer}.json', dict(episodes=count, geometry=data['cache_sha256'],
        source=sha(HERE / 'reconstruction.py'), routine=sha(__file__),
        checkpoint=sha(OUT / f'model_{outer}.joblib')))


def load_fitted(data, outer, settings):
    path = OUT / f'checkpoint_{outer}.json'
    checkpoint = OUT / f'model_{outer}.joblib'
    if path.exists() and checkpoint.exists():
        saved = json.loads(path.read_text())
        expected = dict(episodes=settings['training_episodes'], geometry=data['cache_sha256'],
            source=sha(HERE / 'reconstruction.py'), routine=sha(__file__), checkpoint=sha(checkpoint))
        if saved == expected:
            return joblib.load(checkpoint)
    return fit(data, outer, settings['training_episodes'], [settings['variant']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['profile', 'develop', 'confirm', 'predict', 'fit-audit', 'fit-production'])
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--team', type=Path, default=SOURCE)
    parser.add_argument('--episodes', type=int, default=16)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=4):
        data = load(args.data, args.team)
        if args.mode == 'profile':
            profile(data)
        elif args.mode == 'develop':
            develop(data, args.episodes)
        elif args.mode == 'confirm':
            confirm(data)
        elif args.mode in ('fit-audit', 'fit-production'):
            fit_for_later(data, '27613' if args.mode == 'fit-audit' else 'production', args.episodes)
        else:
            predict(data, args.data)


if __name__ == '__main__':
    main()
