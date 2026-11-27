"""Independent audit of frozen Model13 comparisons; imports no experiment code.

Run ``python Models/Abhay/Model13/verify.py --seed 27613`` after scoring.
This validates persisted provenance, manifests, scores, rankings and metrics.
It does not infer temporal independence merely from timestamps or hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tmp/abhay_deps'))
import joblib
import numpy as np
import pandas as pd

NEW = 'new_individual'
METHODS = ('baseline', 'candidate')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def close(actual, expected, message):
    require(np.allclose(actual, expected, rtol=0, atol=1e-12,
                        equal_nan=False), message)


def reciprocal(predictions, truth):
    values = []
    for row, target in zip(predictions, truth):
        found = np.flatnonzero(np.asarray(row) == target)
        values.append(1. / (int(found[0]) + 1) if len(found) else 0.)
    return np.asarray(values)


def score_probabilities(result, all_candidate=False):
    pc = np.asarray(result['pc'])
    if all_candidate:
        raw = np.asarray(result['raw'])
        pc = raw / np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
    pb = np.asarray(result['pb'])
    return np.column_stack([(1 - pb[:, None]) * pc, pb])


def check_enrichment(original, enriched):
    """PLDA may add numeric evidence, never alter baseline rows or columns."""
    require(set(original) <= set(enriched), 'Enrichment removed baseline frame keys')
    for key in ['order', 'truth']:
        require(np.array_equal(original[key], enriched[key]), 'Enrichment changed ' + key)
    schema = {}
    for key in ['q', 'c']:
        base, extra = original[key], enriched[key]
        require(len(base) == len(extra) and set(base.columns) <= set(extra.columns),
                'Enrichment changed baseline dimensions/columns')
        pd.testing.assert_frame_equal(base, extra.loc[:, base.columns], check_exact=True)
        added = [name for name in extra.columns if name not in base.columns]
        require(bool(added) and all(name.startswith('pl_') for name in added),
                'Expected only additional pl_* evidence columns')
        for name in added:
            require(pd.api.types.is_numeric_dtype(extra[name]), 'Enrichment added nonnumeric data: ' + name)
            require(not np.isinf(extra[name].to_numpy()).any(), 'Enrichment contains infinity: ' + name)
        schema[key] = tuple(added)
    return schema


def reconstructed_candidate(model, recipe, frame, baseline, baseline_result):
    """Reconstruct frozen candidate inference without importing runner helpers."""
    kind = recipe['kind']
    if kind == 'all_candidate':
        return score_probabilities(baseline_result, True)
    q, c = frame['q'], frame['c']
    n = len(q)
    forbidden_q = {'unk', 'truth', 'target', 'identity', 'id', 'label', 'order', 'y'}
    forbidden_c = forbidden_q | {'q', 'c', 'ident'}
    if kind == 'head':
        detector, closed, anyq, (qf, cf) = model
        require(not set(qf) & forbidden_q and not set(cf) & forbidden_c,
                'Candidate model schema contains target/identifier metadata')
        require(not any(re.match(r'^(nb|nw|sb|sw)\d+$', col) for col in qf),
                'Candidate model schema contains neighbour pointers')
        cx = pd.concat([c[list(cf)].reset_index(drop=True),
                        q.iloc[c.q.to_numpy()][list(qf)].reset_index(drop=True)], axis=1)
        pn = detector.predict_proba(q[list(qf)])[:, 1]
        raw = anyq.predict_proba(cx)[:, 1].reshape(n, 15)
        within = closed.predict_proba(cx)[:, 1].reshape(n, 15)
        within /= np.maximum(within.sum(axis=1, keepdims=True), 1e-9)
        pb = (pn + np.clip(1 - raw.sum(axis=1), 0, 1)) / 2
        own = score_probabilities(dict(pc=within, raw=raw, pb=pb), recipe.get('all_candidate', False))
    elif kind == 'joint':
        qf, cf = list(model.query_features), list(model.candidate_features)
        require(model.candidates == 15, 'Joint model shortlist size changed')
        require(not set(qf) & forbidden_q and not set(cf) & forbidden_c,
                'Joint model schema contains target/identifier metadata')
        require(not any(re.match(r'^(nb|nw|sb|sw)\d+$', col) for col in qf),
                'Joint model schema contains neighbour pointers')
        ordered = c.iloc[np.argsort(c.q.to_numpy(), kind='stable')]
        matrix = np.full((n, 16, len(cf) + len(qf) + 1), np.nan, dtype=np.float32)
        matrix[:, :15, :len(cf)] = ordered[cf].to_numpy(dtype=np.float32).reshape(n, 15, len(cf))
        matrix[:, :, len(cf):-1] = q[qf].to_numpy(dtype=np.float32)[:, None, :]
        matrix[:, :, -1] = 0
        matrix[:, -1, -1] = 1
        columns = ['c__' + name for name in cf] + ['q__' + name for name in qf] + ['is_new']
        raw = model.estimator.predict(pd.DataFrame(matrix.reshape(n * 16, -1), columns=columns)).reshape(n, 16)
        if recipe['blend'] == 1:
            return raw
        shifted = raw - raw.max(axis=1, keepdims=True)
        own = np.exp(shifted)
        own /= own.sum(axis=1, keepdims=True)
    else:
        raise AssertionError('Unrecognized frozen recipe kind: ' + str(kind))
    return (1 - recipe['blend']) * baseline + recipe['blend'] * own


def check_episode(part, meta, allowed_queries, context=None, backgrounds=None):
    require(set(part.assignment) == {'gallery', 'query'}, 'Unexpected episode assignments')
    g = part.loc[part.assignment == 'gallery', 'id'].tolist()
    qr = part[part.assignment == 'query'].set_index('id')
    q = qr.index.tolist()
    require(len(g) == len(set(g)) and len(q) == len(set(q)), 'Repeated episode photos')
    require(not set(g) & set(q), 'Query/gallery photo leakage')
    require(set(g + q) <= set(meta.index), 'Non-training photo in an evaluation/training episode')
    require(set(q) <= allowed_queries, 'Query outside its assigned identity role')
    if context is not None:
        require(set(g + q) <= context, 'Episode uses an excluded fitting context')
    if backgrounds is not None:
        require(backgrounds <= set(g), 'Expanded gallery omits a required background photo')
    gi = set(meta.loc[g, 'identity'])
    actual = meta.loc[q, 'identity'].to_numpy()
    truth = qr.truth.to_numpy()
    is_new = truth == NEW
    require(np.all(is_new | (truth == actual)), 'Known truth disagrees with catalogue identity')
    require(not set(actual[is_new]) & gi, 'Unknown query identity is present in gallery')
    require(set(actual[~is_new]) <= gi, 'Known query identity absent from gallery')
    require(is_new.any() and (~is_new).any(), 'Episode needs known and unknown queries')
    return g, qr


def metric_masks(df):
    unknown = df.truth.eq(NEW)
    masks = {
        'overall': np.ones(len(df), bool), 'known': ~unknown, 'unknown': unknown,
        'hyena': df.species.eq('hyena'), 'leopard': df.species.eq('leopard'),
        'known_hyena': ~unknown & df.species.eq('hyena'),
        'known_leopard': ~unknown & df.species.eq('leopard'),
        'small_gallery': ~unknown & df.true_gallery_images.between(1, 3),
    }
    masks.update({str(ep): df.episode.eq(ep) for ep in sorted(df.episode.unique())})
    return masks


def audit(out, seed, data):
    required = ['training_contexts.csv', 'identity_roles.csv', 'training_episode_splits.csv',
                'validation_episode_splits.csv', 'validation_predictions.csv',
                'validation_scores.joblib', 'metrics.json', 'evaluation_preparation.json',
                'training_complete.json', 'baseline_models.joblib']
    for name in required:
        require((out / name).is_file(), 'Missing audit input: ' + name)
    lock_path = HERE / 'recipe_lock.json'
    lock = json.loads(lock_path.read_text())
    recipe = lock['recipe']
    feature_set = recipe.get('feature_set')
    require(feature_set in {None, 'baseline', 'plda_v1'}, 'Unrecognized frozen feature set')
    digest = sha(lock_path)
    metrics = json.loads((out / 'metrics.json').read_text())
    prepared = json.loads((out / 'evaluation_preparation.json').read_text())
    trained = json.loads((out / 'training_complete.json').read_text())
    require(seed in lock['outer_seeds'], 'Seed absent from locked recipe')
    require(metrics['seed'] == prepared['seed'] == trained['seed'] == seed, 'Seed provenance mismatch')
    require(metrics['recipe_lock_sha256'] == prepared['recipe_lock_sha256'] == digest,
            'Recipe changed after evaluation preparation/scoring')
    require(metrics['baseline_model_sha256'] == trained['baseline_sha256'] == sha(out / 'baseline_models.joblib'),
            'Baseline checkpoint hash mismatch')
    require(trained['stages'] == 3 and trained['episodes_per_stage'] == 12,
            'Unexpected baseline fitting budget')
    require(trained['source_sha256'] == sha(HERE / 'experiment.py'),
            'Training source differs from current experiment.py')
    source_hashes = lock.get('source_sha256', lock.get('source_hashes', {}))
    require(isinstance(source_hashes, dict), 'Invalid source hash mapping')
    for name, expected in source_hashes.items():
        path = Path(name)
        if not path.is_absolute():
            path = HERE / path if (HERE / path).exists() else ROOT / path
        require(path.is_file() and sha(path) == expected, 'Locked source changed: ' + name)

    train = pd.read_csv(data / 'train.csv', dtype={'id': str, 'identity': str})
    require(train.id.is_unique, 'Duplicate IDs in source catalogue')
    meta = train.set_index('id')
    roles = pd.read_csv(out / 'identity_roles.csv')
    require(roles.id.is_unique and set(roles.id) == set(meta.index), 'Role rows do not partition catalogue')
    require(set(roles.role) == {'stage1', 'stage2', 'stage3', 'calibration', 'validation'}, 'Unknown roles')
    require(roles.groupby('identity').role.nunique().eq(1).all(), 'Identity crosses roles')
    for name in ['identity', 'species']:
        require(np.array_equal(roles[name], meta.loc[roles.id, name]), 'Role metadata mismatch: ' + name)
    groups = {name: set(part.id) for name, part in roles.groupby('role')}
    contexts = pd.read_csv(out / 'training_contexts.csv')
    require(not contexts.duplicated(['stage', 'id']).any(), 'Duplicate fitting context rows')
    require(set(contexts.stage) == {1, 2, 3}, 'Incorrect stages in contexts')
    training = pd.read_csv(out / 'training_episode_splits.csv', keep_default_na=False)
    require(set(training.stage) == {1, 2, 3}, 'Incorrect stages in training episodes')
    expected_train = {seed + 1000 * s + j for s in [1, 2, 3] for j in range(12)}
    require(set(training.episode) == expected_train, 'Training episode set differs from protocol')
    training_queries, training_galleries = {}, {}
    for stage in [1, 2, 3]:
        bg = set().union(groups['calibration'], *(groups['stage' + str(s)] for s in range(1, stage)))
        expected_context = groups['stage' + str(stage)] | bg
        observed_context = set(contexts.loc[contexts.stage == stage, 'id'])
        require(observed_context == expected_context, 'Fitting context differs at stage ' + str(stage))
        forbidden = set().union(groups['validation'], *(groups['stage' + str(s)] for s in range(stage + 1, 4)))
        require(not observed_context & forbidden, 'Heldout/future-stage exposure')
        for ep, part in training[training.stage == stage].groupby('episode'):
            require(seed + stage * 1000 <= ep < seed + stage * 1000 + 12, 'Episode assigned to wrong stage')
            g, qr = check_episode(part, meta, groups['stage' + str(stage)], observed_context, bg)
            training_queries[int(ep)] = qr
            training_galleries[int(ep)] = g
    require(len(training_queries) == 36, 'Wrong training episode count')

    # Inspect persisted supervised frames, one at a time. This catches stale
    # frame resumes whose photo/truth manifests differ from the recorded split.
    frame_checks = 0
    enriched_checks, enriched_schema = 0, None
    stage3_truth, stage3_ids, stage3_present = [], [], []
    train_ids = train.id.to_numpy()
    for ep, qr in training_queries.items():
        frame_path = out / f'train_{ep}.joblib'
        require(frame_path.is_file(), 'Missing training frame ' + str(ep))
        frame = joblib.load(frame_path)
        order = np.asarray(frame['order'])
        require(np.issubdtype(order.dtype, np.integer) and (order >= 0).all() and (order < len(train)).all(),
                'Training frame indexes competition test rows')
        ids = train_ids[order]
        require(len(ids) == len(qr) and len(set(ids)) == len(ids) and set(ids) == set(qr.index),
                'Training frame/manifest query mismatch')
        truth = qr.loc[ids, 'truth'].to_numpy()
        require(np.array_equal(frame['truth'], truth), 'Training frame truth mismatch')
        require(np.array_equal(frame['q'].unk, truth == NEW), 'Training query targets mismatch')
        c = frame['c']
        ci = c.q.to_numpy()
        require(np.issubdtype(ci.dtype, np.integer) and (ci >= 0).all() and (ci < len(ids)).all(),
                'Training candidate pointers invalid')
        require(np.all(np.bincount(ci, minlength=len(ids)) == 15), 'Training shortlist size mismatch')
        require(np.array_equal(c.y, c.ident.to_numpy() == truth[ci]), 'Training candidate targets mismatch')
        allowed = meta.loc[training_galleries[ep]].groupby('species').identity.agg(set).to_dict()
        for row, species in zip(c.sort_values('q', kind='stable').ident.to_numpy().reshape(-1, 15),
                                meta.loc[ids, 'species']):
            require(len(set(row)) == 15 and set(row) <= allowed[species],
                    'Training candidate absent from same-species gallery or duplicated')
        if seed + 3000 <= ep < seed + 3012:
            if feature_set == 'plda_v1':
                richer_path = out / f'plda_train_{ep}.joblib'
                require(richer_path.is_file(), 'Missing enriched fitting frame ' + str(ep))
                richer = joblib.load(richer_path)
                schema = check_enrichment(frame, richer)
                if enriched_schema is None:
                    enriched_schema = schema
                require(schema == enriched_schema, 'Enriched fitting feature schema changed between episodes')
                enriched_checks += 1
                del richer
            positive = np.bincount(ci, weights=c.y.to_numpy(), minlength=len(ids))
            require((positive <= 1).all(), 'Multiple true candidates in training query')
            stage3_truth.extend(truth.tolist())
            stage3_ids.extend(ids.tolist())
            stage3_present.extend(((positive == 1) | (truth == NEW)).tolist())
        frame_checks += 1
        del frame

    candidate_model = None
    candidate_path = out / 'candidate.joblib'
    if recipe['kind'] in {'head', 'joint'}:
        require(candidate_path.is_file(), 'Missing selected candidate checkpoint')
        candidate_model = joblib.load(candidate_path)
        required.append('candidate.joblib')
        if 'candidate_model_sha256' in metrics:
            require(metrics['candidate_model_sha256'] == sha(candidate_path), 'Candidate checkpoint hash mismatch')
        if recipe['kind'] == 'head':
            if feature_set == 'plda_v1':
                qf, cf = candidate_model[3]
                require(set(enriched_schema['q']) <= set(qf) and set(enriched_schema['c']) <= set(cf),
                        'Candidate omitted declared PLDA feature columns')
            for estimator in candidate_model[:3]:
                stats = estimator.photo_weighting_metadata_
                require(stats['training_appearances'] == len(stage3_ids), 'Head fitting appearance count mismatch')
                require(stats['unique_photos'] == len(set(stage3_ids)), 'Head fitting photo count mismatch')
                require(stats['unknown_queries'] == stage3_truth.count(NEW), 'Head fitting unknown count mismatch')
                require(stats['weight_by_photo'] == recipe['weight_by_photo'], 'Head weighting differs from recipe')
            common = {k: v for k, v in recipe['params'].items()
                      if k not in {'detector_params', 'candidate_params'}}
            for index, estimator in enumerate(candidate_model[:3]):
                expected = {**common, **recipe['params'].get('detector_params' if index == 0 else 'candidate_params', {})}
                actual = estimator.get_params()
                require(all(actual.get(k) == v for k, v in expected.items()), 'Head parameters differ from locked recipe')
        else:
            stats = candidate_model.metadata
            present = np.asarray(stage3_present)
            require(stats['training_query_appearances'] == len(stage3_ids), 'Ranker fitting appearance count mismatch')
            require(stats['used_groups'] == int(present.sum()), 'Ranker group count mismatch')
            require(stats['omitted_missing_true_candidate'] == int((~present).sum()), 'Ranker omission count mismatch')
            require(stats['unique_training_photos'] == len(set(np.asarray(stage3_ids)[present])), 'Ranker photo count mismatch')
            require(stats['weight_by_photo'] == recipe['weight_by_photo'], 'Ranker weighting differs from recipe')
            require(all(candidate_model.estimator.get_params().get(k) == v for k, v in recipe['params'].items()),
                    'Ranker parameters differ from locked recipe')

    validation = pd.read_csv(out / 'validation_episode_splits.csv', keep_default_na=False)
    expected_eval = {seed + 20000 + j for j in range(3)}
    if 'evaluation_seeds' in lock:
        locked = lock['evaluation_seeds']
        if isinstance(locked, dict):
            locked = locked[str(seed)]
        require(set(locked) == expected_eval, 'Evaluation seeds differ from recipe lock')
    if 'episodes_by_seed' in lock:
        require(set(lock['episodes_by_seed'][str(seed)]) == expected_eval, 'Episode mapping differs from lock')
    require(set(validation.episode) == expected_eval, 'Unplanned/missing evaluation episodes')
    require(set(validation.stage) == {'validation'}, 'Evaluation manifest role invalid')
    val_episodes = {}
    background = set(meta.index) - groups['validation']
    for ep, part in validation.groupby('episode'):
        val_episodes[int(ep)] = check_episode(part, meta, groups['validation'], backgrounds=background)

    df = pd.read_csv(out / 'validation_predictions.csv')
    require(not df.duplicated(['episode', 'id']).any(), 'Repeated scored query within episode')
    expected_keys = {(int(ep), iid) for ep, (_, qr) in val_episodes.items() for iid in qr.index}
    actual_keys = set(zip(df.episode, df.id))
    require(actual_keys == expected_keys, 'Scored queries differ from validation manifest')
    require(np.array_equal(df.identity, meta.loc[df.id, 'identity']), 'Prediction identity metadata mismatch')
    require(np.array_equal(df.species, meta.loc[df.id, 'species']), 'Prediction species metadata mismatch')
    require(np.array_equal(df.unknown, df.truth.eq(NEW)), 'Prediction known/unknown flag mismatch')
    score_bundles = joblib.load(out / 'validation_scores.joblib')
    require(len(score_bundles) == 3 and {int(z['episode']) for z in score_bundles} == expected_eval,
            'Score bundles do not match evaluation episodes')
    tie_queries = 0
    tie_changed_top5 = 0
    indexed = df.set_index(['episode', 'id'])
    for bundle in score_bundles:
        ep = int(bundle['episode'])
        ids = np.asarray(bundle['id'])
        lab = np.asarray(bundle['labels'])
        g, qr = val_episodes[ep]
        require(len(ids) == len(qr) and len(set(ids)) == len(ids) and set(ids) == set(qr.index),
                'Score bundle query mismatch')
        require(lab.shape == (len(ids), 16), 'Expected 15 candidates plus NEW')
        require((lab[:, -1] == NEW).all(), 'NEW missing from the final candidate slot')
        require(all(len(set(row)) == 16 for row in lab), 'Duplicate candidate labels')
        z = indexed.loc[[(ep, iid) for iid in ids]]
        truth = qr.loc[ids, 'truth'].to_numpy()
        require(np.array_equal(z.truth, truth), 'Prediction truth differs from manifest')
        allowed = meta.loc[g].groupby('species').identity.agg(set).to_dict()
        for row, species in zip(lab, z.species):
            require(set(row[:-1]) <= allowed[species], 'Candidate absent from same-species gallery')
        counts = meta.loc[g, 'identity'].value_counts()
        expected_counts = [0 if target == NEW else int(counts[target]) for target in truth]
        require(np.array_equal(z.true_gallery_images, expected_counts), 'Gallery size slice metadata wrong')
        require(set(bundle['scores']) == set(METHODS), 'Scored competitor set differs from frozen comparison')
        for method in METHODS:
            score = np.asarray(bundle['scores'][method])
            require(score.shape == lab.shape and np.isfinite(score).all(), 'Invalid scores: ' + method)
            pred = np.take_along_axis(lab, np.argsort(-score, axis=1, kind='stable')[:, :5], axis=1)
            require([' '.join(row) for row in pred] == z[method + '_predictions'].tolist(),
                    'Rank reconstruction failed: ' + method)
            close(z[method + '_rr5'], reciprocal(pred, truth), 'Reciprocal rank mismatch: ' + method)
        baseline = np.asarray(bundle['scores']['baseline'])
        close(baseline.sum(1), 1., 'Baseline score mass does not sum to one')
        tie_queries += int(np.any(np.diff(np.sort(baseline, axis=1), axis=1) == 0, axis=1).sum())
        stable = np.take_along_axis(lab, np.argsort(-baseline, axis=1, kind='stable')[:, :5], axis=1)
        legacy = np.take_along_axis(lab, np.argsort(-baseline, axis=1)[:, :5], axis=1)
        tie_changed_top5 += int(np.any(stable != legacy, axis=1).sum())
        # Both competitors must use the same frozen frame. Recompute the
        # baseline score formula directly from that saved evaluation bundle.
        prepared_bundle = joblib.load(out / f'eval_{ep}.joblib')
        f, r = prepared_bundle['frame'], prepared_bundle['result']
        require(np.array_equal(train_ids[np.asarray(f['order'])], ids), 'Evaluation frame row order changed')
        prepared_labels = np.column_stack([f['c'].ident.to_numpy().reshape(-1, 15), np.full(len(ids), NEW)])
        require(np.array_equal(lab, prepared_labels), 'Competitors do not share the prepared shortlist')
        reconstructed = np.column_stack([(1 - r['pb'][:, None]) * r['pc'], r['pb']])
        close(baseline, reconstructed, 'Baseline score differs from frozen Model12 formula')
        require(set(train_ids[np.asarray(prepared_bundle['gallery'])]) == set(g), 'Prepared gallery differs from manifest')
        candidate_frame = f
        if feature_set == 'plda_v1':
            enriched_path = out / f'plda_eval_{ep}.joblib'
            require(enriched_path.is_file(), 'Missing enriched evaluation frame')
            enriched_bundle = joblib.load(enriched_path)
            require(enriched_bundle['episode'] == ep, 'Enrichment changed episode ID')
            require(np.array_equal(enriched_bundle['gallery'], prepared_bundle['gallery']), 'Enrichment changed gallery')
            for result_key, value in r.items():
                require(result_key in enriched_bundle['result'], 'Enrichment removed baseline result')
                close(enriched_bundle['result'][result_key], value, 'Enrichment changed baseline result')
            candidate_frame = enriched_bundle['frame']
            require(check_enrichment(f, candidate_frame) == enriched_schema, 'Evaluation enrichment schema mismatch')
            enriched_checks += 1
        candidate_reconstruction = reconstructed_candidate(candidate_model, recipe, candidate_frame, baseline, r)
        close(bundle['scores']['candidate'], candidate_reconstruction, 'Candidate score differs from locked checkpoint/recipe')

    require(tie_changed_top5 == 0, 'Stable sorting changes the legacy baseline top-five; disclose/resolve before parity claim')

    require(metrics['appearances'] == len(df), 'Appearance count mismatch')
    require(metrics['unique_photos'] == df.id.nunique(), 'Unique photo count mismatch')
    require(metrics['identities'] == df.identity.nunique(), 'Identity count mismatch')
    delta = df.candidate_rr5 - df.baseline_rr5
    require(metrics['improved'] == int((delta > 0).sum()) and metrics['worsened'] == int((delta < 0).sum()),
            'Improved/worsened counts mismatch')
    slices = metric_masks(df)
    require(set(metrics['slices']) == set(slices), 'Unexpected/missing metric slices')
    for name, mask in slices.items():
        z = df[mask]
        reported = metrics['slices'][name]
        require(reported['n'] == len(z), 'Slice count mismatch: ' + name)
        for method, key in [('baseline', 'baseline'), ('candidate', 'candidate')]:
            close(reported[key], z[method + '_rr5'].mean(), 'Slice MAP mismatch: ' + name)
            close(reported[key + '_top1'], z[method + '_rr5'].eq(1).mean(), 'Slice top1 mismatch: ' + name)
        close(reported['delta'], (z.candidate_rr5 - z.baseline_rr5).mean(), 'Slice delta mismatch: ' + name)
    rng = np.random.default_rng(130726)
    numerator, denominator = np.zeros(5000), np.zeros(5000)
    for species in sorted(df.species.unique()):
        values = []
        for identity in sorted(df.loc[df.species == species, 'identity'].unique()):
            d = delta[(df.species == species) & (df.identity == identity)]
            values.append((float(d.sum()), len(d)))
        values = np.asarray(values)
        sampled = values[rng.integers(len(values), size=(5000, len(values)))]
        numerator += sampled[:, :, 0].sum(1)
        denominator += sampled[:, :, 1].sum(1)
    interval = np.quantile(numerator / denominator, [.025, .975])
    close(metrics['delta_ci95'], interval, 'Identity-cluster bootstrap interval mismatch')
    return {
        'all_checks_passed': True, 'seed': seed, 'training_episodes_checked': 36,
        'training_frames_checked': frame_checks, 'validation_episodes_checked': sorted(expected_eval),
        'heldout_and_future_stage_exclusion': True, 'query_gallery_and_unknown_identity_separation': True,
        'same_query_gallery_shortlists_for_competitors': True, 'rankings_and_metrics_reconstructed': True,
        'candidate_scores_reconstructed_from_checkpoint': True,
        'feature_set': feature_set or 'baseline', 'enriched_frames_checked': enriched_checks,
        'enriched_feature_schema': enriched_schema,
        'recipe_lock_sha256': digest, 'locked_sources_checked': list(source_hashes),
        'appearances': len(df), 'unique_photos': int(df.id.nunique()), 'identities': int(df.identity.nunique()),
        'baseline_map5': float(df.baseline_rr5.mean()), 'candidate_map5': float(df.candidate_rr5.mean()),
        'paired_delta': float(delta.mean()), 'delta_ci95': interval.tolist(),
        'baseline_queries_with_any_score_tie': tie_queries,
        'baseline_top5_changed_by_stable_tie_rule': tie_changed_top5,
        'input_hashes': {name: sha(out / name) for name in required},
        'limitations': [
            'Manifest/score auditing does not independently retrain the models or prove chronological independence.',
            'One outer allocation on a previously studied dataset; episodes reuse identities and photos.',
            'Near-duplicate encounter grouping is approximate and is not independently reconstructed here.',
        ],
    }


def self_test():
    """Small adversarial checks, requiring no project data or fitted models."""
    checks = []
    meta = pd.DataFrame({'id': ['g1', 'q1', 'q2', 'g2', 'future', 'gB'],
                         'identity': ['A', 'A', 'B', 'C', 'F', 'B'],
                         'species': ['hyena'] * 6}).set_index('id')
    episode = pd.DataFrame({'id': ['g1', 'q1', 'q2'],
                            'assignment': ['gallery', 'query', 'query'],
                            'truth': ['', 'A', NEW]})
    g, q = check_episode(episode, meta, {'q1', 'q2'}, {'g1', 'q1', 'q2'}, {'g1'})
    require(g == ['g1'] and q.index.tolist() == ['q1', 'q2'], 'Valid synthetic episode failed')
    checks.append('valid_episode')

    def rejected(name, call):
        try:
            call()
        except AssertionError:
            checks.append(name)
        else:
            raise AssertionError('Failed to reject synthetic corruption: ' + name)

    bad = pd.concat([episode, pd.DataFrame([{'id': 'q2', 'assignment': 'gallery', 'truth': ''}])], ignore_index=True)
    rejected('photo_overlap_rejected', lambda: check_episode(bad, meta, {'q1', 'q2'}))
    bad = pd.concat([episode, pd.DataFrame([{'id': 'gB', 'assignment': 'gallery', 'truth': ''}])], ignore_index=True)
    rejected('unknown_gallery_identity_rejected', lambda: check_episode(bad, meta, {'q1', 'q2'}))
    bad = episode.copy(); bad.loc[0, 'id'] = 'g2'
    rejected('missing_known_identity_rejected', lambda: check_episode(bad, meta, {'q1', 'q2'}))
    bad = episode.copy(); bad.loc[2, 'id'] = 'future'
    rejected('future_query_rejected', lambda: check_episode(bad, meta, {'q1', 'q2'}))
    bad = episode.copy(); bad.loc[1, 'truth'] = 'C'
    rejected('incorrect_target_rejected', lambda: check_episode(bad, meta, {'q1', 'q2'}))
    rejected('background_omission_rejected', lambda: check_episode(episode, meta, {'q1', 'q2'}, backgrounds={'g2'}))
    rejected('future_context_rejected', lambda: check_episode(episode, meta, {'q1', 'q2'}, context={'g1', 'q1'}))

    pred = np.array([['A', 'B', 'C', 'D', NEW], ['C', NEW, 'A', 'B', 'D'],
                     ['A', 'B', 'C', 'D', NEW], ['A', 'B', 'C', 'D', NEW]])
    close(reciprocal(pred, ['A', NEW, NEW, 'absent']), [1., .5, .2, 0.], 'Reciprocal rank check failed')
    checks.append('top_five_reciprocal_rank')

    class FixedHead:
        def __init__(self, probabilities):
            self.probabilities = np.asarray(probabilities)
        def predict_proba(self, matrix):
            require(len(matrix) == len(self.probabilities), 'Synthetic predictor row count mismatch')
            return np.column_stack([1 - self.probabilities, self.probabilities])

    frame = dict(q=pd.DataFrame({'feature': [2., 3.], 'unk': [0, 1]}),
                 c=pd.DataFrame({'q': np.repeat([0, 1], 15), 'feat': np.arange(30),
                                 'ident': np.tile([f'animal_{i}' for i in range(15)], 2), 'y': np.zeros(30)}))
    closed = np.tile(np.linspace(.1, .2, 15), 2)
    model = (FixedHead([.2, .8]), FixedHead(closed), FixedHead(np.full(30, .03)), (['feature'], ['feat']))
    base = np.full((2, 16), 1 / 16)
    own = reconstructed_candidate(model, {'kind': 'head', 'blend': 1., 'all_candidate': False}, frame, base, {})
    expected_pb = np.array([.375, .675])
    expected_pc = closed.reshape(2, 15); expected_pc = expected_pc / expected_pc.sum(axis=1, keepdims=True)
    expected = np.column_stack([(1 - expected_pb[:, None]) * expected_pc, expected_pb])
    close(own, expected, 'Independent head reconstruction failed')
    mix = reconstructed_candidate(model, {'kind': 'head', 'blend': .5, 'all_candidate': False}, frame, base, {})
    close(mix, (base + expected) / 2, 'Head blend reconstruction failed')
    checks.extend(['head_score_reconstruction', 'score_blending'])
    poisoned_model = (*model[:3], (['feature', 'unk'], ['feat']))
    rejected('target_feature_schema_rejected', lambda: reconstructed_candidate(
        poisoned_model, {'kind': 'head', 'blend': 1.}, frame, base, {}))
    import copy
    original = {**frame, 'order': np.arange(2), 'truth': np.array(['animal_0', NEW])}
    enriched = copy.deepcopy(original)
    enriched['q']['pl_density'] = [1., 2.]
    enriched['c']['pl_score'] = np.arange(30, dtype=float)
    require(check_enrichment(original, enriched) == {'q': ('pl_density',), 'c': ('pl_score',)},
            'Valid numeric enrichment rejected')
    checks.append('enrichment_preserves_base_schema')
    bad = copy.deepcopy(enriched); bad['q'].loc[0, 'feature'] = 8.
    rejected('enrichment_base_mutation_rejected', lambda: check_enrichment(original, bad))
    bad = copy.deepcopy(enriched); bad['truth'][0] = 'wrong'
    rejected('enrichment_truth_mutation_rejected', lambda: check_enrichment(original, bad))
    bad = copy.deepcopy(enriched); bad['c']['pl_label'] = 'animal_0'
    rejected('enrichment_nonnumeric_label_rejected', lambda: check_enrichment(original, bad))
    return {'all_checks_passed': True, 'synthetic_checks': checks, 'count': len(checks)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=27613)
    parser.add_argument('--data', type=Path, default=Path.home() / 'Downloads/reid_student_data_main')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    out = HERE / 'local_runs' / f'fresh_{args.seed}'
    try:
        report = audit(out, args.seed, args.data)
    except Exception as exc:
        report = {'all_checks_passed': False, 'seed': args.seed,
                  'error_type': type(exc).__name__, 'error': str(exc)}
        if out.is_dir():
            (out / 'independent_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        raise
    (out / 'independent_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'input_hashes'}, indent=2))


if __name__ == '__main__':
    main()
