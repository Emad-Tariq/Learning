"""Profile the competition queries, validate Model 14, and generate a local CSV."""
import argparse
import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'Model12'))
sys.path.insert(0, str(HERE.parent / 'Model13'))

import experiment as e
from compare import labels, row_frame, summarize
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits
import submission as portable
from graph_model import query_links, score_graph

b, p = e.b, e.p
DEV = e.HERE / 'local_runs/development_2052'
FRESH = e.HERE / 'local_runs/fresh_27613'
OUT = HERE / 'results'
LOCK = OUT / 'recipe.json'
ASSETS = ROOT / 'tmp/model12_submission_notebook/model12_assets'
LINKS = {'near_duplicate': (.90, 30, .97), 'dual_evidence': (.70, 20, .94),
         'strong_geometry': (.50, 35, .93)}
ARMS = {f'{name}_{blend}': dict(link=name, blend=blend, all_candidate_mix=0.)
        for name in LINKS for blend in (.25, .5, .75)}
ARMS['all_candidate'] = dict(link='dual_evidence', blend=0., all_candidate_mix=1.)
ARMS['dual_evidence_mixed'] = dict(link='dual_evidence', blend=.5, all_candidate_mix=.5)


def build_links(order, inliers, name):
    return query_links(b.X[order], b.SP[order], inliers, *LINKS[name])


def cache_matrix(path):
    with np.load(path, allow_pickle=False) as cache:
        a, c = cache['pairs'].T
        values = cache['values'][:, 1]
    return sparse.csr_matrix((np.r_[values, values], (np.r_[a, c], np.r_[c, a])),
                             shape=(len(b.IDS), len(b.IDS)))


def evaluate(bundles, recipes):
    rows = []
    for bundle, inliers in bundles:
        frame, result = bundle['frame'], bundle['result']
        order, truth = frame['order'], frame['truth']
        row = row_frame(bundle)
        baseline = b.rank(frame, result)
        row['baseline_rr5'] = e.rr(baseline, truth)
        graphs = {name: build_links(order, inliers, name)
                  for name in {recipe['link'] for recipe in recipes.values()}}
        for name, recipe in recipes.items():
            graph = graphs[recipe['link']]
            pred = score_graph(labels(frame), result, graph, recipe['blend'], recipe['all_candidate_mix'])
            row[f'{name}_rr5'] = e.rr(pred, truth)
            row[f'{name}_changed'] = (pred != baseline).any(axis=1)
            row[f'{name}_linked'] = graph.getnnz(axis=1) > 0
            if name == 'candidate':
                row['baseline_predictions'] = [' '.join(r) for r in baseline]
                row['candidate_predictions'] = [' '.join(r) for r in pred]
        rows.append(row)
    predictions = pd.concat(rows, ignore_index=True)
    metrics = {}
    for name in recipes:
        score = summarize(predictions, name)
        score.update(changed=int(predictions[f'{name}_changed'].sum()),
                     linked=int(predictions[f'{name}_linked'].sum()))
        metrics[name] = score
    return predictions, metrics


def development(data):
    if LOCK.exists():
        raise FileExistsError(f'Existing frozen recipe at {LOCK}')
    p.load_data(data)
    matrix = cache_matrix(next(DEV.glob('sift_mutual_*.npz')))
    bundles = []
    for path in sorted(DEV.glob('eval_*.joblib')):
        bundle = joblib.load(path)
        order = bundle['frame']['order']
        bundles.append((bundle, matrix[order][:, order].toarray()))
    _, metrics = evaluate(bundles, ARMS)
    best = max(metrics, key=lambda name: metrics[name]['slices']['overall']['delta'])
    e.write_json(OUT / 'development_metrics.json', metrics)
    e.write_json(LOCK, dict(selected=best, recipe=ARMS[best], links=LINKS,
        outer_seed=27613, episodes=[47613, 47614, 47615],
        promotion='Positive overall delta and clustered interval lower bound, positive delta in at least two episodes, and nonnegative known and unknown deltas.'))
    for name, score in sorted(metrics.items(), key=lambda item: -item[1]['slices']['overall']['delta']):
        print(name, 'delta', round(score['slices']['overall']['delta'], 6),
              'linked', score['linked'], 'changed', score['changed'], flush=True)


def validation_bundles(data, lock):
    run = OUT / 'fresh_27613'
    model_path = FRESH / 'baseline_models.joblib'
    provenance = json.loads((FRESH / 'training_complete.json').read_text())
    if e.sha(model_path) != provenance['baseline_sha256']:
        raise ValueError('Fresh Model 12 checkpoint does not match its training record')
    e.DATA = data
    train, _, roles, cache = e.configure(run, lock['outer_seed'], 4)
    source_roles = pd.read_csv(FRESH / 'identity_roles.csv').set_index('id').role.sort_index()
    fitted_roles = pd.read_csv(run / 'identity_roles.csv').set_index('id').role.sort_index()
    if not source_roles.equals(fitted_roles):
        raise ValueError('Fresh identity allocation differs from the checkpoint')
    path = next(FRESH.glob('sift_mutual_*.npz'))
    with np.load(path, allow_pickle=False) as matches:
        cache.done.update(zip(map(tuple, matches['pairs']), map(tuple, matches['values'])))
    models = joblib.load(model_path)
    cache.activate(roles['validation'])
    extra = np.setdiff1d(np.arange(len(train)), roles['validation'])
    episodes = []
    for episode in lock['episodes']:
        gallery, queries, truth = p.split_episode(roles['validation'], episode, link=(.95, 111))
        gallery = np.r_[gallery, extra]
        e.validate_episode(gallery, queries, truth, roles)
        episodes.append((episode, gallery, queries, truth))
    manifests = []
    for episode, gallery, queries, truth in episodes:
        manifests.extend(e.manifest_rows(episode, 'validation', gallery, queries, truth))
        saved = run / f'episode_{episode}.joblib'
        if saved.exists():
            yield joblib.load(saved)
            continue
        cache.active_queries = set(queries)
        cache.activate(np.r_[gallery, queries])
        frame, result = p.through_stack(p.frame(gallery, queries, truth), models)
        order = frame['order']
        bundle = (dict(frame=frame, result=result, gallery=gallery, episode=episode),
                  b.INL[order][:, order].toarray())
        e.dump(bundle, saved)
        print('Prepared fresh episode', episode, flush=True)
        yield bundle
    pd.DataFrame(manifests).to_csv(run / 'validation_manifest.csv', index=False)


def confirm(data):
    lock = json.loads(LOCK.read_text())
    if lock['links'] != {key: list(value) for key, value in LINKS.items()}:
        raise ValueError('Frozen graph thresholds differ from the source')
    predictions, metrics = evaluate(validation_bundles(data, lock), {'candidate': lock['recipe']})
    score = metrics['candidate']
    slices = score['slices']
    promote = (slices['overall']['delta'] > 0 and score['delta_ci95'][0] > 0
               and sum(slices[str(ep)]['delta'] > 0 for ep in lock['episodes']) >= 2
               and all(slices[key]['delta'] >= 0 for key in ['known', 'unknown']))
    score.update(promote=promote, recipe_sha256=e.sha(LOCK), seed=lock['outer_seed'])
    e.write_json(OUT / 'validation_metrics.json', score)
    predictions.to_csv(OUT / 'validation_predictions.csv', index=False)
    print(json.dumps(score, indent=2), flush=True)


def profile_and_predict(data, assets, profile_only=False):
    context = portable.prepare(data, assets)
    train, test, sample = (context[key] for key in ('train', 'test', 'sample'))
    frame, result = p.through_stack(context['frame'], portable.load_models(assets))
    order = frame['order']
    inliers = b.INL[order][:, order].toarray()
    baseline = b.rank(frame, result)
    source = OUT / 'test_profile.json'
    graph_stats = {}
    for name in LINKS:
        graph = build_links(order, inliers, name)
        _, group = connected_components(graph, directed=False)
        graph_stats[name] = dict(edges=graph.nnz // 2, linked_queries=int((graph.getnnz(axis=1) > 0).sum()),
            components=len(np.unique(group)), largest_component=int(np.bincount(group).max()))
    profiles = {}
    for species, subset in test.groupby('species'):
        mask = b.SP[order] == species
        query = frame['q'].loc[mask]
        gallery = train[train.species == species]
        profiles[species] = dict(queries=len(subset), gallery_photos=len(gallery),
            gallery_identities=gallery.identity.nunique(), viewpoints=subset.viewpoint.fillna('missing').value_counts().to_dict(),
            queries_with_at_least_7_gallery_inliers=int((query.s1 >= 7).sum()),
            median_best_gallery_cosine=float(query.r1.median()),
            model12_new_first=int((baseline[mask, 0] == p.NEW).sum()))
    e.write_json(source, dict(rows=len(test), species=profiles, query_graphs=graph_stats))
    print(source.read_text(), flush=True)
    if profile_only:
        return
    lock = json.loads(LOCK.read_text())
    validation = json.loads((OUT / 'validation_metrics.json').read_text())
    if validation['recipe_sha256'] != e.sha(LOCK):
        raise ValueError('Validation metrics do not belong to the frozen recipe')
    recipe = lock['recipe']
    graph = build_links(order, inliers, recipe['link'])
    candidate = score_graph(labels(frame), result, graph, recipe['blend'], recipe['all_candidate_mix'])
    chosen = candidate if validation['promote'] else baseline
    write_submission(train, test, sample, order, chosen, baseline, OUT / 'submission.csv',
                     'Model 14' if validation['promote'] else 'Model 12')
    if not validation['promote']:
        write_submission(train, test, sample, order, candidate, baseline,
                         OUT / 'submission_experimental.csv', 'Model 14 (experimental)')


def write_submission(train, test, sample, order, pred, baseline, output, model_name):
    by_id = pd.DataFrame({'id': b.IDS[order], 'predictions': [' '.join(r) for r in pred]})
    submission = sample[['id']].merge(by_id, on='id', how='left', validate='one_to_one')
    checks = portable.validate_submission(submission, sample, test, train)
    pending = output.with_suffix('.tmp.csv')
    submission.to_csv(pending, index=False, lineterminator='\n')
    portable.validate_submission(pd.read_csv(pending, dtype=str, keep_default_na=False), sample, test, train)
    pending.replace(output)
    checks.update(model=model_name,
                  changed_from_model12=int((pred != baseline).any(axis=1).sum()), csv_sha256=portable.sha(output))
    e.write_json(OUT / 'submission.validation.json', checks)
    print(json.dumps(checks, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['profile', 'development', 'confirm', 'predict'])
    parser.add_argument('--data', type=Path, default=e.DATA)
    parser.add_argument('--assets', type=Path, default=ASSETS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings('ignore', message='Only one sample available.*', module='sklearn.covariance')
    with threadpool_limits(limits=4):
        if args.mode == 'development':
            development(args.data)
        elif args.mode == 'confirm':
            confirm(args.data)
        else:
            profile_and_predict(args.data, args.assets, args.mode == 'profile')


if __name__ == '__main__':
    main()
