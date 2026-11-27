"""Train, compare and export the gallery-conditioned album ranker."""
import argparse
import json
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE.parent / name) for name in ('Model12', 'Model13', 'Model14')]

import experiment as e
from compare import row_frame, summarize
from graph_model import query_links, score_graph
import album as a
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits
import submission as portable

b, p = e.b, e.p
RUN13 = e.HERE / 'local_runs'
RUN12 = HERE.parent / 'Model12/local_runs/large_2052'
RUN14 = HERE.parent / 'Model14/results/fresh_27613'
ASSETS = ROOT / 'tmp/model12_submission_notebook/model12_assets'
LEARNING = Path.home() / 'Documents/Abhay/Semester 5/ML/Project/Learning'
OUT = HERE / 'results'
DEV = 2052
FRESH = 27613
BLENDS = (.15, .30, .50, .75, 1.)
PROTECTED = {f'{scope}_{weight:g}': (scope, weight)
             for scope in ('all', 'leopard') for weight in (.15, .30, .50)}


def feature_table(frame, gallery, sift):
    return a.candidate_frame(frame, gallery, b.X, b.SP, b.Y, b.VIEW, *sift)


def load_matches(path, native=False):
    return a.match_matrices(path, len(b.IDS), native=native)


def source_run(seed):
    return RUN13 / ('development_2052' if seed == DEV else f'fresh_{seed}')


def train(seed):
    run = source_run(seed)
    source = RUN12 if seed == DEV else run
    sift = load_matches(next(source.glob('sift_mutual_*.npz')))
    manifest = pd.read_csv(source / 'training_episode_splits.csv')
    galleries = manifest[(manifest.stage == 3) & manifest.assignment.eq('gallery')].groupby('episode').id
    ids = pd.Index(b.IDS)
    frames, photo_ids = [], []
    for episode, gallery in galleries:
        frame = joblib.load(run / f'train_{episode}.joblib')
        references = ids.get_indexer(gallery)
        order = frame['order']
        if (references < 0).any() or np.intersect1d(order, references).size:
            raise ValueError('Training queries overlap the gallery or references are unavailable')
        if not set(b.Y[order[frame['truth'] == p.NEW]]).isdisjoint(b.Y[references]):
            raise ValueError('Unknown identity present in an episode gallery')
        features = feature_table(frame, references, sift)
        frames.append(features)
        photo_ids.append(b.IDS[order[features.row.to_numpy()]])
        print('Album training episode', episode, len(order), 'queries', len(features), 'candidate rows', flush=True)
    model = a.learn_head(frames, photo_ids)
    e.dump(model, OUT / f'album_head_{seed}.joblib')
    if seed == DEV:
        e.dump(a.learn_head(frames, photo_ids, prior_only=True), OUT / 'prior_ablation.joblib')


def baseline_distribution(frame, result, features):
    original_names = frame['c'].ident.to_numpy().reshape(len(frame['q']), b.K)
    base = ((1 - result['pb'][:, None]) * result['pc']).ravel()
    i = np.repeat(np.arange(len(original_names)), b.K)
    key = pd.MultiIndex.from_arrays([i, original_names.ravel()])
    candidate_keys = pd.MultiIndex.from_frame(features[['row', 'name']])
    scores = pd.Series(base, index=key).reindex(candidate_keys, fill_value=0.).to_numpy()
    is_new = features.is_new.to_numpy()
    scores[is_new] = result['pb'][features.row.to_numpy()[is_new]]
    return scores


def local_bundles(seed):
    if seed == DEV:
        run = source_run(seed)
        sift = load_matches(next(RUN12.glob('sift_mutual_*.npz')))
        for path in sorted(run.glob('eval_*.joblib')):
            bundle = joblib.load(path)
            order = bundle['frame']['order']
            yield bundle, sift[1][order][:, order].toarray(), sift
    else:
        sift = load_matches(next(RUN14.glob('sift_mutual_*.npz')))
        for path in sorted(RUN14.glob('episode_*.joblib')):
            bundle, inliers = joblib.load(path)
            yield bundle, inliers, sift


def model14(frame, result, inliers):
    order = frame['order']
    graph = query_links(b.X[order], b.SP[order], inliers, .70, 20, .94)
    labs = np.column_stack([frame['c'].ident.to_numpy().reshape(len(order), b.K), np.full(len(order), p.NEW)])
    return score_graph(labs, result, graph, .5, .5)


def candidate_recall(frame, features):
    has_label = np.bincount(features.row, weights=features.positive, minlength=len(frame['order'])) > 0
    old = frame['c'].ident.to_numpy().reshape(-1, b.K)
    original = (old == frame['truth'][:, None]).any(1)
    return has_label, original


def protect_novelty(table, base, album, initial, species, scope, weight):
    """Rerank known IDs while keeping the baseline's open-set decision and rank."""
    row = table.row.to_numpy()
    novelty = table.is_new.to_numpy()

    def normalize(values):
        known = np.where(novelty, 0., values)
        return known / np.maximum(np.bincount(row, weights=known)[row], 1e-12)

    combined = (1 - weight) * normalize(base) + weight * normalize(album)
    known = a.rank_scores(table, np.where(novelty, -np.inf, combined))
    predicted = initial.copy()
    targets = np.flatnonzero((species == scope) | (scope == 'all'))
    for query in targets:
        novelty_slots = np.flatnonzero(initial[query] == p.NEW)
        if len(novelty_slots):
            position = int(novelty_slots[0])
            predicted[query] = np.insert(known[query, :4], position, p.NEW)
        else:
            predicted[query] = known[query]
    return predicted


def compare(seed, blends, protected=None):
    fitted = joblib.load(OUT / f'album_head_{seed}.joblib')
    prior_only = joblib.load(OUT / 'prior_ablation.joblib') if seed == DEV else None
    rows = []
    for bundle, inliers, sift in local_bundles(seed):
        frame, result = bundle['frame'], bundle['result']
        truth = frame['truth']
        table = feature_table(frame, bundle['gallery'], sift)
        row = row_frame(bundle)
        row['album_has_label'], row['original_has_label'] = candidate_recall(frame, table)
        baseline = b.rank(frame, result)
        row['baseline_rr5'] = e.rr(baseline, truth)
        row['model14_rr5'] = e.rr(model14(frame, result, inliers), truth)
        base = baseline_distribution(frame, result, table)
        album = a.head_scores(fitted, table)
        for weight in blends:
            method = f'album_{weight:g}'
            prediction = a.rank_scores(table, (1 - weight) * base + weight * album)
            row[method + '_rr5'] = e.rr(prediction, truth)
            if seed != DEV:
                row[method + '_predictions'] = [' '.join(labels) for labels in prediction]
        for name, (scope, weight) in (protected or {}).items():
            method = f'protected_{name}'
            prediction = protect_novelty(table, base, album, baseline, row.species.to_numpy(), scope, weight)
            row[method + '_rr5'] = e.rr(prediction, truth)
            if seed != DEV:
                row[method + '_predictions'] = [' '.join(labels) for labels in prediction]
        if prior_only:
            prior = a.head_scores(prior_only, table)
            row['prior_ablation_rr5'] = e.rr(a.rank_scores(table, .7 * base + .3 * prior), truth)
        rows.append(row)
        print('Evaluated episode', bundle['episode'], 'reference photos', len(bundle['gallery']), flush=True)
    outcome = pd.concat(rows, ignore_index=True)
    scores = {name[:-4]: summarize(outcome, name[:-4])
              for name in outcome if name.endswith('_rr5') and name != 'baseline_rr5'}
    known = outcome.loc[~outcome.unknown]
    recall = dict(known_appearances=len(known), original_hits=int(known.original_has_label.sum()),
                  album_hits=int(known.album_has_label.sum()))
    print(json.dumps({key: value['slices']['overall'] for key, value in scores.items()}, indent=2), flush=True)
    return outcome, scores, recall


def develop():
    lock = OUT / 'recipe.json'
    if lock.exists():
        raise FileExistsError('The validation recipe is already frozen')
    train(DEV)
    _, scores, recall = compare(DEV, BLENDS)
    chosen = max(BLENDS, key=lambda w: scores[f'album_{w:g}']['slices']['overall']['candidate'])
    e.write_json(OUT / 'development_metrics.json', dict(methods=scores, shortlist=recall))
    e.write_json(lock, dict(album_weight=chosen, development_seed=DEV, evaluation_seed=FRESH,
        confirmation_episodes=[47613, 47614, 47615],
        album_source_sha256=e.sha(HERE / 'album.py'),
        selection='Maximum development MAP@5 over fixed blend grid.'))
    print('Frozen blend', chosen, flush=True)


def confirm():
    lock = json.loads((OUT / 'recipe.json').read_text())
    if lock['album_source_sha256'] != e.sha(HERE / 'album.py'):
        raise ValueError('Album method changed after recipe freeze')
    policy = json.loads((OUT / 'policy.json').read_text())
    if policy['base_recipe_sha256'] != e.sha(OUT / 'recipe.json'):
        raise ValueError('Protected-ranking policy differs from its frozen baseline recipe')
    protected = {policy['protected_name']: policy['protected_recipe']}
    outcome, scores, recall = compare(FRESH, [lock['album_weight']], protected)
    if sorted(outcome.episode.unique()) != lock['confirmation_episodes']:
        raise ValueError('Missing frozen evaluation episodes')
    e.write_json(OUT / 'confirmation_metrics.json', dict(methods=scores, shortlist=recall,
        recipe_sha256=e.sha(OUT / 'recipe.json'), policy_sha256=e.sha(OUT / 'policy.json')))
    outcome.to_csv(OUT / 'confirmation_predictions.csv', index=False)


def develop_policy():
    path = OUT / 'policy.json'
    if path.exists():
        raise FileExistsError('The protected-ranking policy is already frozen')
    lock = json.loads((OUT / 'recipe.json').read_text())
    _, scores, recall = compare(DEV, [lock['album_weight']], PROTECTED)
    chosen = max(PROTECTED, key=lambda name: scores[f'protected_{name}']['slices']['overall']['candidate'])
    joint = f"album_{lock['album_weight']:g}"
    main = max([joint, f'protected_{chosen}'], key=lambda name: scores[name]['slices']['overall']['candidate'])
    e.write_json(OUT / 'policy_development_metrics.json', dict(methods=scores, shortlist=recall))
    e.write_json(path, dict(primary=main, protected_name=chosen, protected_recipe=PROTECTED[chosen],
        base_recipe_sha256=e.sha(OUT / 'recipe.json'),
        selection='Maximum previously exposed development MAP@5. Confirmation cannot change the primary recipe.'))


def production_gallery_matches(learning, data):
    """Reuse teammate-computed train-to-train geometry after checking its row mapping."""
    cache = learning / 'Models/Emad/cross_flank_3e3eceb5c8fa/sift_pairs_mutual.npz'
    master = pd.read_csv(learning / 'Processed_Data/master_dataset.csv')
    train = pd.read_csv(data / 'train.csv')
    if len(master) != len(b.IDS) or not master.id.eq(b.IDS).all() or not master.species.eq(b.SP).all():
        raise ValueError('Team SIFT cache row mapping differs from the competition files')
    with np.load(cache, allow_pickle=False) as z:
        pairs = z['pairs']
        if not ((pairs[:, 0] >= 0) & (pairs[:, 0] < pairs[:, 1]) & (pairs[:, 1] < len(master))).all():
            raise ValueError('Team SIFT cache contains invalid pair indices')
        train_only = (pairs < len(train)).all(1)
        x, y = pairs[train_only].T
        rows, cols = np.r_[x, y], np.r_[y, x]
        n = len(master)
        output = []
        for key in ('good', 'inl'):
            values = z[key][train_only]
            output.append(sparse.csr_matrix((np.r_[values, values], (rows, cols)), shape=(n, n)))
        return output


def verify_csv(context, order, prediction, output):
    rows = pd.DataFrame({'id': b.IDS[order], 'predictions': [' '.join(row) for row in prediction]})
    sample, test, train = (context[k] for k in ('sample', 'test', 'train'))
    merged = sample[['id']].merge(rows, on='id', validate='one_to_one', how='left')
    portable.validate_submission(merged, sample, test, train)
    tmp = output.with_suffix('.tmp.csv')
    merged.to_csv(tmp, index=False, lineterminator='\n')
    checks = portable.validate_submission(pd.read_csv(tmp, dtype=str, keep_default_na=False), sample, test, train)
    tmp.replace(output)
    return merged, dict(checks, sha256=e.sha(output))


def predict(data, assets, learning):
    lock = json.loads((OUT / 'recipe.json').read_text())
    if lock['album_source_sha256'] != e.sha(HERE / 'album.py'):
        raise ValueError('Album method changed after recipe freeze')
    ctx = portable.prepare(data, assets)
    frame, result = p.through_stack(ctx['frame'], portable.load_models(assets))
    order = frame['order']
    gallery = np.arange(len(ctx['train']))
    gallery_sift = production_gallery_matches(learning, data)
    sift = (b.GOOD + gallery_sift[0], b.INL + gallery_sift[1])
    table = feature_table(frame, gallery, sift)
    base = baseline_distribution(frame, result, table)
    w = lock['album_weight']
    model = joblib.load(OUT / f'album_head_{DEV}.joblib')
    album = a.head_scores(model, table)
    policy = json.loads((OUT / 'policy.json').read_text())
    if policy['base_recipe_sha256'] != e.sha(OUT / 'recipe.json'):
        raise ValueError('Protected-ranking policy differs from its frozen baseline recipe')
    if policy['primary'].startswith('protected_'):
        scope, w = policy['protected_recipe']
        predicted = protect_novelty(table, base, album, b.rank(frame, result), b.SP[order], scope, w)
    else:
        predicted = a.rank_scores(table, (1 - w) * base + w * album)
    output = OUT / 'submission.csv'
    merged, checks = verify_csv(ctx, order, predicted, output)
    emad = pd.read_csv(learning / 'Models/Emad/submission_model8.csv', dtype=str, keep_default_na=False)
    emad = ctx['sample'][['id']].merge(emad, on='id', validate='one_to_one', how='left')
    portable.validate_submission(emad, ctx['sample'], ctx['test'], ctx['train'])
    old = emad.predictions.str.split().str[0]
    new = merged.predictions.str.split().str[0]
    old_unknown, new_unknown = old.eq(p.NEW), new.eq(p.NEW)
    query_stats = ctx['test'].set_index('id').loc[merged.id].reset_index()
    sift_by_id = pd.Series(frame['q'].s1.to_numpy(), index=b.IDS[order])
    query_stats['gallery_sift_7plus'] = sift_by_id.loc[merged.id].to_numpy() >= 7
    query_stats['top1_disagreement'] = old != new
    query_stats['new_status_disagreement'] = old_unknown != new_unknown
    query_stats['emad_new_first'] = old_unknown
    query_stats['model15_new_first'] = new_unknown
    groups = query_stats.groupby(['species', 'gallery_sift_7plus'], sort=True, observed=True)
    breakdown = groups.agg(queries=('id', 'size'), top1_disagreements=('top1_disagreement', 'sum'),
        known_new_disagreements=('new_status_disagreement', 'sum'),
        emad_new_first=('emad_new_first', 'sum'), model15_new_first=('model15_new_first', 'sum')).reset_index()
    checks.update(emad_top1_disagreements=int((old != new).sum()),
        emad_known_new_disagreements=int((old_unknown != new_unknown).sum()),
        emad_only_new=int((old_unknown & ~new_unknown).sum()),
        model15_only_new=int((new_unknown & ~old_unknown).sum()),
        model15_new_first=int(new_unknown.sum()), emad_new_first=int(old_unknown.sum()),
        expanded_candidate_rows=int(len(table)), emad_exact_prediction_list_changes=int((emad.predictions != merged.predictions).sum()),
        blend=w, policy=policy['primary'], by_species_and_sift=breakdown.to_dict('records'))
    e.write_json(OUT / 'submission_checks.json', checks)
    print(json.dumps(checks, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['develop', 'develop-policy', 'fit-fresh', 'confirm', 'predict'])
    parser.add_argument('--data', type=Path, default=e.DATA)
    parser.add_argument('--assets', type=Path, default=ASSETS)
    parser.add_argument('--learning', type=Path, default=LEARNING)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings('ignore', message='Only one sample available.*')
    with threadpool_limits(limits=4):
        if args.mode == 'predict':
            predict(args.data, args.assets, args.learning)
        else:
            p.load_data(args.data)
            if args.mode == 'develop':
                develop()
            elif args.mode == 'develop-policy':
                develop_policy()
            elif args.mode == 'fit-fresh':
                train(FRESH)
            else:
                confirm()


if __name__ == '__main__':
    main()
