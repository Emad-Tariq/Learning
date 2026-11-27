"""Test album reranking only when the leading identity has few references."""
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run as r

b, p, e, a = r.b, r.p, r.e, r.a
NEW_EPISODES = (58071, 58072, 58073)
LOCK = r.OUT / 'specialist_recipe.json'


def rank(frame, gallery, table, result, learned, links, weight):
    """Route scarce-album cases to the specialist without moving the novelty slot."""
    anchor = r.model14(frame, result, links)
    first = [next(label for label in row if label != p.NEW) for row in anchor]
    names, counts = np.unique(b.Y[gallery], return_counts=True)
    available = dict(zip(names, counts))
    routed = np.array([available[label] <= 3 for label in first])
    base = r.baseline_distribution(frame, result, table)
    advice = r.protect_novelty(table, base, learned, anchor,
                               np.where(routed, 'specialist', 'default'), 'specialist', weight)
    return advice, anchor, routed


def episode_scores(bundles, weights, model):
    outputs = []
    for bundle, links, sift in bundles:
        frame, result, gallery = (bundle[k] for k in ('frame', 'result', 'gallery'))
        table = r.feature_table(frame, gallery, sift)
        learned = a.head_scores(model, table)
        truth = frame['truth']
        rows = r.row_frame(bundle)
        rows['model12_rr5'] = e.rr(b.rank(frame, result), truth)
        for weight in weights:
            pred, anchor, routed = rank(frame, gallery, table, result, learned, links, weight)
            rows['routed'] = routed
            rows['baseline_rr5'] = e.rr(anchor, truth)
            rows[f'weight_{weight:g}_rr5'] = e.rr(pred, truth)
        outputs.append(rows)
        print('Specialist scored episode', bundle['episode'], 'routed', int(routed.sum()), flush=True)
    return pd.concat(outputs, ignore_index=True)


def develop():
    if LOCK.exists():
        raise FileExistsError('Specialist recipe already frozen')
    weights = [.15, .30, .50]
    outcomes = pd.concat([episode_scores(r.local_bundles(seed), weights,
        joblib.load(r.OUT / f'album_head_{seed}.joblib')) for seed in (r.DEV, r.FRESH)], ignore_index=True)
    scores = {w: r.summarize(outcomes, f'weight_{w:g}') for w in weights}
    winner = max(weights, key=lambda w: scores[w]['slices']['overall']['delta'])
    e.write_json(LOCK, dict(weight=winner, additional_episodes=list(NEW_EPISODES),
        scope='Leading known Model 14 identity has at most three eligible gallery photos.',
        selection='Highest pooled development MAP@5 on both previously examined allocations.',
        source_sha256=e.sha(__file__)))
    e.write_json(r.OUT / 'specialist_development.json', {str(w): s for w, s in scores.items()})
    print('Frozen specialist', winner, json.dumps({str(w): s['slices']['overall'] for w, s in scores.items()}), flush=True)


def extra_bundles(data):
    run = r.OUT / 'additional_episodes'
    e.DATA = data
    train, _, roles, cache = e.configure(run, r.FRESH, 6)
    cached = next(r.RUN14.glob('sift_mutual_*.npz'))
    with np.load(cached, allow_pickle=False) as z:
        cache.done.update(zip(map(tuple, z['pairs']), map(tuple, z['values'])))
    b.INL = r.load_matches(cached)[1]
    models = joblib.load(r.source_run(r.FRESH) / 'baseline_models.joblib')
    extra = np.setdiff1d(np.arange(len(train)), roles['validation'])
    manifests = []
    for episode in NEW_EPISODES:
        b.INL = r.load_matches(cached)[1]
        gallery, query, truth = p.split_episode(roles['validation'], episode, link=(.95, 111))
        gallery = np.r_[gallery, extra]
        e.validate_episode(gallery, query, truth, roles)
        manifests.extend(e.manifest_rows(episode, 'validation', gallery, query, truth))
        cache.active_queries = set(query)
        cache.activate(np.r_[gallery, query])
        frame, result = p.through_stack(p.frame(gallery, query, truth), models)
        order = frame['order']
        yield dict(frame=frame, result=result, gallery=gallery, episode=episode), b.INL[order][:, order].toarray(), (b.GOOD, b.INL)
    pd.DataFrame(manifests).to_csv(run / 'validation_manifest.csv', index=False)


def confirm(data, recipe):
    if recipe['additional_episodes'] != list(NEW_EPISODES) or recipe['source_sha256'] != e.sha(__file__):
        raise ValueError('Specialist changed after recipe freeze')
    weight = recipe['weight']
    outcomes = episode_scores(extra_bundles(data), [weight], joblib.load(r.OUT / f'album_head_{r.FRESH}.joblib'))
    key = f'weight_{weight:g}'
    score = r.summarize(outcomes, key)
    group = outcomes[outcomes.routed]
    score.update(model12_map5=float(outcomes.model12_rr5.mean()), routed=int(len(group)),
        routed_baseline=float(group.baseline_rr5.mean()), routed_candidate=float(group[key + '_rr5'].mean()),
        recipe_sha256=e.sha(LOCK))
    e.write_json(r.OUT / 'specialist_confirmation.json', score)
    outcomes.to_csv(r.OUT / 'specialist_comparison.csv', index=False)
    print(json.dumps(score, indent=2), flush=True)


def predict(args, recipe):
    ctx = r.portable.prepare(args.data, args.assets)
    frame, result = p.through_stack(ctx['frame'], r.portable.load_models(args.assets))
    order = frame['order']
    gallery = np.arange(len(ctx['train']))
    added = r.production_gallery_matches(args.learning, args.data)
    table = r.feature_table(frame, gallery, (b.GOOD + added[0], b.INL + added[1]))
    model = joblib.load(r.OUT / f'album_head_{r.DEV}.joblib')
    learned = a.head_scores(model, table)
    pred, anchor, routed = rank(frame, gallery, table, result, learned, b.INL[order][:, order].toarray(), recipe['weight'])
    output = r.OUT / 'submission_specialist.csv'
    submission, checks = r.verify_csv(ctx, order, pred, output)
    ids = submission.id
    emad = pd.read_csv(args.learning / 'Models/Emad/submission_model8.csv').set_index('id').loc[ids]
    first = submission.predictions.str.split().str[0].to_numpy()
    emad_first = emad.predictions.str.split().str[0].to_numpy()
    checks.update(routed=int(routed.sum()), ranked_list_changes_vs_model14=int((pred != anchor).any(1).sum()),
                  emad_top1_disagreements=int((first != emad_first).sum()),
                  emad_known_new_disagreements=int(((first == p.NEW) != (emad_first == p.NEW)).sum()),
                  recipe_sha256=e.sha(LOCK))
    e.write_json(r.OUT / 'specialist_submission_checks.json', checks)
    print(json.dumps(checks, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['develop', 'confirm', 'predict'])
    parser.add_argument('--data', type=r.Path, default=e.DATA)
    parser.add_argument('--assets', type=r.Path, default=r.ASSETS)
    parser.add_argument('--learning', type=r.Path, default=r.LEARNING)
    args = parser.parse_args()
    warnings.filterwarnings('ignore', message='Only one sample available.*')
    with threadpool_limits(limits=4):
        if args.mode == 'develop':
            p.load_data(args.data)
            develop()
        elif args.mode == 'confirm':
            confirm(args.data, json.loads(LOCK.read_text()))
        else:
            predict(args, json.loads(LOCK.read_text()))


if __name__ == '__main__':
    main()
