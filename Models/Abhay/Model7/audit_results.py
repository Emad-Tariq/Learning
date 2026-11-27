"""Recompute paired scores and check identity roles and episode assignments."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=Path('local_runs/paired_2052'))
    args = ap.parse_args()
    root = args.run
    config = json.loads((root / 'config.json').read_text())
    report = json.loads((root / 'metrics.json').read_text())
    train = pd.read_csv(Path(config['data']) / 'train.csv').set_index('id')
    roles = pd.read_csv(root / 'identity_roles.csv').set_index('id')
    episodes = pd.read_csv(root / 'episode_splits.csv')
    rows = pd.read_csv(root / 'validation_predictions.csv')
    assert roles.index.is_unique and set(roles.index) == set(train.index)
    assert roles.groupby('identity').role.nunique().eq(1).all()
    assert not episodes.duplicated(['episode', 'id']).any()
    galleries = {}
    for seed, part in episodes.groupby('episode'):
        assert set(part.role) == set(roles.loc[part.id, 'role'])
        gallery = set(train.loc[part.loc[part.assignment == 'gallery', 'id'], 'identity'])
        known = set(train.loc[part.loc[part.assignment == 'known_query', 'id'], 'identity'])
        unknown = set(train.loc[part.loc[part.assignment == 'unknown_query', 'id'], 'identity'])
        assert known <= gallery and not unknown & gallery
        galleries[int(seed)] = gallery
    assert set(roles.loc[rows.id, 'role']) == {'validation'}
    assert set(rows.episode) == set(config['evaluation_episodes'])
    assert rows.unknown.eq(rows.truth == 'new_individual').all()
    assignments = episodes.set_index(['episode', 'id']).assignment
    species_catalog = train.groupby('species').identity.apply(set).to_dict()
    for row in rows.itertuples():
        assert row.identity == train.loc[row.id, 'identity']
        assert row.species == train.loc[row.id, 'species']
        assert row.truth == ('new_individual' if row.unknown else row.identity)
        assert assignments.loc[(row.episode, row.id)] == ('unknown_query' if row.unknown else 'known_query')
        for method in ('baseline', 'nuisance'):
            labels = getattr(row, method + '_predictions').split()
            assert len(labels) == len(set(labels)) == 5
            assert set(labels) <= galleries[row.episode] | {'new_individual'}
            assert set(labels) <= species_catalog[row.species] | {'new_individual'}
            rr = next((1 / (j + 1) for j, label in enumerate(labels) if label == row.truth), 0.)
            assert np.isclose(rr, getattr(row, method + '_rr5'))
    for method in ('baseline', 'nuisance'):
        assert np.isclose(report[method]['map5'], rows[method + '_rr5'].mean())
    rows['delta'] = rows.nuisance_rr5 - rows.baseline_rr5
    # Resample complete animal identities, preserving repeated query appearances.
    groups = rows.groupby(['species', 'identity']).delta.agg(['sum', 'size'])
    rng = np.random.default_rng(7052)
    bootstrap = []
    for _ in range(2000):
        numerator, denominator = 0., 0
        for _, part in groups.groupby(level='species'):
            sample = part.iloc[rng.integers(len(part), size=len(part))]
            numerator += sample['sum'].sum()
            denominator += sample['size'].sum()
        bootstrap.append(numerator / denominator)
    audit = {'identity_role_and_score_checks': 'passed', 'query_appearances': len(rows),
             'unique_query_images': int(rows.id.nunique()), 'queried_identities': len(groups),
             'pooled_delta': float(rows.delta.mean()),
             'identity_cluster_bootstrap_95_interval': np.quantile(bootstrap, [.025, .975]).tolist(),
             'note': 'Conditional on this fitted model pair and outer identity partition; not a leaderboard confidence interval.'}
    (root / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
