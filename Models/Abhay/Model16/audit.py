"""Independently verify the CSV and pair confirmation scores with existing local runs."""
import csv
from collections import defaultdict

import numpy as np
import pandas as pd

import run as r


def csv_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def verify_submission(path):
    rows = csv_rows(path)
    test = {row['id']: row['species'] for row in csv_rows(r.DATA / 'test.csv')}
    allowed = defaultdict(set)
    for row in csv_rows(r.DATA / 'train.csv'):
        allowed[row['species']].add(row['identity'])
    sample = [row['id'] for row in csv_rows(r.DATA / 'sample_submission.csv')]
    assert [row['id'] for row in rows] == sample and set(sample) == set(test)
    assert len(rows) == len(test) == 2791 and list(rows[0]) == ['id', 'predictions']
    for row in rows:
        labels = row['predictions'].split(' ')
        assert len(labels) == len(set(labels)) == 5 and set(labels) <= allowed[test[row['id']]] | {r.m.NEW}
    return {row['id']: row['predictions'].split() for row in rows}


def paired_ci(frame, difference):
    grouped = frame.assign(delta=difference).groupby(['species', 'animal']).delta.agg(['sum', 'size'])
    rng = np.random.default_rng(1616)
    hits, rows = np.zeros(5000), np.zeros(5000)
    for _, group in grouped.groupby(level=0):
        values = group.to_numpy()
        draws = rng.integers(len(values), size=(len(hits), len(values)))
        hits += values[draws, 0].sum(1)
        rows += values[draws, 1].sum(1)
    return np.quantile(hits / rows, [.025, .975]).tolist()


def main():
    import json

    lock = json.loads((r.OUT / 'recipe.json').read_text())
    key = f"{lock['variant']}_{lock['prior_strength']:g}"
    fitted = pd.read_csv(r.OUT / 'confirmation_predictions.csv')
    reference = pd.read_csv(r.ROOT / 'Models/Abhay/Model15/results/specialist_comparison.csv')
    checked = fitted.merge(reference, on=['id', 'episode', 'truth', 'species'], validate='one_to_one')
    assert len(checked) == len(fitted) == len(reference) == 1402
    recalculated = [next((1 / k for k, name in enumerate(seq.split(), 1) if name == truth), 0.)
        for seq, truth in zip(checked[key + '_prediction'], checked.truth)]
    assert np.allclose(recalculated, checked[key])
    methods = {'Model 12': 'model12_rr5', 'Model 14': 'baseline_rr5', 'Model 15 specialist': 'weight_0.3_rr5'}
    paired = {}
    for name, col in methods.items():
        delta = checked[key] - checked[col]
        paired[name] = dict(map5=float(checked[col].mean()), replacement_map5=float(checked[key].mean()),
            delta=float(delta.mean()), ci95=paired_ci(checked, delta),
            known_delta=float(delta[~checked.novel].mean()), novel_delta=float(delta[checked.novel].mean()))

    result = verify_submission(r.OUT / 'submission.csv')
    emad = {row['id']: row['predictions'].split() for row in csv_rows(r.SOURCE / 'Models/Emad/submission_model8.csv')}
    assert set(result) == set(emad)
    other = {row['id']: row['predictions'].split() for row in csv_rows(r.ROOT / 'Models/Abhay/Model15/results/submission_specialist.csv')}
    outputs = {}
    for name, submitted in [('Emad Model 8', emad), ('Model 15 specialist', other)]:
        outputs[name] = dict(top1_disagreements=sum(result[q][0] != submitted[q][0] for q in result),
            known_new_disagreements=sum((result[q][0] == r.m.NEW) != (submitted[q][0] == r.m.NEW) for q in result),
            identical_ranked_lists=sum(result[q] == submitted[q] for q in result))
    outcome = dict(paired_validation=paired, independent_csv_format_verified=True,
        full_test_prediction_disagreements=outputs, unique_validation_photos=int(checked.id.nunique()),
        validation_identities=int(checked.animal.nunique()), validation_appearances=len(checked))
    r.write_json('independent_audit.json', outcome)
    print(json.dumps(outcome, indent=2))


if __name__ == '__main__':
    main()
