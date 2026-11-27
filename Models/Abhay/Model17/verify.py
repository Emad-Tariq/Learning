"""Verify submission contents and measure what the second-sighting model changed."""
import json

import numpy as np
import pandas as pd
from experiment import OUT, inputs, prepare, benchmarks


def weak_gallery(data,g,q):
    selected = np.zeros(len(q),bool)
    for sp in inputs.SPECIES:
        ix = np.flatnonzero(data['species'][q]==sp)
        qs = q[ix]
        gs = g[data['species'][g]==sp]
        sift = data['inliers'][qs]
        selected[ix] = ((sift[:,gs].max(axis=1).toarray().ravel()<7) &
                        (sift[:,qs].max(axis=1).toarray().ravel()>=7))
    return selected


def main():
    data,dev,audit,simulate = prepare()
    reference = pd.read_csv(inputs.DATA/'sample_submission.csv')
    test = data['test'].set_index('id').loc[reference.id].reset_index()
    target = pd.read_csv(OUT/'submission_model17.csv')
    original = pd.read_csv(inputs.SOURCE/'Models/Emad/submission_model8.csv').set_index('id').loc[reference.id]
    unlinked = pd.read_csv(OUT/'submission_graph.csv').set_index('id').loc[reference.id]
    assert target.columns.equals(reference.columns) and target.id.equals(reference.id) and target.id.is_unique
    allowed = data['train'].groupby('species').identity.agg(set)
    words = target.predictions.str.split()
    assert all(len(row)==len(set(row))==5 and set(row)<=(allowed[sp]|{'new_individual'})
               for row,sp in zip(words,test.species))
    first = words.str[0].to_numpy()
    upstream = unlinked.predictions.str.split().str[0].to_numpy()
    incumbent = original.predictions.str.split().str[0].to_numpy()
    tr = np.arange(len(data['train']))
    q = data['index'].get_indexer(reference.id)
    mask = weak_gallery(data,tr,q)
    result = dict(rows=len(target),sha256=inputs.sha(OUT/'submission_model17.csv'),
        changed_first_vs_model8=int((first!=incumbent).sum()),
        changed_first_from_linking=int((first!=upstream).sum()),no_gallery_strong_peer=int(mask.sum()),
        changed_first_from_linking_in_subgroup=int((first[mask]!=upstream[mask]).sum()),
        model17_new_first=int((first=='new_individual').sum()),model8_new_first=int((incumbent=='new_individual').sum()))
    linked = pd.read_csv(OUT/'linked_holdout_24.csv')
    base = pd.read_csv(OUT/'holdout_24.csv')
    rr = lambda f: inputs.reciprocal(f[[f'p{i}' for i in range(1,6)]].to_numpy(),f.truth.to_numpy())
    segments = []
    for kind,seed,g,q,y in benchmarks(dev,audit,simulate):
        a,b = (f[f.seed==seed] for f in (linked,base))
        assert np.array_equal(a.id,data['ids'][q]) and np.array_equal(a.id,b.id)
        segments.append(pd.DataFrame(dict(size=kind,weak=weak_gallery(data,g,q),base=rr(b),linked=rr(a))))
    result['holdout_strata'] = pd.concat(segments).groupby(['size','weak']).agg(
        appearances=('linked','size'),before=('base','mean'),after=('linked','mean')).reset_index().to_dict('records')
    print(json.dumps(result,indent=2))
    (OUT/'submission_checks.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
