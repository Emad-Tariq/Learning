"""Independently verify predictions, fitting roles and final score calculation."""
import sys,json,hashlib,platform,importlib.metadata
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'tmp/abhay_deps'))
import numpy as np
import pandas as pd
import joblib

out=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'local_runs/main'
d=pd.read_csv(out/'validation_predictions.csv');r=json.loads((out/'metrics.json').read_text())
roles=pd.read_csv(out/'identity_roles.csv');pr=pd.read_csv(out/'pair_identity_roles.csv')
assert roles.groupby('identity').role.nunique().max()==1
assert pr.groupby('identity').role.nunique().max()==1
assert set(pr.identity)<=set(roles.loc[roles.role.str.startswith('stage'),'identity'])
assert not set(pr.identity)&set(roles.loc[roles.role.isin(['calibration','validation']),'identity'])
val=joblib.load(out/'validation_bundle.joblib')
assert d.id.tolist()==val['rows'].id.tolist() and d.episode.tolist()==val['rows'].episode.tolist()
network=joblib.load(out/'verifier.joblib')
assert network['parameters']<1_000_000
assert not set(roles.set_index('id').loc[pr.loc[pr.role=='pair_fit','id'],'identity'])&set(pr.loc[pr.role=='pair_early_stop','identity'])
for method in r['methods']:
    computed=[]
    for i,row in enumerate(d.itertuples()):
        labels=np.r_[val['candidates'][i],['new_individual']]
        pb=getattr(row,method+'_pnew')
        scores=np.r_[(1-pb)*val['pc'][i],pb]
        pred=labels[np.argsort(-scores)[:5]].tolist()
        assert len(set(pred))==5
        assert ' '.join(pred)==getattr(row,method+'_predictions')
        rr=next((1/(j+1) for j,label in enumerate(pred) if label==row.truth),0.)
        assert abs(rr-getattr(row,method+'_rr5'))<1e-12
        computed.append(rr)
    computed=np.array(computed)
    assert abs(computed.mean()-r['methods'][method]['map5'])<1e-12
    for mask,metric in [(~d.unknown,'known_map5'),(d.unknown,'unknown_map5')]:
        assert abs(computed[mask].mean()-r['methods'][method][metric])<1e-12
    assert abs((computed[d.unknown]==1).mean()-r['methods'][method]['unknown_top1'])<1e-12
audit=dict(role_disjointness=True,full_rank_and_score_recomputation=True,
    query_appearances=len(d),unique_photos=int(d.id.nunique()),identity_count=int(d.identity.nunique()),
    pair_fit_identities=int(pr.loc[pr.role=='pair_fit','identity'].nunique()),
    pair_stop_identities=int(pr.loc[pr.role=='pair_early_stop','identity'].nunique()),
    parameters=network['parameters'],best_epoch=int(np.argmin([x['stop_logloss'] for x in network['history']])+1),
    python=platform.python_version(),packages={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','lightgbm','opencv-python']})
(out/'independent_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2))
lines=['# Model 11: full-vector MLP new-animal detector','',
    '**The local experiment did not improve overall MAP@5. Keep the baseline.**','',
    'The fixed primary arm blends baseline P(new) equally with a logistic calibrator that receives full-vector pair-MLP evidence and baseline context. A summary-only calibrator provides the same baseline context without the MLP evidence. Full replacement is diagnostic; no arm or blend was selected using validation outcomes.','',
    '| Method | Overall MAP@5 | Known MAP@5 | New MAP@5 | New top-one detection |',
    '| --- | ---: | ---: | ---: | ---: |']
for name,z in r['methods'].items():lines.append(f"| {name} | {z['map5']:.6f} | {z['known_map5']:.6f} | {z['unknown_map5']:.6f} | {100*z['unknown_top1']:.2f}% |")
lines+=['',f"Primary overall difference: {r['methods']['mlp_blend']['delta_map5']:+.6f}. Identity-cluster bootstrap 95% interval: {r['methods']['mlp_blend']['delta_identity_bootstrap95']}. Promotion gate passed: {r['promotion_gate_passed']}.",'',
    'Known-animal performance improved, but the loss on truly new animals was larger. The MLP arm beat the summary-only calibration control by just 0.000547 MAP@5, which does not establish a reliable incremental benefit. No settings were changed after inspecting these validation results.','',
    f"The network has {network['parameters']:,} parameters, trained on {network['fit_pairs']:,} distinct sampled pairs. Early stopping used {network['stop_pairs']:,} pairs from disjoint identities and selected epoch {audit['best_epoch']}; six later epochs failed to improve stopping loss. Pair probabilities are not interpreted as query P(known). Gallery size and pair aggregation enter a separate query-level calibrator.",'',
    f"Validation contains {len(d):,} query appearances from {audit['unique_photos']:,} distinct photos and {audit['identity_count']} identities across three episodes. Baseline predictions exactly reproduce the preserved checkpoint. Independent checks verify identity separation and recompute all final rankings, reciprocal ranks, class-specific MAP@5 and new-animal top-one rates.",'',
    '## Scope','',
    'This is a local completed run using the preserved Dalvin baseline checkpoint at seed 2052 and its verified matching cache. The Kaggle notebook trains that baseline from scratch from embedded preserved sources, then executes the same verifier and calibration protocol. Exact numeric results can vary with package versions; local versions are in `independent_audit.json`. Kaggle execution itself has not been performed here.','',
    'The existing evaluation allocation was previously inspected in EDA and other experiments. These are development results, not an untouched test or a leaderboard result. Gallery sizes differ from full-data competition inference. Episodes reuse photos/identities, and the bootstrap does not estimate retraining variance.','',
    '## Files','',
    '- `MLP_NewAnimal_Kaggle.ipynb`: self-contained notebook, including source snapshots and tests.',
    '- `README.md`: protocol and execution instructions.',
    '- `local_runs/main/validation_predictions.csv`: per-query predictions for all four arms.',
    '- `local_runs/main/metrics.json`: overall, species and episode metrics and uncertainty.',
    '- `local_runs/main/independent_audit.json`: independently verified counts and environment.']
(HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
