"""Create a compact report and independently verify saved predictions/metrics."""
import json
from pathlib import Path
import csv
from collections import defaultdict

HERE=Path(__file__).resolve().parent
OUT=HERE/'results'
a=json.loads((OUT/'audit_summary.json').read_text())
r=json.loads((OUT/'paired_metrics.json').read_text())
rows=list(csv.DictReader((OUT/'paired_predictions.csv').open(newline='')))
source=list(csv.DictReader((HERE.parent/'Model7/local_runs/paired_2052/validation_predictions.csv').open(newline='')))
reference={(x['episode'],x['id']):x for x in source}
assert len(rows)==len(reference)==a['query_appearances']
assert len({(x['episode'],x['id']) for x in rows})==len(rows)
totals=defaultdict(list)
for row in rows:
    key=(row['episode'],row['id'])
    assert row['baseline_predictions']==reference[key]['baseline_predictions']
    truth='new_individual' if row['unknown']=='True' else row['identity']
    for method in ('baseline','supplement'):
        labels=row[method+'_predictions'].split()
        assert len(labels)==len(set(labels))==5
        score=next((1/(i+1) for i,label in enumerate(labels) if label==truth),0.)
        assert abs(score-float(row[method+'_rr5']))<1e-12
        totals[(method,'map5')].append(score)
        totals[(method,'unknown_map5' if row['unknown']=='True' else 'known_map5')].append(score)
for (method,metric),values in totals.items():
    assert abs(sum(values)/len(values)-r[method][metric])<1e-12
gap=[x for x in rows if x['unknown']=='False' and x['candidate_present']=='True' and int(x['original_true_checked'])==0]
assert len(gap)==a['shortlisted_correct_without_sift']
assert len({x['id'] for x in gap})==a['gap_unique_photos']
audit={'baseline_predictions_reproduced':True,'all_rr5_recomputed':True,'pooled_metrics_recomputed':True,
       'unique_episode_query_keys':len(rows),'gap_unique_photos':len({x['id'] for x in gap})}
(OUT/'independent_audit.json').write_text(json.dumps(audit,indent=2))

lines=['# SIFT coverage: measured result','',
       'The experiment keeps Dalvin Model 3d’s baseline shortlist and final ranking rule. It adds up to two SIFT checks for each shortlisted identity with zero checked gallery photos, adds availability features, and retrains all three stages. No baseline artifacts or submissions are changed.','',
       '## How often was the alleged gap present?','',
       f"Across {a['query_appearances']:,} query appearances, {a['known_queries']:,} were known animals and {a['known_correct_shortlisted']:,} had the correct identity shortlisted. Only **{a['shortlisted_correct_without_sift']} of {a['known_correct_shortlisted']} ({100*a['gap_fraction_shortlisted_known']:.2f}%)** had no checked reference photo for the correct shortlisted identity. These were **{a['gap_unique_photos']} distinct leopard photos**; no hyena query had this gap.",'',
       f"Additional verification found at least seven geometric inliers in {a['gap_extra_ge7']} affected appearances, both the same distinct leopard photo. Four of the five appearances originally missed the top five; the other was ranked fifth. Among all {a['unchecked_candidate_rows']:,} unchecked candidate rows, supplementation produced at least seven inliers in {a['supplemented_candidate_rows_ge7']} rows: two correct and 16 incorrect candidate appearances. Seven inliers is therefore supporting evidence, not an identity decision.",'',
       '## Complete three-stage comparison','',
       '| Metric | Baseline | Added verification + availability | Difference |',
       '| --- | ---: | ---: | ---: |']
for key,label in [('map5','Overall MAP@5'),('known_map5','Known MAP@5'),('unknown_map5','Unknown MAP@5')]:
    x,y=r['baseline'][key],r['supplement'][key]
    lines.append(f'| {label} | {x:.6f} | {y:.6f} | {y-x:+.6f} |')
lines+=['','| Episode | Baseline | Intervention | Difference |','| --- | ---: | ---: | ---: |']
for seed,z in r['episodes'].items():
    lines.append(f"| {seed} | {z['baseline']['map5']:.6f} | {z['supplement']['map5']:.6f} | {z['delta']:+.6f} |")
lines+=['','| Species | Baseline | Intervention | Difference |','| --- | ---: | ---: | ---: |']
for sp,z in r['species'].items():
    lines.append(f"| {sp} | {z['baseline']['map5']:.6f} | {z['supplement']['map5']:.6f} | {z['delta']:+.6f} |")
lo,hi=r['identity_cluster_bootstrap_95pct']
lines+=['',f"The identity-cluster bootstrap 95% interval for the overall difference is [{lo:+.6f}, {hi:+.6f}]. {r['improved']} appearances improved and {r['worsened']} worsened. The predeclared gate {'passed' if r['gate_passed'] else 'failed'}.",'',
        '## Affected queries','',
        '| Episode | Photo | True identity | Extra best inliers | Baseline rank | Intervention rank |',
        '| --- | --- | --- | ---: | ---: | ---: |']
for x in gap:
    rank=lambda name: str(round(1/float(x[name+'_rr5']))) if float(x[name+'_rr5']) else 'outside 5'
    lines.append(f"| {x['episode']} | {x['id']} | {x['identity']} | {x['true_extra_max_inliers']} | {rank('baseline')} | {rank('supplement')} |")
lines+=['','## What this establishes','',
        ('The combined intervention passed the development gate, but it still needs independent confirmation.' if r['gate_passed'] else 'Keep Dalvin’s baseline as the default. This intervention did not meet the development gate.'),'',
        'The specific no-checked-photo gap is real but uncommon under this protocol. Its presence alone does not establish that missing SIFT caused a wrong prediction. The paired comparison tests the combined effect of extra checks, availability features and retraining; it does not isolate these effects. Even perfectly repairing only the five affected appearances, with all other predictions fixed, could add at most about 0.002624 MAP@5 in these episodes. This is not an upper bound for all possible SIFT improvements.','',
        '## Checks and limitations','',
        '- All baseline top-five predictions reproduce the saved checkpoint output exactly. Every saved reciprocal-rank score and pooled metric is independently recomputed from the prediction strings.',
        '- Runtime assertions preserve candidate membership, query order, truth and non-SIFT candidate evidence. Two tests check sparse-zero provenance and label-independent pair selection.',
        '- Validation identities were examined in earlier EDA/experiments. Episodes reuse photos and identities; these are development results, not a pristine final evaluation or leaderboard claim.',
        '- The identity-disjoint roles have smaller galleries than original full-data competition inference. These measurements cannot establish the gap frequency or performance loss for that original model.',
        '- Only previously unchecked candidates receive at most two checks. Partially checked candidates and all-photo verification were not tested.',
        '- The bootstrap reflects these fixed models/identities and does not estimate retraining variance.','',
        'Reproduction and exact intervention: [README.md](README.md). Detailed results and per-query predictions are under `results/`.']
(HERE/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'independent_audit':audit,'overall':r,'report':str(HERE/'RESULTS.md')},indent=2))
