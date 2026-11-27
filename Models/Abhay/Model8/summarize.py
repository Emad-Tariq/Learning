"""Produce reviewable tables, plots and interpretation from the paired run."""
import json, base64, sys
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
out=HERE/'local_runs/union_2052'
j=json.loads((out/'metrics.json').read_text())
d=pd.read_csv(out/'validation_predictions.csv')
audit=json.loads((out/'baseline_audit.json').read_text()) if (out/'baseline_audit.json').exists() else {}
labels={'all':'Overall','known':'Known animals','unknown':'Unknown animals','missing_side':'Missing side (known)',
        'same_side_available':'Same side available','small_gallery_1to3':'1–3 gallery photos (known)',
        'gallery_4plus':'4+ gallery photos (known)','missing_side_small_gallery':'Missing side + 1–3 photos','viewpoint_missing':'Viewpoint missing'}
rows=[]
for key,s in j['slices'].items():
    interval=s['uncertainty']['identity_cluster_bootstrap_95pct'] if s['uncertainty'] else [None,None]
    rows.append({'slice':labels[key],'query_appearances':s['query_appearances'],'unique_queries':s['unique_queries'],
                 'identities':s['unique_identities'],'baseline_MAP5':s['baseline']['map5'],'union_MAP5':s['union']['map5'],
                 'delta':s['delta'],'CI_low':interval[0],'CI_high':interval[1],
                 'baseline_candidate_recall':s['baseline']['candidate_recall_known'],
                 'union_candidate_recall':s['union']['candidate_recall_known'],'improved':s['improved'],'worsened':s['worsened']})
table=pd.DataFrame(rows); table.to_csv(HERE/'slice_results.csv',index=False)
episode_table=pd.DataFrame([{'episode':k,'queries':v['query_appearances'],'baseline':v['baseline']['map5'],'union':v['union']['map5'],'delta':v['delta']} for k,v in j['episodes'].items()])
episode_table.to_csv(HERE/'episode_results.csv',index=False)
species_table=pd.DataFrame([{'species':k,'queries':v['query_appearances'],'baseline':v['baseline']['map5'],'union':v['union']['map5'],'delta':v['delta']} for k,v in j['species'].items()])
species_table.to_csv(HERE/'species_results.csv',index=False)
known=d[~d.unknown]
recovered=known[~known.baseline_candidate_present & known.union_candidate_present]
recovered.to_csv(out/'recovered_candidate_queries.csv',index=False)
changed=d.assign(delta=d.union_rr5-d.baseline_rr5).sort_values('delta')
changed.to_csv(out/'query_changes.csv',index=False)
diagnostics={
    'average_union_candidates':float(d.union_candidate_count.mean()),
    'candidate_count_min':int(d.union_candidate_count.min()),'candidate_count_max':int(d.union_candidate_count.max()),
    'recovered_candidate_appearances':len(recovered),'recovered_unique_queries':int(recovered.id.nunique()),
    'recovered_in_final_top5':int((recovered.union_rr5>0).sum()),
    'recovered_mean_rr5_union':float(recovered.union_rr5.mean()) if len(recovered) else None,
    'query_types':{str(kind):{'mean_baseline_pnew':float(a.baseline_pnew.mean()),'mean_union_pnew':float(a.union_pnew.mean()),
                            'mean_baseline_rawsum':float(a.baseline_rawsum.mean()),'mean_union_rawsum':float(a.union_rawsum.mean())} for kind,a in d.groupby('unknown')},
    'baseline_reference_audit':audit}
(HERE/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(13,5),layout='constrained')
plot=table.iloc[:6].set_index('slice')
plot[['baseline_MAP5','union_MAP5']].rename(columns={'baseline_MAP5':'Dalvin baseline','union_MAP5':'Union'}).plot.barh(ax=ax[0],color=['#176b78','#e7a342'])
ax[0].invert_yaxis(); ax[0].set(xlim=(0,1),xlabel='Full-pipeline MAP@5',ylabel='',title='Paired full-pipeline comparison')
y=np.arange(len(plot)); diff=plot.delta.to_numpy(); lo=plot.CI_low.to_numpy(); hi=plot.CI_high.to_numpy()
ax[1].errorbar(diff,y,xerr=np.vstack([diff-lo,hi-diff]),fmt='o',color='#176b78',capsize=4)
ax[1].axvline(0,color='#666',ls='--'); ax[1].set(yticks=y,yticklabels=plot.index,xlabel='Union − baseline MAP@5',title='Descriptive identity-cluster bootstrap, 95%')
ax[1].invert_yaxis(); fig.savefig(HERE/'comparison.png',dpi=150); plt.close(fig)
def fmt(v): return '—' if v is None else f'{v:.6f}'
headline='The union passed the predeclared overall gate.' if j['gate_passed'] else 'The union did not pass the predeclared overall gate.'
summary=f"Overall MAP@5: {fmt(j['baseline']['map5'])} → {fmt(j['union']['map5'])}; difference {j['delta']:+.6f}."
coverage_note=f"The union newly includes the correct identity for {len(recovered)} query appearances; {diagnostics['recovered_in_final_top5']} of those reach the final top five. Candidate recall is unchanged in both the missing-side and small-gallery slices. Their score changes therefore arise from changed model ranking/unknown decisions rather than recovery of an omitted correct identity."
ci=j['slices']['all']['uncertainty']['identity_cluster_bootstrap_95pct']
interpretation=f'''{headline} {summary} The descriptive 95% identity-cluster bootstrap interval for the paired difference is [{ci[0]:+.6f}, {ci[1]:+.6f}]. Keep Dalvin's baseline as the default: this run does not establish a reliable improvement. The interval reflects these fixed models and validation identities; it does not measure retraining variance. The earlier EDA examined the labelled dataset, and episodes reuse queries, so this is not an untouched final evaluation or a leaderboard claim.'''
lines=['# Model 8: union shortlist — completed full-pipeline comparison','',interpretation,'',coverage_note,'',
       'Both variants were retrained through all three stages using 12 simulations per stage. The baseline keeps its original LDA/SIFT top 15; the union retains those and appends distinct raw top-15 identities. All other model settings and role-local SIFT evidence are shared. No nuisance transformation is used.','',
       '| Slice | Appearances | Unique queries | Baseline MAP@5 | Union MAP@5 | Difference |','| --- | ---: | ---: | ---: | ---: | ---: |']
for row in rows:
 lines.append(f"| {row['slice']} | {row['query_appearances']} | {row['unique_queries']} | {fmt(row['baseline_MAP5'])} | {fmt(row['union_MAP5'])} | {fmt(row['delta'])} |")
lines+=['','## Candidate coverage','',f"Average valid union candidates: {diagnostics['average_union_candidates']:.2f}; range {diagnostics['candidate_count_min']}–{diagnostics['candidate_count_max']}. Newly included correct candidates: {len(recovered)} query appearances ({recovered.id.nunique()} distinct photos). Of those, {diagnostics['recovered_in_final_top5']} finish in the final top five. See `slice_results.csv` for candidate recall by slice.",'','## Validation episodes','', '| Episode | Baseline MAP@5 | Union MAP@5 | Difference |','| --- | ---: | ---: | ---: |']
for row in episode_table.itertuples(): lines.append(f'| {row.episode} | {row.baseline:.6f} | {row.union:.6f} | {row.delta:+.6f} |')
lines+=['','## Baseline parity and limitations','','```json',json.dumps(audit,indent=2),'```','',
        'Synthetic tests verify feature parity with the preserved Dalvin baseline, exact union membership and removal of padded rows from fitting/scoring/ranking. Baseline checkpoint parity is audited separately on validation episode 22052. Bootstrap intervals cluster appearances by identity and stratify by species; targeted slices can contain few independent animals. The protocol uses identity-disjoint training stages and smaller galleries than final competition inference. The result compares complete pipeline components under this protocol, not the original full-data competition training. No submission or baseline replacement is performed.','',
        'Detailed per-query predictions, episode splits, configuration, source hashes and models are retained in `local_runs/union_2052`. Reproduce with `union_eval.py`, then `audit_baseline.py` and `summarize.py`.']
(HERE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
img=base64.b64encode((HERE/'comparison.png').read_bytes()).decode()
html=f'''<!doctype html><html><head><meta charset="utf-8"><title>Full-pipeline union shortlist experiment</title><style>body{{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 24px;color:#172334}}img{{width:100%}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}h1,h2{{color:#174c57}}</style></head><body><h1>Raw + LDA union: full-pipeline results</h1><p>{interpretation}</p><p>{coverage_note}</p><img src="data:image/png;base64,{img}"><h2>Results by slice</h2>{table.to_html(index=False,float_format=lambda v:f'{v:.6f}')}<h2>Episode results</h2>{episode_table.to_html(index=False,float_format=lambda v:f'{v:.6f}')}<h2>Species results</h2>{species_table.to_html(index=False,float_format=lambda v:f'{v:.6f}')}<h2>Candidate coverage and baseline audit</h2><pre>{json.dumps(diagnostics,indent=2)}</pre><h2>Method</h2><p>Three stages, 12 simulations each, separate training for both variants, 400-tree candidate scorers and 300-tree unknown detectors. Shared PCA/LDA and mutual RootSIFT evidence; unchanged unknown score rule and neighbour prediction stacking. Baseline has 15 candidates. Union has the baseline 15 plus distinct raw top-15 identities; masked padding cannot enter the learners or final predictions. Missing side: known left/right query without same-side true-identity gallery coverage. Small gallery: one to three true-identity gallery photos. No nuisance projection, submission generation or automatic baseline replacement.</p></body></html>'''
(HERE/'report.html').write_text(html,encoding='utf-8')
assert len(d)==j['query_appearances'] and not d[['baseline_rr5','union_rr5']].isna().any().any()
assert ((d.baseline_rr5>=0)&(d.baseline_rr5<=1)&(d.union_rr5>=0)&(d.union_rr5<=1)).all()
assert (known.union_candidate_present>=known.baseline_candidate_present).all()
for col in ['baseline_predictions','union_predictions']: assert d[col].map(lambda x:len(x.split())==len(set(x.split()))==5).all()
print(interpretation); print(table.to_string(index=False)); print(json.dumps(diagnostics,indent=2))
