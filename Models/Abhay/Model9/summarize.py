"""Summarize the frozen development-selected coverage experiment and ablations."""
import sys,json,base64
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coverage_heads import adjust,predictions,rr5,RANK_NAMES
HERE=Path(__file__).resolve().parent; out=HERE/'local_runs/coverage_2052'
j=json.loads((out/'metrics.json').read_text()); selection=j['selection']
d=pd.read_csv(out/'validation_predictions.csv')
search=pd.read_csv(out/'development_strength_search.csv')
names={'all':'Overall','known':'Known animals','unknown':'Unknown animals','missing_side':'Missing side',
       'small_gallery_1to3':'Small gallery (1–3 photos)','missing_side_small_gallery':'Missing side + small gallery',
       'same_side_available':'Same side available','gallery_4plus':'4+ gallery photos'}
rows=[]
for key,s in j['slices'].items():
    row={'slice':names[key],'appearances':s['appearances'],'unique_queries':s['unique_queries'],'identities':s['identities']}
    for method,v in s['methods'].items(): row[method]=v['map5']
    row.update(delta=s['methods']['coverage']['delta'],ci_low=s['coverage_delta_ci95'][0] if s['coverage_delta_ci95'] else None,ci_high=s['coverage_delta_ci95'][1] if s['coverage_delta_ci95'] else None)
    rows.append(row)
table=pd.DataFrame(rows); table.to_csv(HERE/'slice_results.csv',index=False)
episodes=pd.DataFrame([{'episode':seed,**{m:v['map5'] for m,v in result.items()},'delta':result['coverage']['delta']} for seed,result in j['episodes'].items()])
episodes.to_csv(HERE/'episode_results.csv',index=False)
species=pd.DataFrame([{'species':sp,**{m:v['map5'] for m,v in result.items()},'delta':result['coverage']['delta']} for sp,result in j['species'].items()])
species.to_csv(HERE/'species_results.csv',index=False)
search.to_csv(HERE/'development_strength_search.csv',index=False)
bundle=joblib.load(out/'validation_bundle.joblib'); heads=joblib.load(out/'coverage_heads.joblib')
rs=selection.get('diagnostic_rank_strength',selection['rank_strength']); us=selection.get('diagnostic_unknown_strength',selection['unknown_strength'])
pc,pb=adjust(bundle,heads['rank'],heads['unknown'],rs,us)
truth=bundle['rows'].truth.to_numpy(); known=truth!='new_individual'
def conditional_score(prob):
    pred=np.take_along_axis(bundle['candidates'],np.argsort(-prob,axis=1)[:,:5],axis=1)
    return rr5(pred,truth)
cb=conditional_score(bundle['pc']); cc=conditional_score(pc)
diag={'selected_strengths':selection,'evaluated_strengths':{'rank':rs,'unknown':us},
      'baseline_parity':j['baseline_parity'],
      'conditional_known_identity_map5':{'baseline':float(cb[known].mean()),'coverage':float(cc[known].mean())},
      'query_decisions':{},'rank_head':{'initial_loss':heads['rank'].initial_loss,'final_loss':heads['rank'].final_loss,
                                     'coefficients':dict(zip(RANK_NAMES,heads['rank'].coef.tolist()))},
      'unknown_head':{'initial_loss':heads['unknown'].initial_loss,'final_loss':heads['unknown'].final_loss},
      'max_observed_rank_correction':float(np.abs(heads['rank'].correction(bundle['rank_features'])).max()),
      'max_observed_unknown_logodds_correction':float(np.abs(heads['unknown'].correction(bundle['unknown_features'])).max())}
for m in ('baseline','rank_only','unknown_only','coverage'):
    first_new=d[m+'_predictions'].str.split().str[0]=='new_individual'
    diag['query_decisions'][m]={'known_false_new_top1':float(first_new[~d.unknown].mean()),
                              'unknown_new_top1':float(first_new[d.unknown].mean()),
                              'mean_known_pnew':float(d.loc[~d.unknown,m+'_pnew'].mean()),
                              'mean_unknown_pnew':float(d.loc[d.unknown,m+'_pnew'].mean())}
(HERE/'diagnostics.json').write_text(json.dumps(diag,indent=2))
delta=j['methods']['coverage']['delta']; ci=j['slices']['all']['coverage_delta_ci95']
if selection['primary_is_baseline']:
    verdict='Development cross-validation selected the unchanged baseline. The full-strength correction below is diagnostic only and is not promoted using validation results.'
elif not j['gate_passed']:
    verdict='The development-selected correction did not pass the validation gate. Keep Dalvin as the default.'
elif ci[0]<=0:
    verdict='The development-selected correction passed the mean-score gate, but its uncertainty interval includes zero. Treat it as promising rather than an established improvement.'
else:
    verdict='The development-selected correction passed the predeclared validation gate with a positive descriptive uncertainty interval. Further independent validation is still needed.'
summary=f"Full-pipeline MAP@5: {j['methods']['baseline']['map5']:.6f} → {j['methods']['coverage']['map5']:.6f}, difference {delta:+.6f}; descriptive 95% paired identity-cluster bootstrap interval [{ci[0]:+.6f}, {ci[1]:+.6f}]."
limitations='The heads were fitted on previously unused calibration identities; the original three-stage stack is frozen. Strengths were selected using four identity-disjoint development folds before validation. Validation identities and the labelled dataset have been examined before, and episodes reuse photos; this is not an untouched holdout or a Kaggle leaderboard result. The experimental heads use additional supervised calibration examples, so this is a test of that addition, not an equal-supervised-data architecture comparison.'
rank_trials=search[(search.rank_strength>0)&(search.unknown_strength==0)]
development_note=f"Development used 5,587 query appearances from 897 distinct photos and 109 query identities across 12 episodes. Ranking-only corrections at strengths 0.25, 0.5 and 1 improved the targeted development slices but reduced overall development MAP@5 by {abs(rank_trials.all_delta.max()):.6f} to {abs(rank_trials.all_delta.min()):.6f}; they failed the selection constraint. The selected ranking strength is therefore zero, so the rank-only validation ablation is exactly the baseline. The selected unknown strength is {selection['unknown_strength']}; it is evaluated without further validation tuning."
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
plot=table.iloc[:6].set_index('slice')
plot[['baseline','coverage']].rename(columns={'baseline':'Dalvin','coverage':'Coverage correction'}).plot.barh(ax=axes[0],color=['#176b78','#e7a342'])
axes[0].invert_yaxis(); axes[0].set(xlim=(0,1),xlabel='Full-pipeline MAP@5',ylabel='',title='Frozen development-selected comparison' if not selection['primary_is_baseline'] else 'Full-strength diagnostic comparison')
x=plot.delta.to_numpy(); lo=plot.ci_low.to_numpy(); hi=plot.ci_high.to_numpy(); yy=np.arange(len(plot))
axes[1].errorbar(x,yy,xerr=np.maximum(np.vstack([x-lo,hi-x]),0),fmt='o',color='#176b78',capsize=4)
axes[1].axvline(0,color='#666',ls='--'); axes[1].set(yticks=yy,yticklabels=plot.index,xlabel='Correction − Dalvin MAP@5',title='Descriptive 95% identity-cluster bootstrap')
axes[1].invert_yaxis(); fig.savefig(HERE/'comparison.png',dpi=150); plt.close(fig)
lines=['# Model 9: coverage-aware ranking and unknown detection','',verdict,'',summary,'',limitations,'',development_note,'',
       'Ranking strength selected on development: '+str(selection['rank_strength'])+'. Unknown strength: '+str(selection['unknown_strength'])+'.','',
       '| Slice | Appearances | Unique photos | Dalvin | Rank only | Unknown only | Combined | Difference |',
       '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in rows:
 lines.append(f"| {r['slice']} | {r['appearances']} | {r['unique_queries']} | {r['baseline']:.6f} | {r['rank_only']:.6f} | {r['unknown_only']:.6f} | {r['coverage']:.6f} | {r['delta']:+.6f} |")
lines+=['','Both heads use observable candidate evidence at inference. The ranking correction is bounded to ±1 log-score; the unknown correction to ±log(2) log odds before blending. The shortlist and all baseline feature extraction, SIFT, three-stage stacking and final ranking rule remain unchanged.','',
        'The baseline exactly reproduces the prior run across all '+str(j['baseline_parity']['queries'])+' validation query appearances. Tests check correction bounds, preservation at zero strength and exclusion of truth columns from inference features.','',
        'Detailed configuration, development strength search, identity folds, fitted heads, source hashes and per-query predictions are retained in `local_runs/coverage_2052`. See `README.md` for the frozen selection procedure.']
(HERE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
img=base64.b64encode((HERE/'comparison.png').read_bytes()).decode()
body=f'''<!doctype html><html><head><meta charset="utf-8"><title>Coverage-aware ranking and unknown detection</title><style>body{{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 24px;color:#172334}}img{{width:100%}}table{{border-collapse:collapse;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}h1,h2{{color:#174c57}}</style></head><body><h1>Coverage-aware ranking and unknown detection</h1><p><b>{verdict}</b></p><p>{summary}</p><p>{limitations}</p><img src="data:image/png;base64,{img}"><h2>Target slices and ablations</h2>{table.to_html(index=False,float_format=lambda a:f'{a:.6f}')}<h2>Development selection</h2><pre>{json.dumps(selection,indent=2)}</pre>{search.to_html(index=False,float_format=lambda a:f'{a:.6f}')}<h2>Validation episodes</h2>{episodes.to_html(index=False,float_format=lambda a:f'{a:.6f}')}<h2>Species</h2>{species.to_html(index=False,float_format=lambda a:f'{a:.6f}')}<h2>Decision diagnostics</h2><pre>{json.dumps(diag,indent=2)}</pre></body></html>'''
body=body.replace('<h2>Development selection</h2>','<h2>Development selection</h2><p>'+development_note+'</p>')
(HERE/'report.html').write_text(body,encoding='utf-8')
assert len(d)==j['slices']['all']['appearances']
assert ((d.coverage_rr5>=0)&(d.coverage_rr5<=1)).all()
assert all(d[m+'_predictions'].map(lambda a:len(a.split())==len(set(a.split()))==5).all() for m in ('baseline','rank_only','unknown_only','coverage'))
print(verdict); print(summary); print(table.to_string(index=False)); print(json.dumps(diag,indent=2))
