"""Publish the completed three-way difficult-episode comparison."""
import sys,json,base64
if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; OUT=HERE/'local_runs/difficult_2052'
METHODS=['baseline','difficult25','difficult50']
LABELS={'baseline':'Dalvin baseline','difficult25':'25% difficult','difficult50':'50% difficult'}
j=json.loads((OUT/'metrics.json').read_text()); selection=json.loads((OUT/'selection.json').read_text())
d=pd.read_csv(OUT/'validation_predictions.csv'); dev=pd.read_csv(OUT/'development_predictions.csv')
slice_names={'all':'Overall','known':'Known animals','unknown':'Unknown animals','missing_side':'Missing side',
             'small_gallery_1to3':'Small gallery (1–3)','missing_side_small_gallery':'Missing side + small gallery',
             'same_side_available':'Same side available','gallery_4plus':'4+ gallery photos'}
rows=[]
for name,s in j['slices'].items():
    row={'slice':slice_names[name],'appearances':s['appearances'],'unique_queries':s['unique_queries'],'identities':s['identities']}
    for m in METHODS: row[m]=s['methods'][m]['map5']
    for m in METHODS[1:]:
        row[m+'_delta']=s['methods'][m]['delta']; row[m+'_CI_low']=s['ci95'][m][0]; row[m+'_CI_high']=s['ci95'][m][1]
    rows.append(row)
table=pd.DataFrame(rows); table.to_csv(HERE/'slice_results.csv',index=False)
eps=pd.DataFrame([{'episode':seed,**{m:stats[m]['map5'] for m in METHODS},**{m+'_delta':stats[m]['delta'] for m in METHODS[1:]}} for seed,stats in j['episodes'].items()])
eps.to_csv(HERE/'episode_results.csv',index=False)
species=pd.DataFrame([{'species':sp,**{m:stats[m]['map5'] for m in METHODS},**{m+'_delta':stats[m]['delta'] for m in METHODS[1:]}} for sp,stats in j['species'].items()])
species.to_csv(HERE/'species_results.csv',index=False)
devrows=[]
for name,results in selection['development'].items():
    devrows.append({'slice':slice_names[name],**{m:results[m]['map5'] for m in METHODS},**{m+'_delta':results[m]['delta'] for m in METHODS[1:]}})
devtable=pd.DataFrame(devrows); devtable.to_csv(HERE/'development_results.csv',index=False)
training=pd.read_csv(OUT/'training_episode_diagnostics.csv')
schedule=pd.read_csv(OUT/'training_method_schedule.csv')
merged=schedule.merge(training,on=['stage','episode','kind'],validate='many_to_one')
realized=merged.groupby('method').agg(episodes=('episode','size'),difficult_episodes=('kind',lambda a:(a=='difficult').sum()),
                                  query_appearances=('queries','sum'),known=('known','sum'),unknown=('unknown','sum'),
                                  missing_side=('missing_side','sum'),small_gallery=('small_gallery_1to3','sum'),
                                  mean_gallery_images=('gallery','mean')).reset_index()
realized.to_csv(HERE/'training_distribution.csv',index=False)
assert (realized.episodes==36).all()
assert realized.set_index('method').difficult_episodes.to_dict()=={'baseline':0,'difficult25':9,'difficult50':18}
assert realized.known.nunique()==realized.unknown.nunique()==realized.query_appearances.nunique()==1

# Diagnostic decomposition: conditional identity ordering versus insertion of new_individual.
diagnostics=[]
for name,mask in {'known':~d.unknown,'missing_side':(~d.unknown)&d.side_query&d.missing_side,
                  'small_gallery':(~d.unknown)&d.true_gallery_images.between(1,3)}.items():
    part=d[mask]
    for m in METHODS:
        firstnew=part[m+'_predictions'].str.split().str[0]=='new_individual'
        diagnostics.append({'slice':name,'method':m,'appearances':len(part),
                            'full_map5':float(part[m+'_rr5'].mean()),'conditional_map5':float(part[m+'_conditional_rr5'].mean()),
                            'new_label_hurts_true_rank':int((part[m+'_conditional_rr5']>part[m+'_rr5']).sum()),
                            'known_false_new_top1':float(firstnew.mean())})
diag=pd.DataFrame(diagnostics); diag.to_csv(HERE/'decision_diagnostics.csv',index=False)

selected=j['selection']['method']; candidate=max(METHODS[1:],key=lambda m:j['methods'][m]['map5'])
if selected=='baseline':
    verdict='Development selected the unchanged baseline. Neither experimental fraction qualified under the development constraints; validation results are reported for comparison, not used to change that choice.'
elif j['gates'][selected]:
    ci=j['slices']['all']['ci95'][selected]
    verdict=('The development-selected mixture passed the point-score validation gate. '+
             ('Its descriptive uncertainty interval is positive.' if ci[0]>0 else 'Its uncertainty interval includes zero, so the improvement is not yet established.'))
else:
    verdict='The development-selected mixture did not pass the validation gate. Keep the original baseline as the default.'
summary='Overall MAP@5: '+', '.join(f'{LABELS[m]} {j["methods"][m]["map5"]:.6f}' for m in METHODS)+'.'
limits='The full pipeline was retrained for all three methods, with identical model settings and 36 total episodes each. Development uses separate identities and selects the mixture before validation. These validation identities and the labelled dataset have been inspected previously. This is one outer identity allocation with repeated photos across episodes; bootstrap intervals do not establish retraining robustness or a Kaggle gain.'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
plot=table.iloc[:6].set_index('slice')
plot[METHODS].rename(columns=LABELS).plot.barh(ax=axes[0],color=['#176b78','#e7a342','#8e67a8'])
axes[0].invert_yaxis(); axes[0].set(xlim=(0,1),xlabel='Full-pipeline MAP@5',ylabel='',title='Matched retraining comparison')
y=np.arange(len(plot))
for m,color,offset in [('difficult25','#e7a342',-.12),('difficult50','#8e67a8',.12)]:
    delta=plot[m+'_delta'].to_numpy(); lo=plot[m+'_CI_low'].to_numpy(); hi=plot[m+'_CI_high'].to_numpy()
    axes[1].errorbar(delta,y+offset,xerr=np.maximum(np.vstack([delta-lo,hi-delta]),0),fmt='o',capsize=4,color=color,label=LABELS[m])
axes[1].axvline(0,ls='--',color='#666'); axes[1].set(yticks=y,yticklabels=plot.index,xlabel='Mixture − baseline MAP@5',title='Descriptive 95% identity-cluster bootstrap')
axes[1].invert_yaxis(); axes[1].legend(); fig.savefig(HERE/'comparison.png',dpi=150); plt.close(fig)
lines=['# Model 10: difficult-episode full-pipeline retraining','',verdict,'',summary,'',limits,'',
       'Development-selected method: **'+LABELS[selected]+'**.','',
       '| Slice | Appearances | Unique photos | Baseline | 25% | Difference | 50% | Difference |',
       '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
for r in rows:
    lines.append(f"| {r['slice']} | {r['appearances']} | {r['unique_queries']} | {r['baseline']:.6f} | {r['difficult25']:.6f} | {r['difficult25_delta']:+.6f} | {r['difficult50']:.6f} | {r['difficult50_delta']:+.6f} |")
lines+=['','## Reliability and parity','']
for m in METHODS[1:]:
    ci=j['slices']['all']['ci95'][m]
    lines.append(f"- {LABELS[m]} overall paired delta interval: [{ci[0]:+.6f}, {ci[1]:+.6f}]; validation gate passed: {j['gates'][m]}.")
lines+=['','The retrained baseline reproduces all '+str(j['baseline_parity']['query_appearances'])+' previous validation top-five predictions exactly. Sampler tests check nested proportions, class/identity conservation, deterministic behavior, gallery subset safety and harder unknown selection. Realized distributions confirm equal known/unknown query counts and exactly 9 or 18 difficult episodes out of 36.','',
        'No original model, competition submission or earlier experiment was overwritten. All three trained stacks, source hashes, configuration, identity roles, exact episode manifests, development selection and per-query predictions are saved in `local_runs/difficult_2052`. See `README.md` to reproduce.']
(HERE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
img=base64.b64encode((HERE/'comparison.png').read_bytes()).decode()
html=f'''<!doctype html><html><head><meta charset="utf-8"><title>Difficult episode retraining: 25% and 50%</title><style>body{{font:16px system-ui;max-width:1400px;margin:40px auto;padding:0 24px;color:#172334}}img{{width:100%}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}h1,h2{{color:#174c57}}</style></head><body><h1>Full-pipeline retraining with difficult episodes</h1><p><b>{verdict}</b></p><p>{summary}</p><p>{limits}</p><p>Development selection: {LABELS[selected]}.</p><img src="data:image/png;base64,{img}"><h2>Validation slices</h2>{table.to_html(index=False,float_format=lambda x:f'{x:.6f}')}<h2>Validation episodes</h2>{eps.to_html(index=False,float_format=lambda x:f'{x:.6f}')}<h2>Species</h2>{species.to_html(index=False,float_format=lambda x:f'{x:.6f}')}<h2>Development selection results</h2>{devtable.to_html(index=False,float_format=lambda x:f'{x:.6f}')}<h2>Realized training distributions</h2>{realized.to_html(index=False,float_format=lambda x:f'{x:.2f}')}<h2>Ranking and unknown-decision diagnostics</h2><p>Conditional MAP@5 removes the new-individual option only for analysis; it is not a deployable score for a known/unknown mixture.</p>{diag.to_html(index=False,float_format=lambda x:f'{x:.6f}')}<h2>Audit</h2><pre>{json.dumps(j['baseline_parity'],indent=2)}</pre></body></html>'''
(HERE/'report.html').write_text(html,encoding='utf-8')
assert len(d)==j['baseline_parity']['query_appearances']
assert all(d[m+'_predictions'].map(lambda x:len(x.split())==len(set(x.split()))==5).all() for m in METHODS)
print(verdict); print(summary); print(table.to_string(index=False)); print(realized.to_string(index=False)); print(diag.to_string(index=False))
