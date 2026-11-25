"""Add interpretation and PCA/LDA diagnostics to the EDA report."""
from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent
t=pd.read_csv(p/'duplicate_groups.csv')
r=pd.read_csv(p/'retrieval_per_photo.csv')
counts=t.groupby('duplicate_group').size()
r=r.drop(columns=['duplicate_group'],errors='ignore').merge(t[['id','duplicate_group']],on='id',validate='one_to_one')
r['available_gallery_images']=r.identity_images-r.duplicate_group.map(counts)
r.to_csv(p/'retrieval_per_photo.csv',index=False)
v=r[r.group_excluded_eligible].copy()
macro=v.groupby(['species','identity']).group_excluded_rr5.mean().groupby(level=0).mean().rename('identity_macro_map5').reset_index()
macro.to_csv(p/'identity_macro_scores.csv',index=False)
v['gallery_size_bin']=pd.cut(v.available_gallery_images,[0,1,3,7,15,float('inf')],labels=['1','2-3','4-7','8-15','16+'])
s=pd.read_csv(p/'retrieval_slices.csv'); s=s[s['slice']!='gallery_size_bin']
rows=[]
for name,d in v.groupby('gallery_size_bin',observed=True):
 rows.append({'slice':'gallery_size_bin','value':str(name),'queries':len(d),'map5':d.group_excluded_rr5.mean(),'top1':(d.group_excluded_rank==1).mean(),'top15':(d.group_excluded_rank<=15).mean()})
s=pd.concat([s,pd.DataFrame(rows)],ignore_index=True); s.to_csv(p/'retrieval_slices.csv',index=False)
side=v[v.flank.isin(['left','right'])]
side_summary=side.groupby(['species','same_flank_available']).agg(queries=('id','size'),map5=('group_excluded_rr5','mean')).reset_index()
side_summary.to_csv(p/'side_coverage_summary.csv',index=False)
strata=v.assign(small_gallery=v.available_gallery_images<=3).groupby(['species','small_gallery','same_view_available']).agg(queries=('id','size'),map5=('group_excluded_rr5','mean')).reset_index()
strata.to_csv(p/'view_and_gallery_size.csv',index=False)
d=pd.read_csv(p/'lda_episode_queries.csv').merge(t[['id','viewpoint']],on='id')
d=d[d.viewpoint.str.contains('left|right',na=False)]
ls=d.groupby(['species','method','same_flank_available']).agg(queries=('id','size'),map5=('rr5','mean'),top15=('rank',lambda a:(a<=15).mean())).reset_index()
ls.to_csv(p/'lda_side_coverage_summary.csv',index=False)
a=pd.read_csv(p/'lda_episode_summary.csv')
union=pd.read_csv(p/'lda_episode_queries.csv').pivot(index=['species','id'],columns='method',values='rank')
union['lda15']=union.pca_lda<=15; union['raw15']=union.raw<=15; union['union15']=union.lda15|union.raw15
union_summary=union.groupby(level=0)[['lda15','raw15','union15']].mean().reset_index()
union_summary.to_csv(p/'candidate_union_recall.csv',index=False)
# Compact figure for the main findings; detailed charts remain in overview.png.
fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
micro=v.groupby('species').group_excluded_rr5.mean()
mf=macro.set_index('species').identity_macro_map5
pd.DataFrame({'Per photo':micro,'Per animal':mf}).plot.bar(ax=axes[0],color=['#176b78','#e7a342'])
axes[0].set(title='Large identities hide harder animals',ylabel='Raw cosine MAP@5',ylim=(0,1),xlabel=''); axes[0].tick_params(axis='x',rotation=0)
lf=ls[ls.method=='pca_lda'].pivot(index='species',columns='same_flank_available',values='map5').rename(columns={False:'Same side absent',True:'Same side present'})
lf.plot.bar(ax=axes[1],color=['#e7a342','#176b78'])
axes[1].set(title='Missing-side queries remain difficult',ylabel='PCA/LDA MAP@5',ylim=(0,1),xlabel=''); axes[1].tick_params(axis='x',rotation=0)
axes[1].legend(title=None)
sf=pd.read_csv(p/'sift_sample_summary.csv')
sf=sf[sf.category!='near_duplicate_positive'].pivot(index='category',columns='species',values='fraction_ge7')
sf.index=sf.index.str.replace('distinct_group_','').str.replace('_',' ').str.replace('flank','side')
sf.plot.barh(ax=axes[2],color=['#176b78','#9062a8'])
axes[2].set(title='Where sampled SIFT pairs verify',xlabel='Fraction with at least 7 inliers',ylabel='',xlim=(0,1))
fig.suptitle('Animal ReID: three improvement targets',fontsize=16,fontweight='bold')
fig.savefig(p/'findings.png',dpi=150); plt.close(fig)
insights='''<h2>What this suggests we should improve</h2>
<ol><li><b>Prioritize missing-side and missing-view coverage.</b> Among left/right query photos, raw matching falls dramatically when the correct identity has no same-side gallery image. This remains difficult with PCA/LDA in the separate episode. An experiment should focus on same-view, same-side and opposite-side evidence separately, and make uncertainty depend on gallery coverage. Gallery size and animal difficulty are confounders; this association does not prove that adding a viewpoint feature will improve the complete model.</li>
<li><b>Test candidate expansion on difficult queries.</b> In the single PCA/LDA episode, correct-identity recall among 15 candidates is 97.34% for hyenas and 95.49% for leopards. A ranker cannot recover an identity outside its shortlist. Compare the current 15 candidates with an adaptive 30–50, or a union from raw and LDA retrieval, on a reserved split. These figures omit Dalvin's SIFT shortlist boost, so they do not measure his actual shortlist recall.</li>
<li><b>Evaluate small galleries explicitly.</b> Many identities have very few photos, whereas a few dominate the image count. Report both per-image and per-identity metrics, plus slices by usable gallery size. The same-view disadvantage persists within small/large gallery strata. Balance training episodes by identity and coverage as a hypothesis to test.</li>
<li><b>Improve unknown-animal decisions with coverage-aware evidence.</b> Across six raw-cosine open-set episodes, accepting about 95% of known queries also accepts 61–84% of unknown queries. A single similarity threshold is inadequate here. Compare the complete baseline's errors across species, viewpoints, gallery sizes and SIFT-support levels before changing its unknown detector.</li>
<li><b>Keep SIFT's role conditional on viewpoint.</b> Inspect the sampled verification table below before treating low inlier counts as evidence for a new animal. A different side or pose can prevent geometric alignment. SIFT sample sizes are small and deliberately selected; test any proposed scoring change in the full pipeline.</li>
<li><b>Account for train/test composition.</b> Hyenas represent 26.69% of training photos and 43.60% of test photos. Use species-specific reporting and calibration checks; do not infer test identity prevalence from this shift.</li></ol>
<p><b>Suggested first experiment:</b> preserve the full Dalvin baseline and compare an adaptive union shortlist against its current shortlist, concentrating analysis on missing-side and small-gallery queries. Measure correct-identity shortlist recall, known-query MAP@5, unknown-query MAP@5 and overall MAP@5 on a newly reserved evaluation split. The present EDA has already examined these data, so it must not be presented as an untouched final evaluation.</p>
<p><b>Scope:</b> these are measured diagnostic weaknesses and proposed experiments. No improvement to the complete Dalvin pipeline or competition score is established.</p>'''
extra='<h2>Identity imbalance</h2><p>204 of 345 leopard identities and 42 of 205 hyena identities have at most three photos. Equal weighting of eligible identities gives substantially lower raw MAP@5 than weighting each query photo equally. Singleton-only identities cannot enter this retrieval average. The feature matrix is finite and nonzero; metadata has missing viewpoints, which should be represented explicitly. Exactly two training rows share an identical feature vector, both labelled LEO0032; there are no exact feature duplicates across train and test.</p>'+macro.to_html(index=False,float_format=lambda v:f'{v:.4f}')
extra+='<h2>Left/right queries: same-side coverage</h2>'+side_summary.to_html(index=False,float_format=lambda v:f'{v:.4f}')
extra+='<h2>Same-view coverage within gallery-size strata</h2>'+strata.to_html(index=False,float_format=lambda v:f'{v:.4f}')
extra+='<h2>PCA/LDA retrieval diagnostic</h2><p>Seed 2059; 25% of identities held out as unknown, remaining identities split by whole near-duplicate groups into gallery/query. PCA sees gallery and unlabelled queries, LDA uses gallery labels. Scores below cover known queries only. LDA warns for singleton gallery identities; shrinkage is retained as in the baseline. This tests the embedding component, without SIFT boosting or learned reranking. One descriptive episode, with no claim of statistical significance.</p>'
extra+=a.to_html(index=False,float_format=lambda v:f'{v:.4f}')+ls.to_html(index=False,float_format=lambda v:f'{v:.4f}')
extra+='<h2>Candidate union diagnostic</h2><p>Union of the top 15 raw identities and top 15 PCA/LDA identities (at most 30 identities). This recovers two additional hyena queries and ten additional leopard queries in the same diagnostic episode. It measures candidate inclusion only, without SIFT shortlist boosting, final ranking or unknown detection.</p>'+union_summary.to_html(index=False,float_format=lambda v:f'{v:.4f}')
text=(p/'report.html').read_text(encoding='utf-8')
start=text.find('<h2>What this suggests we should improve</h2>')
if start>=0:
 end=text.index('<h2>Dataset composition</h2>',start)
 text=text[:start]+text[end:]
text=text.replace('<h2>Dataset composition</h2>',insights+extra+'<h2>Dataset composition</h2>',1)
sift=pd.read_csv(p/'sift_per_photo.csv')
sift_limits=sift.assign(empty=sift.keypoints==0,too_few=sift.keypoints<4).groupby(['split','species'])[['empty','too_few']].sum().reset_index()
text=text.replace('<h2>SIFT coverage</h2>','<h2>SIFT availability limits</h2><p>All 9,954 images have descriptor and coordinate files with consistent shapes. However, 96 have no keypoints and 155 have fewer than four, preventing the preserved baseline from attempting geometric verification for any pair involving those images. Missing evidence must not be treated as a verified non-match. Descriptor dimension is 128 throughout. Training metadata has 136 missing viewpoints and test metadata has 42.</p>'+sift_limits.to_html(index=False)+'<h2>SIFT coverage</h2>',1) if '<h2>SIFT availability limits</h2>' not in text else text
(p/'report.html').write_text(text,encoding='utf-8')
j=json.loads((p/'summary.json').read_text()); j['slices']=s.to_dict('records'); j['lda_episode']=a.to_dict('records'); j['left_right_coverage']=side_summary.to_dict('records'); j['identity_macro']=macro.to_dict('records')
j['sift_availability_limits']=sift_limits.to_dict('records'); j['candidate_union_recall']=union_summary.to_dict('records')
(p/'summary.json').write_text(json.dumps(j,indent=2))
assert len(r)==7163 and r.id.is_unique
assert int(pd.read_csv(p/'composition.csv').images.sum())==9954
assert (p/'overview.png').stat().st_size>10000
assert text.count('<html>')==1 and '</html>' in text
print('Report finalized; row counts, unique IDs, image and HTML checked.')
print(json.dumps({k:j[k] for k in ['missing_metadata','missing_train_identity','identity_distribution','exact_feature_duplicate_rows','sift','sift_sample']},indent=2))
