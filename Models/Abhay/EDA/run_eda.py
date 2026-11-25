"""Read-only source-data EDA. Run from any directory; outputs beside this script."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import json, hashlib, base64, html
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import roc_auc_score
import cv2

OUT=Path(__file__).resolve().parent
ROOT=Path.home()/'Downloads'
DATA=ROOT/'reid_student_data_main'
rng=np.random.default_rng(2059)
train=pd.read_csv(DATA/'train.csv'); test=pd.read_csv(DATA/'test.csv')
meta=pd.concat([train.assign(split='train'),test.assign(split='test')],ignore_index=True)
ids=pd.read_csv(DATA/'miewid_ids.csv')
raw=np.load(DATA/'miewid_features.npy')
assert meta.id.is_unique and ids.id.is_unique
ix=pd.Index(ids.id).get_indexer(meta.id)
assert (ix>=0).all() and len(ids)==len(meta)
raw=raw[ix]; norms=np.linalg.norm(raw,axis=1)
assert np.isfinite(raw).all() and (norms>0).all()
x=raw/norms[:,None]; nt=len(train)
y=train.identity.to_numpy(); sp=meta.species.to_numpy()
view=meta.viewpoint.to_numpy()
flank=np.array(['left' if 'left' in str(v) else 'right' if 'right' in str(v) else 'other' for v in view])
summary={'date':'2026-11-25','seed':2059,'data_path':str(DATA),
         'features':{'shape':list(raw.shape),'dtype':str(raw.dtype),'norm_min':float(norms.min()),'norm_max':float(norms.max()),'finite':True},
         'missing_metadata':meta.drop(columns='identity').isna().sum().to_dict(),
         'missing_train_identity':int(train.identity.isna().sum())}
composition=meta.groupby(['split','species']).size().rename('images').reset_index()
composition.to_csv(OUT/'composition.csv',index=False)
vp=pd.crosstab([meta.split,meta.species],meta.viewpoint)
vp.to_csv(OUT/'viewpoint_counts.csv')
animal=train.groupby(['species','identity']).agg(images=('id','size'),viewpoints=('viewpoint','nunique'))
animal.to_csv(OUT/'identity_counts.csv')
summary['composition']=composition.to_dict('records')
summary['identity_distribution']={s:{'identities':len(d),'min':int(d.images.min()),'median':float(d.images.median()),'max':int(d.images.max()),'single_photo':int((d.images==1).sum()),'up_to_three_photos':int((d.images<=3).sum())} for s,d in animal.groupby(level=0)}
hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in raw]
dups=meta.assign(feature_hash=hashes)
dups=dups[dups.feature_hash.duplicated(keep=False)]
dups.to_csv(OUT/'exact_feature_duplicates.csv',index=False)
summary['exact_feature_duplicate_rows']=len(dups)

# Group near duplicates within each labelled identity using connected components.
group=np.full(nt,-1,int); next_group=0
for _,d in train.groupby('identity',sort=True):
    a=d.index.to_numpy(); sim=x[a]@x[a].T
    _,g=connected_components(csr_matrix(sim>=.97),directed=False)
    group[a]=g+next_group; next_group+=g.max()+1
train.assign(duplicate_group=group).to_csv(OUT/'duplicate_groups.csv',index=False)
ng=train.assign(group=group).groupby(['species','identity']).group.nunique()
summary['near_duplicate_groups']={s:{'groups':int(d.sum()),'identities_with_one_group':int((d==1).sum()),'identities':len(d)} for s,d in ng.groupby(level=0)}
print('Metadata and near-duplicate groups finished',flush=True)

# All training photos act as queries; remove self or the entire near-duplicate group.
# This is a descriptive retrieval stress test, not an untouched validation score.
records=[]; test_records=[]; pair_pool={}; similarities={}
for species in ('hyena','leopard'):
    gi=np.flatnonzero(sp[:nt]==species); names,inv=np.unique(y[gi],return_inverse=True)
    sim=x[gi]@x[gi].T; similarities[species]=(gi,sim)
    for j,a in enumerate(gi):
        ss=sim[j].copy(); ss[j]=-np.inf
        no_group=ss.copy(); no_group[group[gi]==group[a]]=-np.inf
        true=names==y[a]; own=y[gi]==y[a]
        row={'id':meta.id.iloc[a],'species':species,'identity':y[a],'viewpoint':view[a], 'flank':flank[a],
             'identity_images':int(own.sum()),'available_gallery_images':int((own&(group[gi]!=group[a])).sum()),'identity_groups':int(len(np.unique(group[gi[own]])))}
        for label,score in [('self_excluded',ss),('group_excluded',no_group)]:
            best=np.full(len(names),-np.inf); np.maximum.at(best,inv,score)
            available=np.isfinite(best[true][0]); rank=1+int((best>best[true][0]).sum()) if available else None
            row[label+'_eligible']=available
            row[label+'_rank']=rank
            row[label+'_rr5']=1/rank if rank is not None and rank<=5 else 0.
            if label=='group_excluded' and available:
                wrong=np.where(~own,no_group,-np.inf); w=int(np.argmax(wrong)); p=int(np.argmax(np.where(own,no_group,-np.inf)))
                row.update(best_true=float(no_group[p]),best_wrong=float(wrong[w]),margin=float(no_group[p]-wrong[w]),
                           best_true_id=meta.id.iloc[gi[p]],best_wrong_id=meta.id.iloc[gi[w]],
                           predicted_identity=y[gi[int(np.argmax(no_group))]],
                           same_flank_available=bool(((group[gi]!=group[a])&own&(flank[gi]==flank[a])&(flank[a]!='other')).any()),
                           same_view_available=bool(((group[gi]!=group[a])&own&(view[gi]==view[a])).any()))
        records.append(row)
        for category,mask in {
            'near_duplicate_positive':own&(group[gi]==group[a])&(gi!=a),
            'distinct_group_same_flank':own&(group[gi]!=group[a])&(flank[gi]==flank[a])&(flank[a]!='other'),
            'distinct_group_opposite_flank':own&(group[gi]!=group[a])&(flank[gi]!=flank[a])&(flank[gi]!='other')&(flank[a]!='other'),
            'hard_negative':~own}.items():
            if mask.any():
                b=gi[np.argmax(np.where(mask,sim[j],-np.inf))]
                pair_pool.setdefault((species,category),[]).append((int(a),int(b),float(sim[j,np.where(gi==b)[0][0]])))
    # Nearest labelled photo for test distribution diagnostics only.
    ti=np.flatnonzero((sp==species)&(np.arange(len(meta))>=nt))
    ts=x[ti]@x[gi].T
    for a,row in zip(ti,ts):
        b=gi[int(row.argmax())]
        test_records.append({'id':meta.id.iloc[a],'species':species,'viewpoint':view[a], 'max_train_cosine':float(row.max()),'nearest_train_id':meta.id.iloc[b], 'nearest_train_identity':y[b], 'same_view_nearest':bool(view[a]==view[b])})
retr=pd.DataFrame(records); retr.to_csv(OUT/'retrieval_per_photo.csv',index=False)
pd.DataFrame(test_records).to_csv(OUT/'test_nearest_train.csv',index=False)
agg=[]
for species in ('hyena','leopard','all'):
    d=retr if species=='all' else retr[retr.species==species]
    for mode in ('self_excluded','group_excluded'):
        a=d[d[mode+'_eligible']]
        agg.append({'species':species,'mode':mode,'queries':len(a),'excluded':len(d)-len(a),'map5':float(a[mode+'_rr5'].mean()),'top1':float((a[mode+'_rank']==1).mean()),'top5':float((a[mode+'_rank']<=5).mean())})
pd.DataFrame(agg).to_csv(OUT/'retrieval_summary.csv',index=False); summary['retrieval']=agg
valid=retr[retr.group_excluded_eligible].copy()
valid['gallery_size_bin']=pd.cut(valid.available_gallery_images,[0,1,3,7,15,np.inf],labels=['1','2-3','4-7','8-15','16+'])
slices=[]
for key in ['species','viewpoint','same_view_available','same_flank_available','gallery_size_bin']:
    for name,d in valid.groupby(key,observed=True):
        slices.append({'slice':key,'value':str(name),'queries':len(d),'map5':float(d.group_excluded_rr5.mean()),'top1':float((d.group_excluded_rank==1).mean()),'top15':float((d.group_excluded_rank<=15).mean())})
pd.DataFrame(slices).to_csv(OUT/'retrieval_slices.csv',index=False); summary['slices']=slices
valid.sort_values('margin').head(100).to_csv(OUT/'hardest_queries.csv',index=False)
# Matched query subset separates group exclusion from changes in query eligibility.
summary['matched_query_subset']={'queries':len(valid),'self_excluded_map5':float(valid.self_excluded_rr5.mean()),'group_excluded_map5':float(valid.group_excluded_rr5.mean()),'raw_top15_recall':float((valid.group_excluded_rank<=15).mean())}
print('Retrieval diagnostics finished',flush=True)

# Small simulated open-set episodes: hold out 25% identities entirely; keep groups intact.
episodes=[]; open_rows=[]
for seed in (2059,2060,2061):
    rr=np.random.default_rng(seed)
    for species,(gi,sim) in similarities.items():
        names=np.unique(y[gi]); rr.shuffle(names); unknown=set(names[:max(1,len(names)//4)])
        gallery=[]; known_q=[]; unknown_q=[]
        for name in names:
            a=gi[y[gi]==name]
            if name in unknown:
                unknown_q.extend(a); continue
            gs=np.unique(group[a])
            if len(gs)<2: gallery.extend(a); continue
            qg=rr.choice(gs,size=min(max(1,round(.3*len(gs))),len(gs)-1),replace=False)
            known_q.extend(a[np.isin(group[a],qg)]); gallery.extend(a[~np.isin(group[a],qg)])
        gallery=np.array(gallery); known_q=np.array(known_q); unknown_q=np.array(unknown_q)
        def score(q): return (x[q]@x[gallery].T).max(1)
        ks=score(known_q); us=score(unknown_q); threshold=np.quantile(ks,.05)
        episodes.append({'seed':seed,'species':species,'gallery_images':len(gallery),'known_queries':len(ks),'unknown_queries':len(us),'auc_unknown':float(roc_auc_score(np.r_[np.zeros(len(ks)),np.ones(len(us))],-np.r_[ks,us])), 'threshold_for_95pct_known_acceptance':float(threshold),'unknown_false_accept_rate':float((us>=threshold).mean())})
        for q,scores,unk in [(known_q,ks,False),(unknown_q,us,True)]:
            open_rows.extend({'seed':seed,'species':species,'id':meta.id.iloc[a],'unknown':unk,'max_gallery_cosine':float(s)} for a,s in zip(q,scores))
pd.DataFrame(episodes).to_csv(OUT/'open_set_episodes.csv',index=False)
pd.DataFrame(open_rows).to_csv(OUT/'open_set_scores.csv',index=False); summary['open_set']=episodes

# Inspect every SIFT array shape by memory mapping; never load archives into RAM.
def shape(path):
    with path.open('rb') as f:
        version=np.lib.format.read_magic(f)
        if version==(1,0): return np.lib.format.read_array_header_1_0(f)[0]
        if version==(2,0): return np.lib.format.read_array_header_2_0(f)[0]
        raise ValueError(f'Unsupported NPY version {version}')
def sift_header(a):
    split=meta.split.iloc[a]; folder=ROOT/f'sift_features_{split}.npz'
    iid=meta.id.iloc[a]; dp=folder/f'{iid}_desc.npy'; xp=folder/f'{iid}_xy.npy'
    if not dp.exists() or not xp.exists():
        return {'id':iid,'split':split,'species':sp[a],'viewpoint':view[a],'missing':True}
    ds=shape(dp); xs=shape(xp)
    return {'id':iid,'split':split,'species':sp[a],'viewpoint':view[a],'missing':False,'keypoints':ds[0],'dimensions':ds[1] if len(ds)==2 else -1,'xy_shape_ok':xs==(ds[0],2),'bytes':dp.stat().st_size+xp.stat().st_size}
with ThreadPoolExecutor(max_workers=16) as ex:
    sift_rows=list(ex.map(sift_header,range(len(meta))))
sift=pd.DataFrame(sift_rows); sift.to_csv(OUT/'sift_per_photo.csv',index=False)
summary['sift']={'missing':int(sift.missing.sum()),'invalid_xy_shapes':int((sift.xy_shape_ok==False).sum()),'by_split_species':sift.groupby(['split','species']).keypoints.agg(['count','min','median','max']).reset_index().to_dict('records')}
print('All SIFT headers inspected',flush=True)

# Targeted pattern verification sample; not a candidate reranking experiment.
cv2.setNumThreads(1)
@lru_cache(maxsize=400)
def desc(a):
    split=meta.split.iloc[a]; iid=meta.id.iloc[a]; p=ROOT/f'sift_features_{split}.npz'
    d=np.load(p/f'{iid}_desc.npy').astype(np.float32)
    return np.sqrt(d/np.maximum(d.sum(1,keepdims=True),1)),np.load(p/f'{iid}_xy.npy')
def verify(task):
    species,cat,a,b,cos=task; da,xa=desc(a); db,xb=desc(b); good=[]; inliers=0
    if min(len(da),len(db))>=4:
        bf=cv2.BFMatcher(); back=np.array([m.trainIdx for m in bf.match(db,da)])
        pairs=bf.knnMatch(da,db,k=2)
        good=[m[0] for m in pairs if len(m)==2 and m[0].distance<.85*m[1].distance and back[m[0].trainIdx]==m[0].queryIdx]
        if len(good)>=4:
            cv2.setRNGSeed(int((a*100003+b)%(2**31-1)))
            _,mask=cv2.estimateAffinePartial2D(np.float32([xa[m.queryIdx] for m in good]),np.float32([xb[m.trainIdx] for m in good]),method=cv2.RANSAC,ransacReprojThreshold=15.)
            inliers=int(mask.sum()) if mask is not None else 0
    return {'species':species,'category':cat,'a':meta.id.iloc[a],'b':meta.id.iloc[b],'cosine':cos,'ratio_mutual_matches':len(good),'inliers':inliers,'a_keypoints':len(da),'b_keypoints':len(db)}
tasks=[]
for (species,cat),pool in pair_pool.items():
    # Deduplicate unordered pairs before random sampling.
    unique={tuple(sorted((a,b))):(a,b,c) for a,b,c in pool}
    pool=list(unique.values())
    tasks.extend((species,cat,*pool[j]) for j in rng.choice(len(pool),min(40,len(pool)),replace=False))
with ThreadPoolExecutor(max_workers=4) as ex:
    pair_results=list(ex.map(verify,tasks))
pf=pd.DataFrame(pair_results); pf.to_csv(OUT/'sift_sample_pairs.csv',index=False)
sa=pf.groupby(['species','category']).agg(pairs=('inliers','size'),median_inliers=('inliers','median'),mean_inliers=('inliers','mean'),fraction_ge7=('inliers',lambda a:(a>=7).mean())).reset_index()
summary['sift_sample']=sa.to_dict('records'); sa.to_csv(OUT/'sift_sample_summary.csv',index=False)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda a:a.item() if hasattr(a,'item') else str(a)),encoding='utf-8')
print('SIFT pair sample finished',flush=True)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,3,figsize=(16,9),layout='constrained')
for species,color in [('hyena','#1b9e77'),('leopard','#d95f02')]:
    counts=animal.loc[species].images
    axs[0,0].hist(counts,bins=[1,2,4,8,16,32,64,128,256,512],alpha=.6,label=species,color=color)
axs[0,0].set(xscale='log',title='Photos per identity',xlabel='Photos (log scale)',ylabel='Identities'); axs[0,0].legend()
vp.div(vp.sum(1),axis=0).T.plot.bar(ax=axs[0,1],width=.85)
axs[0,1].set(title='Viewpoint composition',ylabel='Fraction',xlabel=''); axs[0,1].tick_params(axis='x',rotation=70); axs[0,1].legend(fontsize=7)
rs=pd.DataFrame(agg); rs=rs[rs.species!='all']
rs.pivot(index='species',columns='mode',values='map5').plot.bar(ax=axs[0,2],color=['#7570b3','#66a61e'])
axs[0,2].set(title='Raw cosine: duplicate sensitivity',ylabel='MAP@5 (eligible queries)',ylim=(0,1)); axs[0,2].tick_params(axis='x',rotation=0)
for label,d in valid.groupby('same_flank_available'):
    axs[1,0].hist(d.margin,bins=np.linspace(-.7,.7,45),density=True,alpha=.55,label=f'Same side available: {label}')
axs[1,0].axvline(0,color='black',ls='--'); axs[1,0].set(title='Correct minus best wrong similarity',xlabel='Margin (negative means wrong wins)',ylabel='Density'); axs[1,0].legend(fontsize=8)
o=pd.DataFrame(open_rows); o=o[o.seed==2059]
for label,d in o.groupby('unknown'):
    axs[1,1].hist(d.max_gallery_cosine,bins=35,density=True,alpha=.55,label='Unknown' if label else 'Known')
axs[1,1].set(title='Open-set score overlap (seed 2059)',xlabel='Best raw cosine',ylabel='Density'); axs[1,1].legend()
sa.pivot(index='category',columns='species',values='fraction_ge7').plot.barh(ax=axs[1,2],color=['#1b9e77','#d95f02'])
axs[1,2].set(title='Sampled RootSIFT / RANSAC verification',xlabel='Fraction with at least 7 inliers',ylabel='',xlim=(0,1))
fig.savefig(OUT/'overview.png',dpi=150); plt.close(fig)
img=base64.b64encode((OUT/'overview.png').read_bytes()).decode()
sections=[('Dataset composition',composition),('Identity distribution',pd.DataFrame(summary['identity_distribution']).T.reset_index()),('Near-duplicate groups',pd.DataFrame(summary['near_duplicate_groups']).T.reset_index()),('Raw retrieval stress test',pd.DataFrame(agg)),('Failure slices',pd.DataFrame(slices)),('Simulated open-set episodes',pd.DataFrame(episodes)),('SIFT coverage',pd.DataFrame(summary['sift']['by_split_species'])),('Sampled SIFT verification',sa)]
body=''.join('<h2>'+title+'</h2>'+frame.to_html(index=False,float_format=lambda v:f'{v:.4f}') for title,frame in sections)
report=f'''<!doctype html><html><head><meta charset="utf-8"><title>Animal ReID: exploratory analysis</title><style>body{{font:16px system-ui;max-width:1400px;margin:40px auto;padding:0 25px;color:#172334}}h1,h2{{color:#174c57}}table{{border-collapse:collapse;font-size:13px;margin:20px 0}}th,td{{padding:7px 10px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%}}li{{margin:8px 0}}.note{{background:#eef5f6;padding:18px}}</style></head><body><h1>Animal ReID: where can matching improve?</h1><p>25 November 2026. Reproducible analysis of supplied metadata, 2,152-dimensional MIEW-ID features and both extracted SIFT directories. Sources were read only.</p><img src="data:image/png;base64,{img}"><div class="note"><b>How to interpret these numbers</b><ul><li>Raw cosine retrieval is a diagnostic baseline, not Dalvin's full PCA/LDA + SIFT + LightGBM pipeline. Candidate recall here is for raw cosine, not Dalvin's shortlist.</li><li>Self-excluded retrieval removes only the query. Group-excluded retrieval removes its entire within-identity near-duplicate component (cosine ≥0.97). Identities without another group are excluded from the eligible average; use the matched-query comparison in summary.json to isolate duplicate effects.</li><li>Training photos rotate through query and gallery roles. Scores are descriptive, correlated and not an untouched validation result. Duplicate groups approximate encounters; they are not confirmed encounter IDs.</li><li>Open-set simulations use 25% held-out identities per species, seeds 2059–2061, and group-separated known queries. Thresholds are measured on each episode's known queries to describe overlap, not independently calibrated deployment thresholds. Episodes reuse photos.</li><li>SIFT verification samples up to 40 unique pairs per species/category, selected from each query's best available pair. Categories are not representative random pair populations. Mutual nearest matching plus a forward 0.85 ratio test and 15-pixel partial-affine RANSAC reproduce the preserved baseline's matching procedure. Seven inliers is a feature threshold, not a standalone identity decision.</li><li>Test identities are unknown. High similarity cannot establish that a test animal is known, nor estimate the test unknown fraction. Metadata coverage and descriptor shapes are checked; all SIFT descriptor values are not scanned for finiteness.</li></ul></div>{body}<h2>Reproduce</h2><p>Run <code>python Models/Abhay/EDA/run_eda.py</code>. All tables, individual query diagnostics and sampled pairs are saved alongside this report. No models or submissions are changed.</p></body></html>'''
(OUT/'report.html').write_text(report,encoding='utf-8')
print(json.dumps({'retrieval':agg,'matched':summary['matched_query_subset'],'slices':slices,'sift_sample':summary['sift_sample'],'identity_distribution':summary['identity_distribution'],'near_duplicates':summary['near_duplicate_groups'],'open_set':episodes},indent=2),flush=True)
