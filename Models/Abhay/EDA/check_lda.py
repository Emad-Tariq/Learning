"""One descriptive, group-separated episode of Dalvin's PCA/LDA retrieval."""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits

out=Path(__file__).resolve().parent
data=Path.home()/'Downloads'/'reid_student_data_main'
t=pd.read_csv(out/'duplicate_groups.csv')
ids=pd.read_csv(data/'miewid_ids.csv')
x=np.load(data/'miewid_features.npy')[pd.Index(ids.id).get_indexer(t.id)]
x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
rng=np.random.default_rng(2059)
rows=[]
with threadpool_limits(limits=4):
 for species in ('hyena','leopard'):
    ix=np.flatnonzero(t.species.to_numpy()==species)
    names=np.unique(t.identity.iloc[ix]); rng.shuffle(names)
    unknown=set(names[:len(names)//4]); gallery=[]; queries=[]
    for name in names:
        a=ix[t.identity.iloc[ix].to_numpy()==name]
        if name in unknown:
            queries.extend(a); continue
        gs=np.unique(t.duplicate_group.iloc[a])
        if len(gs)<2:
            gallery.extend(a); continue
        chosen=rng.choice(gs,size=min(max(1,round(.3*len(gs))),len(gs)-1),replace=False)
        isq=np.isin(t.duplicate_group.iloc[a],chosen)
        queries.extend(a[isq]); gallery.extend(a[~isq])
    g=np.array(gallery); q=np.array(queries)
    pca=PCA(n_components=min(384,len(g)+len(q)-1,x.shape[1]),random_state=2059).fit(np.r_[x[g],x[q]])
    pg=pca.transform(x[g]); pq=pca.transform(x[q])
    lda=LinearDiscriminantAnalysis(solver='eigen',shrinkage=.3).fit(pg,t.identity.iloc[g])
    lg=lda.transform(pg); lq=lda.transform(pq)
    lg/=np.maximum(np.linalg.norm(lg,axis=1,keepdims=True),1e-12); lq/=np.maximum(np.linalg.norm(lq,axis=1,keepdims=True),1e-12)
    y=t.identity.to_numpy(); gn,inv=np.unique(y[g],return_inverse=True)
    for method,scores in [('raw',x[q]@x[g].T),('pca_lda',lq@lg.T)]:
      for a,s in zip(q,scores):
        if y[a] in unknown: continue
        best=np.full(len(gn),-np.inf); np.maximum.at(best,inv,s)
        rank=int(1+(best>best[gn==y[a]][0]).sum())
        vv=str(t.viewpoint.iloc[a]); side='left' if 'left' in vv else 'right' if 'right' in vv else 'other'
        gv=t.viewpoint.iloc[g].astype(str)
        same=(gv.str.contains(side).to_numpy() if side!='other' else np.zeros(len(g),bool))&(y[g]==y[a])
        rows.append({'id':t.id.iloc[a],'identity':y[a],'species':species,'method':method,'rank':rank,'rr5':1/rank if rank<=5 else 0.,'same_flank_available':bool(same.any())})
    print(species,'PCA/LDA checked',flush=True)
d=pd.DataFrame(rows); d.to_csv(out/'lda_episode_queries.csv',index=False)
def agg(d):
 return {'queries':len(d),'map5':float(d.rr5.mean()),'top1':float((d['rank']==1).mean()),'top5':float((d['rank']<=5).mean()),'top15':float((d['rank']<=15).mean())}
results=[]
for (species,method),a in d.groupby(['species','method']):
 results.append(dict(species=species,method=method,slice='all',**agg(a)))
 for same,b in a.groupby('same_flank_available'):
  results.append(dict(species=species,method=method,slice='same_flank_'+str(same),**agg(b)))
pd.DataFrame(results).to_csv(out/'lda_episode_summary.csv',index=False)
(out/'lda_episode_summary.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
