"""Small, bounded residual heads on frozen Dalvin probabilities.

Inference features use candidate/gallery evidence only; true coverage labels are
used solely for training weights and evaluation slices.
"""
import numpy as np
from scipy.special import softmax, expit, logit
from scipy.optimize import minimize

RANK_NAMES=['log_count','small','log_flank','missing_side','log_view','no_same_view',
            'sift_log','sift_fraction','raw_similarity','lda_similarity','centroid',
            'neighbour','stack_neighbour','stack_sift','missing_x_weak_sift','small_x_weak_sift']

def evidence(s,r):
    q,c=s['q'],s['c']; n=len(q); k=15
    get=lambda key: c[key].to_numpy(float).reshape(n,k)
    cnt=get('cnt'); flank=get('n_flank'); nv=get('n_view')
    # View codes 0,1,4,5,6,7 correspond to left/right flanks in Dalvin's VIEWS.
    side=np.isin(q['view'].to_numpy(),[0,1,4,5,6,7])
    missing=(side[:,None] & (flank==0)).astype(float)
    sparse=(cnt<=3).astype(float); sx=get('sx'); support=get('support')
    parts=[np.log1p(cnt),sparse,np.log1p(flank),missing,np.log1p(nv),(nv==0).astype(float),
           np.log1p(sx),support/np.maximum(cnt,1),get('mxr'),get('mxl'),get('cl'),get('nbev'),
           get('r_nbev'),get('r_sbev'),missing/(1+sx),sparse/(1+sx)]
    rank=np.stack(parts,axis=2)
    rank=np.nan_to_num(rank,nan=0.,posinf=0.,neginf=0.)
    top=r['pc'].argmax(1); ii=np.arange(n)
    # Express each candidate relative to the baseline's favourite.
    rank=rank-rank[ii,top][:,None,:]
    pc=r['pc']; sorted_pc=np.sort(pc,axis=1)
    weighted=lambda a:(pc*np.nan_to_num(a,nan=0.)).sum(1)
    topv=lambda a:np.nan_to_num(a[ii,top],nan=0.)
    entropy=-(pc*np.log(np.maximum(pc,1e-300))).sum(1)
    uf=np.column_stack([sorted_pc[:,-1],sorted_pc[:,-2],entropy,r['pn'],r['raw'].sum(1),r['raw'].max(1),
                        np.log1p(topv(cnt)),topv(sparse),np.log1p(topv(flank)),topv(missing),
                        np.log1p(topv(nv)),np.log1p(topv(sx)),topv(support/cnt),weighted(missing),weighted(sparse),
                        weighted(np.log1p(sx)),topv(missing)/(1+topv(sx)),topv(sparse)/(1+topv(sx)),
                        q['qq_minus_g'],q['qs_minus_s'],q['sp'],(q['view']<0).astype(float)])
    return rank,np.nan_to_num(uf,nan=0.,posinf=0.,neginf=0.)

class BoundedResidual:
    def __init__(self,kind,penalty=.1):
        self.kind=kind; self.penalty=penalty; self.cap=1.0 if kind=='rank' else float(np.log(2))

    def prepare(self,features,fit=False):
        f=np.asarray(features,dtype=float)
        if fit:
            flat=f.reshape(-1,f.shape[-1])
            self.mean=np.zeros(f.shape[-1]) if self.kind=='rank' else flat.mean(0)
            self.scale=np.maximum(flat.std(0),.05)
        x=np.clip((f-self.mean)/self.scale,-5,5)
        if self.kind!='rank': x=np.c_[np.ones(len(x)),x]
        return x

    def fit(self,features,offset,labels,weights):
        x=self.prepare(features,True); y=np.asarray(labels); w=np.asarray(weights,float); w=w/w.sum()
        off=np.asarray(offset,float)
        def objective(coef):
            z=x@coef; t=np.tanh(z/self.cap); scores=off+self.cap*t
            if self.kind=='rank':
                probs=softmax(scores,axis=1)
                loss=-np.sum(w*np.log(np.maximum(probs[np.arange(len(y)),y],1e-300)))
                probs[np.arange(len(y)),y]-=1
                grad=np.einsum('nk,nkp->p',probs*w[:,None]*(1-t*t),x)
            else:
                probs=expit(scores)
                loss=np.sum(w*(np.logaddexp(0,scores)-y*scores))
                grad=x.T@(w*(probs-y)*(1-t*t))
            return loss+.5*self.penalty*np.dot(coef,coef),grad+self.penalty*coef
        start=np.zeros(x.shape[-1]); initial=objective(start)[0]
        result=minimize(objective,start,jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-11,'gtol':1e-7})
        if not result.success: raise RuntimeError(f'{self.kind}: {result.message}')
        self.coef=result.x; self.initial_loss=float(initial); self.final_loss=float(result.fun)
        assert self.final_loss<=self.initial_loss+1e-8
        return self

    def correction(self,features):
        return self.cap*np.tanh((self.prepare(features)@self.coef)/self.cap)

def adjust(bundle,rank_head=None,unknown_head=None,rank_strength=0.,unknown_strength=0.):
    pc=bundle['pc']; pb=bundle['pb']
    if rank_strength:
        pc=softmax(np.log(np.maximum(pc,1e-300))+rank_strength*rank_head.correction(bundle['rank_features']),axis=1)
    if unknown_strength:
        pb=expit(logit(np.clip(pb,1e-9,1-1e-9))+unknown_strength*unknown_head.correction(bundle['unknown_features']))
    return pc,pb

def predictions(bundle,pc,pb):
    labels=np.c_[bundle['candidates'],np.full(len(pc),'new_individual',dtype=object)]
    scores=np.c_[(1-pb[:,None])*pc,pb]
    return np.take_along_axis(labels,np.argsort(-scores,axis=1)[:,:5],axis=1)

def rr5(pred,truth):
    return np.array([next((1/(j+1) for j,p in enumerate(row) if p==target),0.) for row,target in zip(pred,truth)])

def train_heads(bundle,indices):
    d=bundle['rows'].iloc[indices]; y=bundle['target'][indices]
    have=(~d.unknown.to_numpy())&(y>=0)
    rank_ix=indices[have]; rd=d.iloc[np.flatnonzero(have)]
    rw=1/np.sqrt(rd.groupby('identity').identity.transform('size').to_numpy())
    rw*=1+rd.missing_side.to_numpy()+(rd.true_gallery_images.to_numpy()<=3)
    rank=BoundedResidual('rank').fit(bundle['rank_features'][rank_ix],np.log(np.maximum(bundle['pc'][rank_ix],1e-300)),bundle['target'][rank_ix],rw)
    uy=d.unknown.to_numpy().astype(int)
    uw=1/np.sqrt(d.groupby('identity').identity.transform('size').to_numpy())
    # Difficulty inferred from top candidate evidence, so both classes are represented.
    uf=bundle['unknown_features'][indices]
    uw*=1+((uf[:,7]>0)|(uf[:,9]>0))
    for label in (0,1):
        mask=uy==label; uw[mask]*=mask.sum()/uw[mask].sum()
    unknown=BoundedResidual('unknown').fit(uf,logit(np.clip(bundle['pb'][indices],1e-9,1-1e-9)),uy,uw)
    return rank,unknown
