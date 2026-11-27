"""Deterministic training-only gallery thinning and hard unknown selection."""
import numpy as np
import dalvin_core as b
import data_protocol as protocol

def hard_positions(stage,fraction):
    base={0,5,10} if fraction==.25 else {0,1,2,5,7,10} if fraction==.5 else set()
    return {(j+stage)%12 for j in base}

def slices(g,q,truth):
    rows=[]
    for i,t in zip(q,truth):
        known=t!=protocol.NEW; v=b.VIEW[i]
        own=g[b.Y[g]==b.Y[i]]
        side=v in (0,1,4,5,6,7)
        same=side and np.any((b.VIEW[own]>=0)&(b.FLANK[b.VIEW[own].clip(0)]==b.FLANK[v]))
        rows.append({'id':b.IDS[i],'identity':b.Y[i],'species':b.SP[i],
                     'truth':t,'unknown':not known,'viewpoint_code':int(v),'side_query':side,
                     'missing_side':bool(known and side and not same),'true_gallery_images':len(own)})
    return rows

def difficult_episode(indices,g,q,truth,seed):
    """Thin half of each species' gallery identities; preserve class counts.

    Known queries are unchanged, and their original gallery is only reduced.
    Thus a duplicate group cannot newly cross gallery and query. Unknown IDs
    stay entirely absent. Within each unknown identity, choose the same number
    of query photos with highest raw similarity to the thinned gallery.
    """
    rng=np.random.default_rng(seed+1000000)
    selected=set()
    for sp in ('hyena','leopard'):
        names=np.unique(b.Y[g[b.SP[g]==sp]])
        selected.update(rng.choice(names,max(1,len(names)//2),replace=False).tolist())
    keep=[]; side_thinned=0; altered=0
    for identity in np.unique(b.Y[g]):
        own=g[b.Y[g]==identity]
        if identity not in selected:
            keep.extend(own); continue
        pool=own
        knownq=q[(b.Y[q]==identity)&(truth!=protocol.NEW)]
        possible=[]
        for flank in (0,1):
            query_on_side=knownq[(b.VIEW[knownq]>=0)&(b.FLANK[b.VIEW[knownq].clip(0)]==flank)]
            other=own[(b.VIEW[own]>=0)&(b.FLANK[b.VIEW[own].clip(0)]!=flank)]
            if len(query_on_side) and len(other): possible.append(other)
        if possible:
            pool=possible[int(rng.integers(len(possible)))]; side_thinned+=1
        count=min(len(pool),int(rng.integers(1,4)))
        picked=rng.choice(pool,count,replace=False)
        keep.extend(picked)
        altered+=int(len(picked)!=len(own) or set(picked)!=set(own))
    # Preserve original gallery ordering for all retained images.
    gnew=g[np.isin(g,np.array(keep,dtype=int))]
    assert set(b.Y[gnew])==set(b.Y[g])
    knownq=q[truth!=protocol.NEW]
    unknownq=[]; unknown_replaced=0
    for identity in np.unique(b.Y[q[truth==protocol.NEW]]):
        old=q[(truth==protocol.NEW)&(b.Y[q]==identity)]
        pool=indices[b.Y[indices]==identity]
        gallery=gnew[b.SP[gnew]==b.SP[pool[0]]]
        sim=(b.X[pool]@b.X[gallery].T).max(1)
        # Stable order makes equal-score choices deterministic.
        picked=pool[np.argsort(-sim,kind='stable')[:len(old)]]
        unknownq.extend(picked); unknown_replaced+=len(set(picked)-set(old))
    qnew=np.r_[knownq,np.array(unknownq,dtype=int)]
    tnew=np.r_[b.Y[knownq],np.full(len(unknownq),protocol.NEW,dtype=object)]
    assert len(qnew)==len(q) and len(np.unique(qnew))==len(qnew)
    assert not set(gnew)&set(qnew)
    assert set(b.Y[knownq])<=set(b.Y[gnew])
    assert not set(b.Y[unknownq])&set(b.Y[gnew])
    assert int((tnew==protocol.NEW).sum())==int((truth==protocol.NEW).sum())
    for sp in ('hyena','leopard'):
        assert int(((b.SP[qnew]==sp)&(tnew==protocol.NEW)).sum())==int(((b.SP[q]==sp)&(truth==protocol.NEW)).sum())
    diagnostics={'gallery_before':len(g),'gallery_after':len(gnew),'identities_modified':altered,
                 'identities_side_thinned':side_thinned,'unknown_photos_replaced':unknown_replaced,
                 'query_count':len(q),'unknown_queries':int((truth==protocol.NEW).sum())}
    return gnew,qnew,tnew,diagnostics
