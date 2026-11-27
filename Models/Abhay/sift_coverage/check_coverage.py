"""Audit and paired retraining for shortlisted identities with no SIFT checks.

The baseline, splits, candidate membership and source caches remain unchanged.
Supplement at most two gallery photos per unchecked candidate, chosen without labels.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'tmp/abhay_deps'))
sys.path.insert(0, str(HERE.parent / 'Model7'))
import argparse, inspect, json, time, warnings, hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2, joblib
from scipy import sparse
from threadpoolctl import threadpool_limits
import dalvin_core as b
import data_protocol as p

SOURCE = HERE.parent / 'Model7/local_runs/paired_2052'
OUT = HERE / 'results'
EPISODES = [22052, 22053, 22054]


def pair(a, c):
    return tuple(sorted((int(a), int(c))))


def checked_matrix(m):
    # Sparse matrices retain explicitly stored zero-inlier pairs. Values alone
    # cannot distinguish a failed verification from an unattempted pair.
    z = m.copy()
    z.data = np.ones_like(z.data, dtype=np.int8)
    return z


def fixed_function():
    source = inspect.getsource(b.species_features)
    old = 'cand = np.argsort(-(lda1 + SIFT_PULL * np.tanh(sift1 / 15)), 1)[:, :K]'
    assert source.count(old) == 1
    source = source.replace('def species_features(', 'def fixed_species_features(', 1)
    source = source.replace(old, 'cand = FIXED_CAND[sp]')
    exec(compile(source, '<fixed Dalvin candidate selection>', 'exec'), b.__dict__)


class Experiment:
    def __init__(self, workers):
        self.workers = workers
        self.train, _, _ = p.load_data(Path.home() / 'Downloads/reid_student_data_main')
        b.NX = b.X
        self.roles = p.allocate_roles(self.train, 3, 2052)
        self.models = joblib.load(SOURCE / 'paired_models.joblib')['baseline']
        self.cache = p.SiftCache(Path.home() / 'Downloads/reid_student_data_main', OUT, workers, 150)
        # Reuse existing verified values in memory; never write source artifacts.
        for path in SOURCE.glob('sift_mutual_*.npz'):
            with np.load(path) as z:
                self.cache.done.update({tuple(a): tuple(v) for a, v in zip(z['pairs'], z['values'])})
        self.original = b.species_features
        self.original_embed = b.embed
        fixed_function()
        self.audit, self.pair_audit = [], []

    def activate(self, role):
        self.cache.activate(self.roles[role])
        self.inl, self.good = b.INL.copy(), b.GOOD.copy()
        self.checked = checked_matrix(self.inl)
        assert self.inl.nnz == self.checked.nnz
        self.active_role = role

    def frames(self, seed, link, audit=False):
        b.INL, b.GOOD, b.species_features = self.inl, self.good, self.original
        embeddings = {}
        def cached_embed(gallery, labels, queries):
            key = (gallery.shape, queries.shape, tuple(labels), gallery.tobytes(), queries.tobytes())
            if key not in embeddings:
                embeddings[key] = self.original_embed(gallery, labels, queries)
            return embeddings[key]
        b.embed = cached_embed
        gi, qi, truth = p.split_episode(self.roles[self.active_role], seed, link=link)
        base = p.frame(gi, qi, truth)
        original_candidates = base['c'].ident.to_numpy().reshape(-1, b.K)
        want = set()
        selected = {}
        for row, query in enumerate(base['order']):
            for candidate in original_candidates[row]:
                gallery = gi[b.Y[gi] == candidate]
                covered = np.asarray(self.checked[query, gallery].toarray()).ravel().astype(bool)
                if covered.any():
                    continue
                # Same flank first, then raw similarity, then global row index.
                v = b.VIEW[query]
                same = ((b.VIEW[gallery] >= 0) & (b.FLANK[b.VIEW[gallery].clip(0)] == b.FLANK[v])) if v >= 0 and b.FLANK[v] in (0, 1) else np.zeros(len(gallery), bool)
                similarity = b.X[gallery] @ b.X[query]
                chosen = gallery[np.lexsort((gallery, -similarity, -same.astype(int)))[:2]]
                selected[(row, candidate)] = chosen
                want.update(pair(query, c) for c in chosen)
        missing = sorted(want - self.cache.done.keys())
        print('VERIFY', self.active_role, seed, 'selected', len(want), 'uncached', len(missing), flush=True)
        if missing:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for done, (key, val) in enumerate(zip(missing, pool.map(self.cache.verify, missing)), 1):
                    self.cache.done[key] = val
                    if done % 2000 == 0:
                        print('VERIFIED', done, '/', len(missing), flush=True)
            self.cache.save()
        # Include only this episode's label-free selected pairs, not diagnostic
        # or cached pairs from other episodes.
        def extra_matrix(index):
            keys = np.array(sorted(want), dtype=int).reshape(-1, 2)
            if not len(keys):
                return sparse.csr_matrix(self.inl.shape, dtype=np.float32)
            values = np.array([self.cache.done[tuple(k)][index] for k in keys], dtype=np.float32)
            return sparse.coo_matrix((np.r_[values, values], (np.r_[keys[:, 0], keys[:, 1]], np.r_[keys[:, 1], keys[:, 0]])), shape=self.inl.shape).tocsr()
        b.INL, b.GOOD = self.inl + extra_matrix(1), self.good + extra_matrix(0)
        b.FIXED_CAND = {}
        for sp in ('hyena', 'leopard'):
            identities = np.unique(b.Y[gi[b.SP[gi] == sp]])
            candidates = original_candidates[b.SP[base['order']] == sp]
            b.FIXED_CAND[sp] = np.searchsorted(identities, candidates)
        b.species_features = b.fixed_species_features
        try:
            improved = p.frame(gi, qi, truth)
        finally:
            b.species_features = self.original
            b.embed = self.original_embed
            b.INL, b.GOOD = self.inl, self.good
        assert np.array_equal(base['order'], improved['order'])
        assert base['c'].ident.equals(improved['c'].ident)
        assert base['truth'].tolist() == improved['truth'].tolist()
        for col in ('mxr','m2r','mxl','m2l','cl','gapl','gapr','same','cnt','nbev','rk_r','rk_c','n_view','n_flank'):
            np.testing.assert_allclose(base['c'][col], improved['c'][col], equal_nan=True,
                                       err_msg='Non-SIFT evidence changed: '+col)
        counts, totals = [], []
        for row, query in enumerate(base['order']):
            for candidate in original_candidates[row]:
                gallery = gi[b.Y[gi] == candidate]
                nchecked = int(self.checked[query, gallery].sum())
                chosen = selected.get((row, candidate), [])
                counts.append(nchecked + len(chosen))
                totals.append(len(gallery))
                if audit:
                    known = base['truth'][row] != p.NEW
                    correct = known and candidate == b.Y[query]
                    vals = [self.cache.done[pair(query, c)][1] for c in chosen]
                    self.pair_audit.append(dict(episode=seed, id=b.IDS[query], candidate=candidate,
                        correct=bool(correct), original_checked=nchecked, gallery_images=len(gallery),
                        extra_checked=len(chosen), extra_max_inliers=max(vals, default=0),
                        extra_ge7=int(sum(v >= 7 for v in vals))))
            if audit:
                known = base['truth'][row] != p.NEW
                gallery = gi[b.Y[gi] == b.Y[query]]
                nchecked = int(self.checked[query, gallery].sum()) if known else 0
                present = bool(b.Y[query] in original_candidates[row]) if known else False
                chosen = selected.get((row, b.Y[query]), [])
                vals = [self.cache.done[pair(query, c)][1] for c in chosen]
                self.audit.append(dict(episode=seed, id=b.IDS[query], identity=b.Y[query], species=b.SP[query],
                    unknown=not known, candidate_present=present, true_gallery_images=len(gallery) if known else 0,
                    original_true_checked=nchecked, true_extra_checked=len(chosen),
                    true_extra_max_inliers=max(vals, default=0), true_extra_ge7=any(v >= 7 for v in vals)))
        improved['c']['verified_photos'] = counts
        improved['c']['verified_fraction'] = np.array(counts) / np.array(totals)
        improved['c']['verification_missing'] = np.array(counts) == 0
        return base, improved, dict(additional_pairs=len(want), newly_computed_pairs=len(missing))


def predictions(s, models):
    s, r = p.through_stack(s, models)
    pred = b.rank(s, r)
    rr = np.array([next((1/(i+1) for i, label in enumerate(row) if label == target), 0.) for row, target in zip(pred, s['truth'])])
    return rr, [' '.join(row) for row in pred], r


def metrics(d):
    result = {'queries':len(d), 'unique_photos':int(d.id.nunique()), 'unique_identities':int(d.identity.nunique())}
    for m in ('baseline', 'supplement'):
        result[m] = {name:float(d.loc[mask, m+'_rr5'].mean()) if np.any(mask) else None for name, mask in (
            ('map5', np.ones(len(d), bool)), ('known_map5', ~d.unknown), ('unknown_map5', d.unknown))}
    diff = d.supplement_rr5-d.baseline_rr5
    result.update(delta=float(diff.mean()), improved=int((diff>0).sum()), worsened=int((diff<0).sum()))
    return result


def bootstrap(d):
    a = d.assign(delta=d.supplement_rr5-d.baseline_rr5).groupby(['species','identity']).delta.agg(['sum','size'])
    rng=np.random.default_rng(2052); sums=np.zeros(3000); counts=np.zeros(3000)
    for _, part in a.groupby(level=0):
        z=part.to_numpy(); ix=rng.integers(0,len(z),(3000,len(z)))
        sums+=z[ix,0].sum(1); counts+=z[ix,1].sum(1)
    return np.quantile(sums/counts,[.025,.975]).tolist()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--audit-only',action='store_true'); ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    warnings.filterwarnings('ignore',message='Only one sample available.*',category=UserWarning)
    b.PARAMS.update(n_estimators=400,n_jobs=args.workers,random_state=2052)
    config={'seed':2052,'stages':3,'simulations_per_stage':12,'evaluation_episodes':EPISODES,
        'intervention':'Only candidates with zero baseline checked gallery photos: check up to two photos, same flank first then raw cosine. Add checked-count/fraction/missing features; retrain all 3 stages.',
        'candidate_membership':'Frozen baseline 15 for every episode; baseline SIFT also used for duplicate splits.',
        'decision_rule':'Unchanged Dalvin final unknown/known combination.',
        'limitations':['Previously examined validation identities; not a pristine final holdout or leaderboard result.',
                       'Repeated queries and identities across episodes; bootstrap clustered by identity.',
                       'Tests two selected photos for unchecked candidates, not every unverified photo.',
                       'Combines added verification and availability features; does not isolate their individual effects.'],
        'gate':'Positive pooled and all episode deltas; no pooled known or unknown MAP5 decrease.',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'config.json').write_text(json.dumps(config,indent=2))
    ex=Experiment(args.workers)
    ex.activate('validation')
    eval_frames=[]
    reference=pd.read_csv(SOURCE/'validation_predictions.csv')
    for seed in EPISODES:
        base, improved, extra = ex.frames(seed,(.95,111),audit=True)
        rr, labels, r=predictions(base,ex.models)
        d=pd.DataFrame([a for a in ex.audit if a['episode']==seed])
        d['baseline_rr5'],d['baseline_predictions'],d['baseline_pnew']=rr,labels,r['pb']
        saved=reference[reference.episode==seed].set_index('id')
        # Legacy paired output calls the original model "baseline".
        assert labels == saved.loc[d.id,'baseline_predictions'].tolist(), 'Baseline prediction parity failed'
        d.to_csv(OUT/f'audit_{seed}.csv',index=False)
        eval_frames.append((d,improved))
        joblib.dump((d,improved), OUT/f'eval_frame_{seed}.joblib')
        print('AUDIT',seed,json.dumps(extra),flush=True)
    audit=pd.concat([d for d,_ in eval_frames],ignore_index=True)
    audit.to_csv(OUT/'query_audit.csv',index=False)
    candidates=pd.DataFrame(ex.pair_audit); candidates.to_csv(OUT/'candidate_audit.csv',index=False)
    eligible=audit[~audit.unknown & audit.candidate_present]
    gap=eligible[eligible.original_true_checked==0]
    report={'query_appearances':len(audit),'known_queries':int((~audit.unknown).sum()),
        'known_correct_shortlisted':len(eligible),'shortlisted_correct_without_sift':len(gap),
        'gap_fraction_shortlisted_known':len(gap)/len(eligible),
        'gap_unique_photos':int(gap.id.nunique()),'gap_extra_ge7':int(gap.true_extra_ge7.sum()),
        'gap_baseline_map5':float(gap.baseline_rr5.mean()) if len(gap) else None,
        'gap_baseline_outside_top5':int((gap.baseline_rr5==0).sum()),
        'all_candidate_rows':len(candidates),'unchecked_candidate_rows':int((candidates.original_checked==0).sum()),
        'supplemented_candidate_rows_ge7':int(((candidates.original_checked==0)&(candidates.extra_max_inliers>=7)).sum()),
        'baseline_prediction_parity':True,
        'species':{sp:{'shortlisted_known':len(z),'without_sift':int((z.original_true_checked==0).sum()),
                       'extra_ge7':int(z.loc[z.original_true_checked==0,'true_extra_ge7'].sum())} for sp,z in eligible.groupby('species')}}
    (OUT/'audit_summary.json').write_text(json.dumps(report,indent=2))
    print('AUDIT SUMMARY',json.dumps(report),flush=True)
    if args.audit_only: return
    checkpoint=OUT/'supplement_models.joblib'
    models=joblib.load(checkpoint) if checkpoint.exists() else []
    for stage in range(3):
        if stage<len(models): continue
        ex.activate(f'stage{stage+1}')
        training=[]
        for j in range(12):
            _, s, extra=ex.frames(2052+1000*(stage+1)+j,b.REGIMES[j%len(b.REGIMES)])
            for model in models:
                s=b.augment(s,b.predict(model,s))
            training.append(s)
            print('TRAIN',stage+1,j+1,json.dumps(extra),flush=True)
        models.append(b.fit(training))
        temp=checkpoint.with_suffix('.tmp'); joblib.dump(models,temp); os.replace(temp,checkpoint)
        del training
        print('FITTED STAGE',stage+1,flush=True)
    results=[]
    for d,s in eval_frames:
        rr, labels, r=predictions(s,models)
        d=d.copy(); d['supplement_rr5'],d['supplement_predictions'],d['supplement_pnew']=rr,labels,r['pb']
        results.append(d)
    d=pd.concat(results,ignore_index=True); d.to_csv(OUT/'paired_predictions.csv',index=False)
    result=metrics(d)
    result['episodes']={str(seed):metrics(z) for seed,z in d.groupby('episode')}
    result['species']={sp:metrics(z) for sp,z in d.groupby('species')}
    gapmask=~d.unknown & d.candidate_present & (d.original_true_checked==0)
    result['unchecked_correct_slice']=metrics(d[gapmask]) if gapmask.any() else None
    result['identity_cluster_bootstrap_95pct']=bootstrap(d)
    result['gate_passed']=bool(result['delta']>0 and all(z['delta']>0 for z in result['episodes'].values()) and all(result['supplement'][k]>=result['baseline'][k] for k in ('known_map5','unknown_map5')))
    (OUT/'paired_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('FINISHED',json.dumps(result),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4,user_api='blas'): main()
