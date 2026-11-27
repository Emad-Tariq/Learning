"""Independently audit saved training boundaries and reconstruct final ranks."""
from pathlib import Path
import json,hashlib,sys,importlib.metadata,platform
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUN=HERE/'local_runs/large_2052'
SOURCE=ROOT/'Models/Abhay/Model7/local_runs/paired_2052'
SHIFT=ROOT/'analysis/domain_shift/results'
sys.path.insert(0,str(ROOT/'tmp/abhay_deps'))
import joblib

def main():
    train=pd.read_csv(Path.home()/'Downloads/reid_student_data_main/train.csv').set_index('id')
    roles=pd.read_csv(RUN/'identity_roles.csv').set_index('id')
    contexts=pd.read_csv(RUN/'training_contexts.csv')
    fitting=pd.read_csv(RUN/'training_episode_splits.csv')
    original=pd.read_csv(SOURCE/'episode_splits.csv')
    validation=pd.read_csv(RUN/'validation_episode_splits.csv')
    pred=pd.read_csv(RUN/'validation_predictions.csv')
    candidates=pd.read_csv(RUN/'validation_candidate_scores.csv')
    report=json.loads((RUN/'metrics.json').read_text())
    cfg=json.loads((RUN/'config.json').read_text())
    match_checks=json.loads((RUN/'independent_matching_checks.json').read_text())
    assert not match_checks['mismatches']
    assert hashlib.sha256((SOURCE/'paired_models.joblib').read_bytes()).hexdigest()==cfg['source_model_hash']
    old_models=joblib.load(SOURCE/'paired_models.joblib')['baseline']
    new_models=joblib.load(RUN/'large_models.joblib')
    assert len(old_models)==len(new_models)==3
    for old,new in zip(old_models,new_models):
        assert old[3]==new[3]
        for old_head,new_head in zip(old[:3],new[:3]):
            assert old_head.get_params()==new_head.get_params()
    assert roles.index.is_unique and set(roles.index)==set(train.index)
    assert roles.groupby('identity').role.nunique().eq(1).all()
    prior=set();episodes=0
    for stage in range(1,4):
        ctx=contexts[contexts.stage==stage]
        expected=set(roles[roles.role.isin(['background']+[f'stage{j}' for j in range(1,stage+1)])].index)
        assert set(ctx.id)==expected and not ctx.id.duplicated().any()
        ctx_names=set(train.loc[ctx.id].identity)
        assert not ctx_names&set(roles[roles.role=='validation'].identity)
        for ep,z in fitting[fitting.stage==stage].groupby('episode'):
            episodes+=1
            old=original[original.episode==ep]
            oq=old[old.assignment!='gallery'].set_index('id').assignment
            nq=z[z.assignment!='gallery'].set_index('id').assignment
            assert oq.equals(nq)
            assert not set(train.loc[nq.index].identity)&prior
            g=z[z.assignment=='gallery'];q=z[z.assignment!='gallery']
            assert set(z.id)<=expected and not z.id.duplicated().any()
            names=set(train.loc[g.id].identity)
            assert not names&set(train.loc[q.loc[q.assignment=='unknown_query','id']].identity)
            assert set(train.loc[q.loc[q.assignment=='known_query','id']].identity)<=names
            old_g=old[old.assignment=='gallery']
            own_role=set(roles[roles.role==f'stage{stage}'].index)
            assert set(g.id)&own_role==set(old_g.id)
        prior|=ctx_names
    assert episodes==36
    rr=[]
    for ep,d in pred.groupby('episode',sort=False):
        manifest=validation[validation.episode==ep]
        g=manifest[manifest.assignment=='gallery'];q=manifest[manifest.assignment=='query']
        assert set(q.id)==set(d.id) and d.id.is_unique and not manifest.id.duplicated().any()
        assert set(roles.loc[q.id].role)=={'validation'}
        gallery_names=set(train.loc[g.id].identity)
        assert not gallery_names&set(d.loc[d.unknown,'identity'])
        assert set(d.loc[~d.unknown,'identity'])<=gallery_names
        target=q.set_index('id').truth.loc[d.id].to_numpy()
        assert np.array_equal(target,d.truth.to_numpy())
        assert np.array_equal(d.identity.to_numpy(),train.loc[d.id].identity.to_numpy())
        assert np.array_equal(d.species.to_numpy(),train.loc[d.id].species.to_numpy())
        assert np.array_equal(d.unknown.to_numpy(),d.truth.to_numpy()=='new_individual')
        frozen=pd.read_csv(SHIFT/f'expanded_{ep}.csv').set_index('id')
        assert d.baseline_predictions.tolist()==frozen.loc[d.id].predictions.tolist()
        prior_candidates=None
        for method in ['baseline','large']:
            c=candidates[(candidates.episode==ep)&(candidates.method==method)].reset_index(drop=True)
            assert len(c)==len(d)*15
            assert np.array_equal(c.id.to_numpy().reshape(-1,15)[:,0],d.id)
            pc=c.pc.to_numpy().reshape(-1,15);raw=c.raw.to_numpy().reshape(-1,15)
            labels=c.ident.to_numpy().reshape(-1,15)
            np.testing.assert_allclose(pc.sum(1),1,atol=1e-8)
            pnew=(d[method+'_detector'].to_numpy()+np.clip(1-raw.sum(1),0,1))/2
            np.testing.assert_allclose(pnew,d[method+'_pnew'],atol=1e-12)
            if prior_candidates is not None:assert np.array_equal(labels,prior_candidates)
            prior_candidates=labels.copy()
            scores=np.c_[(1-pnew[:,None])*pc,pnew]
            labels=np.c_[labels,np.full(len(d),'new_individual',dtype=object)]
            ordered=np.take_along_axis(labels,np.argsort(-scores,axis=1)[:,:5],axis=1)
            assert [' '.join(row) for row in ordered]==d[method+'_predictions'].tolist()
            recomputed=[]
            for row,t,sp in zip(ordered,target,d.species):
                allowed=set(train.loc[g.id][train.loc[g.id].species==sp].identity)|{'new_individual'}
                assert len(set(row))==5 and set(row)<=allowed
                recomputed.append(next((1/(j+1) for j,v in enumerate(row) if v==t),0.))
            np.testing.assert_allclose(recomputed,d[method+'_rr5'],atol=1e-12)
        rr.append(d)
    d=pd.concat(rr,ignore_index=True)
    assert len(d)==1829 and d.id.nunique()==1147 and d.identity.nunique()==105
    slices={r['slice']:r for r in report['slices']}
    for name,mask in [('overall',np.ones(len(d),bool)),('known',~d.unknown),('unknown',d.unknown),
                      ('hyena',d.species=='hyena'),('leopard',d.species=='leopard')]:
        z=d[mask]
        for method in ['baseline','large']:
            assert np.isclose(z[method+'_rr5'].mean(),slices[name][method+'_map5'])
        assert np.isclose((z.large_rr5-z.baseline_rr5).mean(),slices[name]['delta'])
    checks=dict(all_checks_passed=True,training_episodes_checked=episodes,
        future_stage_and_validation_identities_excluded_during_fitting=True,
        original_training_queries_truth_and_own_references_preserved=True,
        earlier_stack_models_never_saw_later_query_animals=True,
        validation_galleries_exclude_queries_and_unknown_animals=True,
        frozen_baseline_predictions_exactly_reproduced=True,
        estimator_settings_and_feature_schemas_identical_to_baseline=True,
        cross_role_matches_independently_checked=match_checks['cross_role_training_pairs_checked'],
        both_methods_share_validation_candidates=True,
        all_final_rankings_and_metrics_independently_recomputed=True,
        appearances=len(d),unique_photos=d.id.nunique(),identities=d.identity.nunique())
    (RUN/'independent_checks.json').write_text(json.dumps(checks,indent=2))
    provenance=dict(python=platform.python_version(),
        packages={name:importlib.metadata.version(name) for name in
            ['numpy','pandas','scipy','scikit-learn','lightgbm','opencv-python','joblib','threadpoolctl']},
        trained_checkpoint_sha256=hashlib.sha256((RUN/'large_models.joblib').read_bytes()).hexdigest(),
        unchanged_baseline_sha256=cfg['source_model_hash'])
    (RUN/'verified_provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
