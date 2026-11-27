"""Compare retrained baseline with the preserved pre-existing three-stage checkpoint."""
from union_eval import *

def main():
    out=HERE/'local_runs/union_2052'
    seed=2052
    cfg=json.loads((out/'config.json').read_text())
    cv2.setNumThreads(1)
    warnings.filterwarnings('ignore',message='Only one sample available.*',category=UserWarning)
    train,_,_=protocol.load_data(Path(cfg['data'])); b.NX=b.X
    roles=protocol.allocate_roles(train,3,seed)
    source=HERE.parent/'Model7/local_runs/paired_2052'
    original=joblib.load(source/'paired_models.joblib')['baseline']
    retrained=joblib.load(out/'paired_models.joblib')['baseline']
    cache=protocol.SiftCache(Path(cfg['data']),out,cfg['workers'],150)
    cache.activate(roles['validation'])
    select('baseline')
    g,q,truth=protocol.split_episode(roles['validation'],22052,link=(.95,111))
    paired=paired_frames(g,q,truth)
    select('baseline'); s=paired['baseline']
    a,ra=protocol.through_stack(s,original); pa=b.rank(a,ra)
    z,rz=protocol.through_stack(s,retrained); pz=b.rank(z,rz)
    audit={'episode':22052,'queries':len(pa),'identical_top5':bool(np.array_equal(pa,pz)),
           'max_abs_final_pnew_difference':float(np.max(np.abs(ra['pb']-rz['pb']))),
           'max_abs_candidate_score_difference':float(np.max(np.abs(ra['raw']-rz['raw']))),
           'reference_checkpoint':str(source/'paired_models.joblib')}
    saved=pd.read_csv(out/'validation_22052.csv')
    audit['reproduces_saved_predictions']=bool(saved.baseline_predictions.tolist()==[' '.join(row) for row in pz])
    (out/'baseline_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=4,user_api='blas'): main()
