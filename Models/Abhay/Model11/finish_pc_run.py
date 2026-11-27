"""Validate local competition CSVs and package reusable checkpoints/cache."""
from pathlib import Path
import json,csv,hashlib
from persistence import save_bundle

HERE=Path(__file__).resolve().parent
run=HERE/'local_runs/pc_retrain_20261126'
data=Path.home()/'Downloads/reid_student_data_main'
with (data/'sample_submission.csv').open(newline='') as f:sample=list(csv.DictReader(f))
with (data/'test.csv').open(newline='') as f:test={r['id']:r for r in csv.DictReader(f)}
allowed={}
with (data/'train.csv').open(newline='') as f:
    for row in csv.DictReader(f):allowed.setdefault(row['species'],set()).add(row['identity'])
verified={}
for method in ('mlp_blend','baseline'):
    path=run/f'submission_{method}.csv'
    with path.open(newline='') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['id','predictions'];rows=list(reader)
    assert len(rows)==len(test)==2791
    assert [r['id'] for r in rows]==[r['id'] for r in sample]
    assert len({r['id'] for r in rows})==len(test)
    for row in rows:
        labels=row['predictions'].split()
        assert len(labels)==len(set(labels))==5
        assert set(labels)<=(allowed[test[row['id']]['species']]|{'new_individual'})
    verified[method]=dict(rows=len(rows),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
metrics=json.loads((run/'metrics.json').read_text())
info=json.loads((run/'submission_info.json').read_text())
train_minutes=((run/'metrics.json').stat().st_mtime-(run/'config.json').stat().st_mtime)/60
audit=dict(submissions=verified,training_and_validation_minutes=train_minutes,inference=info,
    model_validation=metrics['methods'],checks='Exact sample order, unique IDs, 5 distinct labels, species consistency, CSV header passed')
(run/'final_audit.json').write_text(json.dumps(audit,indent=2))
archive=HERE/'Model11_PC_Trained_Checkpoints.zip'
save_bundle(run,HERE,archive)
lines=['# Completed local retraining and competition inference','',
    'CSV files are in `local_runs/pc_retrain_20261126/`. The requested MLP blend and unchanged-decision baseline control are both supplied. Test identities are unknown; no test or leaderboard accuracy is asserted.','',
    f'Fresh three-stage baseline, MLP, calibration and validation completed in {train_minutes:.2f} minutes, reusing previously verified SIFT training comparisons. All classifier stages and the MLP were freshly trained.','',
    f"Full-gallery SIFT: {info['sift_performance']['required_pairs']:,} required query-related pairs; {info['sift_performance']['new_pairs']:,} were computed in the final resumed matching pass, taking {info['sift_performance']['seconds']/60:.2f} minutes. Earlier cached work was reused. Baseline scoring took {info['baseline_seconds']:.1f}s and full-vector verifier/calibration scoring took {info['verifier_seconds']:.1f}s.",'',
    'Optimizations preserve neighbour selection, thresholds and final ranking rules:','',
    '- Skip SIFT edges between two gallery photos; inference reads query rows only.',
    '- Check reverse nearest neighbours only for matches passing the forward ratio test, using the same OpenCV distances/RANSAC. All 5,000 sampled cached pair results reproduced exactly.',
    '- Eight matching workers were faster than four in a 1,500-pair worker benchmark (about 320 versus 249 pairs/s). This is a sampled benchmark, not a guarantee for all pairs.',
    '- Queue at most 1,024 pair tasks per batch. Save completed matches periodically and reuse them on restart.',
    '- Limit the RootSIFT descriptor cache to 384 images and release descriptors/pair dictionary before PCA/LDA scoring.',
    '- Save the trained models, calibrators, metadata and expanded SIFT cache in `Model11_PC_Trained_Checkpoints.zip`.',
    '',
    'The rejected NumPy distance-reuse variant changed one of 5,000 pair results and was not used for final predictions.','',
    '## Running again','',
    '`Run_Local.ps1` skips training when complete checkpoints exist and resumes test inference from the saved cache. After this completed run, the SIFT pairs are cached; do not delete the run folder.','',
    '## Further practical speed improvements','',
    'The PC reports AMD Ryzen 5 5500U, 12 logical CPU threads and about 8 GB installed RAM. Only roughly 1.2 GB was available initially. Close memory-heavy apps, keep the laptop plugged in with adequate cooling, and keep the existing cache on a local SSD. More workers are not automatically faster; eight was chosen from the measured four/eight comparison.','',
    'The current MLP, PCA/LDA and configured LightGBM use CPU. OpenCV receives ordinary NumPy descriptor arrays here. A GPU or GPU notebook session does not automatically accelerate these operations. OpenCL is available on this PC, but a GPU matcher would need separate implementation and output-parity testing; no GPU speedup is claimed. PyTorch/TensorFlow replacements are outside the supplied project rules.','',
    'Validation performance remains a negative experiment: baseline MAP@5 0.934272 and MLP blend 0.932723. Generating the requested submission does not change that finding.']
(HERE/'PC_RUN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(audit,indent=2))
