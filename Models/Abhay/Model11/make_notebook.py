"""Build a self-contained Kaggle notebook with exact embedded source snapshots."""
import json,hashlib,ast
from pathlib import Path
HERE=Path(__file__).resolve().parent
def md(text):return {'cell_type':'markdown','metadata':{},'source':text.splitlines(True)}
def code(text):
    ast.parse(text)
    return {'cell_type':'code','metadata':{},'source':text.splitlines(True),'outputs':[],'execution_count':None}
names=['dalvin_core.py','data_protocol.py','verifier.py','run.py','test_verifier.py','requirements.txt','persistence.py']
sources={name:(HERE/name).read_text(encoding='utf-8') for name in names}
hashes={name:hashlib.sha256(text.encode()).hexdigest() for name,text in sources.items()}
cells=[md('''# Model 11: full-vector MLP evidence for new-animal detection

This is an experiment, not a claimed improvement. Attach the complete `projectwild3` competition data, including both supplied SIFT feature archives/directories. Use a CPU session; GPU is unused. Default settings run the full three-stage Dalvin baseline (12 episodes/stage, 400 candidate trees), then the MLP/calibration and three validation episodes. SIFT verification can take substantial time; output checkpoints/caches support resuming with identical settings.

Only supplied MIEW-ID vectors, SIFT descriptors/coordinates, metadata and labels are used. There are no raw images, pretrained networks, attention layers or deep-learning frameworks. The plain scikit-learn MLP learns symmetric interactions of the full 2,152-dimensional vectors. Dalvin's preserved baseline code is credited to Dalvin; the pair sampling design draws on the earlier Abhay Model 6 experiment, whose saturation failure motivates identity-based early stopping and query-level calibration here.

**Before looking at validation results:** the primary arm is a fixed 50% blend of baseline P(new) and the MLP-evidence calibrator. Candidate lists, known-only ordering, stacking and final ranking formula are shared. A summary-only calibration control tests whether adding the MLP gives useful information beyond recalibration. Full MLP replacement is diagnostic, not selected after validation.

The default seed reproduces an existing development allocation that has already been examined. Supervised training, calibration and validation identities are disjoint, but this is not a pristine final test or a leaderboard estimate. Episodes reuse photos/identities. A different seed requires fresh baseline training and a new output directory; do not repeatedly tune against validation results.'''),
code('''from pathlib import Path
import sys, subprocess, json, importlib.metadata

MODE = 'train'  # 'train' for training/validation; 'submission' restores and skips training.
GENERATE_SUBMISSION = False  # Optional after training; submission mode always generates it.
RESTORE_ARCHIVE = None  # Optional Path('/kaggle/input/.../model11_checkpoint_bundle.zip')
BACKUP = Path('/kaggle/working/model11_checkpoint_bundle.zip')
SEED = 2052
WORKERS = 4
MLP_MAX_EPOCHS = 40
MAX_TRAIN_PAIRS = 24000
BASELINE_EPISODES_PER_STAGE = 12
CALIBRATION_EPISODES = 6
CANDIDATE_TREES = 400
OUTPUT = Path('/kaggle/working/model11_results')
DATA_ROOT = None  # Optional: set the folder containing train.csv and miewid_features.npy.
INSTALL_REQUIREMENTS = False  # Usually unnecessary on Kaggle; enable only if imports fail.
CODE = Path('/kaggle/working/model11_code')
CODE.mkdir(parents=True, exist_ok=True)
'''),
code('import hashlib\nSOURCES = '+repr(sources)+'\nEXPECTED_HASHES = '+repr(hashes)+'''
for name, text in SOURCES.items():
    assert hashlib.sha256(text.encode()).hexdigest() == EXPECTED_HASHES[name]
    (CODE/name).write_text(text, encoding='utf-8')
print('Verified and extracted', len(SOURCES), 'embedded source files')
'''),
code('''import importlib.util
spec = importlib.util.spec_from_file_location('model11_persistence', CODE/'persistence.py')
persistence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(persistence)
if MODE not in ('train', 'submission'):
    raise ValueError("MODE must be 'train' or 'submission'")
if RESTORE_ARCHIVE is None and MODE == 'submission':
    matches = list(Path('/kaggle/input').rglob('model11_checkpoint_bundle.zip'))
    if len(matches) == 1: RESTORE_ARCHIVE = matches[0]
    elif len(matches) > 1: raise ValueError(f'Set RESTORE_ARCHIVE explicitly: {matches}')
if RESTORE_ARCHIVE is not None:
    # Restore exact training sources into a separate folder, protecting current code.
    CODE = Path('/kaggle/working/model11_restored_code')
    restored = persistence.restore_bundle(RESTORE_ARCHIVE, OUTPUT, CODE)
    if MODE == 'submission' and not restored['complete_for_submission']:
        raise RuntimeError(f"This backup is partial: {restored['missing_for_submission']}")
if MODE == 'submission':
    missing = [name for name in persistence.REQUIRED if not (OUTPUT/name).exists()]
    if missing: raise RuntimeError(f'Attach a saved checkpoint bundle and set RESTORE_ARCHIVE. Missing: {missing}')
    print('Submission mode: training and validation will be skipped.')
'''),
code('''if INSTALL_REQUIREMENTS:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', str(CODE/'requirements.txt')], check=True)
import importlib
for name in ['numpy','pandas','scipy','sklearn','lightgbm','cv2','joblib','threadpoolctl']:
    module = importlib.import_module(name)
    print(name, getattr(module, '__version__', 'import OK'))
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(CODE), '-p', 'test_verifier.py', '-v'], check=True)
'''),
code('''root = Path('/kaggle/input')
if DATA_ROOT is None:
    options = [p.parent for p in root.rglob('miewid_features.npy')
               if all((p.parent/name).exists() for name in ['train.csv','test.csv','miewid_ids.csv'])]
    if len(options) != 1:
        raise RuntimeError(f'Set DATA_ROOT explicitly; found metadata folders: {options}')
    DATA_ROOT = options[0]
DATA_ROOT = Path(DATA_ROOT)
# Make a small input view so SIFT assets can be mounted in a separate Kaggle dataset.
INPUT = Path('/kaggle/working/model11_input')
INPUT.mkdir(exist_ok=True)
for name in ['train.csv','test.csv','miewid_ids.csv','miewid_features.npy']:
    target = DATA_ROOT/name
    if not target.exists(): raise FileNotFoundError(target)
    link = INPUT/name
    if link.exists() and link.resolve() != target.resolve(): raise RuntimeError(f'Stale input link: {link}')
    if not link.exists(): link.symlink_to(target)
for split in ['train','test']:
    name = f'sift_features_{split}.npz'
    nearby = [p/name for p in [DATA_ROOT, DATA_ROOT.parent, DATA_ROOT.parent.parent] if (p/name).exists()]
    options = nearby[:1] or list(root.rglob(name))
    if len(options) != 1:
        raise RuntimeError(f'Attach the full competition SIFT input; ambiguous/missing {name}: {options}')
    target = options[0]; link = INPUT/name
    if link.exists() and link.resolve() != target.resolve(): raise RuntimeError(f'Stale SIFT link: {link}')
    if not link.exists(): link.symlink_to(target, target_is_directory=target.is_dir())
    print(split, target)
print('Metadata:', DATA_ROOT)
'''),
md('''Run the next cell to completion. It saves the identity assignments, training history, source hashes, baseline and verifier checkpoints, calibrated models and per-query validation predictions. Changing settings/source requires a new OUTPUT folder. No competition submission is generated or uploaded.'''),
code('''command = [sys.executable, '-u', str(CODE/'run.py'),
    '--data', str(INPUT), '--output', str(OUTPUT), '--seed', str(SEED),
    '--workers', str(WORKERS), '--epochs', str(MLP_MAX_EPOCHS),
    '--max-pairs', str(MAX_TRAIN_PAIRS), '--baseline-sims', str(BASELINE_EPISODES_PER_STAGE),
    '--calibration-episodes', str(CALIBRATION_EPISODES), '--trees', str(CANDIDATE_TREES)]
from IPython.display import FileLink, display
if MODE == 'train':
    try:
        subprocess.run(command, check=True)
    finally:
        if (OUTPUT/'config.json').exists():
            persistence.save_bundle(OUTPUT, CODE, BACKUP)
            display(FileLink(str(BACKUP)))
else:
    print('Using restored models; skipped training/validation.')
'''),
code('''import pandas as pd
report = json.loads((OUTPUT/'metrics.json').read_text())
display(pd.DataFrame(report['methods']).T)
print('MLP gain beyond summary-only calibration:', report['mlp_minus_summary_map5'])
print('Strict primary promotion gate passed:', report['promotion_gate_passed'])
for episode, scores in report['episodes'].items():
    print('Episode', episode)
    display(pd.DataFrame(scores).T)
for species, scores in report['species'].items():
    print('Species', species)
    display(pd.DataFrame(scores).T)
'''),
code('''# Independent score and identity-role checks.
pred = pd.read_csv(OUTPUT/'validation_predictions.csv')
roles = pd.read_csv(OUTPUT/'identity_roles.csv')
assert roles.groupby('identity').role.nunique().max() == 1
pair_roles = pd.read_csv(OUTPUT/'pair_identity_roles.csv')
assert pair_roles.groupby('identity').role.nunique().max() == 1
for method, metrics in report['methods'].items():
    scores = []
    for row in pred.itertuples(index=False):
        labels = getattr(row, method+'_predictions').split()
        assert len(labels) == len(set(labels)) == 5
        score = next((1/(i+1) for i, label in enumerate(labels) if label == row.truth), 0.)
        assert abs(score-getattr(row, method+'_rr5')) < 1e-12
        scores.append(score)
    assert abs(sum(scores)/len(scores)-metrics['map5']) < 1e-12
print('All per-query scores and pooled MAP@5 values recompute correctly.')
from IPython.display import FileLink
display(FileLink(str(OUTPUT/'metrics.json')))
display(FileLink(str(OUTPUT/'validation_predictions.csv')))
'''),
md('''## Keep the trained models across sessions

**Files in `/kaggle/working` alone are temporary. Download `model11_checkpoint_bundle.zip` using the link above, or preserve it in the saved notebook outputs before ending the session.** The archive includes baseline models, MLP weights/scaler, calibrators, the current SIFT cache, configuration, metrics and source snapshots. Every file has a verified SHA-256 checksum.

For a later submission: attach this ZIP as Kaggle input alongside the competition data, set `MODE = 'submission'` in the first settings cell, and Run All. The archive is discovered automatically if there is exactly one matching ZIP. Otherwise set `RESTORE_ARCHIVE` explicitly. Training and validation are skipped. The final submission cell uses the restored checkpoints and reuses cached SIFT pairs. Pairs needed only for full-gallery inference still need computing once.

If training failed, the backup may be partial; its manifest lists missing files. A partial backup can preserve completed stages/cache for resuming identical training, but cannot generate submissions until all required checkpoints exist. Save/download the refreshed bundle after submission too; it includes the expanded full-gallery SIFT cache.'''),
md('''Interpretation: compare `mlp_blend` with `baseline` for the primary result, and with `summary_blend` to assess the incremental value of vector-pair evidence. Promotion requires a positive overall gain with a positive identity-cluster bootstrap lower bound, positive gain in every episode, and no decrease in known MAP@5, actual-new MAP@5 or actual-new top-one detection. A failed gate means retain the baseline. Passing is development evidence and still does not guarantee a competition gain. Report negative results too; do not select a different blend from validation output.''')]
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'},'kaggle':{'accelerator':'none','isInternetEnabled':False}},'nbformat':4,'nbformat_minor':5}
for cell in cells:
    text=''.join(cell['source'])
    if cell['cell_type']=='code' and ('report = json.loads' in text or '# Independent score' in text):
        cell['source']=('if MODE == "train":\n'+''.join('    '+line+'\n' for line in text.splitlines())).splitlines(True)
for i,cell in enumerate(cells):cell['id']=f'model11-{i:02d}'
path=HERE/'MLP_NewAnimal_Kaggle.ipynb';path.write_text(json.dumps(nb,indent=1),encoding='utf-8')
print(path)
import runpy
runpy.run_path(str(HERE/'make_submission_cell.py'))
