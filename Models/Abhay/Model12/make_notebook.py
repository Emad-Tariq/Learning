"""Embed portable trained boosters and the exact full-test SIFT graph."""
import os,sys,json,hashlib,base64,io,zipfile,textwrap,ast
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'tmp/abhay_deps'),str(ROOT/'Models/Abhay/Model7')]
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
import dalvin_core as b
import data_protocol as p
from submission import fingerprint

RUN=HERE/'local_runs/large_2052';DATA=Path.home()/'Downloads/reid_student_data_main'

def digest(value):return hashlib.sha256(value).hexdigest()
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
def code(s):
    ast.parse(s)
    return dict(cell_type='code',metadata={},source=s.splitlines(True),outputs=[],execution_count=None)

def main():
    train,test,all_df=p.load_data(DATA);b.NX=b.X
    models=joblib.load(RUN/'large_models.joblib')
    assets={};schema={'stages':[]}
    for index,model in enumerate(models,1):
        names=[]
        for label,head in zip(['detector','closed','all'],model[:3]):
            assert np.array_equal(head.classes_,[0,1])
            name=f'model/stage{index}_{label}.txt'
            assets[name]=head.booster_.model_to_string(num_iteration=-1).encode('utf-8');names.append(name)
        schema['stages'].append(dict(heads=names,query_features=model[3][0],candidate_features=model[3][1]))
    assets['model_schema.json']=json.dumps(schema,indent=2).encode()
    # Freeze exactly the original 150-neighbour full-data inference graph.
    # Only test/gallery and test/test edges can affect these inference frames.
    wanted=set();n=len(all_df);ntrain=len(train)
    for sp in ['hyena','leopard']:
        ix=np.flatnonzero(b.SP==sp)
        for start in range(0,len(ix),256):
            q=ix[start:start+256];s=b.X[q]@b.X[ix].T
            s[np.arange(len(q)),np.arange(start,start+len(q))]=-np.inf
            nb=np.argsort(-s,axis=1)[:,:150]
            for a,row in zip(q,nb):
                for c in ix[row]:
                    if a>=ntrain or c>=ntrain:wanted.add((min(int(a),int(c)),max(int(a),int(c))))
    pairs=np.array(sorted(wanted),dtype=np.int32)
    with np.load(next(RUN.glob('sift_mutual_*.npz'))) as z:
        saved=z['pairs'];values=z['values']
    keys=saved[:,0].astype(np.int64)*n+saved[:,1]
    query=pairs[:,0].astype(np.int64)*n+pairs[:,1]
    assert (keys[1:]>keys[:-1]).all()
    at=np.searchsorted(keys,query)
    assert (at<len(keys)).all() and np.array_equal(keys[at],query),'Required full-test pairs are missing'
    val=values[at]
    stream=io.BytesIO()
    np.savez_compressed(stream,pairs=pairs,good=val[:,0].astype(np.int32),
        inliers=val[:,1].astype(np.int32),image_count=np.int32(n))
    assets['full_test_sift_graph.npz']=stream.getvalue()
    assets['input_fingerprint.json']=json.dumps(fingerprint(DATA),indent=2).encode()
    assets['training_metrics.json']=(RUN/'metrics.json').read_bytes()
    assets['training_provenance.json']=(RUN/'verified_provenance.json').read_bytes()
    manifest=dict(model='Model 12: larger-gallery Dalvin',training_checkpoint_sha256=digest((RUN/'large_models.joblib').read_bytes()),
        sift_graph_pairs=len(pairs),train_images=ntrain,test_images=len(test),
        files={name:digest(content) for name,content in assets.items()})
    assets['bundle_manifest.json']=json.dumps(manifest,indent=2).encode()
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,content in assets.items():archive.writestr(name,content)
    payload=stream.getvalue()
    sources={name:path.read_text(encoding='utf-8') for name,path in {
        'dalvin_core.py':ROOT/'Models/Abhay/Model7/dalvin_core.py',
        'data_protocol.py':ROOT/'Models/Abhay/Model7/data_protocol.py',
        'submission.py':HERE/'submission.py'}.items()}
    hashes={name:digest(text.encode()) for name,text in sources.items()}
    cells=[md('''# Model 12 — generate the Kaggle submission

**Upload this notebook, attach the original competition dataset, select CPU, and Run All.** The final output is `/kaggle/working/submission.csv`.

This notebook embeds the exact trained larger-gallery Model 12 weights and the already-computed full-test SIFT graph. **No retraining, extra model dataset, raw images, SIFT download, GPU or Internet connection is needed.** Attach the competition `train.csv`, `test.csv`, `sample_submission.csv`, `miewid_ids.csv` and `miewid_features.npy`. If Kaggle provides them as a single competition input, attach that input normally.

The trained stack improved paired larger-gallery local validation MAP@5 from 0.878358 to 0.907654. That is a local result, not a leaderboard estimate. Dalvin's model architecture, PCA/LDA, feature extraction, three-stage stacking and ranking rule are preserved. The supplied train photos all serve as the reference gallery at inference; test identity labels are never used.

Boosters are stored in portable LightGBM text format, rather than a Windows/scikit-learn pickle. The precomputed SIFT graph uses only supplied descriptors and embeddings; input fingerprints prevent accidentally applying its image indices to different data. PCA/LDA and all model scores are computed in this notebook.

The notebook checks both the in-memory predictions and the CSV after writing: exact sample headers and row order, every test ID once, five distinct space-separated labels, and valid species-specific identities or `new_individual`.'''),
    code('''from pathlib import Path
import os, sys, json, hashlib, base64, io, zipfile, importlib

# Defaults work on Kaggle. Set DATA_ROOT only if auto-discovery finds multiple copies.
INPUT_ROOT = Path(os.environ.get('MODEL12_INPUT_ROOT', '/kaggle/input'))
WORK = Path(os.environ.get('MODEL12_WORK', '/kaggle/working'))
DATA_ROOT = Path(os.environ['MODEL12_DATA_ROOT']) if os.environ.get('MODEL12_DATA_ROOT') else None
WORKERS = 4
OUTPUT = WORK / 'submission.csv'
CODE = WORK / 'model12_code'
ASSETS = WORK / 'model12_assets'
CODE.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)
'''),
    code('SOURCES = '+repr(sources)+'\nSOURCE_HASHES = '+repr(hashes)+'\nPAYLOAD_SHA256 = '+repr(digest(payload))+"\nPAYLOAD_B64 = '''\n"+'\n'.join(textwrap.wrap(base64.b64encode(payload).decode(),120))+"\n'''\n"+'''
for name, text in SOURCES.items():
    if hashlib.sha256(text.encode()).hexdigest() != SOURCE_HASHES[name]:
        raise RuntimeError('Embedded source checksum mismatch: ' + name)
    (CODE/name).write_text(text, encoding='utf-8')
archive_bytes = base64.b64decode(PAYLOAD_B64)
if hashlib.sha256(archive_bytes).hexdigest() != PAYLOAD_SHA256:
    raise RuntimeError('Embedded trained-model bundle is corrupted')
with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
    for info in archive.infolist():
        target = (ASSETS/info.filename).resolve()
        if not target.is_relative_to(ASSETS.resolve()):
            raise RuntimeError('Invalid embedded archive path')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(info))
bundle = json.loads((ASSETS/'bundle_manifest.json').read_text())
for name, expected_hash in bundle['files'].items():
    if hashlib.sha256((ASSETS/name).read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError('Extracted asset checksum mismatch: ' + name)
sys.path.insert(0, str(CODE))
print('Verified embedded Model 12:', bundle['test_images'], 'test photos,', bundle['sift_graph_pairs'], 'SIFT pairs')
'''),
    code('''# Standard Kaggle CPU images provide these packages; nothing is downloaded.
for name in ['numpy', 'pandas', 'scipy', 'sklearn', 'lightgbm', 'cv2', 'threadpoolctl']:
    module = importlib.import_module(name)
    print(name, getattr(module, '__version__', 'available'))
print('Versions used for the validated training run:')
print(json.dumps(json.loads((ASSETS/'training_provenance.json').read_text())['packages'], indent=2))

if DATA_ROOT is None:
    options = sorted({p.parent for p in INPUT_ROOT.rglob('miewid_features.npy')
                      if all((p.parent/name).is_file() for name in ['train.csv', 'test.csv', 'miewid_ids.csv'])})
    if len(options) != 1:
        raise RuntimeError(f'Set DATA_ROOT in the settings cell; found these competition folders: {options}')
    DATA_ROOT = options[0]
required = ['train.csv', 'test.csv', 'miewid_ids.csv', 'miewid_features.npy']
for name in required:
    if not (DATA_ROOT/name).is_file(): raise FileNotFoundError(DATA_ROOT/name)
SAMPLE = DATA_ROOT/'sample_submission.csv'
if not SAMPLE.is_file():
    options = list(INPUT_ROOT.rglob('sample_submission.csv'))
    if len(options) != 1: raise RuntimeError('Attach the competition sample_submission.csv alongside the data')
    SAMPLE = options[0]
print('Competition data:', DATA_ROOT)
print('Sample submission:', SAMPLE)
'''),
    code('''import submission as inference
import cv2
from threadpoolctl import threadpool_limits

cv2.setNumThreads(1)
with threadpool_limits(limits=WORKERS):
    context = inference.prepare(DATA_ROOT, ASSETS, SAMPLE)
    models = inference.load_models(ASSETS, WORKERS)
    submission, validation = inference.generate(context, models, OUTPUT)
'''),
    code('''from IPython.display import display, FileLink
display(submission.head(10))
print('Submission format checks:', json.dumps(validation, indent=2))
display(FileLink(str(OUTPUT)))
display(FileLink(str(OUTPUT.with_suffix('.validation.json'))))
print('Submit submission.csv from the notebook Output files. This notebook does not upload it automatically.')
'''),
    md('''## Submit the output

After Run All finishes successfully, save the notebook version to preserve its outputs. Use `submission.csv` in the notebook Output files for the competition submission. Do not submit the notebook or the validation JSON as the prediction file.

The CSV has exactly `id,predictions` as its header. Each predictions field contains five unique identity labels separated by single spaces. `new_individual` is allowed at any ranked position. IDs follow `sample_submission.csv` exactly; the pandas index is excluded.

If the input-fingerprint check fails, verify that the attached data is the original competition input rather than modified feature arrays, reordered feature IDs or a different competition version. The embedded graph belongs to that exact input. Numerical PCA/LDA results can vary slightly with the notebook runtime's library versions; the model weights and matching graph stay fixed.''')]
    for i,cell in enumerate(cells):cell['id']=f'model12-submission-{i:02d}'
    notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.11'),kaggle=dict(accelerator='none',isInternetEnabled=False)),
        nbformat=4,nbformat_minor=5)
    path=HERE/'Model12_LargerGallery_Submission_Kaggle.ipynb'
    path.write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    (HERE/'notebook_bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(path,flush=True)
    print('Notebook bytes',path.stat().st_size,'embedded archive bytes',len(payload),'graph pairs',len(pairs),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
