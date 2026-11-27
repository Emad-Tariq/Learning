"""Recover completed Model 11 outputs and Kaggle inputs without refitting."""
from pathlib import Path
import json,hashlib,shutil,zipfile

REQUIRED=('config.json','baseline_models.joblib','verifier.joblib','calibrators.joblib')


def recover_archives(roots,working):
    """Recognize checkpoint ZIPs by content, including renamed uploaded files."""
    archives=[]
    for root in roots:
        if root.exists():
            archives.extend(path for path in root.rglob('*') if path.is_file() and path.suffix.lower()=='.zip')
    recovered=[]
    for archive in archives:
        try:
            with zipfile.ZipFile(archive) as z:
                if 'manifest.json' not in z.namelist():continue
                manifest=json.loads(z.read('manifest.json'))
                if manifest.get('format')!=1 or not all('run/'+name in manifest.get('files',{}) for name in REQUIRED):continue
                # Verify all payloads before exposing any checkpoint files.
                for name,expected in manifest['files'].items():
                    parts=name.split('/')
                    if len(parts)!=2 or parts[0] not in ('run','code') or parts[1] in ('','..','.') or '\\' in name or ':' in name:
                        raise ValueError('Unsafe checkpoint archive member')
                    if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise ValueError(f'Checkpoint checksum failed: {name}')
                key=hashlib.sha256(z.read('manifest.json')).hexdigest()[:12]
                run=working/f'model11_restored_{key}'
                for name,expected in manifest['files'].items():
                    group,filename=name.split('/')
                    target=run/filename if group=='run' else run/'saved_sources'/filename
                    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
                        raise ValueError(f'Restore would overwrite another file: {target}')
                for name in manifest['files']:
                    group,filename=name.split('/')
                    target=run/filename if group=='run' else run/'saved_sources'/filename
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
                print('Restored checkpoint ZIP:',archive,'->',run)
                recovered.append(run)
        except zipfile.BadZipFile:continue
    return sorted(set(recovered))


def find_runs(roots):
    complete=[];partial=[]
    for root in roots:
        if not root.exists():continue
        for file in root.rglob('config.json'):
            try:cfg=json.loads(file.read_text())
            except (OSError,ValueError):continue
            if cfg.get('primary')!='mlp_blend':continue
            missing=[name for name in REQUIRED if not (file.parent/name).is_file()]
            if missing:partial.append((str(file.parent),missing))
            else:complete.append(file.parent)
    return sorted(set(complete)),partial


def prepare(sources,preferred=None,roots=None,working=Path('/kaggle/working')):
    roots=roots or [working,Path('/kaggle/input')]
    runs,partial=find_runs(roots)
    if not runs:
        recover_archives(roots,working)
        runs,partial=find_runs(roots)
    preferred=Path(preferred) if preferred is not None else None
    if preferred in runs:selected=preferred
    elif len(runs)==1:selected=runs[0]
    elif not runs:
        raise FileNotFoundError('No complete Model 11 checkpoints found in /kaggle/working or /kaggle/input. '
            'Attach the saved OUTPUT of your completed training notebook via Kaggle Add Input, then rerun this cell. '
            'If those checkpoint files were never saved, the metric table cannot recreate them and training must be rerun. '
            f'Partial runs and missing files: {partial}. '
            f'Attached input folders: {[str(path) for root in roots if root.exists() for path in root.iterdir() if path.is_dir()]}')
    else:raise ValueError(f'Multiple completed runs found. Set OUTPUT to the intended folder and rerun: {runs}')
    print('Found completed checkpoints:',selected)
    output=selected
    if not selected.is_relative_to(working):
        output=working/'model11_submission_from_saved';output.mkdir(parents=True,exist_ok=True)
        for name in list(REQUIRED)+[f.name for f in selected.glob('sift_mutual_*.npz')]:
            shutil.copy2(selected/name,output/name)
        print('Copied checkpoints into writable folder:',output)
    config=json.loads((output/'config.json').read_text())
    code=working/'model11_submission_code';code.mkdir(parents=True,exist_ok=True)
    for name,text in sources.items():
        expected=config['sources'][name]
        variants=[text.encode(),text.replace('\n','\r\n').encode()]
        chosen=next((raw for raw in variants if hashlib.sha256(raw).hexdigest()==expected),None)
        if chosen is None:raise ValueError(f'Training source differs for {name}; use the sources from that saved notebook version')
        (code/name).write_bytes(chosen)
    input_root=Path('/kaggle/input')
    options=[file.parent for file in input_root.rglob('miewid_features.npy')
             if all((file.parent/name).exists() for name in ('train.csv','test.csv','miewid_ids.csv'))]
    if len(options)!=1:raise ValueError(f'Attach the competition feature data (one metadata folder expected): {options}')
    data=options[0];inputs=working/'model11_submission_input';inputs.mkdir(exist_ok=True)
    for name in ('train.csv','test.csv','miewid_ids.csv','miewid_features.npy'):
        dest=inputs/name;target=data/name
        if dest.is_symlink() and not dest.exists():dest.unlink()
        if not dest.exists():dest.symlink_to(target)
        if dest.resolve()!=target.resolve():raise ValueError(f'Stale input link: {dest}')
    for split in ('train','test'):
        name=f'sift_features_{split}.npz'
        nearby=[base/name for base in (data,data.parent,data.parent.parent) if (base/name).exists()]
        options=nearby[:1] or list(input_root.rglob(name))
        if len(options)!=1:raise ValueError(f'Attach full competition SIFT input: {name}: {options}')
        target=options[0];dest=inputs/name
        if dest.is_symlink() and not dest.exists():dest.unlink()
        if not dest.exists():dest.symlink_to(target,target_is_directory=target.is_dir())
        if dest.resolve()!=target.resolve():raise ValueError(f'Stale SIFT link: {dest}')
    import pandas as pd
    ids=set(pd.read_csv(inputs/'test.csv').id)
    candidates=[data/'sample_submission.csv'] if (data/'sample_submission.csv').exists() else list(input_root.rglob('sample_submission.csv'))
    matches=[]
    for file in candidates:
        frame=pd.read_csv(file)
        if 'id' in frame and frame.id.is_unique and set(frame.id)==ids:matches.append(file)
    if len(matches)!=1:raise ValueError(f'Cannot locate an unambiguous sample submission: {matches}')
    return code,output,inputs,matches[0]
