"""Portable, checksummed Model 11 checkpoint bundles; never includes input data."""
from pathlib import Path,PurePosixPath
import json,hashlib,zipfile,os,platform,tempfile,shutil
REQUIRED=('config.json','baseline_models.joblib','verifier.joblib','calibrators.joblib')

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def save_bundle(run,code,archive):
    run,code,archive=map(Path,(run,code,archive));archive.parent.mkdir(parents=True,exist_ok=True)
    if (run/'config.json').exists():
        config=json.loads((run/'config.json').read_text())
        for name in ('dalvin_core.py','data_protocol.py','verifier.py'):
            expected=config.get('sources',{}).get(name)
            if expected and (not (code/name).exists() or digest(code/name)!=expected):
                raise ValueError(f'Backup source does not match the trained checkpoint: {name}')
    files=[]
    for label,folder in [('run',run),('code',code)]:
        if folder.exists():
            for path in sorted(folder.iterdir()):
                if path.is_file() and path.suffix not in ('.tmp','.zip','.pyc'):
                    files.append((f'{label}/{path.name}',path))
    if not files:raise FileNotFoundError('No checkpoints or source files to save')
    manifest=dict(format=1,python=platform.python_version(),
        complete_for_submission=all((run/name).is_file() for name in REQUIRED),
        missing_for_submission=[name for name in REQUIRED if not (run/name).is_file()],files={})
    temp=archive.with_suffix('.tmp')
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for name,path in files:
                before=digest(path);z.write(path,name)
                if digest(path)!=before:raise RuntimeError(f'File changed during backup: {path}; retry after computation stops')
                manifest['files'][name]=before
            z.writestr('manifest.json',json.dumps(manifest,indent=2))
        # Verify the actual archive contents before publishing the backup.
        with zipfile.ZipFile(temp) as z:
            for name,expected in manifest['files'].items():
                if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise ValueError('Backup checksum mismatch')
        os.replace(temp,archive)
    finally:
        if temp.exists():temp.unlink()
    print('Saved checkpoint bundle:',archive,'complete for submission:',manifest['complete_for_submission'])
    return manifest

def restore_bundle(archive,run,code):
    run,code=Path(run),Path(code)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('manifest.json'))
        if manifest.get('format')!=1:raise ValueError('Unknown checkpoint bundle format')
        if len(z.namelist())!=len(set(z.namelist())):raise ValueError('Duplicate archive members')
        if set(z.namelist())!=set(manifest['files'])|{'manifest.json'}:raise ValueError('Unexpected archive members')
        for name,expected in manifest['files'].items():
            parts=PurePosixPath(name).parts
            if len(parts)!=2 or parts[0] not in ('run','code') or parts[1] in ('.','..') or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive path')
            if hashlib.sha256(z.read(name)).hexdigest()!=expected:raise ValueError(f'Corrupt bundle: {name}')
            destination=(run if parts[0]=='run' else code)/parts[1]
            if destination.exists() and digest(destination)!=expected:
                raise ValueError(f'Restore would overwrite different files at {destination}. Use empty OUTPUT/CODE folders.')
        # All members and existing destinations are verified before writing.
        for name in manifest['files']:
            group,filename=PurePosixPath(name).parts
            destination=(run if group=='run' else code)/filename
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(z.read(name))
    print('Restored verified checkpoints and sources:',run)
    return manifest
