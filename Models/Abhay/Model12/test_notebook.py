"""Execute every notebook code cell locally and independently audit the CSV."""
import sys,os,json,ast,csv,hashlib,shutil
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tmp/abhay_deps'))
import numpy as np
import pandas as pd
import joblib

def main():
    path=HERE/'Model12_LargerGallery_Submission_Kaggle.ipynb'
    nb=json.loads(path.read_text(encoding='utf-8'))
    assert nb['nbformat']==4 and nb['nbformat_minor']==5
    assert len({c['id'] for c in nb['cells']})==len(nb['cells'])
    work=ROOT/'tmp/model12_submission_notebook'
    os.environ['MODEL12_WORK']=str(work)
    os.environ['MODEL12_INPUT_ROOT']=str(Path.home()/'Downloads')
    os.environ['MODEL12_DATA_ROOT']=str(Path.home()/'Downloads/reid_student_data_main')
    env={'__name__':'__notebook__'}
    for i,cell in enumerate(nb['cells']):
        assert isinstance(cell['source'],list) and cell['cell_type'] in ['code','markdown']
        if cell['cell_type']=='code':
            assert cell['outputs']==[] and cell['execution_count'] is None
            source=''.join(cell['source']);ast.parse(source)
            print('EXECUTING CELL',i,flush=True)
            exec(compile(source,f'<notebook-cell-{i}>','exec'),env)
    inference=env['inference']
    assert Path(inference.__file__).resolve()==(work/'model12_code/submission.py').resolve()
    context=env['context'];submission=env['submission']
    # Full-test parity: portable native boosters versus the actual fitted pickle.
    import data_protocol as protocol
    import dalvin_core as b
    native=env['models'];original=joblib.load(HERE/'local_runs/large_2052/large_models.joblib')
    s=context['frame'];native_frame=context['frame'];max_error=0.
    for index,(fitted,portable) in enumerate(zip(original,native)):
        r=b.predict(fitted,s);n=b.predict(portable,native_frame)
        for key in ['pn','raw','pc','pb']:
            err=float(np.max(np.abs(r[key]-n[key])));max_error=max(max_error,err)
            np.testing.assert_allclose(r[key],n[key],atol=1e-12,rtol=0)
        assert np.array_equal(b.rank(s,r),b.rank(native_frame,n))
        if index<2:s=b.augment(s,r);native_frame=b.augment(native_frame,n)
    # Independent standard-library parsing against the supplied sample and labels.
    root=Path(os.environ['MODEL12_DATA_ROOT'])
    with (root/'sample_submission.csv').open(newline='',encoding='utf-8-sig') as f:sample=list(csv.DictReader(f))
    with (root/'train.csv').open(newline='',encoding='utf-8-sig') as f:train=list(csv.DictReader(f))
    with (root/'test.csv').open(newline='',encoding='utf-8-sig') as f:test=list(csv.DictReader(f))
    catalog={}
    for row in train:catalog.setdefault(row['species'],set()).add(row['identity'])
    species={r['id']:r['species'] for r in test}
    with (work/'submission.csv').open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['id','predictions'];rows=list(reader)
    assert len(rows)==len(test)==len(sample)==2791
    assert [r['id'] for r in rows]==[r['id'] for r in sample]
    assert len({r['id'] for r in rows})==len(rows)
    for row in rows:
        labels=row['predictions'].split(' ')
        assert len(labels)==len(set(labels))==5 and all(labels)
        assert set(labels)<=catalog[species[row['id']]]|{'new_individual'}
    # Reject plausible bad formats, including a correctly named but indexed CSV.
    bad=[]
    bad.append(submission.iloc[:-1].copy())
    x=submission.copy();x.loc[1,'id']=x.loc[0,'id'];bad.append(x)
    bad.append(submission.iloc[::-1].reset_index(drop=True))
    x=submission.copy();x['Unnamed: 0']=range(len(x));bad.append(x)
    x=submission.copy();x.loc[0,'predictions']='new_individual';bad.append(x)
    x=submission.copy();x.loc[0,'predictions']=' '.join(['new_individual']*5);bad.append(x)
    x=submission.copy();labels=x.loc[0,'predictions'].split();labels[0]='new_individual_typo';x.loc[0,'predictions']=' '.join(labels);bad.append(x)
    x=submission.copy();sp=species[x.loc[0,'id']];other=next(v for v in catalog if v!=sp)
    labels=x.loc[0,'predictions'].split();labels[0]=next(iter(catalog[other]));x.loc[0,'predictions']=' '.join(labels);bad.append(x)
    x=submission.copy();x.loc[0,'predictions']=x.loc[0,'predictions'].replace(' ','  ');bad.append(x)
    rejected=0
    for frame in bad:
        try:inference.validate_submission(frame,context['sample'],context['test'],context['train'])
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid submission format was accepted')
    report=dict(all_checks_passed=True,all_notebook_code_cells_executed=True,
        native_vs_fitted_model_max_probability_error=max_error,
        native_and_original_top5_identical_all_three_stages=True,
        independent_csv_rows=len(rows),exact_sample_headers_and_order=True,
        five_unique_species_valid_labels_every_row=True,bad_format_cases_rejected=rejected,
        csv_sha256=hashlib.sha256((work/'submission.csv').read_bytes()).hexdigest(),
        notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        kaggle_cloud_execution=False)
    (HERE/'notebook_checks.json').write_text(json.dumps(report,indent=2))
    shutil.copy2(work/'submission.csv',HERE/'submission.csv')
    shutil.copy2(work/'submission.validation.json',HERE/'submission.validation.json')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
