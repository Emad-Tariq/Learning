"""Create one pasteable cell for the already-completed Kaggle notebook."""
from pathlib import Path
import json,ast
HERE=Path(__file__).resolve().parent
source=(HERE/'predict_submission.py').read_text(encoding='utf-8')
helper=(HERE/'prepare_submission.py').read_text(encoding='utf-8')
sources={name:(HERE/name).read_text(encoding='utf-8') for name in ('dalvin_core.py','data_protocol.py','verifier.py')}
persistence_source=(HERE/'persistence.py').read_text(encoding='utf-8')
cell='''# Paste this ENTIRE cell at the end of the completed Model 11 Kaggle notebook.
from pathlib import Path
import subprocess, sys
from IPython.display import FileLink, display

# Optional: if several saved runs are attached, set OUTPUT to the desired folder.
# Otherwise checkpoints are discovered automatically in working files and saved inputs.
CHECKPOINT_CODE = globals().get('CODE')
'''
cell+='\nRECOVERY_SOURCE = '+repr(helper)+'\n'
cell+='TRAINING_SOURCES = '+repr(sources)+'\n'
cell+="exec(RECOVERY_SOURCE)\nCODE, OUTPUT, INPUT, SAMPLE = prepare(TRAINING_SOURCES, preferred=globals().get('OUTPUT'))\n"
cell+='\nSUBMISSION_SOURCE = '+repr(source)+'\n'
cell+='\nPERSISTENCE_SOURCE = '+repr(persistence_source)+'\n'
cell+='''(CODE/'predict_submission.py').write_text(SUBMISSION_SOURCE, encoding='utf-8')
exec(PERSISTENCE_SOURCE)
try:
    subprocess.run([sys.executable, '-u', str(CODE/'predict_submission.py'),
        '--data', str(INPUT), '--sample', str(SAMPLE), '--run', str(OUTPUT),
        '--workers', '4'], check=True)
finally:
    backup_code = Path(CHECKPOINT_CODE) if CHECKPOINT_CODE and (Path(CHECKPOINT_CODE)/'run.py').exists() else CODE
    bundle = Path('/kaggle/working/model11_checkpoint_bundle.zip')
    save_bundle(OUTPUT, backup_code, bundle)
    display(FileLink(str(bundle)))
display(FileLink(str(OUTPUT/'submission_mlp_blend.csv')))
display(FileLink(str(OUTPUT/'submission_baseline.csv')))
'''
ast.parse(cell)
(HERE/'Kaggle_Submission_Cell.py').write_text(cell,encoding='utf-8')
# Also append the same cell to the self-contained notebook. Earlier cells and
# training sources remain byte-for-byte unchanged, allowing checkpoint reuse.
path=HERE/'MLP_NewAnimal_Kaggle.ipynb';nb=json.loads(path.read_text(encoding='utf-8'))
nb['cells']=[c for c in nb['cells'] if c.get('id') not in ['submission-help','submission-generate']]
nb['cells']+=[dict(cell_type='markdown',id='submission-help',metadata={},source=[
    '## Optional: generate competition submissions\n',
    'Run the next cell only when you want test predictions. It uses the saved models and the full labelled gallery. Full-population SIFT verification can take substantial time; its cache is saved under the existing output folder. This creates the requested MLP-blend CSV and a baseline control. It does not upload to Kaggle.\n']),
    dict(cell_type='code',id='submission-generate',metadata={},execution_count=None,outputs=[],
        source=('if globals().get("MODE") == "submission" or globals().get("GENERATE_SUBMISSION", False):\n'+
                ''.join('    '+line+'\n' for line in cell.splitlines())+
                'else:\n    print("Submission skipped. Set GENERATE_SUBMISSION=True or MODE=\\"submission\\" to generate it.")\n').splitlines(True))]
path.write_text(json.dumps(nb,indent=1),encoding='utf-8')
print('Updated notebook and created pasteable submission cell')
