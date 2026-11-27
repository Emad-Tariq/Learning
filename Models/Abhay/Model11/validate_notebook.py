"""Check notebook structure, cell syntax and embedded source extraction."""
import json,ast
from pathlib import Path
HERE=Path(__file__).resolve().parent
nb=json.loads((HERE/'MLP_NewAnimal_Kaggle.ipynb').read_text(encoding='utf-8'))
assert nb['nbformat']==4 and nb['nbformat_minor']==5
assert len({cell['id'] for cell in nb['cells']})==len(nb['cells'])
for cell in nb['cells']:
    assert cell['cell_type'] in ('code','markdown') and isinstance(cell['metadata'],dict)
    assert isinstance(cell['source'],list)
    if cell['cell_type']=='code':
        assert cell['outputs']==[] and cell['execution_count'] is None
        ast.parse(''.join(cell['source']))
code=HERE.parents[2]/'tmp/model11_notebook_check';code.mkdir(parents=True,exist_ok=True)
env={'CODE':code}
exec(''.join(nb['cells'][2]['source']),env)
for name,source in env['SOURCES'].items():
    assert (code/name).read_text(encoding='utf-8')==source
    assert (HERE/name).read_text(encoding='utf-8')==source
print('Notebook structure, every code cell, embedded hashes, extraction and current-source parity passed')
