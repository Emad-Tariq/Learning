"""Use the team notebook's identity allocation and simulator, without its scorers."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'Model16'))
import run as inputs


def select_functions(source, names, namespace):
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'team_model8_split', 'exec'), namespace)


def prepare():
    data = inputs.load(inputs.DATA, inputs.SOURCE)
    path = inputs.SOURCE / 'Models/Emad/model8_multi_sighting.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    parse = ast.parse(''.join(notebook['cells'][2]['source']))
    scripts = {node.targets[0].id: ast.literal_eval(node.value) for node in parse.body
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id in ('_MODEL4C_SOURCE', '_MODEL3E_SOURCE')}
    core = scripts['_MODEL4C_SOURCE']
    constants = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(core).body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ('VIEWS', 'REGIMES', 'FRAC')}
    meta = pd.concat([data['train'], data['test']], ignore_index=True)
    view = meta.viewpoint.map({v: i for i, v in enumerate(constants['VIEWS'])}).fillna(-1).to_numpy(int)
    ctx = dict(np=np, sparse=sparse, connected_components=connected_components,
               X=data['x'], INL=data['inliers'])
    select_functions(core, ['duplicate_groups'], ctx)

    def gather(g, q):
        order = np.concatenate([q[data['species'][q] == sp] for sp in inputs.SPECIES])
        return dict(gallery=g, order=order, q=pd.DataFrame(index=np.arange(len(order))),
                    c=pd.DataFrame({'ident': np.array([], object), 'q': np.array([], int)}))

    c4 = SimpleNamespace(**constants, X=data['x'], Y=data['identity'], SP=data['species'],
        VIEW=view, duplicate_groups=ctx['duplicate_groups'], build_frames=gather)
    ns = dict(np=np, c4=c4, SPECIES=inputs.SPECIES, NEW='new_individual')
    select_functions(scripts['_MODEL3E_SOURCE'], ['identity_folds', 'viewpoint_weights', 'simulate'], ns)
    tr = np.arange(len(data['train']))
    te = np.arange(len(tr), len(meta))
    weights = ns['viewpoint_weights'](tr, te)
    folds = ns['identity_folds'](tr, 5, 7719)
    audit = tr[[folds[c4.Y[i]] == 0 for i in tr]]
    dev = np.setdiff1d(tr, audit)
    links = HERE / 'audit_links.npz'
    if links.exists():
        with np.load(links) as z:
            assert np.array_equal(z['audit'], audit)
            a, b = z['pairs'].T
            exact = sparse.csr_matrix((np.tile(z['inliers'], 2), (np.r_[a,b], np.r_[b,a])),
                                      shape=data['inliers'].shape)
        def split_groups(rows, link):
            if link is None:
                return np.arange(len(rows))
            matrix = exact if folds[c4.Y[rows[0]]] == 0 else data['inliers']
            edges = (c4.X[rows] @ c4.X[rows].T > link[0]) | (matrix[rows][:,rows].toarray() >= link[1])
            return connected_components(sparse.csr_matrix(edges))[1]
        c4.duplicate_groups = split_groups
    data['view'] = view
    return data, dev, audit, lambda seed, ix: ns['simulate'](seed, ix, weights)


def benchmarks(dev, audit, simulate):
    for size, episodes in [('small', range(8550, 8556)), ('large', range(91531, 91534))]:
        for seed in episodes:
            split = simulate(seed, audit)
            gallery = split['gallery'] if size == 'small' else np.r_[split['gallery'], dev]
            yield size, seed, gallery, split['order'], split['truth']


if __name__ == '__main__':
    data, dev, audit, simulate = prepare()
    for size, seed, g, q, truth in benchmarks(dev, audit, simulate):
        print(size, seed, len(g), len(q), flush=True)
