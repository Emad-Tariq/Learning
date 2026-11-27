"""Frozen comparison against the Kaggle specialist on its separate identity split."""
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from experiment import CACHE, OUT, HERE, inputs, prepare, fit_coordinates, episodes, cached_frame, train
from sightings import link_model, linked_ranks


def main():
    data,_,_,simulate = prepare()
    pool = inputs.roles(data,'27613')
    z = fit_coordinates(data,pool)
    path = CACHE/'confirm_model_24.pkl'
    if not path.exists():
        frames = [cached_frame(f'confirm_{i}',data,z,g,q,y)
                  for i,(g,q,y) in enumerate(episodes(data,pool,simulate,24))]
        model = train(frames)
        joblib.dump(model,path)
        del frames
    else:
        model = joblib.load(path)
    verifier = link_model(data,z,pool,simulate,'confirm')
    rows = []
    for seed,split in inputs.manifest('27613').groupby('episode'):
        query = split[split.assignment!='gallery']
        q = data['index'].get_indexer(query.id)
        g = data['index'].get_indexer(split.loc[split.assignment=='gallery','id'])
        y = query.truth.to_numpy()
        assert q.min()>=0 and g.min()>=0 and not np.intersect1d(g,q).size
        assert not set(data['identity'][pool]).intersection(data['identity'][q])
        f = cached_frame(f'confirm_hold_{seed}',data,z,g,q,y)
        pred = linked_ranks(data,z,g,q,f,model,verifier)
        rr = inputs.reciprocal(pred,y)
        rows.append(pd.DataFrame(dict(episode=seed,id=data['ids'][q],truth=y,graph=rr)))
    known = pd.read_csv(HERE.parent/'Model15/results/specialist_comparison.csv')
    df = pd.concat(rows).merge(known,on=['episode','id','truth'],validate='one_to_one')
    assert len(df)==len(known)==sum(map(len,rows))
    baseline = 'weight_0.3_rr5'
    df['delta'] = df.graph-df[baseline]
    df.to_csv(OUT/'paired_specialist.csv',index=False)
    rng = np.random.default_rng(776)
    sample = np.zeros(4000)
    total = np.zeros(4000)
    for _,grp in df.groupby('species'):
        animals = grp.groupby('identity').delta.agg(['sum','count'])
        ids = rng.integers(0,len(animals),size=(4000,len(animals)))
        sample += animals['sum'].to_numpy()[ids].sum(1)
        total += animals['count'].to_numpy()[ids].sum(1)
    report = dict(queries=len(df),animals=df.identity.nunique(),model17=float(df.graph.mean()),
                  specialist=float(df[baseline].mean()),delta=float(df.delta.mean()),
                  interval=np.quantile(sample/total,[.025,.975]).tolist())
    (OUT/'confirmation.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=3,user_api='blas'):
        main()
