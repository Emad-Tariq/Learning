"""Train the graph scorer and evaluate the published Model 8 holdout."""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from protocol import HERE, inputs, prepare, benchmarks
from graph_model import NEW, fit_coordinates, make_frame
import reconstruction

OUT = HERE / 'results'
CACHE = HERE.parents[2] / 'tmp/model17'


def episodes(data, pool, simulate, count):
    names = np.unique(data['identity'][pool])
    for i in range(count):
        rng = np.random.default_rng(47101+i)
        frac = (.27,.45,.73,.38,.82,.55)[i%6]
        keep = rng.choice(names,int(frac*len(names)),replace=False)
        target = pool[np.isin(data['identity'][pool],keep)]
        background = np.setdiff1d(pool,target)
        split = simulate(3300+i,target)
        g,q,y = split['gallery'],split['order'],split['truth']
        if i%4 != 0:
            g = np.r_[g,background]
        if i%5 == 3:
            # Vary the number of repeat encounters when learning the unknown class.
            select = rng.random(len(q)) < .6
            q,y = q[select],y[select]
        yield g,q,y


def metrics(pred, y):
    ap = ((pred == y[:,None]) / np.arange(1,6)).sum(1)
    return dict(n=len(y), map5=float(ap.mean()), known=float(ap[y!=NEW].mean()),
                novel=float(ap[y==NEW].mean()),top1=float((pred[:,0]==y).mean()))


def cached_frame(key,data,z,g,q,y):
    path = CACHE / (key+'.pkl')
    if path.exists():
        return joblib.load(path)
    start = time.monotonic()
    frame = make_frame(data,z,g,q,y)
    joblib.dump(frame,path,compress=1)
    print(key,len(g),len(q),round(time.monotonic()-start,1),'seconds',flush=True)
    return frame


def train(frames):
    # One conditional distribution and an independently learned binary scorer.
    # Combine their log probabilities before ranking.
    primary = reconstruction.train(frames)
    cols = primary[1]
    matrix = pd.concat([f[cols].astype('f') for f in frames],ignore_index=True)
    target = np.concatenate([f.hit for f in frames])
    classifier = lgb.LGBMClassifier(n_estimators=430,num_leaves=23,min_child_samples=85,
        reg_lambda=7,learning_rate=.045,verbosity=-1,n_jobs=5,random_state=715,
        colsample_bytree=.86).fit(matrix,target)
    return primary,classifier


def score(model,frame):
    main, binary = model
    logits = reconstruction.logits(main,frame)
    sizes = frame.groupby('row',sort=True).size().to_numpy()
    prob = reconstruction.grouped_probability(logits,sizes)
    binary_prob = binary.predict_proba(frame[main[1]].astype('f'))[:,1]
    return .54*np.log(np.maximum(prob,1e-12)) + .46*np.log(np.maximum(binary_prob,1e-12))


def main(mode, count):
    data,dev,audit,simulate = prepare()
    production = mode=='production'
    pool = np.arange(len(data['train'])) if production else dev
    z = fit_coordinates(data,pool)
    OUT.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True,exist_ok=True)
    prefix = 'prod' if production else 'dev'
    modelfile = OUT/'identity_rankers.joblib' if production and count==24 else CACHE / f'{prefix}_model_{count}.pkl'
    if not modelfile.exists():
        frames = [cached_frame(f'{prefix}_{i}',data,z,g,q,y)
                  for i,(g,q,y) in enumerate(episodes(data,pool,simulate,count))]
        model = train(frames)
        joblib.dump(model,modelfile)
        del frames
    else:
        model = joblib.load(modelfile)
    if production:
        q = np.arange(len(pool),len(data['x']))
        frame = cached_frame('test',data,z,pool,q,None)
        preds = reconstruction.top5(score(model,frame),frame)
        sample = pd.read_csv(inputs.DATA/'sample_submission.csv')
        order = data['index'].get_indexer(sample.id)-len(pool)
        assert len(sample)==len(q) and np.array_equal(np.sort(order),np.arange(len(q)))
        lookup = data['train'].groupby('species').identity.agg(set).to_dict()
        for row,sp in zip(preds,data['species'][q]):
            assert len(set(row))==5 and set(row)<=(lookup[sp]|{NEW})
        sample['predictions'] = [' '.join(row) for row in preds[order]]
        dest = OUT / 'submission_graph.csv'
        sample.to_csv(dest,index=False)
        assert pd.read_csv(dest).equals(sample)
        print(dest,flush=True)
        return
    if not (HERE/'audit_links.npz').exists():
        raise RuntimeError('Run geometry.py first to reproduce Model 8 holdout episodes.')
    result = []
    for size,seed,g,q,y in benchmarks(dev,audit,simulate):
        f = cached_frame(f'hold_{seed}',data,z,g,q,y)
        pred = reconstruction.top5(score(model,f),f)
        print(seed,metrics(pred,y),flush=True)
        result.append(pd.DataFrame(dict(size=size,seed=seed,id=data['ids'][q],
            animal=data['identity'][q],truth=y,**{f'p{i+1}':pred[:,i] for i in range(5)})))
    result = pd.concat(result,ignore_index=True)
    result.to_csv(OUT/f'holdout_{count}.csv',index=False)
    sums = {kind:metrics(grp[[f'p{i}' for i in range(1,6)]].to_numpy(),grp.truth.to_numpy())
            for kind,grp in result.groupby('size')}
    print(sums,flush=True)
    (OUT/f'metrics_{count}.json').write_text(json.dumps(sums,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['evaluate','production'])
    parser.add_argument('--count',type=int,default=24)
    args = parser.parse_args()
    with threadpool_limits(limits=4,user_api='blas'):
        main(args.mode,args.count)
