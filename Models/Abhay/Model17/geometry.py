"""Recompute Model 8's duplicate links for the held-out animals."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from protocol import prepare, inputs, HERE


def main():
    data, _, audit, _ = prepare()
    cv2.setNumThreads(1)
    base = inputs.DATA.parent / 'sift_features_train.npz'

    @lru_cache(maxsize=500)
    def desc(i):
        name = data['ids'][i]
        d = np.load(base / (name + '_desc.npy')).astype('f')
        xy = np.load(base / (name + '_xy.npy')).astype('f')[:, :2]
        return np.sqrt(d / np.maximum(d.sum(1, keepdims=True), 1e-12)), xy

    def inliers(pair):
        a, b = map(int, pair)
        da, xa = desc(a)
        db, xb = desc(b)
        if min(len(da), len(db)) < 4:
            return 0
        bf = cv2.BFMatcher()
        reverse = [m.trainIdx for m in bf.match(db, da)]
        matches = [m for m, n in bf.knnMatch(da, db, k=2)
                   if m.distance < .85 * n.distance and reverse[m.trainIdx] == m.queryIdx]
        if len(matches) < 60:
            return 0
        cv2.setRNGSeed((a * 100003 + b) % (2**31 - 1))
        _, mask = cv2.estimateAffinePartial2D(xa[[m.queryIdx for m in matches]],
            xb[[m.trainIdx for m in matches]], method=cv2.RANSAC, ransacReprojThreshold=12.)
        return int(mask.sum()) if mask is not None else 0

    # Enumerate the notebook's full-population top-150 geometry pool first.
    pairs = set()
    held = set(audit)
    for sp in inputs.SPECIES:
        ids = np.flatnonzero(data['species'] == sp)
        for rows in np.array_split(audit[data['species'][audit] == sp], 10):
            sim = data['x'][rows] @ data['x'][ids].T
            sim[np.arange(len(rows)), np.searchsorted(ids, rows)] = -2
            for a, near in zip(rows, ids[np.argpartition(-sim, 149)[:, :150]]):
                for b in near:
                    if b in held and data['identity'][a] == data['identity'][b]:
                        pairs.add(tuple(sorted((a, b))))
    pairs = np.array(sorted(pairs))
    # The earlier cache used the same mutual ratio test, so its good count
    # bounds the number of possible inliers. Uncached pairs are tested fully.
    counts = np.asarray(data['good'][pairs[:, 0], pairs[:, 1]]).ravel()
    pairs = pairs[(counts >= 60) | (counts == 0)]
    print('Audit pairs above possible duplicate threshold:', len(pairs), flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        values = list(pool.map(inliers, pairs))
    np.savez(HERE / 'audit_links.npz', pairs=pairs, inliers=values, audit=audit)
    print('Duplicate graph ready', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=3):
        main()
