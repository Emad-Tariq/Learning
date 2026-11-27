"""Data loading, role-local mutual SIFT and identity-disjoint episode helpers."""
import os
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from threading import Lock
import numpy as np
import pandas as pd
import cv2
from scipy import sparse
import dalvin_core as b
NEW='new_individual'
def find_sift(root, split):
    """Kaggle mounts SIFT beside the metadata folder, sometimes as extracted NPZ directories."""
    name = f"sift_features_{split}.npz"
    for parent in (root, root.parent, root.parent.parent):
        candidate = parent / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find {name} in {root} or its two parent directories. "
                            "Attach the full projectwild3 competition input, including both SIFT folders.")


class SiftSource:
    def __init__(self, path):
        self.archive = None
        self.files = {}
        if path.is_file():
            self.archive = np.load(path)
        else:
            for file in path.rglob("*.npy"):
                if file.stem in self.files:
                    raise ValueError(f"Duplicate SIFT key {file.stem} in {path}")
                self.files[file.stem] = file
            if not self.files:
                raise ValueError(f"No extracted .npy SIFT arrays in {path}")

    def __getitem__(self, key):
        return self.archive[key] if self.archive is not None else np.load(self.files[key], allow_pickle=False)


def load_data(root):
    train = pd.read_csv(root / "train.csv")
    test = pd.read_csv(root / "test.csv")
    ids = pd.read_csv(root / "miewid_ids.csv")
    all_df = pd.concat([train, test.assign(identity=None)], ignore_index=True)
    if not all_df.id.is_unique or not ids.id.is_unique:
        raise ValueError("Duplicate image IDs")
    for col in ("species", "viewpoint", "identity"):
        if col not in all_df:
            raise ValueError(f"Missing column: {col}")
    if train.identity.isna().any() or not set(all_df.species) <= {"hyena", "leopard"}:
        raise ValueError("Invalid identity/species metadata")
    raw = np.load(root / "miewid_features.npy", mmap_mode="r")
    if len(raw) != len(ids) or set(ids.id) != set(all_df.id):
        raise ValueError("Feature ID mapping does not match metadata")
    ix = pd.Index(ids.id).get_indexer(all_df.id)
    X = np.asarray(raw[ix], dtype=np.float32)
    if not np.isfinite(X).all() or np.any(np.linalg.norm(X, axis=1) == 0):
        raise ValueError("Features contain nonfinite values or zero vectors")
    b.X = b.unit(X)
    b.IDS = all_df.id.to_numpy()
    b.Y = all_df.identity.to_numpy(dtype=object)
    b.SP = all_df.species.to_numpy()
    b.VIEW = all_df.viewpoint.map({v: i for i, v in enumerate(b.VIEWS)}).fillna(-1).to_numpy(int)
    return train, test, all_df


class SiftCache:
    """Role-local neighbour selection; thread workers share lazily loaded descriptors."""
    def __init__(self, root, out, workers, neighbours):
        paths = [find_sift(root, s) for s in ("train", "test")]
        print("SIFT sources:", *paths, sep="\n", flush=True)
        self.z = [SiftSource(path) for path in paths]
        self.read_lock = Lock()
        self.workers, self.neighbours = workers, neighbours
        fingerprint = hashlib.sha256(b.IDS.tobytes() if b.IDS.dtype != object else "\n".join(b.IDS).encode())
        fingerprint.update(np.ascontiguousarray(b.X).tobytes())
        self.path = out / ("sift_mutual_" + fingerprint.hexdigest()[:16] + ".npz")
        self.done = {}
        if self.path.exists():
            with np.load(self.path) as z:
                self.done = {tuple(p): tuple(v) for p, v in zip(z["pairs"], z["values"])}

    @lru_cache(maxsize=1200)
    def descriptors(self, i):
        iid = b.IDS[i]
        z = self.z[0 if iid.startswith("tr_") else 1]
        # NpzFile's shared ZIP handle is not safe for concurrent reads.
        with self.read_lock:
            d = np.asarray(z[f"{iid}_desc"], dtype=np.float32)
            xy = np.asarray(z[f"{iid}_xy"], dtype=np.float32)
        if d.ndim != 2 or xy.shape != (len(d), 2):
            raise ValueError(f"Malformed SIFT arrays for {iid}")
        d = np.sqrt(d / np.maximum(d.sum(1, keepdims=True), 1))
        return d, xy

    def verify(self, pair):
        a, c = pair
        da, xa = self.descriptors(a)
        dc, xc = self.descriptors(c)
        if min(len(da), len(dc)) < 4:
            return 0, 0
        bf = cv2.BFMatcher()
        back = np.array([m.trainIdx for m in bf.match(dc, da)])
        matches = bf.knnMatch(da, dc, k=2)
        good = [m[0] for m in matches if len(m) == 2 and m[0].distance < .85*m[1].distance
                and back[m[0].trainIdx] == m[0].queryIdx]
        if len(good) < 4:
            return len(good), 0
        # Local RNG seed per task. OpenCV RNG state is thread-local.
        cv2.setRNGSeed(int((int(a)*100003 + int(c)) % (2**31-1)))
        _, mask = cv2.estimateAffinePartial2D(
            np.float32([xa[m.queryIdx] for m in good]),
            np.float32([xc[m.trainIdx] for m in good]),
            method=cv2.RANSAC, ransacReprojThreshold=15.)
        return len(good), int(mask.sum()) if mask is not None else 0

    def activate(self, indices):
        wanted = set()
        for sp in ("hyena", "leopard"):
            ix = indices[b.SP[indices] == sp]
            # Chunked similarities avoid allocating the full role x role matrix.
            for start in range(0, len(ix), 256):
                q = ix[start:start+256]
                s = b.X[q] @ b.X[ix].T
                s[np.arange(len(q)), np.arange(start, start+len(q))] = -np.inf
                k = min(self.neighbours, len(ix)-1)
                nb = np.argsort(-s, axis=1)[:, :k]
                for a, row in zip(q, nb):
                    wanted.update((min(int(a), int(c)), max(int(a), int(c))) for c in ix[row])
        missing = sorted(wanted - self.done.keys())
        print(f"SIFT: {len(wanted)} role-local pairs; {len(missing)} uncached", flush=True)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for j, (pair, value) in enumerate(zip(missing, pool.map(self.verify, missing)), 1):
                self.done[pair] = value
                if j % 5000 == 0:
                    self.save()
                    print(f"  verified {j}/{len(missing)}", flush=True)
        self.save()
        # Only expose requested pairs, even when cache contains other roles/test.
        pp = np.array(sorted(wanted), dtype=int).reshape(-1, 2)
        vv = np.array([self.done[tuple(p)] for p in pp]).reshape(-1, 2)
        n = len(b.X)
        def matrix(v):
            a, c = pp.T
            return sparse.coo_matrix((np.r_[v, v], (np.r_[a, c], np.r_[c, a])), shape=(n, n)).tocsr()
        b.GOOD, b.INL = matrix(vv[:, 0]), matrix(vv[:, 1])

    def save(self):
        """Persist completed verifications atomically so interrupted runs can resume."""
        pairs = sorted(self.done)
        tmp = self.path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, pairs=np.array(pairs, dtype=np.int64).reshape(-1, 2),
                            values=np.array([self.done[p] for p in pairs]).reshape(-1, 2))
        os.replace(tmp, self.path)


def allocate_roles(train, rounds, seed):
    rng = np.random.default_rng(seed)
    roles = {f"stage{i+1}": [] for i in range(rounds)} | {"calibration": [], "validation": []}
    for sp in ("hyena", "leopard"):
        names = np.array(sorted(train.loc[train.species == sp, "identity"].unique()))
        rng.shuffle(names)
        n = max(1, int(round(.2*len(names))))
        roles["validation"].extend(names[:n])
        roles["calibration"].extend(names[n:2*n])
        for i, block in enumerate(np.array_split(names[2*n:], rounds)):
            roles[f"stage{i+1}"].extend(block)
    sets = [set(v) for v in roles.values()]
    assert all(not a & c for i, a in enumerate(sets) for c in sets[i+1:])
    return {k: np.flatnonzero(train.identity.isin(v).to_numpy()) for k, v in roles.items()}


def split_episode(indices, seed, cap=32, link=(.95, 100)):
    """Unknown identities absent from gallery; known-query duplicate groups kept whole."""
    rng = np.random.default_rng(seed)
    gallery, known, unknown = [], [], []
    for sp in ("hyena", "leopard"):
        ix = indices[b.SP[indices] == sp]
        names = np.unique(b.Y[ix])
        rng.shuffle(names)
        nu = max(1, int(round(rng.uniform(.2, .35)*len(names))))
        unseen = set(names[:nu])
        for name in names:
            rows = ix[b.Y[ix] == name].copy()
            rng.shuffle(rows)
            if name in unseen:
                unknown.extend(rows[:cap])
                continue
            held = []
            groups = b.duplicate_groups(rows, link)
            target = min(max(1, int(round(.35*len(rows)))), cap)
            for c in rng.permutation(np.unique(groups)):
                members = rows[groups == c].tolist()
                if len(held) < target and len(held)+len(members) < len(rows):
                    held.extend(members)
            # Never truncate a duplicate group after selecting it.
            known.extend(held)
            gallery.extend(i for i in rows if i not in set(held))
    gi, qi = np.array(gallery, dtype=int), np.array(known+unknown, dtype=int)
    if not len(known) or not len(unknown):
        raise ValueError("Episode lacks known/unknown queries; use a larger role or different seed")
    for sp in ("hyena", "leopard"):
        if len(np.unique(b.Y[gi[b.SP[gi] == sp]])) < b.K or np.sum(b.SP[qi] == sp) < 6:
            raise ValueError("Too few identities/queries per species; reduce --rounds or --candidates")
    assert not set(b.Y[unknown]) & set(b.Y[gi])
    assert set(b.Y[known]) <= set(b.Y[gi])
    return gi, qi, np.where(np.isin(qi, unknown), NEW, b.Y[qi])


def frame(gi, qi, truth=None):
    s = b.build_frames(gi, qi)
    if truth is not None:
        lookup = dict(zip(qi, truth))
        s["truth"] = np.array([lookup[i] for i in s["order"]])
        s["q"]["unk"] = (s["truth"] == NEW).astype(int)
        s["c"]["y"] = (s["c"].ident.to_numpy() == s["truth"][s["c"].q.to_numpy()]).astype(int)
    return s


def through_stack(s, models):
    for i, m in enumerate(models):
        r = b.predict(m, s)
        if i < len(models)-1:
            s = b.augment(s, r)
    return s, r
