"""Checks for baseline parity, union membership and padding exclusion."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tmp'/'abhay_deps'))
import unittest
from unittest.mock import patch
import importlib.util
import numpy as np
import pandas as pd
from scipy import sparse
import dalvin_core as b

class UnionTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(5)
        b.X=b.unit(rng.normal(size=(100,24)).astype(np.float32)); b.NX=b.X
        b.Y=np.array([f'id{i//4}' for i in range(100)])
        b.SP=np.array(['hyena']*100); b.VIEW=np.arange(100)%10
        b.INL=sparse.csr_matrix((100,100)); b.GOOD=b.INL.copy()
        self.g=np.r_[np.arange(0,100,4),np.arange(1,100,4)]
        self.q=np.arange(2,100,4)

    def test_parity_membership_and_unique_valid_candidates(self):
        path=ROOT/'Models/Abhay/Model7/dalvin_core.py'
        spec=importlib.util.spec_from_file_location('original_core',path)
        original=importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
        for k in ('X','NX','Y','SP','VIEW','INL','GOOD'): setattr(original,k,getattr(b,k))
        b.K=15; b.SELECTOR='baseline'
        expected=original.species_features(self.g,self.q,'hyena')
        actual=b.species_features(self.g,self.q,'hyena')
        for a,e in zip(actual,expected): pd.testing.assert_frame_equal(a,e)
        b.K=30; b.SELECTOR='union'
        q,c=b.species_features(self.g,self.q,'hyena')
        pd.testing.assert_frame_equal(q,expected[0])
        first=c[c['rank']<15].drop(columns='_valid').reset_index(drop=True)
        pd.testing.assert_frame_equal(first,expected[1])
        names,inv=np.unique(b.Y[self.g],return_inverse=True)
        raw,_=b.per_identity(b.X[self.q]@b.X[self.g].T,inv,len(names))
        for j,d in c.groupby('q'):
            chosen=d.loc[d._valid,'ident'].tolist()
            want=set(expected[1].loc[expected[1].q==j,'ident'])|set(names[np.argsort(-raw[j])[:15]])
            self.assertEqual(set(chosen),want); self.assertEqual(len(chosen),len(set(chosen)))

    def test_padding_is_not_fitted_scored_or_ranked(self):
        b.K=30; b.SELECTOR='union'
        q,c=b.species_features(self.g,self.q,'hyena')
        q['unk']=0
        c['y']=(c.ident.to_numpy()==b.Y[self.q][c.q.to_numpy()]).astype(int)
        s={'q':q,'c':c}; calls=[]
        class FakeClassifier:
            def __init__(self,**kwargs): pass
            def fit(self,x,y): calls.append(len(x)); return self
            def predict_proba(self,x): return np.tile([.3,.7],(len(x),1))
        with patch.object(b.lgb,'LGBMClassifier',FakeClassifier):
            models=b.fit([s]); r=b.predict(models,s)
        self.assertEqual(calls,[len(q),int(c._valid.sum()),int(c._valid.sum())])
        mask=c._valid.to_numpy().reshape(len(q),30)
        self.assertTrue((r['raw'][~mask]==0).all()); self.assertTrue((r['pc'][~mask]==0).all())
        np.testing.assert_allclose(r['pc'].sum(1),1)
        self.assertNotIn('_valid',models[-1][1])
        preds=b.rank(s,r)
        self.assertTrue(all(len(set(row))==5 for row in preds))
        augmented=b.augment(s,r)
        self.assertTrue(augmented['c']._valid.equals(c._valid))

if __name__=='__main__': unittest.main()
