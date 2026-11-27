import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
import verifier as v


class VerifierTests(unittest.TestCase):
    def test_full_vector_features_are_symmetric(self):
        x=np.array([[.1,.3,.5],[.8,.4,.2]],dtype=np.float32)
        a=v.pair_features(x,np.array([0]),np.array([1]))
        b=v.pair_features(x,np.array([1]),np.array([0]))
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(1,7))

    def test_query_truth_is_not_an_inference_input(self):
        rng=np.random.default_rng(2); x=rng.normal(size=(31,4));y=np.array([f'a{i//2}' for i in range(30)]+['query'])
        ids=np.array([f'a{i}' for i in range(15)])
        c=pd.DataFrame({'ident':ids,**{name:np.arange(15,dtype=float) for name in ['mxr','mxl','sx','n_flank','n_view']}})
        s={'c':c,'q':pd.DataFrame({'sp':[1],'view':[0],'unk':[0]}),'order':np.array([30]),'truth':np.array(['a0'])}
        network=SimpleNamespace(predict_proba=lambda z:np.tile([.4,.6],(len(z),1)))
        scaler=SimpleNamespace(transform=lambda a:a)
        model={'network':network,'scaler':scaler};r={'pc':np.full((1,15),1/15),'pn':np.array([.4]),'pb':np.array([.5]),'raw':np.full((1,15),.03)}
        a=v.query_features(model,x,y,np.arange(30),s,r)
        y[-1]='different';s['truth'][0]='new_individual';s['q']['unk']=1
        b=v.query_features(model,x,y,np.arange(30),s,r)
        for aa,bb in zip(a,b):np.testing.assert_array_equal(aa,bb)

    def test_unknown_and_known_final_ranking(self):
        ids=np.array([[f'a{i}' for i in range(15)]]);pc=np.array([[.8]+[.2/14]*14])
        self.assertEqual(v.rank(ids,pc,np.array([.2]))[0,0],'a0')
        self.assertEqual(v.rank(ids,pc,np.array([.8]))[0,0],'new_individual')
        self.assertEqual(len(set(v.rank(ids,pc,np.array([.2]))[0])),5)


if __name__=='__main__':unittest.main()
