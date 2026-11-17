import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from cs2pred.splits import series_split
from cs2pred.research_features import geometry,temporal_deltas,KEY
from research_experiment import score,threshold
from research_report import paired

class ResearchTests(unittest.TestCase):
    def test_fixed_class_macro(self):
        self.assertEqual(score([1,1],[.9,.8])['macro_f1'],.5)
        self.assertIsNone(score([1,1],[.9,.8])['roc_auc'])

    def test_threshold_tie(self):
        self.assertEqual(threshold([0,1],[.1,.9])[0],.5)

    def test_macro_matches_sklearn(self):
        from sklearn.metrics import f1_score
        rng=np.random.default_rng(7); y=rng.integers(2,size=100); p=rng.random(100)
        self.assertAlmostEqual(score(y,p)['macro_f1'],f1_score(y,p>=.5,labels=[0,1],average='macro'))

    def test_split_stable_under_row_permutation(self):
        d=pd.DataFrame({'match_id':np.repeat(np.arange(40),3)})
        pd.testing.assert_series_equal(series_split(d),series_split(d.sample(frac=1,random_state=3)))

    def test_actual_pair_distance(self):
        d=pd.DataFrame({'match_id':[1,1,1],'map_name':['m']*3,'round':[1]*3,'t_s':[5]*3,
            'player_slot':[0,1,2],'team':[3,3,2],'hp':[100]*3,'x':[0,6,3],'y':[0,0,4],'spd_win_mean':[0]*3})
        f=geometry(d)
        self.assertEqual(f.geo_enemy_distance_min.iloc[0],5)
        self.assertEqual(f.geo_enemy_distance_mean.iloc[0],5)
        self.assertEqual(f.geo_ct_pair_mean.iloc[0],6)
        with self.assertRaises(ValueError): geometry(pd.concat([d,d.iloc[:1]]))

    def test_future_mutation_invariant(self):
        d=pd.DataFrame({'match_id':[1]*3,'map_name':['m']*3,'round':[1]*3,'prediction_time_s':[5,10,15],'x':[1.,3.,9.]})
        a=temporal_deltas(d,['x']); d.loc[d.prediction_time_s>10,'x']=999
        b=temporal_deltas(d,['x'])
        pd.testing.assert_frame_equal(a[a.prediction_time_s<=10],b[b.prediction_time_s<=10])
        self.assertEqual(a.delta_x.iloc[1],.4)

    def test_single_cluster_no_inference(self):
        d=pd.DataFrame({'match_id':[1,1],'map_name':['m','m'],'round':[1,2],'y_ct_win':[0,1],'prediction':[0,1]})
        self.assertTrue(np.isnan(paired(d,d)['p_value']))
        self.assertEqual(paired(d,d)['difference'],0)

if __name__=='__main__': unittest.main()
