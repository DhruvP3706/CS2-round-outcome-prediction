"""Independent artifact validation: cohorts, metrics, thresholds and group isolation."""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from research_experiment import ROOT,KEY,ROUND_KEY,LABEL_COL,DEFAULT_OUT,threshold,write_json

def verify(out):
    manifest=pd.read_csv(out/'split_manifest.csv'); assert not manifest.duplicated(ROUND_KEY).any()
    assert manifest.groupby('match_id').split.nunique().max()==1
    labels=manifest.set_index(ROUND_KEY)[LABEL_COL]
    checked=0
    for path in sorted((out/'checkpoints').glob('*.json')):
        records=json.loads(path.read_text())['metrics']
        p=pd.read_csv(out/'predictions'/f'{path.stem}.csv.gz')
        assert not p.duplicated(KEY).any()
        assert p.p_ct.between(0,1).all() and p.p_ct.notna().all()
        actual=labels.reindex(pd.MultiIndex.from_frame(p[ROUND_KEY])).to_numpy()
        assert np.array_equal(actual,p[LABEL_COL])
        t=int(p.prediction_time_s.iloc[0]); assert (p.prediction_time_s==t).all()
        for split in ['train','val','test']:
            d=p[p.split==split]
            wanted=manifest[(manifest.split==split)&manifest[f'eligible_{t}s']]
            assert set(map(tuple,d[ROUND_KEY].to_numpy()))==set(map(tuple,wanted[ROUND_KEY].to_numpy())),path
        va=p[p.split=='val']; th,quality=threshold(va[LABEL_COL],va.p_ct)
        assert np.allclose(p.threshold,th),path
        for r in records:
            d=p[p.split==r['split']]; pred=(d.p_ct>=r['threshold']).astype(int)
            assert abs(f1_score(d[LABEL_COL],pred,labels=[0,1],average='macro',zero_division=0)-r['macro_f1'])<1e-12
            assert abs(accuracy_score(d[LABEL_COL],pred)-r['accuracy'])<1e-12
            assert len(d)==r['n']
        checked+=1
    write_json(out/'artifact_verification.json',{'runs_verified':checked,'split_isolation':'passed','cohort_equality':'passed',
        'labels':'passed','validation_threshold_recalculation':'passed','sklearn_accuracy_and_macro_f1':'passed'})
    print('Verified',checked,'run artifacts')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default=DEFAULT_OUT); args=p.parse_args(); verify(ROOT/args.out)
