"""Prespecified convex tree ensemble; weights/threshold selected on validation only."""
import argparse
import json
import time
import numpy as np
import pandas as pd
from research_experiment import ROOT,KEY,LABEL_COL,TIMES,DEFAULT_OUT,threshold,score,write_json

def run(out):
    models=['lightgbm','catboost','gradient_boosting']
    weights=np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[.5,0,.5],[0,.5,.5],[1/3,1/3,1/3]])
    manifest=pd.read_csv(out/'split_manifest.csv')
    for t in TIMES:
        tag=f'convex_ensemble__baseline__{t}__42'; done=out/'checkpoints'/f'{tag}.json'
        if done.exists(): continue
        frames=[pd.read_csv(out/'predictions'/f'{m}__baseline__{t}__42.csv.gz').sort_values(KEY).reset_index(drop=True) for m in models]
        for f in frames[1:]: pd.testing.assert_frame_equal(frames[0][KEY+[LABEL_COL,'split']],f[KEY+[LABEL_COL,'split']])
        reference=frames[0]; va=reference.split=='val'
        probs=np.column_stack([f.p_ct for f in frames]); y=reference[LABEL_COL].to_numpy()
        begin=time.perf_counter(); candidates=[]
        for w in weights:
            th,quality=threshold(y[va],probs[va]@w)
            candidates.append({'weights':w.tolist(),'threshold':th,'validation_macro_f1':quality})
        chosen=max(candidates,key=lambda x:x['validation_macro_f1']); cost=time.perf_counter()-begin
        write_json(out/'selection'/f'{tag}.json',{'base_models':models,'candidates':candidates,'selected':chosen,'seed':42,
            'note':'Fixed seven convex combinations; underlying training cost reported separately in base model records'})
        p=probs@np.array(chosen['weights']); th=chosen['threshold']; saved=reference[KEY+[LABEL_COL,'split']].copy()
        saved['p_ct']=p; saved['threshold']=th; saved['prediction']=(p>=th).astype(int)
        saved.to_csv(out/'predictions'/f'{tag}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        metrics=[]
        for label in ['train','val','test']:
            mask=reference.split==label
            for mode,thr in [('default',.5),('validation_selected',th)]:
                metrics.append(dict(model='convex_ensemble',feature_set='baseline',t_s=t,seed=42,split=label,threshold_mode=mode,
                    threshold=thr,coverage=int(mask.sum())/int((manifest.split==label).sum()),search_seconds=cost,predict_seconds=0.,**score(y[mask],p[mask],thr)))
        write_json(done,{'metrics':metrics})
        print('ENSEMBLE',t,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default=DEFAULT_OUT); args=p.parse_args(); run(ROOT/args.out)
