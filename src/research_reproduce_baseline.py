"""Historical baseline reproduction using available cached state features, no new imputation."""
import warnings
import pandas as pd
from research_experiment import ROOT,KEY,LABEL_COL,TIMES,threshold,score
from cs2pred.splits import series_split
from cs2pred.modeling import fit_predict

def main():
    out=ROOT/'results/main_experiment/historical_reproduction'; (out/'predictions').mkdir(parents=True,exist_ok=True)
    r=pd.read_parquet(ROOT/'data/processed/round_table.parquet')
    meta=pd.read_parquet(ROOT/'data/features/step2_metadata_features.parquet')
    ps=pd.read_parquet(ROOT/'data/features/step4_full/player_state_features.parquet')
    d=meta.merge(ps,on=KEY,validate='one_to_one'); d['split']=d.match_id.map(series_split(r))
    cols=(ROOT/'results/step2/feature_list.txt').read_text().splitlines()+[c for c in ps if c.startswith('ps_')]
    old=pd.read_csv(ROOT/'results/step6/baseline_reproduction.csv').set_index('t_s'); rows=[]
    for t in TIMES:
        frames={s:d[(d.prediction_time_s==t)&(d.split==s)] for s in ['train','val','test']}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m,pv,pt,iteration=fit_predict('lightgbm',frames['train'][cols],frames['train'][LABEL_COL],frames['val'][cols],frames['val'][LABEL_COL],frames['test'][cols])
        th,vf=threshold(frames['val'][LABEL_COL],pv); met=score(frames['test'][LABEL_COL],pt,th)
        rows.append(dict(t_s=t,threshold=th,validation_macro_f1=vf,best_iteration=iteration,old_macro_f1=old.loc[t,'test_macro_f1'],difference=met['macro_f1']-old.loc[t,'test_macro_f1'],**met))
        saved=frames['test'][KEY+[LABEL_COL]].copy(); saved['p_ct']=pt; saved['threshold']=th; saved['prediction']=(pt>=th).astype(int)
        saved.to_csv(out/'predictions'/f'{t}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    print(pd.DataFrame(rows)[['t_s','macro_f1','old_macro_f1','difference']].to_string(index=False))

if __name__=='__main__': main()
