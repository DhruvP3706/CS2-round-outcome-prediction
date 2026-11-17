"""Auditable grouped benchmark; selection never accepts test arrays.

Run python src/research_experiment.py --stage all. Checkpoints are per model/time.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import traceback
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score, confusion_matrix, balanced_accuracy_score)
from threadpoolctl import threadpool_limits
from cs2pred.config import ROUND_KEY, LABEL_COL, OUTCOME_COLUMNS
from cs2pred.splits import series_split
from cs2pred.research_features import KEY, build_geometry, temporal_deltas
from cs2pred.modeling_extended import make_model
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
TIMES = [5,10,15,20,25,30,35,40,45,50,60,75]
MODELS = ['logreg','decision_tree','random_forest','extra_trees','svm','knn','naive_bayes',
          'gradient_boosting','hist_gradient_boosting','xgboost','lightgbm','catboost']

def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)

def score(y, p, threshold=.5):
    y = np.asarray(y, dtype=int); pred = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    ct = 2*tp/max(2*tp+fp+fn,1); tt = 2*tn/max(2*tn+fp+fn,1)
    two = len(np.unique(y)) == 2
    return dict(accuracy=float(accuracy_score(y,pred)), macro_f1=float((ct+tt)/2), ct_f1=float(ct), t_f1=float(tt),
                precision=float(precision_score(y,pred,zero_division=0)), recall=float(recall_score(y,pred,zero_division=0)),
                weighted_f1=float(f1_score(y,pred,average='weighted',zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y,pred)) if two else None,
                roc_auc=float(roc_auc_score(y,p)) if two else None,
                pr_auc=float(average_precision_score(y,p)) if two else None,
                tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),n=len(y),ct_count=int(y.sum()),t_count=int(len(y)-y.sum()))

def threshold(y, p):
    grid = np.round(np.arange(.05,.951,.005),3)
    y = np.asarray(y, dtype=bool)[:,None]; pr = np.asarray(p)[:,None] >= grid
    tp=(pr&y).sum(0); tn=(~pr&~y).sum(0); fp=(pr&~y).sum(0); fn=(~pr&y).sum(0)
    f=tp/np.maximum(2*tp+fp+fn,1)+tn/np.maximum(2*tn+fp+fn,1)
    ix=np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-12))
    i=ix[np.argmin(abs(grid[ix]-.5))]
    return float(grid[i]),float(f[i])

def load_data():
    rounds=pd.read_parquet(ROOT/'data/processed/round_table.parquet')
    meta=pd.read_parquet(ROOT/'data/features/step2_metadata_features.parquet')
    tick=pd.read_parquet(ROOT/'data/features/step4_full/player_state_features.parquet')
    mc=(ROOT/'results/step2/feature_list.txt').read_text().splitlines()
    ps=[c for c in tick if c.startswith('ps_')]
    assert not set(mc+ps)&OUTCOME_COLUMNS
    assert not rounds.duplicated(ROUND_KEY).any()
    assert (rounds[LABEL_COL] == (rounds.winner_side == 'ct').astype(int)).all()
    assert not meta.duplicated(KEY).any() and not tick.duplicated(KEY).any()
    # Keep ALL eligible metadata rows, including absent snapshots, with train-fitted imputation.
    d=meta.merge(tick[KEY+ps],on=KEY,how='left',validate='one_to_one')
    geo=build_geometry(ROOT); gc=[c for c in geo if c not in KEY]
    d=d.merge(geo,on=KEY,how='left',validate='one_to_one')
    d=d[d.prediction_time_s.isin(TIMES)].copy()
    d=temporal_deltas(d, gc+['ps_alive_diff','ps_hp_diff'])
    dc=[c for c in d if c.startswith('delta_')]
    families={'baseline':mc+ps,'metadata':mc,'state':ps,'geometry':mc+ps+gc,
              'temporal_spatial':mc+ps+dc,'combined':mc+ps+gc+dc,
              'without_combat':[c for c in mc+ps if not any(x in c for x in ['kill','death','trade','opening','hs','awp','smoke','blind','wall','clutch','1v1'])]}
    return rounds,d,families

def assignments(rounds):
    return series_split(rounds)

def prepare(out):
    rounds,d,families=load_data()
    split=assignments(rounds)
    rounds['split']=rounds.match_id.map(split); d['split']=d.match_id.map(split)
    assert rounds.groupby('match_id').split.nunique().max()==1
    assert set(d.split)=={'train','val','test'}
    maps=pd.read_parquet(ROOT/'data/opencs2/metadata/matches.parquet',columns=['match_id','map_name','map_id','hltv_demo_id'])
    manifest=rounds.merge(maps,on=['match_id','map_name'],validate='many_to_one')
    manifest['series_id']=manifest.match_id
    for t in TIMES: manifest[f'eligible_{t}s']=manifest.playable_duration_s>t
    manifest.to_csv(out/'split_manifest.csv',index=False)
    summary=manifest.groupby('split').agg(series=('match_id','nunique'),maps=('map_id','nunique'),rounds=('round','size'),ct=(LABEL_COL,'sum'))
    summary['t']=summary['rounds']-summary.ct
    (out/'split_report.md').write_text('# Grouped split\n\n'+summary.to_string()+'\n\nSeries = match_id; maps nested within series. Seed 42. All models retain the metadata eligibility cohort; unavailable snapshots are imputed using training data only.\n\n'+manifest.groupby(['split','map_name']).size().to_string(),encoding='utf-8')
    write_json(out/'feature_sets.json',families)
    for c in families['combined']:
        if not pd.api.types.is_numeric_dtype(d[c]): raise TypeError(c)
    write_json(out/'data_quality.json',{'rounds':len(rounds),'series':rounds.match_id.nunique(),
        'missing_snapshot_rows':int(d.ps_ct_alive.isna().sum()),'duplicate_keys':int(d.duplicated(KEY).sum()),
        'label_check':'passed','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/processed/round_table.parquet',ROOT/'data/features/step2_metadata_features.parquet',ROOT/'data/features/step4_full/player_state_features.parquet']}})
    train=d[d.split=='train'][families['combined']]
    pd.DataFrame({'feature':train.columns,'missing_fraction':train.isna().mean().values,'unique':train.nunique().values}).to_csv(out/'feature_quality.csv',index=False)
    corr=train.corr().abs(); upper=np.triu(np.ones(corr.shape),1).astype(bool)
    corr.where(upper).stack().loc[lambda x:x>.98].rename('absolute_correlation').to_csv(out/'redundant_features.csv')
    return rounds,d,families

def estimator(name, variant, seed):
    if name=='svm': return make_pipeline(StandardScaler(),SVC(C=[.1,1.][variant],kernel='linear',probability=False,random_state=seed))
    m=make_model(name,seed)
    if name=='hist_gradient_boosting': m.set_params(early_stopping=False,max_iter=200)
    if variant:
        options={'logreg':{'logisticregression__C':.1},'decision_tree':{'max_depth':4,'min_samples_leaf':60},
                 'random_forest':{'max_depth':8,'min_samples_leaf':30},'extra_trees':{'max_depth':None,'min_samples_leaf':30},
                 'knn':{'kneighborsclassifier__n_neighbors':75},'naive_bayes':{'gaussiannb__var_smoothing':1e-6},
                 'gradient_boosting':{'max_depth':2,'n_estimators':300},'hist_gradient_boosting':{'max_leaf_nodes':7,'l2_regularization':10},
                 'xgboost':{'max_depth':2,'reg_lambda':10},'lightgbm':{'num_leaves':7,'max_depth':-1,'reg_lambda':10},
                 'catboost':{'depth':6,'l2_leaf_reg':10}}
        m.set_params(**options[name])
    if name=='catboost': m.set_params(allow_writing_files=False)
    return m

def probabilities(m,x,name):
    if name=='svm':
        from scipy.special import expit
        return expit(m.decision_function(x))
    return m.predict_proba(x)[:,1]

def fit(m,name,x,y,v,vy):
    if name in ['lightgbm','original_baseline']:
        m.fit(x,y,eval_set=[(v,vy)],callbacks=[lgb.early_stopping(100,verbose=False)])
    elif name=='catboost': m.fit(x,y,eval_set=(v,vy),early_stopping_rounds=60,verbose=False)
    else: m.fit(x,y)

def select_model(name,x,y,v,vy,seed=42,variants=2):
    """Only training and validation are accepted here; no test access."""
    trials=[]; best=None; started=time.perf_counter()
    for variant in range(variants):
        if name=='original_baseline':
            from cs2pred.modeling import make_model as old_model
            m=old_model('lightgbm'); m.set_params(random_state=seed)
        else: m=estimator(name,variant,seed)
        begin=time.perf_counter(); fit(m,name,x,y,v,vy)
        p=probabilities(m,v,name); thr,quality=threshold(vy,p)
        trial={'variant':variant,'threshold':thr,'validation_macro_f1':quality,'validation_accuracy':score(vy,p,thr)['accuracy'],
               'fit_seconds':time.perf_counter()-begin,'params':m.get_params(),'best_iteration':getattr(m,'best_iteration_',None)}
        trials.append(trial)
        if best is None or quality>best[0]: best=(quality,m,thr,trial)
    return best[1],best[2],best[3],trials,time.perf_counter()-started

def run_one(out,rounds,d,families,name,t,family='baseline',seed=42):
    tag=f'{name}__{family}__{t}__{seed}'
    done=out/'checkpoints'/f'{tag}.json'
    if done.exists(): return
    print('RUN',tag,flush=True)
    part=d[d.prediction_time_s==t].copy(); cols=families[family]
    tr=part[part.split=='train']; va=part[part.split=='val']; te=part[part.split=='test']
    imp=SimpleImputer(strategy='median',keep_empty_features=True,add_indicator=True)
    x=imp.fit_transform(tr[cols]); v=imp.transform(va[cols]); z=imp.transform(te[cols])
    y=tr[LABEL_COL].to_numpy(); vy=va[LABEL_COL].to_numpy()
    with threadpool_limits(limits=6), warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        m,thr,best,trials,cost=select_model(name,x,y,v,vy,seed,1 if name=='original_baseline' else 2)
        # Persist the decision BEFORE producing test predictions.
        write_json(out/'selection'/f'{tag}.json',{'selected':best,'trials':trials,'seed':seed,'features':cols,
            'imputer_statistics':imp.statistics_.tolist(),'selection_rule':'maximum validation Macro F1; ties prefer first candidate'})
        predictions=[]; metrics=[]
        for label,frame,xx in [('train',tr,x),('val',va,v),('test',te,z)]:
            begin=time.perf_counter(); p=probabilities(m,xx,name); latency=time.perf_counter()-begin
            saved=frame[KEY+[LABEL_COL,'split']].copy(); saved['p_ct']=p; saved['threshold']=thr
            saved['prediction']=(p>=thr).astype(int); predictions.append(saved)
            for mode,th in [('default',.5),('validation_selected',thr)]:
                met=score(frame[LABEL_COL],p,th)
                metrics.append(dict(model=name,feature_set=family,t_s=t,seed=seed,split=label,threshold_mode=mode,threshold=th,
                    coverage=len(frame)/int((rounds.split==label).sum()),search_seconds=cost,predict_seconds=latency,**met))
        pd.concat(predictions).to_csv(out/'predictions'/f'{tag}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        if name=='lightgbm' and family=='combined' and t in [30,40,50,60,75] and seed==42:
            from sklearn.inspection import permutation_importance
            # Explain validation data; do not drive further selection using test attribution.
            pi=permutation_importance(m,v,vy,scoring='f1_macro',n_repeats=3,random_state=42,n_jobs=1)
            names=imp.get_feature_names_out(cols)
            shap=m.booster_.predict(v,pred_contrib=True)[:,:-1]
            pd.DataFrame({'feature':names,'permutation_mean':pi.importances_mean,'permutation_std':pi.importances_std,
                          'mean_absolute_tree_shap':np.abs(shap).mean(0)}).to_csv(out/'tables'/f'importance_{t}.csv',index=False)
    write_json(done,{'metrics':metrics})

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['all','prepare','classical','ablation','temporal','report'],default='all')
    parser.add_argument('--out',default='results/main_experiment'); parser.add_argument('--times',type=int,nargs='+',default=TIMES)
    args=parser.parse_args(); out=ROOT/args.out
    for p in [out,*[out/x for x in ['checkpoints','selection','predictions','plots','tables']]]: p.mkdir(parents=True,exist_ok=True)
    if args.stage=='report':
        from research_report import report
        report(out); return
    rounds,d,families=prepare(out)
    write_json(out/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),'cpu_count':os.cpu_count(),
        'packages':{m:importlib.metadata.version(m) for m in ['numpy','pandas','scikit-learn','lightgbm','xgboost','catboost','torch']}})
    if args.stage=='prepare': return
    if args.stage in ['all','classical']:
        for t in args.times:
            for model in ['original_baseline']+MODELS:
                try: run_one(out,rounds,d,families,model,t)
                except Exception:
                    with (out/'failures.log').open('a') as f: f.write(f'{model} {t}\n'+traceback.format_exc())
                    raise
    if args.stage in ['all','ablation']:
        for t in args.times:
            for family in ['metadata','state','geometry','temporal_spatial','combined','without_combat']:
                run_one(out,rounds,d,families,'lightgbm',t,family)
        for seed in [1,2]:
            for t in [30,40,50,60,75]:
                for family in ['baseline','combined']: run_one(out,rounds,d,families,'lightgbm',t,family,seed)
    if args.stage in ['all','temporal']:
        from research_temporal import run
        run(out,rounds,d,families,args.times)
    if args.stage=='all':
        from research_report import report
        report(out)

if __name__=='__main__': main()
