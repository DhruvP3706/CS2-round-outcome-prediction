"""Reports generated only from saved predictions and validation-selected models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research_experiment import ROOT, ROUND_KEY, LABEL_COL, score

def table(out,name,d):
    d.to_csv(out/f'{name}.csv',index=False)
    # No optional tabulate dependency.
    def fmt(x):
        if isinstance(x,float): return f'{x:.5f}' if np.isfinite(x) else 'NA'
        return str(x).replace('|','/').replace('\n',' ')
    lines=['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']
    lines+=['| '+' | '.join(map(fmt,row))+' |' for row in d.itertuples(index=False,name=None)]
    (out/'tables'/f'{name}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def counts(y,p,groups):
    rows=[]
    for g in np.unique(groups):
        m=groups==g; yy=y[m]; pp=p[m]
        rows.append([((yy==0)&(pp==0)).sum(),((yy==0)&(pp==1)).sum(),((yy==1)&(pp==0)).sum(),((yy==1)&(pp==1)).sum()])
    return np.asarray(rows)

def f1_counts(c):
    tn,fp,fn,tp=np.moveaxis(c,-1,0)
    return tp/np.maximum(2*tp+fp+fn,1)+tn/np.maximum(2*tn+fp+fn,1)

def paired(a,b,seed=42):
    z=a.merge(b,on=ROUND_KEY,validate='one_to_one',suffixes=('_a','_b'))
    assert len(z)==len(a)==len(b) and (z.y_ct_win_a==z.y_ct_win_b).all()
    y=z.y_ct_win_a.to_numpy(); g=z.match_id.to_numpy()
    ca=counts(y,z.prediction_a.to_numpy(),g); cb=counts(y,z.prediction_b.to_numpy(),g)
    effect=float(f1_counts(cb.sum(0))-f1_counts(ca.sum(0)))
    if len(ca)<2: return dict(difference=effect,ci_low=np.nan,ci_high=np.nan,p_value=np.nan,n_series=len(ca))
    rng=np.random.default_rng(seed); n=len(ca)
    draws=rng.integers(n,size=(2000,n)); delta=f1_counts(cb[draws].sum(1))-f1_counts(ca[draws].sum(1))
    swaps=rng.integers(2,size=(5000,n,1)).astype(bool)
    null=f1_counts(np.where(swaps,cb,ca).sum(1))-f1_counts(np.where(swaps,ca,cb).sum(1))
    return dict(difference=effect,ci_low=float(np.quantile(delta,.025)),ci_high=float(np.quantile(delta,.975)),
        p_value=float((1+(abs(null)>=abs(effect)-1e-12).sum())/(1+len(null))),n_series=n)

def savefig(out,name):
    plt.tight_layout()
    for ext in ['png','svg']: plt.savefig(out/'plots'/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close()

def report(out):
    records=[]
    for p in sorted((out/'checkpoints').glob('*.json')): records.extend(json.loads(p.read_text())['metrics'])
    allm=pd.DataFrame(records); table(out,'all_metrics',allm)
    chosen=allm[(allm.threshold_mode=='validation_selected')&(allm.seed==42)]
    test=chosen[chosen.split=='test'].copy(); val=chosen[chosen.split=='val'].copy()
    keys=['model','feature_set','t_s','seed']
    test=test.merge(val[keys+['macro_f1','accuracy']],on=keys,suffixes=('','_validation'))
    table(out,'test_model_comparison',test); table(out,'model_comparison',test); table(out,'test_metrics',test)
    # Frozen per-time model and feature choice based exclusively on validation.
    best=test.sort_values(['t_s','macro_f1_validation','model','feature_set'],ascending=[True,False,True,True]).groupby('t_s').head(1)
    table(out,'best_model_by_time',best)
    old=pd.read_csv(ROOT/'results/step6/baseline_reproduction.csv')
    reproduced=test[test.model=='original_baseline'][['t_s','macro_f1','n']].merge(old[['t_s','test_macro_f1','n_test']],on='t_s')
    reproduced['difference']=reproduced.macro_f1-reproduced.test_macro_f1
    table(out,'baseline_reproduction',reproduced)
    for metric in ['accuracy','macro_f1']:
        table(out,f'test_{metric}_by_time',test[['model','feature_set','t_s',metric,'n','coverage']])
    abb=test[test.model=='lightgbm'].copy()
    base=abb[abb.feature_set=='baseline'].set_index('t_s').macro_f1
    abb['difference_from_baseline']=abb.macro_f1-abb.t_s.map(base)
    table(out,'feature_ablation',abb)
    hp=[]
    for p in sorted((out/'selection').glob('*.json')):
        j=json.loads(p.read_text()); hp.append({'run':p.stem,'selection':json.dumps(j)})
    table(out,'hyperparameters',pd.DataFrame(hp))
    predictions={}; maps=[]; matched=[]; statistics=[]; cis=[]
    manifest=pd.read_csv(out/'split_manifest.csv'); total=int((manifest.split=='test').sum())
    eligible=set(map(tuple,manifest.loc[(manifest.split=='test')&manifest.eligible_75s,ROUND_KEY].to_numpy()))
    for r in test.itertuples():
        tag=f'{r.model}__{r.feature_set}__{r.t_s}__42'
        d=pd.read_csv(out/'predictions'/f'{tag}.csv.gz'); d=d[d.split=='test']; predictions[tag]=d
        for mp,grp in d.groupby('map_name'):
            maps.append(dict(model=r.model,feature_set=r.feature_set,t_s=r.t_s,map_name=mp,**score(grp[LABEL_COL],grp.p_ct,r.threshold)))
        mm=d[[tuple(k) in eligible for k in d[ROUND_KEY].to_numpy()]]
        if len(mm): matched.append(dict(model=r.model,feature_set=r.feature_set,t_s=r.t_s,coverage=len(mm)/total,**score(mm[LABEL_COL],mm.p_ct,r.threshold)))
        cc=counts(d[LABEL_COL].to_numpy(),d.prediction.to_numpy(),d.match_id.to_numpy())
        if len(cc)>1:
            rng=np.random.default_rng(42); boot=f1_counts(cc[rng.integers(len(cc),size=(2000,len(cc)))].sum(1))
            lo,hi=np.quantile(boot,[.025,.975])
        else: lo=hi=np.nan
        cis.append(dict(model=r.model,feature_set=r.feature_set,t_s=r.t_s,macro_f1=r.macro_f1,ci_low=lo,ci_high=hi,n_series=len(cc)))
    for r in best.itertuples():
        b=f'original_baseline__baseline__{r.t_s}__42'; p=f'{r.model}__{r.feature_set}__{r.t_s}__42'
        if b in predictions:
            statistics.append(dict(baseline='original_baseline',proposed_model=r.model,feature_set=r.feature_set,t_s=r.t_s,**paired(predictions[b],predictions[p])))
    stat=pd.DataFrame(statistics)
    if not stat.empty:
        stat['corrected_p_value']=np.nan; good=stat.p_value.dropna().sort_values(); n=len(good)
        if n: stat.loc[good.index,'corrected_p_value']=np.minimum(1,np.maximum.accumulate(good.to_numpy()*(n-np.arange(n))))
    table(out,'statistical_tests',stat); table(out,'confidence_intervals',pd.DataFrame(cis))
    table(out,'map_analysis',pd.DataFrame(maps)); table(out,'matched_cohort',pd.DataFrame(matched))
    seeds=allm[(allm.split=='test')&(allm.threshold_mode=='validation_selected')]
    sd=seeds.groupby(['model','feature_set','t_s']).macro_f1.agg(['mean','std','count']).reset_index()
    table(out,'seed_summary',sd)
    comparison=test[test.feature_set=='baseline']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for metric in ['macro_f1','accuracy','ct_f1','t_f1','coverage']:
        plt.figure(figsize=(10,6))
        for name,g in comparison.groupby('model'): plt.plot(g.t_s,g[metric],'.-',label=name)
        plt.xlabel('Seconds after freeze end'); plt.ylabel('Test '+metric.replace('_',' ')); plt.legend(fontsize=7,ncol=3)
        if metric=='macro_f1': plt.axhline(.8,color='gray',ls='--')
        savefig(out,metric+'_vs_time')
    for t in [60,75]:
        c=comparison[comparison.t_s==t].sort_values('macro_f1_validation',ascending=False)
        plt.figure(figsize=(10,5)); plt.bar(c.model,c.macro_f1); plt.xticks(rotation=60,ha='right'); plt.ylim(0,1)
        plt.ylabel('Test Macro F1'); plt.title(f'{t}s; ordered by validation Macro F1'); savefig(out,f'model_comparison_{t}')
    for metric in ['macro_f1','accuracy']:
        plt.figure(figsize=(10,6))
        for name,g in pd.DataFrame(matched).query("feature_set=='baseline'").groupby('model'): plt.plot(g.t_s,g[metric],'.-',label=name)
        plt.legend(fontsize=7,ncol=3); plt.xlabel('Seconds after freeze end'); plt.ylabel('Matched test '+metric); savefig(out,'matched_'+metric)
    plt.figure(figsize=(10,6))
    for name,g in abb.groupby('feature_set'): plt.plot(g.t_s,g.macro_f1,'.-',label=name)
    plt.legend(fontsize=8); plt.xlabel('Seconds after freeze end'); plt.ylabel('Test Macro F1'); savefig(out,'ablation')
    for t in [30,40,50,60,75]:
        f=out/'tables'/f'importance_{t}.csv'
        if f.exists():
            importance=pd.read_csv(f)
            for metric in ['permutation_mean','mean_absolute_tree_shap']:
                top=importance.nlargest(20,metric).sort_values(metric)
                plt.figure(figsize=(9,7)); plt.barh(top.feature,top[metric]); plt.xlabel(metric+' (validation)'); savefig(out,f'{metric}_{t}')
    mapdf=pd.DataFrame(maps).query("model=='lightgbm' and feature_set=='baseline'")
    for metric in ['macro_f1','accuracy','n','ct_count']:
        p=mapdf.pivot(index='map_name',columns='t_s',values=metric)
        plt.figure(figsize=(10,5)); plt.imshow(p,aspect='auto'); plt.xticks(range(len(p.columns)),p.columns); plt.yticks(range(len(p)),p.index)
        plt.colorbar(label=metric); plt.xlabel('Seconds after freeze end'); savefig(out,'map_time_'+metric)
    p=comparison.pivot(index='model',columns='t_s',values='macro_f1')
    plt.figure(figsize=(10,6)); plt.imshow(p,aspect='auto'); plt.xticks(range(len(p.columns)),p.columns); plt.yticks(range(len(p)),p.index); plt.colorbar(label='Test Macro F1'); savefig(out,'model_time_heatmap')
    for r in best[best.t_s.isin([30,60,75])].itertuples():
        cm=np.array([[r.tn,r.fp],[r.fn,r.tp]])
        plt.figure(figsize=(4,4)); plt.imshow(cm,cmap='Blues'); plt.xticks([0,1],['T','CT']); plt.yticks([0,1],['T','CT'])
        for i in range(2):
            for j in range(2): plt.text(j,i,str(cm[i,j]),ha='center')
        plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title(f'Validation-selected model, {r.t_s}s'); savefig(out,f'confusion_{r.t_s}')
    ci=pd.DataFrame(cis).merge(best[keys[:-1]],on=keys[:-1])
    plt.figure(figsize=(8,5)); plt.plot(ci.t_s,ci.macro_f1,'o-'); plt.fill_between(ci.t_s,ci.ci_low,ci.ci_high,alpha=.2)
    plt.xlabel('Seconds'); plt.ylabel('Test Macro F1 (95% series bootstrap)'); savefig(out,'confidence_intervals')
    plt.figure(figsize=(9,5)); c=comparison[comparison.t_s==60]; plt.bar(c.model,c.search_seconds); plt.xticks(rotation=60,ha='right'); plt.ylabel('Search seconds at 60s'); savefig(out,'computational_cost')
    training_ids=set(manifest.loc[manifest.split=='train','match_id'])
    paths=[p for p in sorted((ROOT/'data/features/step4/snapshots').glob('*.parquet')) if int(p.stem.split('_')[0]) in training_ids]
    if paths:
        obs=pd.read_parquet(paths[0]); rnd=obs.loc[obs.t_s==60,'round'].min()
        fig,axes=plt.subplots(1,2,figsize=(10,5))
        for ax,t in zip(axes,[5,60]):
            for team,col,label in [(3,'tab:blue','CT'),(2,'tab:red','T')]:
                sub=obs[(obs['round']==rnd)&(obs.t_s==t)&(obs.team==team)]
                ax.scatter(sub.x,sub.y,c=col,label=label,s=60)
            ax.set_title(f'Observed round {rnd}, {t}s'); ax.set_xlabel('World x'); ax.set_ylabel('World y'); ax.legend(); ax.set_aspect('equal')
        fig.suptitle(paths[0].stem); savefig(out,'observed_occupancy')
    selected_lines=best[['t_s','model','feature_set','accuracy','macro_f1','n','coverage']].to_string(index=False)
    report_text='# Executed evaluation\n\nAll model/feature choices below maximize validation Macro F1; test scores are descriptive.\n\n```\n'+selected_lines+'\n```\n\n'
    report_text+='Macro F1 always averages CT and T F1, including single-class samples. Precision, recall and PR-AUC use CT as positive. PR-AUC is average precision. Linear SVM outputs sigmoid-transformed margins, not calibrated probabilities.\n\n'
    report_text+='Confidence intervals resample whole series (2,000 replicates). Paired comparisons swap model predictions at series level (5,000 permutations); Holm correction covers the reported cutoff comparisons. These intervals condition on the fitted models; they do not include model-selection or data-collection uncertainty. A single-series holdout has no population cluster interval or inferential p-value.\n\n'
    report_text+='The historical test set has already been inspected by earlier research. These results are an exploratory re-evaluation, not a new untouched confirmatory test. A random grouped split does not establish future-date generalization. Rankings are excluded because team identity alignment is unverified. Transformer and broad Optuna searches are deferred; two prespecified candidates per classical model are compared on the fixed grouped validation set. Temporal models use a 20-epoch cap.\n'
    (out/'evaluation_report.md').write_text(report_text,encoding='utf-8')
    print(selected_lines,flush=True)
