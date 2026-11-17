"""Compact numerical report and execution manifest; run after artifact verification."""
import argparse
import hashlib
import json
import subprocess
import pandas as pd
from research_experiment import ROOT,DEFAULT_OUT,write_json
from research_report import table

def finalize(out):
    verification=json.loads((out/'artifact_verification.json').read_text())
    assert verification['runs_verified']==296
    d=pd.read_csv(out/'test_model_comparison.csv'); best=pd.read_csv(out/'best_model_by_time.csv')
    overall=d.groupby(['model','feature_set']).agg(mean_validation_macro_f1=('macro_f1_validation','mean'),mean_test_macro_f1=('macro_f1','mean'),mean_test_accuracy=('accuracy','mean'),cutoffs=('t_s','nunique')).reset_index()
    overall=overall[overall.cutoffs==12].sort_values('mean_validation_macro_f1',ascending=False)
    table(out,'overall_model_comparison',overall)
    top=overall.iloc[0]
    selected=best[['t_s','model','feature_set','accuracy','macro_f1','ct_f1','t_f1','n','coverage']]
    table(out,'selected_summary',selected)
    stats=pd.read_csv(out/'statistical_tests.csv')
    descriptive=pd.read_csv(out/'descriptive_test_maxima_not_model_selection.csv')
    cross=best[best.macro_f1>.8]; anycross=d[d.macro_f1>.8]
    text='# Research experiment results\n\n'
    text+=f"Overall selection by mean validation Macro F1 across all 12 cutoffs: **{top['model']} / {top['feature_set']}** (mean validation F1 {top['mean_validation_macro_f1']:.4f}). This is an average across distinct prediction tasks, not one pooled test score.\n\n"
    text+=(out/'tables/selected_summary.md').read_text(encoding='utf-8')
    text+='\n\nSelected-model earliest F1 above 0.80: '+('none' if cross.empty else str(int(cross.t_s.min()))+'s')+'.\n'
    text+='Descriptive earliest crossing among all tested configurations: '+('none' if anycross.empty else str(int(anycross.t_s.min()))+'s')+'. Test maxima are not used for selection.\n\n'
    for t in [50,60,75]:
        z=descriptive[(descriptive.t_s==t)&(descriptive.diagnostic_test_maximum_only=='macro_f1')].iloc[0]
        text+=f"At {t}s the descriptive test-F1 maximum is {z['model']} / {z['feature_set']}: accuracy {z['accuracy']:.4f}, Macro F1 {z['macro_f1']:.4f}.\n\n"
    text+='## Paired statistical comparison\n\n'+(out/'tables/statistical_tests.md').read_text(encoding='utf-8')
    text+='\n\nDifferences compare the validation-selected system with original-configuration LightGBM on identical keys under the new retained-cohort pipeline. Historical-cache reproduction is a different diagnostic. Holm correction covers twelve cutoff comparisons. One test series permits no meaningful population-level cluster inference.\n\n'
    feat=overall[overall.model=='lightgbm'].iloc[0]
    text+=f"Best LightGBM feature configuration by mean validation F1 is {feat['feature_set']}. Per-cutoff feature choices can differ. Full ablation, map strata, map-specific fits, matched cohorts and default-threshold comparisons are in the adjacent CSV/Markdown tables.\n\n"
    mp=pd.read_csv(out/'map_analysis.csv'); maps=mp[(mp.t_s==60)&(mp.model=='lightgbm')&(mp.feature_set=='baseline')]
    table(out,'map_summary_60s',maps[['map_name','accuracy','macro_f1','n','ct_count','t_count']])
    text+='## Map findings at 60 seconds (fixed pooled LightGBM)\n\n'+(out/'tables/map_summary_60s.md').read_text(encoding='utf-8')
    text+='\n\nMap-wise rankings are descriptive and sensitive to class balance and sample size. No map is dropped to improve a score.\n\n'
    text+='## Executed and deferred work\n\nExecuted: 12 classical model families, original-config baseline, LSTM/GRU/causal CNN, validation-weighted tree ensemble, six LightGBM feature ablations at every cutoff, two additional seeds for baseline/combined at five focus cutoffs, map-specific fits where feasible, paired statistics, native TreeSHAP/permutation importance, observed occupancy, plots and manuscript. All 296 global run artifacts independently verified.\n\nDeferred: Transformer, broad Optuna/randomized nested searches, external historical player/team ratings, region-level utility and navigation-aware spatial control, full raw-tick re-extraction, prospective test collection, and primary full-method verification of two inaccessible IEEE papers. No results are fabricated for these gaps. Training search is deliberately small; the study is exploratory because earlier research already used this test partition.\n'
    (out/'FINAL_REPORT.md').write_text(text,encoding='utf-8')
    write_json(out/'execution_manifest.json',{'runs':296,'map_specific_runs':len(list((out/'map_specific').rglob('checkpoints/*.json'))),
        'code_commit_before_result_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').glob('research_*.py'))},
        'verification':verification,'status':'executed; deferred items explicitly listed in FINAL_REPORT.md'})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default=DEFAULT_OUT); args=p.parse_args(); finalize(ROOT/args.out)
