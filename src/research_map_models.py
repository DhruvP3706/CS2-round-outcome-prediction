"""Map-specific versus pooled LightGBM; fixed cutoffs and validation tuning."""
import argparse
import pandas as pd
from research_experiment import ROOT,DEFAULT_OUT,prepare,run_one
from research_report import table

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default=DEFAULT_OUT); args=p.parse_args()
    out=ROOT/args.out; rounds,d,families=prepare(out); rows=[]
    for mp in sorted(d.map_name.unique()):
        child=out/'map_specific'/mp
        for sub in ['checkpoints','selection','predictions','tables']: (child/sub).mkdir(parents=True,exist_ok=True)
        rr=rounds[rounds.map_name==mp]; dd=d[d.map_name==mp]
        for t in [30,40,50,60,75]:
            dt=dd[dd.prediction_time_s==t]; sizes=dt.groupby('split').size()
            if sizes.get('train',0)<200 or sizes.get('val',0)<50 or sizes.get('test',0)<1:
                rows.append(dict(map_name=mp,t_s=t,status='insufficient training/validation/test data')); continue
            run_one(child,rr,dd,families,'lightgbm',t)
            import json
            met=json.loads((child/'checkpoints'/f'lightgbm__baseline__{t}__42.json').read_text())['metrics']
            for row in met:
                if row['split']=='test' and row['threshold_mode']=='validation_selected': rows.append(dict(map_name=mp,status='executed',**row))
    table(out,'map_specific_comparison',pd.DataFrame(rows))

if __name__=='__main__': main()
