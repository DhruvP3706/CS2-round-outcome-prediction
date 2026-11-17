"""Machine-readable source/notebook inventory and executed data validity checks."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from cs2pred.config import ROUND_KEY
from cs2pred.features import load_kills,load_plants,pre_round_features
from step2_metadata_baseline import lookahead_test
from cs2pred.validation import validate
from cs2pred.splits import series_split

ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'results/main_experiment'; out.mkdir(exist_ok=True,parents=True)
    inventory=[]
    for path in sorted((ROOT/'src').rglob('*.py')):
        text=path.read_text(encoding='utf-8'); tree=ast.parse(text)
        inventory.append({'file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(text.encode()).hexdigest(),'lines':len(text.splitlines()),
            'definitions':'; '.join(n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef)))})
    pd.DataFrame(inventory).to_csv(ROOT/'research/source_inventory.csv',index=False)
    notebooks=[]
    for path in ROOT.rglob('*.ipynb'):
        j=json.loads(path.read_text(encoding='utf-8')); text='\n'.join(''.join(c.get('source',[])) for c in j['cells'])
        notebooks.append({'file':str(path.relative_to(ROOT)),'cells':len(j['cells']),'sha256':hashlib.sha256(text.encode()).hexdigest(),
            'uses_random_row_split':'train_test_split' in text,'uses_group_split':any(s in text for s in ['GroupKFold','GroupShuffleSplit','series_split']),
            'random_forest':'RandomForest' in text,'spatial':'STOM' in text or 'stom' in text,
            'evaluation_lines':' | '.join(line.strip() for line in text.splitlines() if any(s in line for s in ['train_test_split','RandomForestClassifier','f1_score','accuracy_score']))[:2000]})
    pd.DataFrame(notebooks).to_csv(ROOT/'research/notebook_inventory.csv',index=False)
    r=pd.read_parquet(ROOT/'data/processed/round_table.parquet'); checks=validate(r); checks.to_csv(out/'data_validation.csv',index=False)
    k=load_kills(); pl=load_plants(); pre=pre_round_features(r)
    rows=[]
    for t in [10,30,60,75]:
        ok,n,p=lookahead_test(r,k,pl,pre,t); rows.append({'time':t,'future_invariant':ok,'future_kills':n,'future_plants':p}); assert ok
    pd.DataFrame(rows).to_csv(out/'leakage_checks.csv',index=False)
    raw=pd.read_parquet(ROOT/'data/opencs2/metadata/kills.parquet',columns=ROUND_KEY+['tick','event_seconds'])
    raw=raw.merge(r[ROUND_KEY+['freeze_end_tick']],on=ROUND_KEY,validate='many_to_one')
    delta=raw.tick-raw.freeze_end_tick; mask=raw.event_seconds>1
    rate=delta[mask]/raw.loc[mask,'event_seconds']
    timing={'n_events':len(raw),'median_tick_rate':float(rate.median()),'min_tick_rate':float(rate.min()),'max_tick_rate':float(rate.max()),
            'max_event_seconds_error_at_64':float(abs(raw.event_seconds-delta/64).max()),
            'limitation':'event_seconds may itself be derived from ticks; this is timeline consistency, not independent wall-clock verification'}
    (out/'tick_rate_check.json').write_text(json.dumps(timing,indent=2))
    existing=pd.read_csv(ROOT/'results/step2/series_split.csv').set_index('match_id').split.sort_index()
    pd.testing.assert_series_equal(existing,series_split(r).sort_index(),check_names=False,check_index_type=False)
    print('Original grouped assignments reproduced; future-event tests passed.'); print(timing)

if __name__=='__main__': main()
