"""Split-independent, observed geometry. No learned bounds or future snapshots."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ROUND_KEY

KEY = ROUND_KEY + ['prediction_time_s']

def geometry(s):
    s = s.rename(columns={'t_s': 'prediction_time_s'})
    s = s[s.team.isin([2, 3]) & (s.hp > 0)].copy()
    if s.duplicated(KEY + ['player_slot']).any():
        raise ValueError('Duplicate player snapshot')
    s['cell_x'] = np.floor(s.x / 1024).astype(int)
    s['cell_y'] = np.floor(s.y / 1024).astype(int)
    out = s[KEY].drop_duplicates().set_index(KEY)
    for team, name in [(3, 'ct'), (2, 't')]:
        a = s[s.team == team]
        g = a.groupby(KEY)
        for col in ['x', 'y', 'spd_win_mean']:
            out[f'geo_{name}_{col}'] = g[col].mean()
        counts = a.groupby(KEY + ['cell_x', 'cell_y']).size().rename('n').reset_index()
        counts['p'] = counts.n / counts.groupby(KEY).n.transform('sum')
        counts['entropy'] = -counts.p * np.log(counts.p)
        out[f'occ_{name}_entropy'] = counts.groupby(KEY).entropy.sum()
        out[f'occ_{name}_cells'] = counts.groupby(KEY).size()
        pairs = a.merge(a, on=KEY, suffixes=('_a', '_b'))
        pairs = pairs[pairs.player_slot_a < pairs.player_slot_b].copy()
        pairs['distance'] = np.hypot(pairs.x_a - pairs.x_b, pairs.y_a - pairs.y_b)
        out[f'geo_{name}_pair_mean'] = pairs.groupby(KEY).distance.mean()
    pairs = s[s.team == 3].merge(s[s.team == 2], on=KEY, suffixes=('_ct', '_t'))
    pairs['distance'] = np.hypot(pairs.x_ct - pairs.x_t, pairs.y_ct - pairs.y_t)
    pairs['near'] = (pairs.distance < 512).astype(float)
    pairs['same_cell'] = ((pairs.cell_x_ct == pairs.cell_x_t) & (pairs.cell_y_ct == pairs.cell_y_t)).astype(float)
    for col, agg in [('distance', 'min'), ('distance', 'mean'), ('near', 'mean'), ('same_cell', 'mean')]:
        out[f'geo_enemy_{col}_{agg}'] = pairs.groupby(KEY)[col].agg(agg)
    return out.reset_index()

def temporal_deltas(d, cols):
    d = d.sort_values(KEY).copy()
    group = d.groupby(ROUND_KEY, sort=False)
    dt = group.prediction_time_s.diff()
    for c in cols:
        d['delta_' + c] = group[c].diff() / dt
    return d

def build_geometry(root):
    cache = root / 'data/features/research_exact_geometry_v1.parquet'
    if cache.exists():
        return pd.read_parquet(cache)
    parts = []
    files = sorted((root / 'data/features/step4/snapshots').glob('*.parquet'))
    for i, path in enumerate(files):
        parts.append(geometry(pd.read_parquet(path)))
        if i % 100 == 0:
            print(f'geometry {i}/{len(files)}', flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out
