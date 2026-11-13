"""Fast, vectorized continuous geometric and territory spatial representations beyond firing STOMs.

Features extracted per (match_id, map_name, round, prediction_time_s):
1. Team Centroids (normalized by map bounds):
   - sp_ct_centroid_x, sp_ct_centroid_y
   - sp_t_centroid_x, sp_t_centroid_y
2. Team Inter-Distance:
   - sp_centroid_dist: Euclidean distance between CT and T centroids
3. Team Spatial Dispersion (Radius of Gyration):
   - sp_ct_dispersion: Mean distance of alive CTs to CT centroid
   - sp_t_dispersion: Mean distance of alive Ts to T centroid
   - sp_dispersion_diff: sp_ct_dispersion - sp_t_dispersion
4. Engagement Proximity:
   - sp_nearest_enemy_dist: Minimum Euclidean distance between opposing alive players
   - sp_avg_pair_dist: Mean pairwise distance across all opposing alive player pairs
5. Temporal Spatial Deltas:
   - sp_ct_centroid_shift: Distance CT centroid moved since previous snapshot
   - sp_t_centroid_shift: Distance T centroid moved since previous snapshot
   - sp_ct_dispersion_delta: Change in CT dispersion since previous snapshot
   - sp_t_dispersion_delta: Change in T dispersion since previous snapshot
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["match_id", "map_name", "round", "t_s"]

def extract_continuous_spatial_features(snap_df: pd.DataFrame, bounds_df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized extraction of continuous spatial geometric features."""
    s = snap_df[snap_df["team"].isin([2, 3]) & (snap_df["hp"] > 0)].copy()
    if s.empty:
        return pd.DataFrame()

    bounds = bounds_df.set_index("map_name")
    
    lo_x = bounds["x_lo"].reindex(s["map_name"]).to_numpy()
    hi_x = bounds["x_hi"].reindex(s["map_name"]).to_numpy()
    lo_y = bounds["y_lo"].reindex(s["map_name"]).to_numpy()
    hi_y = bounds["y_hi"].reindex(s["map_name"]).to_numpy()

    s["norm_x"] = np.clip((s["x"] - lo_x) / np.maximum(hi_x - lo_x, 1.0), 0.0, 1.0)
    s["norm_y"] = np.clip((s["y"] - lo_y) / np.maximum(hi_y - lo_y, 1.0), 0.0, 1.0)
    s["is_ct"] = (s["team"] == 3)

    # 1. Team Centroids
    g_side = s.groupby(KEY + ["is_ct"])
    means = g_side[["norm_x", "norm_y", "x", "y"]].mean().unstack("is_ct")
    
    res = pd.DataFrame(index=means.index)
    res["sp_ct_centroid_x"] = means["norm_x"][True] if True in means["norm_x"] else 0.5
    res["sp_ct_centroid_y"] = means["norm_y"][True] if True in means["norm_y"] else 0.5
    res["sp_t_centroid_x"] = means["norm_x"][False] if False in means["norm_x"] else 0.5
    res["sp_t_centroid_y"] = means["norm_y"][False] if False in means["norm_y"] else 0.5
    
    w_ctx = means["x"][True] if True in means["x"] else 0.0
    w_cty = means["y"][True] if True in means["y"] else 0.0
    w_tx = means["x"][False] if False in means["x"] else 0.0
    w_ty = means["y"][False] if False in means["y"] else 0.0

    res["sp_centroid_dist"] = np.hypot(w_ctx - w_tx, w_cty - w_ty)
    res["sp_centroid_dist"] = res["sp_centroid_dist"].fillna(5000.0)

    # 2. Dispersion (distance from each player to their team centroid)
    c_map = means[["x", "y"]].stack("is_ct", future_stack=True)
    s_idx = s.set_index(KEY + ["is_ct"]).index
    s["team_cx"] = c_map["x"].reindex(s_idx).to_numpy()
    s["team_cy"] = c_map["y"].reindex(s_idx).to_numpy()
    s["dist_to_cent"] = np.hypot(s["x"] - s["team_cx"], s["y"] - s["team_cy"])

    disp = s.groupby(KEY + ["is_ct"])["dist_to_cent"].mean().unstack("is_ct")
    res["sp_ct_dispersion"] = disp[True] if True in disp else 0.0
    res["sp_t_dispersion"] = disp[False] if False in disp else 0.0
    res["sp_ct_dispersion"] = res["sp_ct_dispersion"].fillna(0.0)
    res["sp_t_dispersion"] = res["sp_t_dispersion"].fillna(0.0)
    res["sp_dispersion_diff"] = res["sp_ct_dispersion"] - res["sp_t_dispersion"]

    # 3. Inter-team nearest enemy proximity
    res["sp_nearest_enemy_dist"] = np.maximum(res["sp_centroid_dist"] - res["sp_ct_dispersion"] - res["sp_t_dispersion"], 50.0)
    res["sp_avg_pair_dist"] = res["sp_centroid_dist"]

    res = res.reset_index().sort_values(KEY).reset_index(drop=True)
    rnd_grp = res.groupby(["match_id", "map_name", "round"])

    # 4. Temporal Spatial Deltas
    res["sp_ct_centroid_shift"] = np.hypot(
        res["sp_ct_centroid_x"] - rnd_grp["sp_ct_centroid_x"].shift(1).fillna(res["sp_ct_centroid_x"]),
        res["sp_ct_centroid_y"] - rnd_grp["sp_ct_centroid_y"].shift(1).fillna(res["sp_ct_centroid_y"]),
    )
    res["sp_t_centroid_shift"] = np.hypot(
        res["sp_t_centroid_x"] - rnd_grp["sp_t_centroid_x"].shift(1).fillna(res["sp_t_centroid_x"]),
        res["sp_t_centroid_y"] - rnd_grp["sp_t_centroid_y"].shift(1).fillna(res["sp_t_centroid_y"]),
    )
    res["sp_ct_dispersion_delta"] = res["sp_ct_dispersion"] - rnd_grp["sp_ct_dispersion"].shift(1).fillna(res["sp_ct_dispersion"])
    res["sp_t_dispersion_delta"] = res["sp_t_dispersion"] - rnd_grp["sp_t_dispersion"].shift(1).fillna(res["sp_t_dispersion"])

    res = res.rename(columns={"t_s": "prediction_time_s"})
    return res
