"""Chronologically leak-free historical team rankings and strength indicators.

Strict Temporal Rule:
For any match M played at match_date D, statistics for its teams are computed
EXCLUSIVELY from matches played strictly before D (match_date < D).
Zero information from match M or future matches is ever used.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .config import META_DIR, ROUND_KEY

def compute_historical_rankings(meta_dir=META_DIR) -> pd.DataFrame:
    """Build per-round historical ranking features for CT and T teams.

    Features generated:
    - ct_hist_winrate, t_hist_winrate: Overall historical match win rate
    - hist_winrate_diff: ct_hist_winrate - t_hist_winrate
    - ct_map_winrate, t_map_winrate: Map-specific historical win rate
    - hist_map_winrate_diff: ct_map_winrate - t_map_winrate
    - match_stars: HLTV tournament tier / stars (0-5)
    - hist_exp_diff: log(1 + n_prior_ct) - log(1 + n_prior_t)
    """
    matches = pd.read_parquet(f"{meta_dir}/matches.parquet")
    rounds = pd.read_parquet(f"{meta_dir}/rounds.parquet", columns=ROUND_KEY)
    rp = pd.read_parquet(f"{meta_dir}/round_player.parquet", columns=["match_id", "map_name", "round", "player_slot", "start_side"])

    # Determine which team (team1 or team2) is CT and T in each round
    # In round 1, slots 0-4 are on one team and slots 5-9 on the other
    r1 = rp[rp["round"] == 1].groupby(["match_id", "map_name", "player_slot"])["start_side"].first().unstack()
    # If slot 0 is 't', team2 started T and team1 started CT
    slot0_side = r1[0].map(lambda s: str(s).lower())
    
    # Merge with matches
    m = matches.copy()
    m["match_date"] = pd.to_datetime(m["match_date"])
    m = m.sort_values("match_date").reset_index(drop=True)
    m["team1_started_ct"] = m.set_index(["match_id", "map_name"]).index.map(slot0_side == "t")

    # Chronological match processing
    team_history = {} # team -> list of {'date': d, 'won': bool, 'map': map}
    
    match_features = []
    for row in m.itertuples():
        d = row.match_date
        t1, t2 = row.team1, row.team2
        mp = row.map_name
        won1 = (row.winner == "team1")
        won2 = (row.winner == "team2")
        stars = float(row.stars) if hasattr(row, "stars") and not pd.isna(row.stars) else 0.0

        # Query history strictly before d
        h1 = [x for x in team_history.get(t1, []) if x["date"] < d]
        h2 = [x for x in team_history.get(t2, []) if x["date"] < d]

        w1 = np.mean([x["won"] for x in h1]) if h1 else 0.50
        w2 = np.mean([x["won"] for x in h2]) if h2 else 0.50

        mw1 = np.mean([x["won"] for x in h1 if x["map"] == mp]) if any(x["map"] == mp for x in h1) else 0.50
        mw2 = np.mean([x["won"] for x in h2 if x["map"] == mp]) if any(x["map"] == mp for x in h2) else 0.50

        n1, n2 = len(h1), len(h2)

        match_features.append({
            "match_id": row.match_id,
            "map_name": row.map_name,
            "t1_hist_winrate": w1,
            "t2_hist_winrate": w2,
            "t1_map_winrate": mw1,
            "t2_map_winrate": mw2,
            "t1_n_matches": n1,
            "t2_n_matches": n2,
            "match_stars": stars,
            "team1_started_ct": row.team1_started_ct,
        })

        # Update history
        if t1 not in team_history:
            team_history[t1] = []
        if t2 not in team_history:
            team_history[t2] = []
        team_history[t1].append({"date": d, "won": won1, "map": mp})
        team_history[t2].append({"date": d, "won": won2, "map": mp})

    mf_df = pd.DataFrame(match_features)

    # Now map to individual rounds
    # In regulation MR12: rounds 1-12 team1 is starting side; rounds 13-24 sides swap
    # In overtime (MR3): rounds 25-27 start side; rounds 28-30 swap; etc.
    res = rounds.merge(mf_df, on=["match_id", "map_name"], how="left")

    def is_t1_ct(r, t1_start_ct):
        if r <= 12:
            return t1_start_ct
        elif r <= 24:
            return not t1_start_ct
        else:
            ot_round = (r - 25) % 6
            return t1_start_ct if ot_round < 3 else (not t1_start_ct)

    t1_is_ct = [is_t1_ct(r, sct) for r, sct in zip(res["round"], res["team1_started_ct"])]
    t1_is_ct = np.array(t1_is_ct)

    res["ct_hist_winrate"] = np.where(t1_is_ct, res["t1_hist_winrate"], res["t2_hist_winrate"])
    res["t_hist_winrate"] = np.where(t1_is_ct, res["t2_hist_winrate"], res["t1_hist_winrate"])
    res["hist_winrate_diff"] = res["ct_hist_winrate"] - res["t_hist_winrate"]

    res["ct_map_winrate"] = np.where(t1_is_ct, res["t1_map_winrate"], res["t2_map_winrate"])
    res["t_map_winrate"] = np.where(t1_is_ct, res["t2_map_winrate"], res["t1_map_winrate"])
    res["hist_map_winrate_diff"] = res["ct_map_winrate"] - res["t_map_winrate"]

    n_ct = np.where(t1_is_ct, res["t1_n_matches"], res["t2_n_matches"])
    n_t = np.where(t1_is_ct, res["t2_n_matches"], res["t1_n_matches"])
    res["hist_exp_diff"] = np.log1p(n_ct) - np.log1p(n_t)

    cols = ROUND_KEY + [
        "ct_hist_winrate", "t_hist_winrate", "hist_winrate_diff",
        "ct_map_winrate", "t_map_winrate", "hist_map_winrate_diff",
        "match_stars", "hist_exp_diff",
    ]
    return res[cols].reset_index(drop=True)
