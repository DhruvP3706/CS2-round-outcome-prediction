"""Leakage-safe metadata features (kills, bomb, pre-round context) at a prediction time t.

Every event feature at time t only uses events with tick <= prediction_tick, where
prediction_tick = freeze_end_tick + t * 64.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import LABEL_COL, META_DIR, ROUND_KEY, TICK_RATE
from .round_table import eligible_rounds

MAPS = ["de_ancient", "de_anubis", "de_dust2", "de_inferno", "de_mirage", "de_nuke", "de_overpass"]
TRADE_WINDOW_TICKS = 5 * TICK_RATE
_KILL_COLS = ROUND_KEY + [
    "kill_ordinal", "tick", "attacker_side", "victim_side", "attacker_player_slot",
    "victim_player_slot", "weapon", "headshot", "noscope", "through_smoke", "wallbang",
    "attacker_blind",
]
# Deliberately NOT loaded from kills.parquet: time_to_next_kill_s (future), *_alive_after and
# is_trade_within_5s / is_clutch_context (definition not verified), dmg/distance (unused).


def load_kills(meta_dir=META_DIR) -> pd.DataFrame:
    k = pd.read_parquet(f"{meta_dir}/kills.parquet", columns=_KILL_COLS)
    for c in ["map_name", "attacker_side", "victim_side", "weapon"]:
        k[c] = k[c].astype(str)
    return k.sort_values(ROUND_KEY + ["kill_ordinal"]).reset_index(drop=True)


def load_plants(meta_dir=META_DIR) -> pd.DataFrame:
    """Bomb plant tick per round. Only a tick is kept; consumers must mask it by prediction_tick."""
    r = pd.read_parquet(f"{meta_dir}/rounds.parquet", columns=ROUND_KEY + ["bomb_plant_tick"])
    r["map_name"] = r["map_name"].astype(str)
    return r


def kill_flags(kills: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Per-kill flags. Each flag depends only on that kill and earlier kills of the round."""
    k = kills.merge(table[ROUND_KEY + ["freeze_end_tick"]], on=ROUND_KEY, how="inner")
    k = k.sort_values(ROUND_KEY + ["kill_ordinal"]).reset_index(drop=True)
    k["rel_tick"] = k["tick"] - k["freeze_end_tick"]
    ct_atk = (k.attacker_side == "ct") & (k.victim_side == "t")
    t_atk = (k.attacker_side == "t") & (k.victim_side == "ct")
    f = pd.DataFrame(index=k.index)
    f["ct_kill"], f["t_kill"] = ct_atk.astype(int), t_atk.astype(int)
    f["ct_death"], f["t_death"] = (k.victim_side == "ct").astype(int), (k.victim_side == "t").astype(int)
    for name, cond in {"hs": k.headshot, "awp": k.weapon == "awp", "smoke": k.through_smoke,
                       "blind": k.attacker_blind, "wall": k.wallbang, "noscope": k.noscope}.items():
        f[f"ct_{name}"] = (ct_atk & cond).astype(int)
        f[f"t_{name}"] = (t_atk & cond).astype(int)

    # Trade kill: the killer's team-mate was killed by this kill's victim <= 5 s earlier.
    prev = k[ROUND_KEY + ["kill_ordinal", "tick", "victim_side", "attacker_player_slot"]].rename(
        columns={"kill_ordinal": "o_prev", "tick": "t_prev", "victim_side": "vs_prev",
                 "attacker_player_slot": "as_prev"})
    j = k[ROUND_KEY + ["kill_ordinal", "tick", "attacker_side", "victim_player_slot"]].merge(prev, on=ROUND_KEY)
    j = j[(j.o_prev < j.kill_ordinal) & (j.tick - j.t_prev <= TRADE_WINDOW_TICKS)
          & (j.vs_prev == j.attacker_side) & (j.as_prev == j.victim_player_slot)]
    traded = set(map(tuple, j[ROUND_KEY + ["kill_ordinal"]].drop_duplicates().to_numpy()))
    is_trade = pd.Series([tuple(x) in traded for x in k[ROUND_KEY + ["kill_ordinal"]].to_numpy()], index=k.index)
    f["ct_trade"], f["t_trade"] = (ct_atk & is_trade).astype(int), (t_atk & is_trade).astype(int)

    # Alive counts after each kill (from the kill feed itself).
    ct_alive = 5 - f["ct_death"].groupby([k[c] for c in ROUND_KEY], sort=False).cumsum()
    t_alive = 5 - f["t_death"].groupby([k[c] for c in ROUND_KEY], sort=False).cumsum()
    f["ct_alive_after"], f["t_alive_after"] = ct_alive, t_alive
    f["alive_diff_after"] = ct_alive - t_alive
    f["state_1v1"] = ((ct_alive == 1) & (t_alive == 1)).astype(int)
    f["state_ct_clutch"] = ((ct_alive == 1) & (t_alive >= 2)).astype(int)
    f["state_t_clutch"] = ((t_alive == 1) & (ct_alive >= 2)).astype(int)
    return pd.concat([k[ROUND_KEY + ["kill_ordinal", "rel_tick", "attacker_side"]], f], axis=1)


_SUM_COLS = ["ct_kill", "t_kill", "ct_death", "t_death", "ct_hs", "t_hs", "ct_awp", "t_awp",
             "ct_smoke", "t_smoke", "ct_blind", "t_blind", "ct_wall", "t_wall", "ct_noscope",
             "t_noscope", "ct_trade", "t_trade"]


def kill_features(flags: pd.DataFrame, t: float) -> pd.DataFrame:
    cut = int(round(t * TICK_RATE))
    m = flags[flags.rel_tick <= cut]
    g = m.groupby(ROUND_KEY)
    out = g[_SUM_COLS].sum()
    out["n_deaths"] = out.ct_death + out.t_death
    out["ct_alive"], out["t_alive"] = 5 - out.ct_death, 5 - out.t_death
    out["alive_diff"] = out.ct_alive - out.t_alive
    out["kill_diff"] = out.ct_kill - out.t_kill
    out["hs_total"], out["awp_total"] = out.ct_hs + out.t_hs, out.ct_awp + out.t_awp
    out["smoke_total"], out["blind_total"] = out.ct_smoke + out.t_smoke, out.ct_blind + out.t_blind
    out["wall_total"], out["trade_total"] = out.ct_wall + out.t_wall, out.ct_trade + out.t_trade
    out["ever_1v1"] = g["state_1v1"].max()
    out["ever_ct_clutch"], out["ever_t_clutch"] = g["state_ct_clutch"].max(), g["state_t_clutch"].max()
    out["max_alive_adv_ct"] = np.maximum(g["alive_diff_after"].max(), 0)
    out["max_alive_adv_t"] = np.maximum(-g["alive_diff_after"].min(), 0)
    first = g.first()
    out["opening_occurred"] = 1
    out["opening_side"] = first["attacker_side"].map({"ct": 1, "t": -1}).fillna(0)  # 0 = none/world
    out["opening_time_s"] = first["rel_tick"] / TICK_RATE
    last_s = g["rel_tick"].max() / TICK_RATE
    out["time_since_last_kill_s"] = t - last_s
    for w in (5, 10):
        rec = m[m.rel_tick > cut - w * TICK_RATE].groupby(ROUND_KEY)[["ct_kill", "t_kill", "ct_death", "t_death"]].sum()
        out[f"recent{w}_ct_kills"], out[f"recent{w}_t_kills"] = rec.ct_kill, rec.t_kill
        out[f"recent{w}_deaths"] = rec.ct_death + rec.t_death
    return out.reset_index()


def bomb_features(plants: pd.DataFrame, table: pd.DataFrame, t: float) -> pd.DataFrame:
    d = table[ROUND_KEY + ["freeze_end_tick"]].merge(plants, on=ROUND_KEY, how="left")
    pred_tick = d["freeze_end_tick"] + int(round(t * TICK_RATE))
    planted = d["bomb_plant_tick"].le(pred_tick)  # NaN -> False
    d["bomb_planted"] = planted.astype(int)
    d["since_plant_s"] = np.where(planted, (pred_tick - d["bomb_plant_tick"]) / TICK_RATE, 0.0)
    d["plant_time_s"] = np.where(planted, (d["bomb_plant_tick"] - d["freeze_end_tick"]) / TICK_RATE, 0.0)
    return d[ROUND_KEY + ["bomb_planted", "since_plant_s", "plant_time_s"]]


def pre_round_features(table: pd.DataFrame, meta_dir=META_DIR) -> pd.DataFrame:
    """Context known before the round: map, round number, running score, loss streaks.

    Score/streak use only PRIOR rounds of the same map (never the current round's winner).
    """
    pov = pd.read_parquet(f"{meta_dir}/pov_rounds.parquet", columns=ROUND_KEY + ["player_slot", "player_side"])
    pov = pov[pov.player_slot == 0].copy()
    pov["map_name"] = pov["map_name"].astype(str)
    pov["a_side"] = pov["player_side"].astype(str).str.lower()  # side of slot-0..4 team (team A)
    d = table[ROUND_KEY + ["map_index", "winner_side"]].merge(pov[ROUND_KEY + ["a_side"]], on=ROUND_KEY, how="left")
    d = d.sort_values(ROUND_KEY).reset_index(drop=True)
    rows = []
    for _, g in d.groupby(["match_id", "map_name"], sort=False):
        wa = wb = 0            # prior wins of team A / B
        sa = sb = 0            # current loss streaks within the half
        seg_prev = None
        for r in g.itertuples():
            seg = 0 if r.round <= 12 else (1 if r.round <= 24 else 2)
            if seg != seg_prev:
                sa = sb = 0
                seg_prev = seg
            a_ct = r.a_side == "ct"
            rows.append((r.match_id, r.map_name, r.round,
                         wa if a_ct else wb, wb if a_ct else wa,
                         sa if a_ct else sb, sb if a_ct else sa))
            a_won = r.winner_side == r.a_side   # used only AFTER the row was emitted (future rounds)
            if a_won:
                wa, sa, sb = wa + 1, 0, sb + 1
            else:
                wb, sb, sa = wb + 1, 0, sa + 1
    p = pd.DataFrame(rows, columns=ROUND_KEY + ["ct_prior_score", "t_prior_score", "ct_loss_streak", "t_loss_streak"])
    p = p.merge(d[ROUND_KEY + ["map_index"]], on=ROUND_KEY)
    p["score_diff"] = p.ct_prior_score - p.t_prior_score
    p["is_pistol_round"] = p["round"].isin([1, 13]).astype(int)
    p["is_second_half"] = (p["round"] > 12).astype(int)
    p["is_overtime"] = (p["round"] > 24).astype(int)
    for m in MAPS:
        p[f"map_{m[3:]}"] = (p["map_name"] == m).astype(int)
    return p


def build_features(table, flags, plants, pre, t) -> pd.DataFrame:
    """Feature frame for all rounds eligible at time t (keys, label, prediction_time_s, features)."""
    base = eligible_rounds(table, t)[ROUND_KEY + ["prediction_time_s", "prediction_tick", LABEL_COL]]
    kf = kill_features(flags, t)
    df = base.merge(kf, on=ROUND_KEY, how="left").merge(bomb_features(plants, table, t), on=ROUND_KEY, how="left")
    df = df.merge(pre, on=ROUND_KEY, how="left")
    # rounds with no kill by t: counts are 0, alive 5v5, no opening kill
    fill0 = [c for c in kf.columns if c not in ROUND_KEY + ["opening_time_s", "time_since_last_kill_s",
                                                            "ct_alive", "t_alive", "alive_diff"]]
    df[fill0] = df[fill0].fillna(0)
    df["ct_alive"], df["t_alive"] = df["ct_alive"].fillna(5), df["t_alive"].fillna(5)
    df["alive_diff"] = df["alive_diff"].fillna(0)
    df["opening_time_s"] = df["opening_time_s"].fillna(-1.0)  # -1 = no opening kill yet
    df["time_since_last_kill_s"] = df["time_since_last_kill_s"].fillna(t)  # no kill: time since clock start
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = set(ROUND_KEY) | {"prediction_time_s", "prediction_tick", LABEL_COL}
    return [c for c in df.columns if c not in drop]
