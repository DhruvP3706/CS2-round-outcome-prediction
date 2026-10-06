"""Step 5 feature families (backward-looking only). Every feature at prediction time t uses only
kill events with tick <= freeze_end_tick + t*64, the plant tick if <= that tick, and player snapshots
taken at prediction times <= t (snapshots exist only at the 15 prediction ticks, so 'state change'
features compare with the previous prediction snapshot).

Prefixes / families:  cm_ combat momentum | pa_ player advantage | fw_ weapon/firepower | rs_ round state |
                      bb_ bomb/time | pr_ pre-round x live | sc_ state changes
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PREDICTION_TIMES_S, ROUND_KEY, TICK_RATE
from .features import build_features, feature_columns
from .round_table import eligible_rounds

SNAP_KEY = ROUND_KEY + ["t_s"]
# Fixed, a-priori firepower mapping over the 7 coarse active-weapon classes of Step 4
# (index = wclass code of step4_extract_snapshots.WCLASSES). Never tuned.
WCLASSES = ["rifle", "sniper", "smg", "heavy", "pistol", "knife", "utility"]
FIREPOWER = {"rifle": 4.0, "sniper": 4.5, "smg": 2.5, "heavy": 2.5, "pistol": 1.0, "knife": 0.0, "utility": 0.5}
FP_ARR = np.array([FIREPOWER[c] for c in WCLASSES])
BOMB_TIMER_S = 40.0     # CS2 default mp_c4timer (assumed constant)
FAMILIES = ["cm", "pa", "fw", "rs", "bb", "pr", "sc"]
W_WINDOWS = (5, 10, 15, 20)


# ---------------------------------------------------------------- family 1: combat momentum (kill feed)
def streak_table(flags: pd.DataFrame) -> pd.DataFrame:
    """Per team-kill event: causal streak info (depends only on that kill and earlier kills of the round)."""
    ev = flags[(flags.ct_kill + flags.t_kill) > 0][ROUND_KEY + ["kill_ordinal", "rel_tick", "ct_kill", "t_kill"]].copy()
    ev = ev.sort_values(ROUND_KEY + ["kill_ordinal"]).reset_index(drop=True)
    ev["side"] = ev.ct_kill - ev.t_kill                                   # +1 CT kill, -1 T kill
    rid = ev.groupby(ROUND_KEY, sort=False).ngroup()
    new = (ev.side != ev.side.groupby(rid).shift(1)).astype(int)
    run = new.groupby(rid).cumsum()
    ev["streak_len"] = ev.groupby([rid, run]).cumcount() + 1
    ev["streak_start_rel"] = ev.groupby([rid, run]).rel_tick.transform("first")
    ev["ct_max_streak"] = (ev.streak_len * (ev.side == 1)).groupby(rid).cummax()
    ev["t_max_streak"] = (ev.streak_len * (ev.side == -1)).groupby(rid).cummax()
    return ev


def combat_momentum(flags: pd.DataFrame, ev: pd.DataFrame, t: float) -> pd.DataFrame:
    cut = int(round(t * TICK_RATE))
    m = flags[flags.rel_tick <= cut]
    g = m.groupby(ROUND_KEY)
    out = pd.DataFrame(index=g.size().index)
    out["cm_hs_diff"] = g.ct_hs.sum() - g.t_hs.sum()
    out["cm_awp_diff"] = g.ct_awp.sum() - g.t_awp.sum()
    out["cm_trade_diff"] = g.ct_trade.sum() - g.t_trade.sum()
    out["cm_special_diff"] = sum(g[f"ct_{c}"].sum() - g[f"t_{c}"].sum() for c in ("smoke", "blind", "wall"))
    cum_diff = g.ct_kill.sum() - g.t_kill.sum()
    out["cm_kill_diff"] = cum_diff
    for w in W_WINDOWS:
        r = m[m.rel_tick > cut - w * TICK_RATE].groupby(ROUND_KEY)[["ct_kill", "t_kill", "ct_death", "t_death"]].sum()
        out[f"cm_w{w}_ct_kills"], out[f"cm_w{w}_t_kills"] = r.ct_kill, r.t_kill
        out[f"cm_w{w}_ct_deaths"], out[f"cm_w{w}_t_deaths"] = r.ct_death, r.t_death
        out[f"cm_w{w}_kill_diff"] = r.ct_kill - r.t_kill
    # momentum: kill-rate difference in the last 10 s minus the round's average rate so far
    out["cm_rate_diff_recent10_vs_cum"] = out["cm_w10_kill_diff"].fillna(0) / min(10.0, t) - cum_diff / t
    e = ev[ev.rel_tick <= cut]
    last = e.groupby(ROUND_KEY).last()
    tc = e[e.side == 1].groupby(ROUND_KEY).rel_tick.max()
    tt = e[e.side == -1].groupby(ROUND_KEY).rel_tick.max()
    out["cm_since_ct_kill_s"] = t - tc / TICK_RATE
    out["cm_since_t_kill_s"] = t - tt / TICK_RATE
    out["cm_streak_side"] = last.side                                       # +1 CT, -1 T, 0 no kill yet
    out["cm_streak_len"] = last.streak_len
    out["cm_ct_streak"] = last.streak_len.where(last.side == 1, 0)
    out["cm_t_streak"] = last.streak_len.where(last.side == -1, 0)
    out["cm_ct_max_streak"], out["cm_t_max_streak"] = last.ct_max_streak, last.t_max_streak
    out["cm_since_streak_start_s"] = t - last.streak_start_rel / TICK_RATE
    out = out.reset_index()
    z = [c for c in out.columns if c.startswith("cm_") and c not in ("cm_since_ct_kill_s", "cm_since_t_kill_s", "cm_since_streak_start_s")]
    out[z] = out[z].fillna(0)                                   # no event yet -> 0
    for c in ("cm_since_ct_kill_s", "cm_since_t_kill_s", "cm_since_streak_start_s"):
        out[c] = out[c].fillna(t)                               # no event yet -> time since clock start (as Step 2)
    out["cm_since_kill_diff_s"] = out.cm_since_ct_kill_s - out.cm_since_t_kill_s
    return out


def recent_deaths(flags: pd.DataFrame, t: float) -> pd.DataFrame:
    """Deaths per side in the last 1 s / 2 s (kill feed; family 7)."""
    cut = int(round(t * TICK_RATE))
    m = flags[flags.rel_tick <= cut]
    out = None
    for w in (1, 2):
        r = m[m.rel_tick > cut - w * TICK_RATE].groupby(ROUND_KEY)[["ct_death", "t_death"]].sum()
        r.columns = [f"sc_ct_deaths_{w}s", f"sc_t_deaths_{w}s"]
        out = r if out is None else out.join(r, how="outer")
    return out.reset_index()


# ---------------------------------------------------------------- families 2, 3, 7: snapshot-derived
def snapshot_features(snap: pd.DataFrame, times=PREDICTION_TIMES_S) -> pd.DataFrame:
    """Team aggregates at each prediction snapshot + differences to the previous prediction snapshot."""
    s = snap[snap.team.isin([2, 3]) & snap.t_s.isin(times)].copy()
    s["ct"] = s.team == 3
    s["fp"] = FP_ARR[s.wclass.to_numpy()]
    s["low25"], s["low50"], s["high75"], s["full"] = s.hp < 25, s.hp < 50, s.hp > 75, s.hp >= 100
    s["rifle"], s["sniper"] = s.wclass == 0, s.wclass == 1
    a = s.groupby(SNAP_KEY + ["ct"]).agg(n=("hp", "size"), hp=("hp", "sum"), armor=("armor", "sum"), fp=("fp", "sum"),
                                         low25=("low25", "sum"), low50=("low50", "sum"), high75=("high75", "sum"),
                                         full=("full", "sum"), rifle=("rifle", "sum"), sniper=("sniper", "sum"))
    idx = a.index.droplevel("ct").unique()
    cols = {}
    for ct, nm in ((True, "ct"), (False, "t")):
        d = a.xs(ct, level="ct").reindex(idx, fill_value=0)
        for c in d.columns:
            cols[f"{nm}_{c}"] = d[c].to_numpy(float)
    f = pd.DataFrame(cols, index=idx)
    o = pd.DataFrame(index=idx)
    ctn, tn = f.ct_n.replace(0, np.nan), f.t_n.replace(0, np.nan)
    ad = f.ct_n - f.t_n
    avg_hp_diff = (f.ct_hp / ctn).fillna(0) - (f.t_hp / tn).fillna(0)
    avg_ar_diff = (f.ct_armor / ctn).fillna(0) - (f.t_armor / tn).fillna(0)
    # family 2: player advantage
    o["pa_abs_alive_adv"] = ad.abs()
    o["pa_hp_ratio"] = (f.ct_hp / (f.ct_hp + f.t_hp).replace(0, np.nan)).fillna(0.5)
    o["pa_avg_hp_diff_alive"] = avg_hp_diff
    o["pa_armor_ratio"] = (f.ct_armor / (f.ct_armor + f.t_armor).replace(0, np.nan)).fillna(0.5)
    for c in ("low25", "low50", "high75", "full"):
        o[f"pa_ct_{c}"], o[f"pa_t_{c}"] = f[f"ct_{c}"], f[f"t_{c}"]
    o["pa_alive_x_hp"] = ad * avg_hp_diff
    o["pa_alive_x_armor"] = ad * avg_ar_diff
    # family 3: firepower (alive players only by construction: snapshots contain alive players)
    o["fw_ct_fp"], o["fw_t_fp"], o["fw_fp_diff"] = f.ct_fp, f.t_fp, f.ct_fp - f.t_fp
    o["fw_ct_avg_fp"], o["fw_t_avg_fp"] = (f.ct_fp / ctn).fillna(0), (f.t_fp / tn).fillna(0)
    o["fw_avg_fp_diff"] = o.fw_ct_avg_fp - o.fw_t_avg_fp
    o["fw_fp_x_alive"] = o.fw_fp_diff * ad
    # family 7: change vs the PREVIOUS prediction snapshot (backward only)
    prev_t = {t: p for t, p in zip(times[1:], times[:-1])}
    base = f[["ct_n", "t_n", "ct_hp", "t_hp", "ct_armor", "t_armor", "ct_rifle", "t_rifle", "ct_sniper", "t_sniper"]].reset_index()
    pv = base.copy()
    pv["t_s"] = pv.t_s.map({p: t for t, p in prev_t.items()})           # shift snapshot forward to the next time
    pv = pv.dropna(subset=["t_s"]).assign(t_s=lambda d: d.t_s.astype(int))
    m = base.merge(pv, on=SNAP_KEY, how="left", suffixes=("", "_prev"))
    m = m.set_index(SNAP_KEY).reindex(idx)
    for sd in ("ct", "t"):
        o[f"sc_{sd}_d_alive"] = m[f"{sd}_n"] - m[f"{sd}_n_prev"]
        o[f"sc_{sd}_hp_lost"] = m[f"{sd}_hp_prev"] - m[f"{sd}_hp"]
        o[f"sc_{sd}_armor_lost"] = m[f"{sd}_armor_prev"] - m[f"{sd}_armor"]
        o[f"sc_{sd}_d_rifle"] = m[f"{sd}_rifle"] - m[f"{sd}_rifle_prev"]
        o[f"sc_{sd}_d_sniper"] = m[f"{sd}_sniper"] - m[f"{sd}_sniper_prev"]
    o["sc_hp_lost_diff"] = o.sc_t_hp_lost - o.sc_ct_hp_lost               # >0: T lost more HP than CT
    o["sc_alive_change_diff"] = o.sc_ct_d_alive - o.sc_t_d_alive
    # keep aggregate helpers needed for interactions (not features): alive from snapshots
    o["_hp_diff"] = f.ct_hp - f.t_hp
    o = o.reset_index().rename(columns={"t_s": "prediction_time_s"})
    return o


# ---------------------------------------------------------------- families 4, 5, 6 (from existing columns)
def state_interactions(df: pd.DataFrame) -> pd.DataFrame:
    ad = df.ct_alive - df.t_alive
    o = pd.DataFrame(index=df.index)
    # family 4: round state
    o["rs_open_since_s"] = np.where(df.opening_time_s >= 0, df.prediction_time_s - df.opening_time_s, -1.0)
    o["rs_open_side_x_alive"] = df.opening_side * ad
    o["rs_both_lost"] = ((df.ct_death > 0) & (df.t_death > 0)).astype(int)
    o["rs_alive_adv_sign"] = np.sign(ad)
    o["rs_equal_numbers"] = (ad == 0).astype(int)
    o["rs_state_code"] = df.ct_alive * 10 + df.t_alive
    o["rs_total_alive"], o["rs_min_alive"] = df.ct_alive + df.t_alive, np.minimum(df.ct_alive, df.t_alive)
    o["rs_3v3"] = ((df.ct_alive == 3) & (df.t_alive == 3)).astype(int)
    o["rs_2v2"] = ((df.ct_alive == 2) & (df.t_alive == 2)).astype(int)
    o["rs_1v1_now"] = ((df.ct_alive == 1) & (df.t_alive == 1)).astype(int)
    o["rs_ct_clutch_now"] = ((df.ct_alive == 1) & (df.t_alive >= 2)).astype(int)
    o["rs_t_clutch_now"] = ((df.t_alive == 1) & (df.ct_alive >= 2)).astype(int)
    # family 5: bomb / time (bomb site is NOT used: rounds.bomb_site is not reliable, see summary)
    pl, sp = df.bomb_planted, df.since_plant_s
    o["bb_time_to_explosion_s"] = np.where(pl == 1, np.maximum(BOMB_TIMER_S - sp, 0.0), -1.0)
    o["bb_planted_x_alive"] = pl * ad
    o["bb_planted_x_hp"] = pl * df["_hp_diff"]
    o["bb_planted_x_fp"] = pl * df.fw_fp_diff
    o["bb_since_plant_x_alive"] = sp * ad
    o["bb_since_plant_x_fp"] = sp * df.fw_fp_diff
    # family 6: pre-round x live
    o["pr_round_no"] = df["round"]
    o["pr_score_x_alive"] = df.score_diff * ad
    o["pr_lossstreak_diff_x_alive"] = (df.ct_loss_streak - df.t_loss_streak) * ad
    o["pr_round_x_alive"] = df["round"] * ad
    o["pr_pistol_x_alive"] = df.is_pistol_round * ad
    o["pr_half_x_alive"] = df.is_second_half * ad
    for mname in ("ancient", "anubis", "dust2", "inferno", "mirage", "nuke", "overpass"):
        o[f"pr_map_{mname}_x_alive"] = df[f"map_{mname}"] * ad
    return o


def build_all(table, flags, plants, pre, snap, times=PREDICTION_TIMES_S) -> pd.DataFrame:
    """Step 2 metadata features + all Step 5 families for every eligible round at each t in `times`."""
    ev = streak_table(flags)
    sf = snapshot_features(snap, list(PREDICTION_TIMES_S))
    parts = []
    for t in times:
        base = build_features(table, flags, plants, pre, t)
        d = base.merge(combat_momentum(flags, ev, t), on=ROUND_KEY, how="left").merge(recent_deaths(flags, t), on=ROUND_KEY, how="left")
        d = d.merge(sf[sf.prediction_time_s == t], on=ROUND_KEY + ["prediction_time_s"], how="inner")   # inner: need the tick snapshot
        cmc = [c for c in d.columns if c.startswith("cm_")]
        d[cmc] = d[cmc].fillna({c: (t if c in ("cm_since_ct_kill_s", "cm_since_t_kill_s", "cm_since_streak_start_s") else 0) for c in cmc})
        d[["sc_ct_deaths_1s", "sc_t_deaths_1s", "sc_ct_deaths_2s", "sc_t_deaths_2s"]] = d[["sc_ct_deaths_1s", "sc_t_deaths_1s", "sc_ct_deaths_2s", "sc_t_deaths_2s"]].fillna(0)
        d = pd.concat([d, state_interactions(d)], axis=1)
        parts.append(d)
    out = pd.concat(parts, ignore_index=True).drop(columns=["_hp_diff"])
    return out


def family_columns(df: pd.DataFrame) -> dict:
    return {f: [c for c in df.columns if c.startswith(f + "_")] for f in FAMILIES}
