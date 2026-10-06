"""Step 7: short-window (0.5 / 1 / 2 s) backward-only dynamics from raw ticks + kill feed.

Every window ends exactly at the prediction tick  pred = freeze_end_tick + t*64  and starts w ticks earlier
(w = 32, 64, 128). Rows with tick > pred are never selected (checked in extract_dynamics).
Per-player sets: 'survivors' = players alive at BOTH window start and pred (so changes are not composition effects);
HP/death features also use players alive at window start who died inside the window.

Groups: dyn_b_ player-state changes | dyn_c_ movement dynamics | dyn_d_ team spatial dynamics | dyn_e_ engagement dynamics
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ROUND_KEY, TICK_RATE

WINDOWS = {"w05": 32, "w10": 64, "w20": 128}
MAXW = 128
MOVING_MIN = 30.0                   # units/s (same fixed threshold as Step 4)
PAIR_THRESH = (500, 1000, 2000)     # fixed distance thresholds (world units) for CT-T pair counts
RAW_COLS = ["round", "player_slot", "tick", "team_num", "is_alive", "health", "armor_value", "x", "y", "velocity_x", "velocity_y"]


def _spread(p):
    return np.linalg.norm(p - p.mean(0), axis=1).mean()


def _grid(tick, slot, vals, lo):
    off = tick - lo
    out = []
    for v, fill in vals:
        g = np.full((MAXW + 1, 10), fill, dtype=float)
        g[off, slot] = v
        out.append(g)
    return out


def _features_one(A, HP, AR, X, Y, VX, VY, T):
    f = {}
    ct, tt = T == 3, T == 2
    sides = {"ct": ct, "t": tt}
    a_now = A[MAXW]
    for wn, w in WINDOWS.items():
        i0 = MAXW - w
        a_then = A[i0]
        hp_then, hp_now = np.where(a_then, HP[i0], 0.0), np.where(a_now, HP[MAXW], 0.0)
        loss = np.where(a_then, hp_then - hp_now, 0.0)
        surv = a_now & a_then
        S = np.hypot(VX[i0:], VY[i0:])
        S = np.where(A[i0:], S, np.nan)
        pn, pt_ = np.c_[X[MAXW], Y[MAXW]], np.c_[X[i0], Y[i0]]
        pos = {}
        for sd, m in sides.items():
            mm = m & surv
            f[f"dyn_b_{sd}_hp_lost_{wn}"] = loss[m].sum()
            f[f"dyn_b_{sd}_n_hurt_{wn}"] = int((loss[m] > 0).sum())
            f[f"dyn_b_{sd}_max_hp_loss_{wn}"] = loss[m].max() if m.any() else 0.0
            f[f"dyn_b_{sd}_deaths_{wn}"] = int((a_then & ~a_now)[m].sum())
            f[f"dyn_b_{sd}_armor_lost_{wn}"] = float((AR[i0][mm] - AR[MAXW][mm]).sum())
            if mm.any():
                pm = np.nanmean(S[:, mm], axis=0)
                disp = np.linalg.norm(pn[mm] - pt_[mm], axis=1)
                f[f"dyn_c_{sd}_mean_speed_{wn}"] = pm.mean()
                f[f"dyn_c_{sd}_max_speed_{wn}"] = np.nanmax(S[:, mm])
                f[f"dyn_c_{sd}_speed_var_{wn}"] = np.nanvar(S[:, mm])
                f[f"dyn_c_{sd}_n_moving_{wn}"] = int((pm >= MOVING_MIN).sum())
                f[f"dyn_c_{sd}_mean_disp_{wn}"] = disp.mean()
                f[f"dyn_c_{sd}_max_disp_{wn}"] = disp.max()
                f[f"dyn_c_{sd}_centroid_shift_{wn}"] = np.linalg.norm(pn[mm].mean(0) - pt_[mm].mean(0))
                f[f"dyn_d_{sd}_spread_chg_{wn}"] = _spread(pn[mm]) - _spread(pt_[mm])
                pos[sd] = (pn[mm], pt_[mm])
            else:
                for k in ("mean_speed", "max_speed", "speed_var", "mean_disp", "max_disp", "centroid_shift"):
                    f[f"dyn_c_{sd}_{k}_{wn}"] = np.nan
                f[f"dyn_c_{sd}_n_moving_{wn}"] = 0
                f[f"dyn_d_{sd}_spread_chg_{wn}"] = np.nan
        keys = ("ct_nearest_chg", "t_nearest_chg", "mean_pair_chg", "min_dist_chg")
        if "ct" in pos and "t" in pos:
            Dn = np.linalg.norm(pos["ct"][0][:, None] - pos["t"][0][None], axis=2)
            Dt = np.linalg.norm(pos["ct"][1][:, None] - pos["t"][1][None], axis=2)
            f[f"dyn_d_ct_nearest_chg_{wn}"] = Dn.min(1).mean() - Dt.min(1).mean()
            f[f"dyn_d_t_nearest_chg_{wn}"] = Dn.min(0).mean() - Dt.min(0).mean()
            f[f"dyn_d_mean_pair_chg_{wn}"] = Dn.mean() - Dt.mean()
            f[f"dyn_d_min_dist_chg_{wn}"] = Dn.min() - Dt.min()
        else:
            for k in keys:
                f[f"dyn_d_{k}_{wn}"] = np.nan
    # current-state pair geometry (alive at pred): compact thresholds only, no grid
    mc, mt = ct & a_now, tt & a_now
    if mc.any() and mt.any():
        D = np.linalg.norm(np.c_[X[MAXW], Y[MAXW]][mc][:, None] - np.c_[X[MAXW], Y[MAXW]][mt][None], axis=2)
        f["dyn_d_min_dist"] = D.min()
        for th in PAIR_THRESH:
            f[f"dyn_d_pairs_lt{th}"] = int((D < th).sum())
    else:
        f["dyn_d_min_dist"] = np.nan
        for th in PAIR_THRESH:
            f[f"dyn_d_pairs_lt{th}"] = 0
    # ticks since the last HP decrease of any player (cap 2 s = none in window)
    HPf = np.where(A, HP, 0.0)
    dec = np.flatnonzero((HPf[1:] < HPf[:-1]).any(axis=1))
    f["dyn_e_since_damage_s"] = (MAXW - (dec[-1] + 1)) / TICK_RATE if len(dec) else MAXW / TICK_RATE
    return f


def extract_dynamics(tb: pd.DataFrame, rt: pd.DataFrame, times) -> pd.DataFrame:
    """tb: map-level ticks (RAW_COLS); rt: round table rows of this map indexed by round. Returns one row per (round, t)."""
    tb = tb.sort_values(["round", "tick"], kind="stable")
    rnd = tb["round"].to_numpy(); tick = tb["tick"].to_numpy().astype(np.int64)
    slot = tb["player_slot"].to_numpy().astype(int)
    cols = {c: tb[c].to_numpy() for c in ("team_num", "is_alive", "health", "armor_value", "x", "y", "velocity_x", "velocity_y")}
    out = []
    ur, starts = np.unique(rnd, return_index=True)
    ends = np.r_[starts[1:], len(rnd)]
    for r, a, b in zip(ur, starts, ends):
        if r not in rt.index:
            continue
        fe, dur = int(rt.at[r, "freeze_end_tick"]), rt.at[r, "playable_duration_s"]
        tk = tick[a:b]
        for t in times:
            if not dur > t:
                continue
            pred = fe + t * TICK_RATE
            lo = pred - MAXW
            r0, r1 = a + np.searchsorted(tk, lo, "left"), a + np.searchsorted(tk, pred, "right")
            if r1 <= r0 or tick[r1 - 1] != pred or tick[r0] < lo:
                continue
            assert tick[r0:r1].max() == pred and tick[r0:r1].min() >= lo      # window never crosses the cutoff
            sl, tkw = slot[r0:r1], tick[r0:r1]
            A, HP, AR, X, Y, VX, VY = _grid(tkw, sl, [(cols["is_alive"][r0:r1].astype(float), 0.0), (cols["health"][r0:r1].astype(float), 0.0),
                                                       (cols["armor_value"][r0:r1].astype(float), 0.0), (cols["x"][r0:r1], np.nan),
                                                       (cols["y"][r0:r1], np.nan), (cols["velocity_x"][r0:r1], np.nan),
                                                       (cols["velocity_y"][r0:r1], np.nan)], lo)
            T = np.zeros(10, int); T[sl] = cols["team_num"][r0:r1]
            f = _features_one(A > 0.5, HP, AR, X, Y, VX, VY, T)
            f.update({"round": int(r), "t_s": t, "_win_last_rel": int(tkw.max() - pred), "_win_first_rel": int(tkw.min() - pred)})
            out.append(f)
    return pd.DataFrame(out)


def engagement_features(flags: pd.DataFrame, t: float) -> pd.DataFrame:
    """Kill-feed engagement features over the previous 0.5/1/2 s (rel_tick in (cut-w, cut])."""
    cut = int(round(t * TICK_RATE))
    m = flags[flags.rel_tick <= cut]
    res = None
    for wn, w in WINDOWS.items():
        r = m[m.rel_tick > cut - w].groupby(ROUND_KEY)[["ct_kill", "t_kill"]].sum()
        r.columns = [f"dyn_e_ct_kills_{wn}", f"dyn_e_t_kills_{wn}"]
        r[f"dyn_e_kill_diff_{wn}"] = r.iloc[:, 0] - r.iloc[:, 1]
        r[f"dyn_e_engaged_{wn}"] = ((r.iloc[:, 0] + r.iloc[:, 1]) > 0).astype(int)
        res = r if res is None else res.join(r, how="outer")
    res["dyn_e_since_last_kill_s"] = (cut - m.groupby(ROUND_KEY).rel_tick.max()) / TICK_RATE
    res = res.reset_index()
    z = [c for c in res.columns if c.startswith("dyn_e_") and "since" not in c]
    res[z] = res[z].fillna(0)
    res["dyn_e_since_last_kill_s"] = res["dyn_e_since_last_kill_s"].fillna(t)
    return res


GROUP_PREFIX = {"B": "dyn_b_", "C": "dyn_c_", "D": "dyn_d_", "E": "dyn_e_"}
