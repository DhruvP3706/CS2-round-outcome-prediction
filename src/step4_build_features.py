"""Step 4b: map bounds + compact tick-derived feature rows (player state, spatial occupancy, movement)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cs2pred.config import RESULTS_DIR  # noqa: E402
from step4_extract_snapshots import BSAMP, OUT, SNAP, WCLASSES  # noqa: E402

import os  # noqa: E402
FULL = os.environ.get("STEP4_FULL") == "1"
RES = RESULTS_DIR.parent / ("step4_full" if FULL else "step4")
OUT_FEAT = OUT.parent / "step4_full" if FULL else OUT      # full-data features never overwrite the pilot ones
KEY = ["match_id", "map_name", "round", "t_s"]
GRID = 10
BOUND_Q = (0.005, 0.995)   # robust percentile bounds
BOUND_PAD = 0.02           # pad each side by 2% of the (percentile) range
MOVING_MIN, STATIONARY_MAX = 30.0, 5.0   # units/s of window-mean horizontal speed (fixed, not tuned)


def map_bounds():
    """Per-map x/y bounds = [q0.5%, q99.5%] of 1 Hz alive-player positions in TRAIN pilot files, padded by 2%."""
    rows = []
    for mp in sorted({f.stem.split("_", 1)[1] for f in BSAMP.glob("*.parquet")}):
        s = pd.concat([pd.read_parquet(f) for f in BSAMP.glob(f"*_{mp}.parquet")])
        r = {"map_name": mp, "n_samples": len(s), "abs_min_x": s.x.min(), "abs_max_x": s.x.max(),
             "abs_min_y": s.y.min(), "abs_max_y": s.y.max()}
        for a in "xy":
            lo, hi = s[a].quantile(BOUND_Q[0]), s[a].quantile(BOUND_Q[1])
            pad = BOUND_PAD * (hi - lo)
            r[f"{a}_lo"], r[f"{a}_hi"] = lo - pad, hi + pad
        rows.append(r)
    b = pd.DataFrame(rows)
    RES.mkdir(parents=True, exist_ok=True)
    OUT_FEAT.mkdir(parents=True, exist_ok=True)
    b.to_csv(RES / "map_bounds.csv", index=False); b.to_csv(OUT_FEAT / "map_bounds.csv", index=False)
    return b.set_index("map_name")


def build():
    bounds = map_bounds()
    s = pd.concat([pd.read_parquet(f) for f in sorted(SNAP.glob("*.parquet"))], ignore_index=True)
    s = s[s.team.isin([2, 3])].copy()
    lo_x, hi_x = bounds.x_lo.reindex(s.map_name).to_numpy(), bounds.x_hi.reindex(s.map_name).to_numpy()
    lo_y, hi_y = bounds.y_lo.reindex(s.map_name).to_numpy(), bounds.y_hi.reindex(s.map_name).to_numpy()
    s["nx"] = np.clip((s.x - lo_x) / (hi_x - lo_x), 0, 1)       # positions outside the bounds are clipped to the edge
    s["ny"] = np.clip((s.y - lo_y) / (hi_y - lo_y), 0, 1)
    s["cell"] = (np.minimum((s.nx * GRID).astype(int), GRID - 1) * GRID + np.minimum((s.ny * GRID).astype(int), GRID - 1))
    s["ct"] = (s.team == 3)
    s = s.sort_values(KEY).reset_index(drop=True)
    side = {True: "ct", False: "t"}

    # ---- player state + movement (vectorised group aggregations)
    g = s.groupby(KEY + ["ct"])
    a = g.agg(alive=("hp", "size"), hp=("hp", "sum"), armor=("armor", "sum"), spd_mean=("spd_win_mean", "mean"),
              spd_max=("spd_win_mean", "max")).reset_index()
    mv = s.assign(mv=(s.spd_win_mean >= MOVING_MIN).astype(int), st=(s.spd_win_mean < STATIONARY_MAX).astype(int))
    a = a.merge(mv.groupby(KEY + ["ct"]).agg(moving=("mv", "sum"), stationary=("st", "sum")).reset_index(), on=KEY + ["ct"])
    wide = {}
    for ct in (True, False):
        d = a[a.ct == ct].drop(columns="ct").set_index(KEY)
        d.columns = [f"{side[ct]}_{c}" for c in d.columns]
        wide[ct] = d
    f = wide[True].join(wide[False], how="outer")
    for c in ["alive", "hp", "armor", "moving", "stationary"]:
        for sd in ("ct", "t"):
            f[f"{sd}_{c}"] = f[f"{sd}_{c}"].fillna(0)
    out = pd.DataFrame(index=f.index)
    out["ps_ct_alive"], out["ps_t_alive"] = f.ct_alive, f.t_alive
    out["ps_alive_diff"] = f.ct_alive - f.t_alive
    out["ps_ct_hp"], out["ps_t_hp"], out["ps_hp_diff"] = f.ct_hp, f.t_hp, f.ct_hp - f.t_hp
    out["ps_ct_avg_hp"], out["ps_t_avg_hp"] = f.ct_hp / f.ct_alive.replace(0, np.nan), f.t_hp / f.t_alive.replace(0, np.nan)
    out["ps_ct_armor"], out["ps_t_armor"], out["ps_armor_diff"] = f.ct_armor, f.t_armor, f.ct_armor - f.t_armor
    out["ps_ct_avg_armor"] = f.ct_armor / f.ct_alive.replace(0, np.nan)
    out["ps_t_avg_armor"] = f.t_armor / f.t_alive.replace(0, np.nan)
    wcnt = s.groupby(KEY + ["ct", "wclass"]).size().unstack("wclass", fill_value=0).reindex(columns=range(len(WCLASSES)), fill_value=0)
    for ct in (True, False):
        w = wcnt.xs(ct, level="ct").reindex(f.index, fill_value=0)
        for i, n in enumerate(WCLASSES):
            out[f"ps_{side[ct]}_w_{n}"] = w[i]
    for n in WCLASSES:
        out[f"ps_wdiff_{n}"] = out[f"ps_ct_w_{n}"] - out[f"ps_t_w_{n}"]
    out["mv_ct_avg_speed"], out["mv_t_avg_speed"] = f.ct_spd_mean, f.t_spd_mean
    out["mv_ct_max_speed"], out["mv_t_max_speed"] = f.ct_spd_max, f.t_spd_max
    out["mv_ct_moving"], out["mv_t_moving"] = f.ct_moving, f.t_moving
    out["mv_ct_stationary"], out["mv_t_stationary"] = f.ct_stationary, f.t_stationary

    # ---- occupancy grid counts
    cc = s.groupby(KEY + ["ct", "cell"]).size().unstack("cell", fill_value=0).reindex(columns=range(GRID * GRID), fill_value=0)
    for ct in (True, False):
        gd = cc.xs(ct, level="ct").reindex(f.index, fill_value=0)
        gd.columns = [f"sp_{side[ct]}_g{c // GRID}{c % GRID}" for c in gd.columns]
        out = out.join(gd)
    out[[c for c in out.columns if c.startswith("sp_") and "_g" in c]] = out[[c for c in out.columns if "_g" in c and c.startswith("sp_")]].fillna(0)

    # ---- compact spatial summaries (normalised centroid; world-unit distances; x/y plane only)
    sm = {k: np.full(len(f), np.nan) for k in ["ctx", "cty", "tx", "ty", "dist", "ct_spread", "t_spread", "nearest", "avgpair"]}
    idx = {k: i for i, k in enumerate(f.index)}
    gid = np.array([idx[k] for k in zip(*[s[c].to_numpy() for c in KEY])])
    order = np.argsort(gid, kind="stable")
    gid, X, Y, NX, NY, CT = gid[order], s.x.to_numpy()[order], s.y.to_numpy()[order], s.nx.to_numpy()[order], s.ny.to_numpy()[order], s.ct.to_numpy()[order]
    bnd = np.flatnonzero(np.diff(gid)) + 1
    for lo, hi in zip(np.r_[0, bnd], np.r_[bnd, len(gid)]):
        i = gid[lo]
        c, p = CT[lo:hi], np.c_[X[lo:hi], Y[lo:hi]]
        pc, pt = p[c], p[~c]
        if len(pc):
            sm["ctx"][i], sm["cty"][i] = NX[lo:hi][c].mean(), NY[lo:hi][c].mean()
            sm["ct_spread"][i] = np.linalg.norm(pc - pc.mean(0), axis=1).mean()
        if len(pt):
            sm["tx"][i], sm["ty"][i] = NX[lo:hi][~c].mean(), NY[lo:hi][~c].mean()
            sm["t_spread"][i] = np.linalg.norm(pt - pt.mean(0), axis=1).mean()
        if len(pc) and len(pt):
            sm["dist"][i] = np.linalg.norm(pc.mean(0) - pt.mean(0))
            dm = np.linalg.norm(pc[:, None] - pt[None], axis=2)
            sm["nearest"][i], sm["avgpair"][i] = dm.min(), dm.mean()
    names = {"ctx": "sp_ct_centroid_x", "cty": "sp_ct_centroid_y", "tx": "sp_t_centroid_x", "ty": "sp_t_centroid_y",
             "dist": "sp_centroid_dist", "ct_spread": "sp_ct_spread", "t_spread": "sp_t_spread",
             "nearest": "sp_ct_t_nearest_dist", "avgpair": "sp_ct_t_avg_pair_dist"}
    for k, n in names.items():
        out[n] = sm[k]
    out = out.reset_index().rename(columns={"t_s": "prediction_time_s"})
    out.to_parquet(OUT_FEAT / "step4_tick_features.parquet", index=False)
    groups = {"player_state": [c for c in out.columns if c.startswith("ps_")],
              "spatial": [c for c in out.columns if c.startswith("sp_")],
              "movement": [c for c in out.columns if c.startswith("mv_")]}
    (RES / "feature_groups.txt").write_text("\n".join(f"[{k}] ({len(v)})\n" + "\n".join(v) for k, v in groups.items()))
    print({k: len(v) for k, v in groups.items()}, out.shape)
    return out


if __name__ == "__main__":
    build()
