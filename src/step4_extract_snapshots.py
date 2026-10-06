"""Step 4a: stream map-level tick files (Phase-A pilot) and keep ONLY per-player snapshots at the prediction ticks.

Raw ticks are never written to disk: each file is range-read from the Hub (column projection) into RAM,
reduced to (round, t, player) snapshot rows (+ a 1 Hz position sample of TRAIN files for the map bounds)
and discarded. Output is resumable: data/features/step4/snapshots/{match_id}_{map}.parquet
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfFileSystem  # noqa: E402

from cs2pred.config import PREDICTION_TIMES_S, PROCESSED_DIR, TICK_RATE  # noqa: E402
from cs2pred.splits import series_split  # noqa: E402
from step3_tick_probe import WDS, list_tree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FULL = "full" in sys.argv[1:]   # full-data validation: all 794 map files (pilot files are cached and skipped)
OUT = ROOT / "data" / "features" / "step4"
SNAP, BSAMP = OUT / "snapshots", OUT / "bounds_samples"
COLS = ["round", "player_slot", "tick", "team_num", "is_alive", "health", "armor_value", "input_weapon",
        "x", "y", "velocity_x", "velocity_y"]
WINDOW_TICKS = TICK_RATE // 2          # backward-only look-back: ticks (t-32 .. t] plus t  => 0.5 s

WCLASSES = ["rifle", "sniper", "smg", "heavy", "pistol", "knife", "utility"]
_W = {}
for c, names in {
    "rifle": "ak47 m4a1 m4a1_silencer galilar famas sg556 aug",
    "sniper": "awp ssg08 scar20 g3sg1",
    "smg": "mp9 mac10 mp7 mp5sd ump45 p90 bizon",
    "heavy": "nova xm1014 mag7 sawedoff m249 negev",
    "pistol": "glock usp_silencer usp hkp2000 p2000 p250 fiveseven tec9 cz75a deagle elite revolver",
}.items():
    for n in names.split():
        _W[n] = WCLASSES.index(c)


def wclass(name):
    if name.startswith("knife") or name in ("bayonet", "knife_t"):
        return WCLASSES.index("knife")
    return _W.get(name, WCLASSES.index("utility"))   # grenades, c4, taser, unknown -> utility


def process(job):
    mid, mp, path, split = job
    fn = SNAP / f"{mid}_{mp}.parquet"
    if fn.exists():
        return mid, mp, "cached", 0, {}
    rt = ROUND_TABLE[(ROUND_TABLE.match_id == mid) & (ROUND_TABLE.map_name == mp)].set_index("round")
    for attempt in range(4):
        try:
            with HfFileSystem().open(path, "rb") as fh:
                tb = pq.ParquetFile(fh).read(columns=COLS).to_pandas()
            break
        except Exception as e:  # network hiccup
            if attempt == 3:
                return mid, mp, f"FAILED {e!r}", 0, {}
            time.sleep(5 * (attempt + 1))
    rnd = tb["round"].to_numpy().astype(np.int64)
    tick = tb["tick"].to_numpy().astype(np.int64)
    alive = tb["is_alive"].to_numpy()
    spd = np.hypot(tb["velocity_x"].to_numpy(), tb["velocity_y"].to_numpy()).astype(np.float32)
    w = tb["input_weapon"]
    if isinstance(w.dtype, pd.CategoricalDtype):
        wc = np.array([wclass(str(c)) for c in w.cat.categories], dtype=np.int8)[w.cat.codes.to_numpy()]
    else:
        wc = w.astype(str).map(wclass).to_numpy(np.int8)
    nr = int(max(rnd.max(), rt.index.max())) + 1
    fe = np.full(nr, -1, np.int64); dur = np.full(nr, -1.0)
    fe[rt.index.to_numpy()] = rt.freeze_end_tick.to_numpy(); dur[rt.index.to_numpy()] = rt.playable_duration_s.to_numpy()
    out = []
    for t in PREDICTION_TIMES_S:
        pred = np.where(dur > t, fe + t * TICK_RATE, -10**9)[rnd]       # per-row prediction tick of its round
        rel = tick - pred
        cur = (rel == 0) & alive
        win = (rel <= 0) & (rel >= -WINDOW_TICKS) & alive
        c = pd.DataFrame({"round": rnd[cur], "player_slot": tb["player_slot"].to_numpy()[cur],
                          "team": tb["team_num"].to_numpy()[cur], "hp": tb["health"].to_numpy()[cur],
                          "armor": tb["armor_value"].to_numpy()[cur], "wclass": wc[cur],
                          "x": tb["x"].to_numpy()[cur], "y": tb["y"].to_numpy()[cur]})
        ws = pd.DataFrame({"round": rnd[win], "player_slot": tb["player_slot"].to_numpy()[win], "s": spd[win]})
        ws = ws.groupby(["round", "player_slot"]).s.agg(spd_win_mean="mean", n_win="size").reset_index()
        c = c.merge(ws, on=["round", "player_slot"], how="left")
        c.insert(1, "t_s", t)
        out.append(c)
    snap = pd.concat(out, ignore_index=True)
    snap.insert(0, "map_name", mp); snap.insert(0, "match_id", mid)
    OUT.mkdir(parents=True, exist_ok=True); SNAP.mkdir(exist_ok=True); BSAMP.mkdir(exist_ok=True)
    if split == "train":    # 1 Hz alive-player position sample for map bounds (TRAIN files only)
        rel0 = tick - fe[rnd]
        m = alive & (rel0 >= 0) & (rel0 % TICK_RATE == 0)
        pd.DataFrame({"x": tb["x"].to_numpy()[m], "y": tb["y"].to_numpy()[m]}).to_parquet(BSAMP / f"{mid}_{mp}.parquet")
    first_ok = float((tb.groupby("round").tick.min().reindex(rt.index) == rt.freeze_end_tick).mean())
    snap.to_parquet(fn.with_suffix(".tmp")); fn.with_suffix(".tmp").rename(fn)
    return mid, mp, "ok", len(tb), {"first_tick_eq_freeze_end": first_ok}


def pilot_jobs():
    """Same selection as Step 3 (Phase-A pilot): all val+test series + 1/3 of train series (rng seed 0)."""
    table = pd.read_parquet(PROCESSED_DIR / "round_table.parquet")
    split = series_split(table)
    fs = pd.DataFrame(list_tree(WDS, "ticks"))
    fs["match_id"] = fs.path.str.extract(r"match_id=(\d+)")[0].astype(int)
    fs["map_name"] = fs.path.str.extract(r"map_name=(de_\w+)/")[0]
    fs["split"] = fs.match_id.map(split)
    assert fs.split.notna().all()
    ser = fs[fs.split == "train"].match_id.unique()
    pilot = np.random.default_rng(0).choice(ser, size=len(ser) // 3, replace=False)
    pil = fs.copy() if FULL else fs[(fs.split != "train") | fs.match_id.isin(pilot)].copy()
    pil["hf_path"] = [f"datasets/{WDS}/{p}" for p in pil.path]
    return table, pil


ROUND_TABLE = None


def _init(table):
    global ROUND_TABLE
    ROUND_TABLE = table


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else None
    table, pil = pilot_jobs()
    print(f"pilot files {len(pil)}, {pil['size'].sum() / 1e9:.2f} GB; split counts {pil.split.value_counts().to_dict()}", flush=True)
    pil.drop(columns=["hf_path"]).to_csv(OUT.parent.parent.parent / "results" / "step4_pilot_files.csv", index=False) if False else None
    jobs = [(r.match_id, r.map_name, r.hf_path, r.split) for r in pil.itertuples()]
    if limit:
        jobs = jobs[:limit]
    t0, done, bad = time.time(), 0, []
    with Pool(8, initializer=_init, initargs=(table,)) as pool:
        for mid, mp, st, n, info in pool.imap_unordered(process, jobs):
            done += 1
            if st.startswith("FAILED"):
                bad.append((mid, mp, st))
            if done % 10 == 0 or st.startswith("FAILED") or limit:
                print(f"[{done}/{len(jobs)}] {mid} {mp} {st} rows={n} {info} elapsed={time.time() - t0:.0f}s", flush=True)
    print("DONE", done, "failed", bad, f"{time.time() - t0:.0f}s", flush=True)
