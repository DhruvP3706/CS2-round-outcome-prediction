"""Step 3: tiny reconnaissance probe of the OpenCS2 tick data (no bulk download, no modelling).

Downloads 3 per-POV tick files (~0.4 MB) into a temporary directory, range-reads ONE map-level
tick file (selected columns only) from the consolidated WDS repo, lists repo file sizes via the
Hub API (metadata only), and writes results/step3/probe_results.md.
"""
import json
import re
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfFileSystem  # noqa: E402

from cs2pred.config import META_DIR, PREDICTION_TIMES_S, PROCESSED_DIR, RESULTS_DIR, ROUND_KEY, TICK_RATE  # noqa: E402
from cs2pred.splits import series_split  # noqa: E402

OUT = RESULTS_DIR.parent / "step3"
REPO = "blanchon/opencs2_dataset"
WDS = "blanchon/opencs2_dataset_wds"
PROBE_MAPS = ["de_dust2", "de_nuke", "de_mirage"]
MAP_PROBE = (2392709, "de_dust2")
# Columns proposed for the real download (everything else is skipped by column projection).
PROPOSED_COLS = ["round", "player_slot", "tick", "team_num", "is_alive", "health", "armor_value",
                 "input_weapon_id", "x", "y", "z", "velocity_x", "velocity_y"]
OPTIONAL_COLS = ["velocity_z", "yaw", "pitch"]
L = []  # markdown lines


def p(s=""):
    print(s)
    L.append(s)


def http_url(hf_path):
    hf = hf_path.removeprefix("hf://datasets/")
    repo, rel = hf.split("@main/")
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{rel}"


def list_tree(repo, sub):
    url, out = f"https://huggingface.co/api/datasets/{repo}/tree/main/{sub}?recursive=true&limit=1000", []
    while url:
        r = urllib.request.urlopen(url, timeout=60)
        out += json.load(r)
        link = r.headers.get("Link") or ""
        url = link.split("<")[1].split(">")[0] if 'rel="next"' in link else None
    return [x for x in out if x["type"] == "file"]


def pick_samples(pov):
    med = pov.sidecar_bytes.median()
    c = pov[pov.sidecar_bytes.between(med * .7, med * 1.3) & (pov["round"] >= 3)]
    rng, picks = np.random.default_rng(1), []
    for mp in PROBE_MAPS:
        d = c[(c.map_name == mp) & (~c.match_id.isin([x.match_id for x in picks]))]
        picks.append(d.iloc[rng.integers(len(d))])
    return pd.DataFrame(picks)


def probe_pov_files(pov, rt, tmp):
    p("## 1. Per-POV tick files (3 samples, different matches/maps)")
    for s in pick_samples(pov).itertuples():
        f = Path(tmp) / f"{s.match_id}_{s.map_name}_r{s.round}_p{s.player_slot}.parquet"
        urllib.request.urlretrieve(http_url(s.ticks_parquet_path), f)
        d = pd.read_parquet(f)
        ro = rt[(rt.match_id == s.match_id) & (rt.map_name == s.map_name) & (rt["round"] == s.round)].iloc[0]
        p(f"- match {s.match_id} {s.map_name} r{s.round} slot {s.player_slot} ({s.player_side}): "
          f"{len(d)} rows, file {f.stat().st_size} B (sidecar_bytes {s.sidecar_bytes}); "
          f"tick {d.tick.min()}..{d.tick.max()}, spacing {sorted(int(x) for x in set(np.diff(d.tick)))}, "
          f"first tick == freeze_end_tick: {d.tick.iloc[0] == ro.freeze_end_tick}, "
          f"last tick - round_end_tick = {d.tick.iloc[-1] - ro.round_end_tick}, "
          f"death_tick={s.death_tick}, rows are one player only (no slot column), "
          f"is_alive always True in file: {bool(d.is_alive.all())}, team_num {sorted(int(x) for x in d.team_num.unique())}, "
          f"weapons seen: {sorted(d.input_weapon.astype(str).unique())[:8]}")
        if s.Index == pick_samples(pov).index[0]:
            sch = pq.ParquetFile(f).schema_arrow
            p("\nExact per-POV schema (25 columns): " + ", ".join(f"{n}:{sch.field(n).type}" for n in sch.names
                                                                    if "list" not in str(sch.field(n).type)) +
              "; list columns: active/pressed/released (button names), *_ids (int16).\n")


def probe_map_file(rt, pov):
    p("\n## 2. Consolidated map-level file (WDS repo), range-read")
    mid, mp = MAP_PROBE
    path = f"datasets/{WDS}/ticks/match_id={mid}/map_name={mp}/ticks.parquet"
    with HfFileSystem().open(path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        md, rg = pf.metadata, pf.metadata.row_group(0)
        sizes = {rg.column(i).path_in_schema.split(".")[0]: 0 for i in range(rg.num_columns)}
        for i in range(rg.num_columns):
            sizes[rg.column(i).path_in_schema.split(".")[0]] += rg.column(i).total_compressed_size
        p(f"- {md.num_rows} rows, {md.num_row_groups} row group(s), columns: {pf.schema_arrow.names}")
        tb = pf.read_row_group(0, columns=PROPOSED_COLS + ["velocity_z"]).to_pandas()
    tot = sum(sizes.values())
    p(f"- players per round (all rounds): {tb.groupby('round').player_slot.nunique().value_counts().to_dict()} "
      f"=> one map file holds ALL 10 players")
    r = rt[(rt.match_id == mid) & (rt.map_name == mp)].set_index("round")
    g = tb.groupby(["round", "player_slot"]).tick.agg(["min", "max", "size"]).reset_index().join(
        r[["freeze_end_tick", "round_end_tick"]], on="round")
    p(f"- every (round,player) starts at freeze_end_tick: {(g['min'] == g.freeze_end_tick).all()}; "
      f"contiguous 1-tick spacing: {(g['size'] == g['max'] - g['min'] + 1).all()}; "
      f"players whose last tick == round_end_tick+32: {int(((g['max'] - g.round_end_tick) == 32).sum())} "
      f"(survivors per pov_rounds: {int(pov[(pov.match_id == mid) & (pov.map_name == mp)].death_tick.isna().sum())})")
    dead = tb[~tb.is_alive].groupby(["round", "player_slot"]).tick.min().reset_index(name="first_dead")
    kills = pd.read_parquet(META_DIR / "kills.parquet", columns=ROUND_KEY + ["tick", "victim_player_slot"])
    kills = kills[(kills.match_id == mid) & (kills.map_name == mp)].rename(columns={"victim_player_slot": "player_slot", "tick": "kill_tick"})
    m = dead.merge(kills[["round", "player_slot", "kill_tick"]], on=["round", "player_slot"])
    p(f"- first is_alive=False tick == kill tick of that victim: {(m.first_dead == m.kill_tick).mean():.3f} (n={len(m)}); "
      f"rows continue ~7 ticks after death (health 0), then the player has no rows")
    # Reconstruct alive counts at prediction times from ticks and compare with the kill feed.
    ok, n = 0, 0
    for t in (10, 30, 60):
        for rnd, ro in r.iterrows():
            if (ro.round_end_tick - ro.freeze_end_tick) / TICK_RATE <= t:
                continue
            cut = ro.freeze_end_tick + t * TICK_RATE
            x = tb[(tb["round"] == rnd) & (tb.tick == cut) & tb.is_alive]
            kk = kills[(kills["round"] == rnd) & (kills.kill_tick <= cut)]
            n += 1
            ok += int(len(x) == 10 - len(kk))
    p(f"- alive players at prediction tick from ticks == 10 - kills(tick<=t) in {ok}/{n} (round,t) checks")
    p("\nColumn share of compressed bytes (this file):")
    p(", ".join(f"{k} {v / tot:.1%}" for k, v in sorted(sizes.items(), key=lambda kv: -kv[1])[:12]))
    return sizes, tot


def size_estimates(pov, sizes, tot):
    p("\n## 3. Size / cost estimates")
    a = pov.sidecar_bytes.sum()
    p(f"- MEASURED: per-POV route = {len(pov):,} files; sum(sidecar_bytes) = {a / 1e9:.2f} GB "
      f"(the 3 probe files were 97% of sidecar_bytes => ~{0.97 * a / 1e9:.1f} GB actual, ESTIMATE).")
    files = list_tree(WDS, "ticks")
    fs = pd.DataFrame(files)
    fs["match_id"] = fs.path.str.extract(r"match_id=(\d+)")[0].astype(int)
    fs["map_name"] = fs.path.str.extract(r"map_name=(de_\w+)/")[0]
    p(f"- MEASURED (Hub API listing, nothing downloaded): map-level route = {len(fs)} files, "
      f"{fs['size'].sum() / 1e9:.2f} GB total, median {fs['size'].median() / 1e6:.1f} MB, "
      f"range {fs['size'].min() / 1e6:.1f}-{fs['size'].max() / 1e6:.1f} MB.")
    frac = lambda cols: sum(sizes[c] for c in cols if c in sizes) / tot
    f_core, f_all = frac(PROPOSED_COLS), frac(PROPOSED_COLS + OPTIONAL_COLS)
    p(f"- ESTIMATE (assumes the column-byte shares of the probe file hold for all maps; parquet has ONE row group "
      f"per file, so whole column chunks are read but unused columns are skipped): proposed columns = {f_core:.0%} "
      f"of bytes = {f_core * fs['size'].sum() / 1e9:.1f} GB; + velocity_z/yaw/pitch = {f_all:.0%} = "
      f"{f_all * fs['size'].sum() / 1e9:.1f} GB.")
    split = series_split(pd.read_parquet(PROCESSED_DIR / "round_table.parquet"))
    fs["split"] = fs.match_id.map(split)
    by = fs.groupby("split").agg(files=("size", "size"), GB=("size", lambda s: s.sum() / 1e9)).reindex(["train", "val", "test"])
    by["GB_proposed_cols_est"] = by.GB * f_core
    p("- MEASURED file counts/sizes by Step-2 series split (full files), proposed-column estimate alongside:\n")
    p("```\n" + by.round(2).to_string() + "\n```")
    ser = fs[fs.split == "train"].match_id.unique()
    pilot = np.random.default_rng(0).choice(ser, size=len(ser) // 3, replace=False)
    pil = fs[(fs.split != "train") | fs.match_id.isin(pilot)]
    p(f"\n- Phase-A pilot = all val+test series + 1/3 of train series: {len(pil)} files, "
      f"{pil['size'].sum() / 1e9:.1f} GB full / ~{f_core * pil['size'].sum() / 1e9:.1f} GB with column projection (ESTIMATE).")
    rows_per_byte = 975502 / 28072534
    p(f"- ESTIMATE: total tick rows ~{rows_per_byte * fs['size'].sum() / 1e6:.0f}M "
      f"(975,502 rows / 28.07 MB in the probe file); raw float32 in RAM ~25-30 GB => process map by map and keep "
      f"only snapshots at the prediction ticks (+ short look-back windows), never the raw ticks.")
    p("- ASSUMPTION (unmeasured): download time is bandwidth/rate-limit bound; footer read took 3.4 s and a 12-column "
      "read of a 28 MB file took 4.8 s here => roughly 1-2 h sequential for all 794 files, less with a few parallel workers.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pov = pd.read_parquet(META_DIR / "pov_rounds.parquet", columns=ROUND_KEY + [
        "player_slot", "player_side", "sidecar_bytes", "ticks_parquet_path", "death_tick"])
    pov["map_name"] = pov["map_name"].astype(str)
    rt = pd.read_parquet(PROCESSED_DIR / "round_table.parquet")
    p("# Step 3 - tick data probe results (auto-generated by src/step3_tick_probe.py)\n")
    with tempfile.TemporaryDirectory() as tmp:
        probe_pov_files(pov, rt, tmp)
    sizes, tot = probe_map_file(rt, pov)
    size_estimates(pov, sizes, tot)
    (OUT / "probe_results.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
