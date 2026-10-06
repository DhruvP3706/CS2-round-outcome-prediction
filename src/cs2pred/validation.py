"""Validation checks for the round table and its relationships to the OpenCS2 files."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (LABEL_COL, MAP_KEY, META_DIR, OUTCOME_COLUMNS, PREDICTION_TIMES_S, ROUND_KEY,
                     TICK_RATE)
from .round_table import assert_no_outcome_columns, eligible_rounds


def _res(name, n_bad, severity="error", detail=""):
    return {"check": name, "n_violations": int(n_bad), "severity": severity,
            "status": "PASS" if n_bad == 0 else ("FAIL" if severity == "error" else "WARN"),
            "detail": detail}


def validate(table: pd.DataFrame, meta_dir=META_DIR) -> pd.DataFrame:
    out = []
    add = lambda *a, **k: out.append(_res(*a, **k))

    # --- keys / labels
    add("duplicate round keys", table.duplicated(ROUND_KEY).sum())
    add("null in key columns", table[ROUND_KEY].isna().any(axis=1).sum())
    add("missing/invalid winner_side (not ct/t)", (~table["winner_side"].isin(["ct", "t"])).sum())
    add("missing label", table[LABEL_COL].isna().sum())

    # --- tick ordering / durations
    tk = table[["round_start_tick", "freeze_end_tick", "round_end_tick", "official_end_tick"]]
    add("null ticks", tk.isna().any(axis=1).sum())
    add("round_start_tick > freeze_end_tick", (tk.round_start_tick > tk.freeze_end_tick).sum())
    add("freeze_end_tick >= round_end_tick", (tk.freeze_end_tick >= tk.round_end_tick).sum())
    add("round_end_tick > official_end_tick", (tk.round_end_tick > tk.official_end_tick).sum())
    d = table["playable_duration_s"]
    add("playable_duration_s null/non-finite/<=0", (~np.isfinite(d) | (d <= 0)).sum())
    add("playable_duration_s > 175 s (implausible)", (d > 175).sum())
    add("playable_duration_s < 10 s (suspiciously short)", (d < 10).sum(), "warn")
    add("playable_duration_s == (round_end-freeze_end)/64", (
        ~np.isclose(d, (table.round_end_tick - table.freeze_end_tick) / TICK_RATE)).sum())
    add("freeze_duration_s > 25 s (timeouts/pauses; clock still anchored on freeze_end)",
        (table.freeze_duration_s > 25).sum(), "warn")
    add("rounds ticks non-monotonic across rounds of a map (freeze_end not increasing)", (
        table.sort_values(ROUND_KEY).groupby(MAP_KEY)["freeze_end_tick"].diff() <= 0).sum())

    # --- relationships
    matches = pd.read_parquet(f"{meta_dir}/matches.parquet")
    matches["map_name"] = matches["map_name"].astype(str)
    add("duplicate (match_id,map_name) in matches", matches.duplicated(MAP_KEY).sum())
    add("rounds without a matches row (null map_index)", table["map_index"].isna().sum())
    tmaps = table[MAP_KEY].drop_duplicates()
    add("matches maps with no rounds", len(matches[MAP_KEY].merge(tmaps, how="left", indicator=True)
                                           .query("_merge=='left_only'")))
    cnt = table.groupby(MAP_KEY).size().rename("n").reset_index().merge(
        matches[MAP_KEY + ["rounds_played"]], on=MAP_KEY, how="outer")
    add("rounds count != matches.rounds_played", (cnt.n != cnt.rounds_played).sum())
    contig = table.groupby(MAP_KEY)["round"].agg(lambda s: (s.min() != 1) or (s.max() != s.nunique()))
    add("round numbers not contiguous 1..N within a map", contig.sum())

    pov = pd.read_parquet(f"{meta_dir}/pov_rounds.parquet", columns=ROUND_KEY + [
        "player_slot", "player_side", "round_start_tick", "freeze_end_tick", "round_end_tick"])
    pov["map_name"] = pov["map_name"].astype(str)
    g = pov.groupby(ROUND_KEY)
    add("rounds not having exactly 10 pov rows", (g.size() != 10).sum() + (len(table) - g.ngroups))
    add("rounds without exactly 5 CT / 5 T pov rows",
        (g["player_side"].agg(lambda s: ((s == "CT").sum(), (s == "T").sum())) != (5, 5)).sum())
    pj = pov.drop_duplicates(ROUND_KEY).merge(table, on=ROUND_KEY, suffixes=("_pov", ""))
    for c in ["round_start_tick", "freeze_end_tick", "round_end_tick"]:
        add(f"pov_rounds.{c} != rounds.{c}", (pj[c + "_pov"] != pj[c]).sum())
    pov_only = pov[ROUND_KEY].drop_duplicates().merge(table[ROUND_KEY], how="left", indicator=True)
    add("pov_rounds keys not in rounds", (pov_only["_merge"] == "left_only").sum())

    kills = pd.read_parquet(f"{meta_dir}/kills.parquet", columns=ROUND_KEY + ["tick", "event_seconds"])
    kills["map_name"] = kills["map_name"].astype(str)
    kj = kills.merge(table[ROUND_KEY + ["freeze_end_tick", "round_end_tick"]], on=ROUND_KEY, how="left")
    add("kills keys not in rounds", kj["freeze_end_tick"].isna().sum())
    add("kills before freeze_end_tick", (kj.tick < kj.freeze_end_tick).sum())
    add("kills.event_seconds != (tick-freeze_end)/64",
        (~np.isclose(kj.event_seconds, (kj.tick - kj.freeze_end_tick) / TICK_RATE, atol=0.02)).sum())
    add("kills after round_end_tick (post-round c4/world deaths; must be masked by tick<=t)",
        (kj.tick > kj.round_end_tick).sum(), "warn")

    rpl = pd.read_parquet(f"{meta_dir}/round_player.parquet", columns=ROUND_KEY + ["player_slot"])
    rpl["map_name"] = rpl["map_name"].astype(str)
    rj = rpl[ROUND_KEY].drop_duplicates().merge(table[ROUND_KEY], how="left", indicator=True)
    add("round_player keys not in rounds", (rj["_merge"] == "left_only").sum())
    add("round_player rows with player_slot == -1 (pseudo-player; drop)", (rpl.player_slot == -1).sum(), "warn")

    # --- eligibility invariants
    prev, bad_nested, bad_tick, bad_dur, leak = None, 0, 0, 0, 0
    by_key = table.set_index(ROUND_KEY)
    for t in PREDICTION_TIMES_S:
        e = eligible_rounds(table, t)
        idx = pd.MultiIndex.from_frame(e[ROUND_KEY])
        keys = set(idx)
        if prev is not None and not keys <= prev:
            bad_nested += 1
        prev = keys
        ref = by_key.loc[idx]
        bad_dur += int((ref["playable_duration_s"].to_numpy() <= t).sum())
        bad_tick += int((e["prediction_tick"].to_numpy() != ref["freeze_end_tick"].to_numpy() + t * TICK_RATE).sum())
        bad_tick += int((e["prediction_tick"].to_numpy() >= ref["round_end_tick"].to_numpy()).sum())
        if t == PREDICTION_TIMES_S[0]:
            leak = len((set(e.columns) & OUTCOME_COLUMNS) - {LABEL_COL})
    add("eligible sets not nested (live at later t => live at earlier t)", bad_nested)
    add("eligible rounds with playable_duration_s <= t", bad_dur)
    add("prediction_tick != freeze_end+t*64 or >= round_end_tick", bad_tick)
    add("eligible frame exposes outcome columns other than the label", leak)
    add("eligible frame passes assert_no_outcome_columns after dropping label",
        0 if _guard_ok(eligible_rounds(table, PREDICTION_TIMES_S[0]).drop(columns=[LABEL_COL])) else 1)
    return pd.DataFrame(out)


def _guard_ok(df):
    try:
        assert_no_outcome_columns(df)
        return True
    except ValueError:
        return False
