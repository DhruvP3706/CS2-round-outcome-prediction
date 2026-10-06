"""Canonical round table, prediction clock and eligibility."""
from __future__ import annotations

import pandas as pd

from .config import (ELIGIBILITY_COLS, LABEL_COL, MAP_KEY, META_DIR, OUTCOME_COLUMNS,
                     PRE_ROUND_CONTEXT, PREDICTION_TIMES_S, ROUND_KEY, TICK_RATE)

_ROUND_COLS = ROUND_KEY + [
    "round_start_tick", "freeze_end_tick", "round_end_tick", "official_end_tick", "winner_side",
]


def build_round_table(meta_dir=META_DIR) -> pd.DataFrame:
    """One row per (match_id, map_name, round).

    Contains only: keys, pre-round context, clock/eligibility timing, and the label.
    Outcome-describing columns (reason, kills, bomb info, ...) are deliberately not carried over.
    """
    rounds = pd.read_parquet(f"{meta_dir}/rounds.parquet", columns=_ROUND_COLS)
    matches = pd.read_parquet(f"{meta_dir}/matches.parquet", columns=MAP_KEY + PRE_ROUND_CONTEXT)
    df = rounds.merge(matches, on=MAP_KEY, how="left", validate="many_to_one")

    df["map_name"] = df["map_name"].astype(str)
    df["winner_side"] = df["winner_side"].astype(str)
    df[LABEL_COL] = df["winner_side"].map({"ct": 1, "t": 0}).astype("Int8")
    df["freeze_duration_s"] = (df["freeze_end_tick"] - df["round_start_tick"]) / TICK_RATE
    # Prediction clock: t = 0 at freeze_end_tick (start of playable time).
    df["playable_duration_s"] = (df["round_end_tick"] - df["freeze_end_tick"]) / TICK_RATE
    return df.sort_values(ROUND_KEY).reset_index(drop=True)


def eligible_rounds(table: pd.DataFrame, t: float, include_audit: bool = False) -> pd.DataFrame:
    """Rounds still live at prediction time t (playable_duration_s > t).

    Returned frame: keys, pre-round context, prediction_time_s, prediction_tick (the last tick
    whose data may be used as a feature) and the label. Final-round timing columns are
    dropped unless include_audit=True (for diagnostics only, never as features).
    """
    live = table[table["playable_duration_s"] > t].copy()
    live["prediction_time_s"] = t
    live["prediction_tick"] = live["freeze_end_tick"] + int(round(t * TICK_RATE))
    cols = ROUND_KEY + _pre_round_cols(live) + ["prediction_time_s", "prediction_tick", LABEL_COL]
    if include_audit:
        cols += ELIGIBILITY_COLS
    return live[cols].reset_index(drop=True)


def _pre_round_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in PRE_ROUND_CONTEXT if c in df.columns]


def eligibility_summary(table: pd.DataFrame, times=PREDICTION_TIMES_S) -> pd.DataFrame:
    """Per-time counts: eligible rounds, coverage and class balance (positive = CT win)."""
    total = len(table)
    rows = []
    for t in times:
        e = eligible_rounds(table, t)
        n = len(e)
        n_ct = int(e[LABEL_COL].sum())
        rows.append({
            "t_s": t, "n_total": total, "n_eligible": n, "coverage": n / total,
            "n_ct_win": n_ct, "n_t_win": n - n_ct, "ct_win_rate": n_ct / n if n else float("nan"),
            "n_maps_eligible": e[MAP_KEY[:2]].drop_duplicates().shape[0],
        })
    return pd.DataFrame(rows)


def assert_no_outcome_columns(feature_df: pd.DataFrame) -> None:
    """Guard for feature matrices: raise if any final-round/outcome column slipped in."""
    bad = sorted(set(feature_df.columns) & OUTCOME_COLUMNS)
    if bad:
        raise ValueError(f"Outcome/leaky columns present in feature frame: {bad}")
