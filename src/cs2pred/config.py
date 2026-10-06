"""Constants shared across the pipeline."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
META_DIR = PROJECT_ROOT / "data" / "opencs2" / "metadata"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "step1"

TICK_RATE = 64
ROUND_KEY = ["match_id", "map_name", "round"]
MAP_KEY = ["match_id", "map_name"]

# Prediction times (seconds after freeze_end_tick).
PREDICTION_TIMES_S = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 90, 105, 120]

# Positive class of the binary label.
LABEL_COL = "y_ct_win"

# Columns known before the round becomes playable; safe to use as features.
PRE_ROUND_CONTEXT = ["map_index", "match_date", "event"]

# Timing columns needed to define eligibility and to mask future events.
# They contain final-round information (round_end_tick) and are NOT features.
ELIGIBILITY_COLS = [
    "round_start_tick", "freeze_end_tick", "round_end_tick", "official_end_tick",
    "freeze_duration_s", "playable_duration_s",
]

# Raw columns that describe the finished round. Never use as features
# (event-level versions may only be used after masking to tick <= prediction_tick).
OUTCOME_COLUMNS = {
    # rounds.parquet
    "winner_side", "winner_side_id", "reason", "reason_id", "round_duration_ticks",
    "round_duration_s", "round_end_tick", "official_end_tick", "bomb_plant_tick",
    "bomb_plant_seconds", "bomb_site", "bomb_site_id", "has_bomb_plant", "n_kills",
    "n_headshots", "n_awp_kills", "n_smoke_kills", "n_blind_kills", "n_noscope_kills",
    "n_wallbang_kills", "n_trade_kills_5s", "n_1v1_kills", "ct_kills", "t_kills",
    "ct_headshots", "t_headshots", "ct_survivors", "t_survivors", "opening_kill_tick",
    "opening_kill_seconds", "opening_kill_side", "opening_kill_side_id",
    "opening_death_side", "opening_death_side_id", "opening_weapon", "opening_weapon_id",
    "had_clutch_context", "had_1v1",
    # round_player.parquet (per-round final stats)
    "kills", "deaths", "assists", "headshots", "kast",
    # pov_rounds.parquet (final-round player info)
    "survived_round", "death_tick", "capture_end_tick", "primary_weapon",
    "primary_weapon_id", "weapons_used", "weapons_used_ids",
    # matches.parquet (final match result)
    "score1", "score2", "winner", "rounds_played",
    # derived here
    "playable_duration_s", LABEL_COL,
}
