"""Step 5a: build all Step 5 features (+ quality checks + leakage sanity test). Reads Step 4 snapshots; no downloads."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cs2pred.config import OUTCOME_COLUMNS, PREDICTION_TIMES_S, PROCESSED_DIR, ROUND_KEY, TICK_RATE  # noqa: E402
from cs2pred.features import kill_flags, load_kills, load_plants, pre_round_features  # noqa: E402
from cs2pred.step5_features import FAMILIES, FIREPOWER, build_all, family_columns  # noqa: E402

RES, FEAT = ROOT / "results" / "step5", ROOT / "data" / "features" / "step5"


def load_snap():
    return pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "data/features/step4/snapshots").glob("*.parquet"))], ignore_index=True)


def lookahead_test(table, kills, plants, pre, snap, T, ref):
    """Corrupt kills/plants/snapshots strictly after the cutoff of T; every feature for t <= T must be unchanged."""
    rng = np.random.default_rng(0)
    fe = table.set_index(ROUND_KEY)["freeze_end_tick"]
    k = kills.copy()
    cut = fe.reindex(pd.MultiIndex.from_frame(k[ROUND_KEY])).to_numpy() + int(round(T * TICK_RATE))
    fut = k["tick"].to_numpy() > cut
    for c in ["attacker_side", "victim_side", "weapon", "headshot", "through_smoke", "wallbang", "attacker_blind", "noscope"]:
        k.loc[fut, c] = rng.permutation(k.loc[fut, c].to_numpy())
    p = plants.copy()
    pc = fe.reindex(pd.MultiIndex.from_frame(p[ROUND_KEY])).to_numpy() + int(round(T * TICK_RATE))
    late = p["bomb_plant_tick"].to_numpy() > pc
    p.loc[late, "bomb_plant_tick"] += rng.integers(1, 500, late.sum())
    s = snap.copy()
    fs = (s.t_s > T).to_numpy()
    for c in ["team", "hp", "armor", "wclass", "x", "y", "spd_win_mean"]:
        s.loc[fs, c] = rng.permutation(s.loc[fs, c].to_numpy())
    times = [t for t in PREDICTION_TIMES_S if t <= T]
    mut = build_all(table, kill_flags(k, table), p, pre, s, times)
    r = ref[ref.prediction_time_s <= T].sort_values(ROUND_KEY + ["prediction_time_s"]).reset_index(drop=True)
    mut = mut.sort_values(ROUND_KEY + ["prediction_time_s"]).reset_index(drop=True)
    cols = [c for c in r.columns if c not in ROUND_KEY + ["prediction_time_s", "prediction_tick"]]
    same_rows = len(r) == len(mut) and (r[ROUND_KEY].to_numpy() == mut[ROUND_KEY].to_numpy()).all()
    ok = same_rows and np.allclose(r[cols].to_numpy(float), mut[cols].to_numpy(float), equal_nan=True)
    bad = [c for c in cols if not np.allclose(r[c].to_numpy(float), mut[c].to_numpy(float), equal_nan=True)] if same_rows else ["ROWS"]
    return {"T_s": T, "future_kills_corrupted": int(fut.sum()), "future_plants_shifted": int(late.sum()),
            "future_snapshot_rows_scrambled": int(fs.sum()), "features_unchanged": bool(ok), "n_changed_cols": len(bad), "changed": ";".join(bad[:5])}


def main():
    RES.mkdir(parents=True, exist_ok=True); FEAT.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(PROCESSED_DIR / "round_table.parquet")
    kills, plants, pre = load_kills(), load_plants(), pre_round_features(table)
    flags, snap = kill_flags(kills, table), load_snap()
    df = build_all(table, flags, plants, pre, snap)
    fam = family_columns(df)
    meta_cols = (ROOT / "results/step2/feature_list.txt").read_text().split("\n")
    # --- Step 2 metadata must be reproduced exactly (no modification of Step 2)
    s2 = pd.read_parquet(ROOT / "data/features/step2_metadata_features.parquet")
    j = s2.merge(df[ROUND_KEY + ["prediction_time_s"] + meta_cols], on=ROUND_KEY + ["prediction_time_s"], suffixes=("", "_n"))
    assert all(np.allclose(j[c].to_numpy(float), j[c + "_n"].to_numpy(float), equal_nan=True) for c in meta_cols), "meta mismatch"
    # --- quality checks
    new = [c for l in fam.values() for c in l]
    assert not (set(new) | set(meta_cols)) & OUTCOME_COLUMNS
    X = df[new].astype(float)
    q = pd.DataFrame({"family": [c.split("_")[0] for c in new], "feature": new, "n_nan": X.isna().sum().to_numpy(),
                      "n_inf": np.isinf(X).sum().to_numpy(), "n_unique": X.nunique().to_numpy(), "mean": X.mean().to_numpy(),
                      "std": X.std().to_numpy()})
    q.to_csv(RES / "feature_quality.csv", index=False)
    print("rows", len(df), "| NaN cells in new feats by col (nonzero):", q[q.n_nan > 0][["feature", "n_nan"]].to_dict("records"), "| inf:", int(q.n_inf.sum()),
          "| constant:", q[q.n_unique <= 1].feature.tolist())
    # redundancy: |spearman| > 0.98 at t=50 among new feats, and vs existing (meta + ps) feats
    d50 = df[df.prediction_time_s == 50]
    ps = [c for c in pd.read_parquet(ROOT / "data/features/step4_full/step4_tick_features.parquet").columns if c.startswith("ps_")]
    ps_df = pd.read_parquet(ROOT / "data/features/step4_full/step4_tick_features.parquet")
    ps_df = ps_df[ps_df.prediction_time_s == 50].rename(columns={"t_s": "prediction_time_s"})
    d50 = d50.merge(ps_df[ROUND_KEY + ["prediction_time_s"] + ps], on=ROUND_KEY + ["prediction_time_s"])
    C = d50[new + meta_cols + ps].astype(float).loc[:, lambda d: d.std() > 0].corr(method="spearman").abs()
    pairs = [(a, b, C.loc[a, b]) for i, a in enumerate(C.columns) for b in C.columns[i + 1:] if C.loc[a, b] > 0.98 and (a in new or b in new)]
    pd.DataFrame(pairs, columns=["a", "b", "abs_spearman_t50"]).sort_values("abs_spearman_t50", ascending=False).to_csv(RES / "redundant_pairs.csv", index=False)
    print("near-duplicate pairs (|rho|>0.98, t=50):", len(pairs))
    # --- leakage sanity
    leak = pd.DataFrame([lookahead_test(table, kills, plants, pre, snap, T, df) for T in (10, 30, 60, 90)])
    leak.to_csv(RES / "leakage_checks.csv", index=False); print(leak.to_string(index=False))
    df.to_parquet(FEAT / "step5_features.parquet", index=False)
    (RES / "feature_groups.txt").write_text("\n".join(f"[{f}] ({len(c)})\n" + "\n".join(c) for f, c in fam.items()) +
                                             f"\n\n[firepower mapping fixed a priori] {FIREPOWER}\n")
    print({f: len(c) for f, c in fam.items()})


if __name__ == "__main__":
    main()
