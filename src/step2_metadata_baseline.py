"""Step 2: metadata-only baseline (LogReg + LightGBM) over all prediction times."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cs2pred.config import LABEL_COL, PREDICTION_TIMES_S, PROCESSED_DIR, ROUND_KEY, OUTCOME_COLUMNS  # noqa: E402
from cs2pred.features import (build_features, feature_columns, kill_flags, load_kills,  # noqa: E402
                              load_plants, pre_round_features)
from cs2pred.modeling import evaluate_time  # noqa: E402
from cs2pred.splits import series_split  # noqa: E402

RES = ROOT / "results" / "step2"
FEAT_DIR = ROOT / "data" / "features"
PLOTS = ROOT / "plots"


def lookahead_test(table, kills, plants, pre, t):
    """Corrupt everything after the cutoff; features at t must not change."""
    rng = np.random.default_rng(0)
    base_flags = kill_flags(kills, table)
    ref = build_features(table, base_flags, plants, pre, t)
    fe = table.set_index(ROUND_KEY)["freeze_end_tick"]
    k = kills.copy()
    cut = fe.reindex(pd.MultiIndex.from_frame(k[ROUND_KEY])).to_numpy() + int(round(t * 64))
    fut = (k["tick"].to_numpy() > cut)
    for c in ["attacker_side", "victim_side", "weapon"]:
        k.loc[fut, c] = rng.permutation(k.loc[fut, c].to_numpy())
    for c in ["headshot", "through_smoke", "wallbang", "attacker_blind", "noscope"]:
        k.loc[fut, c] = rng.permutation(k.loc[fut, c].to_numpy())
    p = plants.copy()
    pc = fe.reindex(pd.MultiIndex.from_frame(p[ROUND_KEY])).to_numpy() + int(round(t * 64))
    late = p["bomb_plant_tick"].to_numpy() > pc
    p.loc[late, "bomb_plant_tick"] = p.loc[late, "bomb_plant_tick"] + rng.integers(1, 500, late.sum())
    mut = build_features(table, kill_flags(k, table), p, pre, t)
    cols = feature_columns(ref)
    return bool(np.allclose(ref[cols].to_numpy(float), mut[cols].to_numpy(float), equal_nan=True)), int(fut.sum()), int(late.sum())


def main():
    RES.mkdir(parents=True, exist_ok=True); FEAT_DIR.mkdir(parents=True, exist_ok=True); PLOTS.mkdir(exist_ok=True)
    table = pd.read_parquet(PROCESSED_DIR / "round_table.parquet")
    kills, plants = load_kills(), load_plants()
    pre = pre_round_features(table)
    flags = kill_flags(kills, table)

    # ---- features for every t
    feats = {t: build_features(table, flags, plants, pre, t) for t in PREDICTION_TIMES_S}
    cols = feature_columns(feats[PREDICTION_TIMES_S[0]])
    bad = set(cols) & OUTCOME_COLUMNS
    assert not bad, bad
    assert all(feature_columns(f) == cols for f in feats.values())
    pd.concat(feats.values()).to_parquet(FEAT_DIR / "step2_metadata_features.parquet", index=False)
    (RES / "feature_list.txt").write_text("\n".join(cols))
    print(f"features: {len(cols)}")

    # ---- leakage checks
    leak_rows = []
    for t in (10, 30, 60, 90):
        ok, nf, npl = lookahead_test(table, kills, plants, pre, t)
        leak_rows.append({"t_s": t, "future_kills_corrupted": nf, "future_plants_shifted": npl, "features_unchanged": ok})
    leak = pd.DataFrame(leak_rows); leak.to_csv(RES / "leakage_checks.csv", index=False)
    print(leak.to_string(index=False))
    # own kill-feed alive counts vs. round-level survivors at end of round (sanity only)
    # ---- split
    split = series_split(table)
    tt = table.assign(split=table.match_id.map(split))
    ss = tt.groupby("split").agg(n_series=("match_id", "nunique"), n_rounds=("round", "size"),
                                 ct_win_rate=(LABEL_COL, "mean")).reindex(["train", "val", "test"])
    ss["n_maps"] = tt.groupby("split").apply(lambda d: d[["match_id", "map_name"]].drop_duplicates().shape[0], include_groups=False)
    ss.to_csv(RES / "split_sizes.csv"); print(ss)
    assert tt.groupby("match_id")["split"].nunique().max() == 1
    n_total = ss["n_rounds"].to_dict()
    split.rename("split").to_frame().to_csv(RES / "series_split.csv", index_label="match_id")

    # ---- train / tune / test
    rows = []
    for t in PREDICTION_TIMES_S:
        rows += evaluate_time(feats[t], split, cols, t, n_total)
        r = [x for x in rows if x["t_s"] == t]
        print(t, " | ".join(f"{x['model']}: thr={x['val_threshold']:.3f} val={x['val_macro_f1']:.3f} test={x['test_macro_f1']:.3f}" for x in r), flush=True)
    res = pd.DataFrame(rows)
    res["coverage_overall"] = res["t_s"].map({t: len(feats[t]) / len(table) for t in PREDICTION_TIMES_S})
    res.to_csv(RES / "results.csv", index=False)

    # ---- plot
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for name, c, lab in (("logreg", "#8a8f98", "Logistic Regression"), ("lightgbm", "#1f6feb", "LightGBM")):
        d = res[res.model == name]
        a.plot(d.t_s, d.test_macro_f1, marker="o", color=c, label=lab)
        a.fill_between(d.t_s, d.test_macro_f1_ci_lo, d.test_macro_f1_ci_hi, color=c, alpha=0.15, lw=0)
    a.axhline(0.80, ls="--", color="#c0392b", lw=1); a.text(PREDICTION_TIMES_S[0], 0.805, "target 0.80", color="#c0392b", fontsize=8)
    a.set_ylabel("Test macro F1 (frozen val threshold)"); a.set_title("Metadata-only baseline: F1 vs prediction time (live rounds only)")
    a.grid(alpha=0.25); a.legend(loc="lower right"); a.set_ylim(0.4, 1.0)
    d = res[res.model == "lightgbm"]
    b.bar(d.t_s, d.n_test, width=3.5, color="#8a8f98"); b.set_ylabel("eligible test rounds"); b.set_xlabel("prediction time t (s after freeze end)")
    for x, n, cv in zip(d.t_s, d.n_test, d.coverage_test):
        b.text(x, n, f"{cv:.0%}", ha="center", va="bottom", fontsize=7)
    b.grid(alpha=0.25, axis="y"); fig.tight_layout(); fig.savefig(PLOTS / "step2_f1_vs_time.png", dpi=150)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    show = ["model", "t_s", "n_train", "n_val", "n_test", "coverage_test", "val_threshold", "test_macro_f1", "test_ct_f1", "test_t_f1", "test_auc"]
    print(res[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
