"""Step 4c: LightGBM metadata baseline vs. + tick-derived feature groups (Phase-A pilot). No tuning."""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402

from cs2pred.config import LABEL_COL, PREDICTION_TIMES_S, PROCESSED_DIR, ROUND_KEY  # noqa: E402
from cs2pred.modeling import fit_predict, macro_f1, tune_threshold  # noqa: E402
from cs2pred.splits import series_split  # noqa: E402

RES, FEAT = ROOT / "results" / "step4", ROOT / "data" / "features"
SETS = {"A_meta_pilot": [], "B_meta+player": ["ps_"], "C_meta+spatial": ["sp_"],
        "D_meta+player+spatial+move": ["ps_", "sp_", "mv_"], "E_meta+player+spatial (suppl.)": ["ps_", "sp_"]}


def boot_delta(y, p_new, thr_new, p_ref, thr_ref, groups, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in ug}
    d = []
    for _ in range(n_boot):
        k = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        d.append(macro_f1(y[k], p_new[k], thr_new) - macro_f1(y[k], p_ref[k], thr_ref))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    RES.mkdir(parents=True, exist_ok=True); (ROOT / "plots").mkdir(exist_ok=True)
    table = pd.read_parquet(PROCESSED_DIR / "round_table.parquet")
    split = series_split(table)
    n_test_total = int((table.match_id.map(split) == "test").sum())
    meta = pd.read_parquet(FEAT / "step2_metadata_features.parquet")
    meta_cols = (ROOT / "results" / "step2" / "feature_list.txt").read_text().split("\n")
    tick = pd.read_parquet(FEAT / "step4" / "step4_tick_features.parquet")
    tick_cols = [c for c in tick.columns if c[:3] in ("ps_", "sp_", "mv_")]
    df = meta.merge(tick, on=ROUND_KEY + ["prediction_time_s"], how="left", indicator=True)
    df["split"] = df.match_id.map(split)
    have = df["_merge"] == "both"
    # pilot membership: series for which at least one tick file was processed
    pilot_series = set(tick.match_id.unique())
    df["in_pilot"] = df.match_id.isin(pilot_series)
    # sanity: kill-feed alive counts vs. tick alive counts
    both = df[have]
    agree = ((both.ct_alive == both.ps_ct_alive) & (both.t_alive == both.ps_t_alive)).mean()
    cov = df[df.in_pilot & (df.split != "train") | (df.in_pilot & (df.split == "train"))].groupby("split").apply(
        lambda d: have[d.index].mean(), include_groups=False)
    print("alive agreement kill-feed vs ticks:", round(agree, 4), "| tick-feature coverage of eligible rounds in pilot series:", cov.round(4).to_dict())
    pd.DataFrame({"alive_count_agreement": [agree], **{f"pilot_tick_coverage_{k}": [v] for k, v in cov.items()}}).to_csv(RES / "sanity_checks.csv", index=False)
    sets = {k: meta_cols + [c for c in tick_cols if c[:3] in v] for k, v in SETS.items()}
    sizes = {k: len(v) for k, v in sets.items()}
    print(sizes)
    d = df[have & (df.split.isin(["val", "test"]) | (df.split == "train"))].copy()   # pilot rows only (train rows exist only for pilot series)
    rows = []
    for t in PREDICTION_TIMES_S:
        dt = d[d.prediction_time_s == t]
        y = {s: dt[dt.split == s][LABEL_COL].astype(int).to_numpy() for s in ("train", "val", "test")}
        gte = dt[dt.split == "test"].match_id.to_numpy()
        pt, thr_ = {}, {}
        for name, cols in sets.items():
            X = {s: dt[dt.split == s][cols].astype(float) for s in y}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, pva, pte, bi = fit_predict("lightgbm", X["train"], y["train"], X["val"], y["val"], X["test"])
            thr, vf1 = tune_threshold(y["val"], pva)
            pred = (pte >= thr).astype(int); yt = y["test"]
            pt[name], thr_[name] = pte, thr
            rows.append({"model": name, "t_s": t, "n_features": len(cols), "n_train": len(y["train"]), "n_val": len(y["val"]),
                         "n_test": len(yt), "coverage_test": len(yt) / n_test_total, "val_threshold": thr, "val_macro_f1": vf1,
                         "test_macro_f1": f1_score(yt, pred, average="macro", labels=[0, 1], zero_division=0),
                         "test_ct_f1": f1_score(yt, pred, pos_label=1, zero_division=0),
                         "test_t_f1": f1_score(yt, pred, pos_label=0, zero_division=0),
                         "test_auc": roc_auc_score(yt, pte), "best_iter": bi})
        ref = "A_meta_pilot"
        for name in sets:
            r = [x for x in rows if x["t_s"] == t and x["model"] == name][0]
            if name == ref:
                r["delta_vs_A"], r["delta_lo"], r["delta_hi"] = 0.0, 0.0, 0.0
            else:
                lo, hi = boot_delta(y["test"], pt[name], thr_[name], pt[ref], thr_[ref], gte)
                r["delta_vs_A"] = r["test_macro_f1"] - [x for x in rows if x["t_s"] == t and x["model"] == ref][0]["test_macro_f1"]
                r["delta_lo"], r["delta_hi"] = lo, hi
        print(t, {x["model"][:1]: round(x["test_macro_f1"], 3) for x in rows if x["t_s"] == t}, flush=True)
    res = pd.DataFrame(rows)
    s2 = pd.read_csv(ROOT / "results" / "step2" / "results.csv")
    s2 = s2[s2.model == "lightgbm"].set_index("t_s")
    res["step2_full_train_macro_f1"] = res.t_s.map(s2.test_macro_f1)
    res["delta_vs_step2_full"] = res.test_macro_f1 - res.step2_full_train_macro_f1
    res.to_csv(RES / "results.csv", index=False)

    piv = res.pivot(index="t_s", columns="model", values="test_macro_f1")
    piv.insert(0, "Step2_full_train", s2.test_macro_f1)
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    cols = {"Step2_full_train": "#8a8f98", "A_meta_pilot": "#5b6470", "B_meta+player": "#1f6feb", "C_meta+spatial": "#d97706",
            "D_meta+player+spatial+move": "#c0392b"}
    for k, c in cols.items():
        a.plot(piv.index, piv[k], marker="o", ms=4, color=c, ls="--" if k.startswith("Step2") else "-", label=k)
    a.set_ylabel("Test macro F1"); a.set_title("Step 4 pilot: LightGBM F1 vs prediction time"); a.grid(alpha=.25); a.legend(fontsize=8, loc="lower right")
    for k in ["B_meta+player", "C_meta+spatial", "D_meta+player+spatial+move"]:
        r = res[res.model == k]
        b.errorbar(r.t_s, r.delta_vs_A, yerr=[r.delta_vs_A - r.delta_lo, r.delta_hi - r.delta_vs_A], marker="o", ms=4, capsize=2, color=cols[k], label=k, alpha=.9)
    b.axhline(0, color="k", lw=.8); b.set_ylabel("ΔF1 vs A (same pilot train)\n95% series-bootstrap CI"); b.set_xlabel("prediction time (s after freeze end)")
    b.grid(alpha=.25); b.legend(fontsize=8); fig.tight_layout(); fig.savefig(ROOT / "plots" / "step4_f1_vs_time.png", dpi=150)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(res[["model", "t_s", "n_train", "n_test", "coverage_test", "val_threshold", "test_macro_f1", "test_ct_f1", "test_t_f1", "delta_vs_A", "delta_lo", "delta_hi", "delta_vs_step2_full"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
