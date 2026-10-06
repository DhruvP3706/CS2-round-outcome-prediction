"""Step 6: small controlled LightGBM tuning on the fixed 97-feature set (Step 2 metadata + Step 4 player state).

Stage 0  reproduce Step 4 baseline config (all 15 times) and compare with results/step4_full (model B)
Stage 1  search ~13 configs at the focus times 30/40/50/60/75 s using TRAIN + VALIDATION only (test is never loaded into a model call)
Stage 2  select the config with the highest mean validation macro F1 over the focus times (rule fixed in advance), freeze it
Stage 3  evaluate the frozen config (and the baseline) once on test for all 15 times; extra seeds at focus times
"""
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lightgbm as lgb  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, log_loss, roc_auc_score  # noqa: E402

from cs2pred.config import LABEL_COL, PREDICTION_TIMES_S, PROCESSED_DIR, ROUND_KEY  # noqa: E402
from cs2pred.splits import series_split  # noqa: E402
from cs2pred.modeling import tune_threshold  # noqa: E402

RES = ROOT / "results" / "step6"
FOCUS = [30, 40, 50, 60, 75]
BASE = dict(num_leaves=15, max_depth=-1, min_child_samples=40, learning_rate=0.03, feature_fraction=0.8,
            bagging_fraction=0.8, lambda_l1=0.0, lambda_l2=2.0)   # == cs2pred.modeling.make_model('lightgbm')
CONFIGS = {
    "c00_baseline": {},
    "c01_nl7_mcs60": dict(num_leaves=7, min_child_samples=60),
    "c02_nl4_md3_mcs80": dict(num_leaves=4, max_depth=3, min_child_samples=80),
    "c03_nl31_mcs20": dict(num_leaves=31, min_child_samples=20),
    "c04_nl15_md4_mcs100": dict(max_depth=4, min_child_samples=100),
    "c05_lr0.01": dict(learning_rate=0.01),
    "c06_ff0.5": dict(feature_fraction=0.5),
    "c07_ff0.3_bf0.7": dict(feature_fraction=0.3, bagging_fraction=0.7),
    "c08_l1_1_l2_10": dict(lambda_l1=1.0, lambda_l2=10.0),
    "c09_nl7_mcs80_l2_30": dict(num_leaves=7, min_child_samples=80, lambda_l2=30.0),
    "c10_nl7_md3_ff0.5_l2_10_lr0.02": dict(num_leaves=7, max_depth=3, feature_fraction=0.5, lambda_l2=10.0, learning_rate=0.02),
    "c11_nl4_ff0.5_l2_10_mcs60": dict(num_leaves=4, feature_fraction=0.5, lambda_l2=10.0, min_child_samples=60),
    "c12_nl11_bf0.6_ff0.6_l1_0.5_l2_5_mcs50": dict(num_leaves=11, bagging_fraction=0.6, feature_fraction=0.6, lambda_l1=0.5, lambda_l2=5.0, min_child_samples=50),
}


def mf1(y, pred):
    f = []
    for c in (0, 1):
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c)); fn = np.sum((pred != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return (f[0] + f[1]) / 2


def make(cfg, seed):
    p = {**BASE, **cfg}
    return lgb.LGBMClassifier(objective="binary", n_estimators=2000 if not cfg else 5000, learning_rate=p["learning_rate"],
                              num_leaves=p["num_leaves"], max_depth=p["max_depth"], min_child_samples=p["min_child_samples"],
                              subsample=p["bagging_fraction"], subsample_freq=1, colsample_bytree=p["feature_fraction"],
                              reg_alpha=p["lambda_l1"], reg_lambda=p["lambda_l2"], random_state=seed, n_jobs=8, verbose=-1)


def fit(cfg, seed, Xtr, ytr, Xva, yva):
    m = make(cfg, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss", callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def load():
    table = pd.read_parquet(PROCESSED_DIR / "round_table.parquet")
    split = series_split(table)
    meta = pd.read_parquet(ROOT / "data/features/step2_metadata_features.parquet")
    cols = (ROOT / "results/step2/feature_list.txt").read_text().split("\n")
    tk = pd.read_parquet(ROOT / "data/features/step4_full/step4_tick_features.parquet")
    ps = [c for c in tk.columns if c.startswith("ps_")]
    df = meta.merge(tk[ROUND_KEY + ["prediction_time_s"] + ps], on=ROUND_KEY + ["prediction_time_s"], how="inner")
    df["split"] = df.match_id.map(split)
    feats = cols + ps
    assert len(feats) == 97
    return df, feats, int((table.match_id.map(split) == "test").sum())


def part(df, t, feats, which):
    d = df[(df.prediction_time_s == t) & (df.split == which)]
    return d[feats].astype(float), d[LABEL_COL].astype(int).to_numpy(), d


def main():
    RES.mkdir(parents=True, exist_ok=True); (ROOT / "plots").mkdir(exist_ok=True)
    df, feats, n_test_total = load()
    s4 = pd.read_csv(ROOT / "results/step4_full/results.csv")
    s4 = s4[s4.model.str.startswith("B_")].set_index("t_s")

    # ---- Stage 1: search (train + validation only)
    rows = []
    for name, cfg in CONFIGS.items():
        for t in FOCUS:
            Xtr, ytr, _ = part(df, t, feats, "train"); Xva, yva, _ = part(df, t, feats, "val")
            m = fit(cfg, 42, Xtr, ytr, Xva, yva)
            pva = m.predict_proba(Xva)[:, 1]
            thr, vf1 = tune_threshold(yva, pva)
            rows.append({"config": name, "t_s": t, "val_macro_f1": vf1, "val_logloss": log_loss(yva, pva), "val_auc": roc_auc_score(yva, pva),
                         "val_threshold": thr, "best_iter": int(m.best_iteration_ or m.n_estimators), **{k: {**BASE, **cfg}[k] for k in BASE}})
        print(name, round(np.mean([r["val_macro_f1"] for r in rows if r["config"] == name]), 4), flush=True)
    tr = pd.DataFrame(rows); tr.to_csv(RES / "tuning_results.csv", index=False)
    summ = tr.groupby("config").agg(mean_val_macro_f1=("val_macro_f1", "mean"), mean_val_logloss=("val_logloss", "mean"),
                                    mean_val_auc=("val_auc", "mean")).sort_values("mean_val_macro_f1", ascending=False)
    summ.to_csv(RES / "tuning_summary.csv")
    # ---- Stage 2: selection by mean validation macro F1 over focus times (no test involved)
    sel = summ.index[0]
    cfg = {**BASE, **CONFIGS[sel]}
    json.dump({"selected": sel, "params": cfg, "mean_val_macro_f1": float(summ.iloc[0].mean_val_macro_f1),
               "baseline_mean_val_macro_f1": float(summ.loc["c00_baseline"].mean_val_macro_f1), "rule": "max mean val macro F1 over 30/40/50/60/75s, seed 42",
               "n_estimators_cap": 2000 if not CONFIGS[sel] else 5000, "early_stopping_rounds": 100}, open(RES / "selected_config.json", "w"), indent=2)
    print("selected", sel, summ.iloc[0].to_dict(), flush=True)

    # ---- Stage 0/3: all 15 times, baseline and selected, seed 42; test evaluated once per (config, time, seed)
    def run(name, seed, times):
        out = []
        for t in times:
            Xtr, ytr, _ = part(df, t, feats, "train"); Xva, yva, _ = part(df, t, feats, "val"); Xte, yte, dte = part(df, t, feats, "test")
            m = fit(CONFIGS[name], seed, Xtr, ytr, Xva, yva)
            pva, pte = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
            thr, vf1 = tune_threshold(yva, pva)
            pred = (pte >= thr).astype(int)
            out.append({"config": name, "seed": seed, "t_s": t, "n_train": len(ytr), "n_val": len(yva), "n_test": len(yte),
                        "coverage_test": len(yte) / n_test_total, "val_threshold": thr, "val_macro_f1": vf1,
                        "test_macro_f1": mf1(yte, pred), "test_ct_f1": f1_score(yte, pred, pos_label=1, zero_division=0),
                        "test_t_f1": f1_score(yte, pred, pos_label=0, zero_division=0), "test_auc": roc_auc_score(yte, pte),
                        "best_iter": int(m.best_iteration_ or m.n_estimators), "_pte": pte, "_yte": yte, "_g": dte.match_id.to_numpy()})
        return out
    base42, sel42 = run("c00_baseline", 42, PREDICTION_TIMES_S), run(sel, 42, PREDICTION_TIMES_S)
    extra = []
    for sd in (1, 2):
        extra += run("c00_baseline", sd, FOCUS) + (run(sel, sd, FOCUS) if sel != "c00_baseline" else [])
    # baseline reproduction check vs Step 4 model B
    rep = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in base42])
    rep["step4_B_test_macro_f1"] = rep.t_s.map(s4.test_macro_f1); rep["step4_B_val_threshold"] = rep.t_s.map(s4.val_threshold)
    rep["diff"] = rep.test_macro_f1 - rep.step4_B_test_macro_f1
    rep.to_csv(RES / "baseline_reproduction.csv", index=False)
    print("baseline reproduction max |diff|:", rep["diff"].abs().max().round(4), "mean diff", rep["diff"].mean().round(5), flush=True)
    # paired bootstrap of selected - baseline (seed 42)
    rng = np.random.default_rng(0)
    final = []
    for b, s in zip(base42, sel42):
        ug = np.unique(b["_g"]); idx = {u: np.flatnonzero(b["_g"] == u) for u in ug}
        d = []
        for _ in range(500):
            kk = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
            d.append(mf1(s["_yte"][kk], (s["_pte"][kk] >= s["val_threshold"]).astype(int)) - mf1(b["_yte"][kk], (b["_pte"][kk] >= b["val_threshold"]).astype(int)))
        r = {k: v for k, v in s.items() if not k.startswith("_")}
        r.update(baseline_test_macro_f1=b["test_macro_f1"], delta_vs_baseline=s["test_macro_f1"] - b["test_macro_f1"],
                 delta_lo=float(np.percentile(d, 2.5)), delta_hi=float(np.percentile(d, 97.5)),
                 step4_B_test_macro_f1=float(s4.test_macro_f1[s["t_s"]]), step4_best_test_macro_f1=float(pd.read_csv(ROOT / "results/step4_full/results.csv").query("t_s==@s['t_s']").test_macro_f1.max()))
        final.append(r)
    fin = pd.DataFrame(final); fin.to_csv(RES / "final_results.csv", index=False)
    ex = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in extra + base42 + sel42])
    ex = ex[ex.t_s.isin(FOCUS)]; ex.to_csv(RES / "seed_robustness.csv", index=False)
    # plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(fin.t_s, fin.step4_B_test_macro_f1, "o--", color="#8a8f98", label="Step 4 B (97 feats)")
    ax.plot(rep.t_s, rep.test_macro_f1, "s:", color="#5b6470", label="baseline config (reproduced)")
    ax.plot(fin.t_s, fin.test_macro_f1, "o-", color="#1f6feb", label=f"selected: {sel}")
    ax.axhline(0.8, ls=":", color="#c0392b"); ax.grid(alpha=.25); ax.legend(); ax.set_xlabel("prediction time (s)"); ax.set_ylabel("Test macro F1")
    ax.set_title("Step 6: tuned vs baseline LightGBM (97 features)"); fig.tight_layout(); fig.savefig(ROOT / "plots" / "step6_f1_vs_time.png", dpi=150)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(summ.round(4).to_string()); print(fin.drop(columns=["config", "seed"]).round(3).to_string(index=False))
    print(ex.groupby(["config", "t_s"]).test_macro_f1.agg(["mean", "min", "max"]).round(3).unstack(0).to_string())


if __name__ == "__main__":
    main()
