"""Step 5b: cumulative feature-family ablation (A..H) with the fixed Step 2/4 LightGBM configuration."""
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
from cs2pred.modeling import fit_predict, make_model, tune_threshold  # noqa: E402
from cs2pred.splits import series_split  # noqa: E402
from cs2pred.step5_features import FAMILIES, family_columns  # noqa: E402
import lightgbm as lgb  # noqa: E402

RES, FEAT = ROOT / "results" / "step5", ROOT / "data" / "features"
ORDER = list("ABCDEFGH")
FAM_OF = dict(zip(ORDER[1:], FAMILIES))          # B+cm, C+pa, D+fw, E+rs, F+bb, G+pr, H+sc
FAM_NAME = {"cm": "combat momentum", "pa": "player advantage", "fw": "weapon/firepower", "rs": "round state",
            "bb": "bomb/time", "pr": "pre-round x live", "sc": "state changes"}
FOCUS = [20, 30, 40, 50, 60, 75]


def mf1(y, pred):
    f = []
    for c in (0, 1):
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c)); fn = np.sum((pred != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return (f[0] + f[1]) / 2


def main():
    RES.mkdir(parents=True, exist_ok=True); (ROOT / "plots").mkdir(exist_ok=True)
    table = pd.read_parquet(PROCESSED_DIR / "round_table.parquet")
    split = series_split(table)
    n_test_total = int((table.match_id.map(split) == "test").sum())
    df = pd.read_parquet(FEAT / "step5" / "step5_features.parquet")
    meta_cols = (ROOT / "results/step2/feature_list.txt").read_text().split("\n")
    tk = pd.read_parquet(FEAT / "step4_full" / "step4_tick_features.parquet")
    ps = [c for c in tk.columns if c.startswith("ps_")]
    df = df.merge(tk[ROUND_KEY + ["prediction_time_s"] + ps], on=ROUND_KEY + ["prediction_time_s"], how="inner")
    fam = family_columns(df)
    sets, cur = {}, meta_cols + ps
    sets["A"] = list(cur)
    for k in ORDER[1:]:
        cur = cur + fam[FAM_OF[k]]; sets[k] = list(cur)
    df["split"] = df.match_id.map(split)
    print({k: len(v) for k, v in sets.items()}, flush=True)
    rows, preds, boot_store, seedrows = [], {}, [], []
    rng = np.random.default_rng(0)
    for t in PREDICTION_TIMES_S:
        dt = df[df.prediction_time_s == t]
        part = {s: dt[dt.split == s] for s in ("train", "val", "test")}
        y = {s: part[s][LABEL_COL].astype(int).to_numpy() for s in part}
        pte_all, thr_all = {}, {}
        for k in ORDER:
            X = {s: part[s][sets[k]].astype(float) for s in part}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, pva, pte, bi = fit_predict("lightgbm", X["train"], y["train"], X["val"], y["val"], X["test"])
            thr, vf1 = tune_threshold(y["val"], pva)
            pred = (pte >= thr).astype(int); yt = y["test"]
            pte_all[k], thr_all[k] = pte, thr
            rows.append({"set": k, "t_s": t, "n_features": len(sets[k]), "n_train": len(y["train"]), "n_val": len(y["val"]),
                         "n_test": len(yt), "coverage_test": len(yt) / n_test_total, "val_threshold": thr, "val_macro_f1": vf1,
                         "test_macro_f1": mf1(yt, pred), "test_ct_f1": f1_score(yt, pred, pos_label=1, zero_division=0),
                         "test_t_f1": f1_score(yt, pred, pos_label=0, zero_division=0), "test_auc": roc_auc_score(yt, pte), "best_iter": bi})
        # paired series-cluster bootstrap over all sets at once
        g = part["test"].match_id.to_numpy(); ug = np.unique(g); idx = {u: np.flatnonzero(g == u) for u in ug}
        B = np.empty((400, len(ORDER)))
        for b in range(400):
            kk = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
            for j, k in enumerate(ORDER):
                B[b, j] = mf1(y["test"][kk], (pte_all[k][kk] >= thr_all[k]).astype(int))
        for j, k in enumerate(ORDER[1:], 1):
            for refname, rj in (("A", 0), ("prev", j - 1)):
                d = B[:, j] - B[:, rj]
                r = [x for x in rows if x["t_s"] == t and x["set"] == k][0]
                r[f"delta_vs_{refname}"] = r["test_macro_f1"] - [x for x in rows if x["t_s"] == t and x["set"] == ORDER[rj]][0]["test_macro_f1"]
                r[f"delta_vs_{refname}_lo"], r[f"delta_vs_{refname}_hi"] = np.percentile(d, 2.5), np.percentile(d, 97.5)
        # seed-noise floor (model config otherwise fixed) for A and H at focus times
        if t in FOCUS:
            for k in ("A", "H"):
                for sd in (1, 2):
                    m = make_model("lightgbm").set_params(random_state=sd)
                    X = {s: part[s][sets[k]].astype(float) for s in part}
                    m.fit(X["train"], y["train"], eval_set=[(X["val"], y["val"])], eval_metric="binary_logloss",
                          callbacks=[lgb.early_stopping(100, verbose=False)])
                    thr, _ = tune_threshold(y["val"], m.predict_proba(X["val"])[:, 1])
                    seedrows.append({"set": k, "t_s": t, "seed": sd, "test_macro_f1": mf1(y["test"], (m.predict_proba(X["test"])[:, 1] >= thr).astype(int))})
        print(t, {k: round([x for x in rows if x["t_s"] == t and x["set"] == k][0]["test_macro_f1"], 3) for k in ORDER}, flush=True)
    res = pd.DataFrame(rows)
    s4 = pd.read_csv(ROOT / "results/step4_full/results.csv")
    best4 = s4.groupby("t_s").test_macro_f1.max()
    s4b = s4[s4.model.str.startswith("B_")].set_index("t_s").test_macro_f1
    res["step4_best_macro_f1"] = res.t_s.map(best4)
    res["step4_B_macro_f1"] = res.t_s.map(s4b)
    res["delta_vs_step4_best"] = res.test_macro_f1 - res.step4_best_macro_f1
    res.to_csv(RES / "results.csv", index=False)
    # ablation: increments per family, restricted view
    ab = res[res.set != "A"].copy()
    ab["family_added"] = ab.set.map(lambda k: FAM_OF[k]); ab["family_name"] = ab.family_added.map(FAM_NAME)
    ab[["set", "family_added", "family_name", "t_s", "n_features", "test_macro_f1", "delta_vs_A", "delta_vs_A_lo", "delta_vs_A_hi",
        "delta_vs_prev", "delta_vs_prev_lo", "delta_vs_prev_hi"]].to_csv(RES / "ablation.csv", index=False)
    pd.DataFrame(seedrows).to_csv(RES / "seed_noise.csv", index=False)
    # plot
    piv = res.pivot(index="t_s", columns="set", values="test_macro_f1")
    dv = res[res.set != "A"].pivot(index="set", columns="t_s", values="delta_vs_A")
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [2.4, 2]})
    cm = plt.get_cmap("viridis")
    a.plot(best4.index, best4.values, "k--", lw=1.5, label="Step 4 best (full)")
    for i, k in enumerate(ORDER):
        a.plot(piv.index, piv[k], marker="o", ms=3, color=cm(i / 7), label=f"{k} ({len(sets[k])})", lw=1.3)
    a.axhline(0.80, ls=":", color="#c0392b"); a.set_ylabel("Test macro F1"); a.grid(alpha=.25); a.legend(fontsize=7, ncol=3, loc="lower right")
    a.set_title("Step 5 cumulative ablation: LightGBM test macro F1 vs prediction time"); a.set_xlabel("prediction time (s)")
    im = b.imshow(dv.values, aspect="auto", cmap="RdBu", vmin=-0.04, vmax=0.04)
    b.set_xticks(range(len(dv.columns))); b.set_xticklabels(dv.columns); b.set_yticks(range(len(dv.index)))
    b.set_yticklabels([f"{k} +{FAM_OF[k]}" for k in dv.index])
    for i in range(dv.shape[0]):
        for j in range(dv.shape[1]):
            b.text(j, i, f"{dv.values[i, j] * 100:+.1f}", ha="center", va="center", fontsize=6)
    b.set_title("ΔF1 vs A (x100, cumulative)"); b.set_xlabel("prediction time (s)"); fig.colorbar(im, ax=b, fraction=.02)
    fig.tight_layout(); fig.savefig(ROOT / "plots" / "step5_f1_vs_time.png", dpi=150)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(res[res.t_s.isin(FOCUS)][["set", "t_s", "n_features", "val_threshold", "test_macro_f1", "test_ct_f1", "test_t_f1", "delta_vs_A", "delta_vs_A_lo", "delta_vs_A_hi", "delta_vs_prev", "delta_vs_prev_lo", "delta_vs_prev_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
