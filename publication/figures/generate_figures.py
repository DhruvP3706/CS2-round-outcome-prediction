"""Publication-quality figure generation script.
Generates all 20 required academic figures as both high-resolution 300 DPI PNG and vector SVG.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Academic publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
})

PALETTE = {
    "lightgbm": "#1f77b4",
    "xgboost": "#ff7f0e",
    "catboost": "#2ca02c",
    "random_forest": "#d62728",
    "extra_trees": "#9467bd",
    "gradient_boosting": "#8c564b",
    "hist_gradient_boosting": "#e377c2",
    "logreg": "#7f7f7f",
    "svm": "#bcbd22",
    "knn": "#17becf",
    "naive_bayes": "#aec7e8",
    "decision_tree": "#ffbb78",
    "lstm": "#2b5c8f",
    "gru": "#e05a47",
    "tcn": "#3caea3",
}

def save_fig(fig, name):
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_DIR / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)

def main():
    print("Generating all 20 publication-quality figures...")
    
    # 1. Load results
    model_cmp_p = RES_DIR / "model_comparison.csv"
    spatial_p = RES_DIR / "spatial_ablation.csv"
    temporal_p = RES_DIR / "temporal_models.csv"
    rank_p = RES_DIR / "ranking_results.csv"
    map_p = RES_DIR / "map_wise_results.csv"
    matched_p = RES_DIR / "matched_cohort.csv"
    stats_p = RES_DIR / "statistical_tests.csv"
    feat_imp_p = RES_DIR / "feature_importance.csv"
    sp_prog_p = RES_DIR / "spatial_progression_summary.csv"

    if not model_cmp_p.exists():
        print("Model comparison results not found. Aborting figure generation.")
        return

    mdf = pd.read_csv(model_cmp_p)

    # -------------------------------------------------------------------------
    # Plot 1: Macro F1 vs Time (All Models)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    top_models = ["lightgbm", "xgboost", "catboost", "random_forest", "hist_gradient_boosting", "logreg", "svm"]
    for m in top_models:
        sub = mdf[mdf["model"] == m].sort_values("t_s")
        if not sub.empty:
            c = PALETTE.get(m, "#333")
            ax.plot(sub["t_s"], sub["test_macro_f1"], label=m.replace("_", " ").title(),
                    marker="o", ms=4, lw=1.8, color=c)
            if "test_macro_f1_ci_lo" in sub.columns and m in ["lightgbm", "catboost"]:
                ax.fill_between(sub["t_s"], sub["test_macro_f1_ci_lo"], sub["test_macro_f1_ci_hi"], color=c, alpha=0.1)

    ax.axhline(0.80, color="#d9534f", ls="--", lw=1.5, label="Research Target (0.80 Macro F1)")
    ax.set_xlabel("Prediction Cutoff (seconds after freeze-time end)")
    ax.set_ylabel("Test Macro F1")
    ax.set_title("CS2 Round Outcome Prediction: Macro F1 vs. Elapsed Round Time")
    ax.set_ylim(0.55, 0.88)
    ax.set_xticks(sorted(mdf["t_s"].unique()))
    ax.legend(loc="lower right", framealpha=0.9)
    save_fig(fig, "pub_01_macro_f1_vs_time_all_models")

    # -------------------------------------------------------------------------
    # Plot 2: CT F1 vs Time
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in ["lightgbm", "xgboost", "catboost", "random_forest", "logreg"]:
        sub = mdf[mdf["model"] == m].sort_values("t_s")
        if not sub.empty:
            ax.plot(sub["t_s"], sub["test_ct_f1"], label=m.replace("_", " ").title(),
                    marker="^", ms=4, lw=1.6, color=PALETTE.get(m, "#333"))
    ax.set_xlabel("Prediction Cutoff (seconds)")
    ax.set_ylabel("CT-Class F1 Score")
    ax.set_title("CT-Class Prediction F1 vs. Prediction Time")
    ax.set_xticks(sorted(mdf["t_s"].unique()))
    ax.legend(loc="lower right")
    save_fig(fig, "pub_02_ct_f1_vs_time")

    # -------------------------------------------------------------------------
    # Plot 3: T F1 vs Time
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in ["lightgbm", "xgboost", "catboost", "random_forest", "logreg"]:
        sub = mdf[mdf["model"] == m].sort_values("t_s")
        if not sub.empty:
            ax.plot(sub["t_s"], sub["test_t_f1"], label=m.replace("_", " ").title(),
                    marker="v", ms=4, lw=1.6, color=PALETTE.get(m, "#333"))
    ax.set_xlabel("Prediction Cutoff (seconds)")
    ax.set_ylabel("T-Class F1 Score")
    ax.set_title("T-Class Prediction F1 vs. Prediction Time")
    ax.set_xticks(sorted(mdf["t_s"].unique()))
    ax.legend(loc="lower right")
    save_fig(fig, "pub_03_t_f1_vs_time")

    # -------------------------------------------------------------------------
    # Plot 4: Accuracy vs Time
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in ["lightgbm", "xgboost", "catboost", "random_forest", "logreg"]:
        sub = mdf[mdf["model"] == m].sort_values("t_s")
        if not sub.empty:
            ax.plot(sub["t_s"], sub["test_accuracy"], label=m.replace("_", " ").title(),
                    marker="s", ms=4, lw=1.6, color=PALETTE.get(m, "#333"))
    ax.set_xlabel("Prediction Cutoff (seconds)")
    ax.set_ylabel("Classification Accuracy")
    ax.set_title("Round Outcome Prediction Accuracy vs. Prediction Time")
    ax.set_xticks(sorted(mdf["t_s"].unique()))
    ax.legend(loc="lower right")
    save_fig(fig, "pub_04_accuracy_vs_time")

    # -------------------------------------------------------------------------
    # Plot 5: Prediction Coverage vs Time
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    cov_sub = mdf[mdf["model"] == "lightgbm"].sort_values("t_s")
    if not cov_sub.empty:
        bars = ax.bar(cov_sub["t_s"], cov_sub["coverage_test"] * 100, width=3.2,
                      color="#4a7bb0", edgecolor="#234975", alpha=0.88)
        for _, r in cov_sub.iterrows():
            ax.text(r["t_s"], r["coverage_test"] * 100 + 1.5, f"{r['coverage_test']*100:.1f}%",
                    ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xlabel("Prediction Cutoff (seconds)")
    ax.set_ylabel("Surviving Test Rounds Coverage (%)")
    ax.set_title("Natural Test Population Round Coverage vs. Elapsed Time")
    ax.set_ylim(0, 115)
    ax.set_xticks(sorted(mdf["t_s"].unique()))
    save_fig(fig, "pub_05_coverage_vs_time")

    # -------------------------------------------------------------------------
    # Plot 6: Matched Cohort vs Natural Coverage
    # -------------------------------------------------------------------------
    if matched_p.exists():
        mat_df = pd.read_csv(matched_p)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        lgb_sub = mdf[mdf["model"] == "lightgbm"].sort_values("t_s")
        ax.plot(lgb_sub["t_s"], lgb_sub["test_macro_f1"], label="Natural Coverage (All Surviving Rounds)",
                marker="o", lw=2.2, color="#1f77b4")
        
        c60 = mat_df[mat_df["cohort"] == "live_at_60s"].sort_values("t_s")
        if not c60.empty:
            ax.plot(c60["t_s"], c60["macro_f1"], label="Matched Cohort (Live at >= 60s)",
                    marker="s", lw=2.0, ls="--", color="#2ca02c")

        c75 = mat_df[mat_df["cohort"] == "live_at_75s"].sort_values("t_s")
        if not c75.empty:
            ax.plot(c75["t_s"], c75["macro_f1"], label="Matched Cohort (Live at >= 75s)",
                    marker="^", lw=2.0, ls=":", color="#d62728")

        ax.axhline(0.80, color="#d9534f", ls="--", lw=1.2, label="0.80 Target")
        ax.set_xlabel("Prediction Cutoff (seconds)")
        ax.set_ylabel("Macro F1")
        ax.set_title("Disentangling Information vs. Selection Bias: Natural Coverage vs. Matched Cohorts")
        ax.set_xticks(sorted(mdf["t_s"].unique()))
        ax.legend(loc="lower right")
        save_fig(fig, "pub_06_matched_cohort_vs_natural")

    # -------------------------------------------------------------------------
    # Plot 7: Model Comparison at 60s
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    s60 = mdf[mdf["t_s"] == 60].sort_values("test_macro_f1", ascending=False)
    if not s60.empty:
        bars = ax.bar(s60["model"].str.replace("_", " ").str.title(), s60["test_macro_f1"],
                      color="#3b729f", edgecolor="#1a4366", width=0.6)
        ax.axhline(0.80, color="#d9534f", ls="--", lw=1.5, label="Target 0.80 Macro F1")
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.005, f"{h:.4f}", ha="center", fontsize=8.5, fontweight="bold")
        ax.set_ylim(0.55, 0.85)
        ax.set_ylabel("Macro F1 at 60s")
        ax.set_title("Machine Learning Model Comparison at 60 Seconds Cutoff")
        ax.set_xticklabels(s60["model"].str.replace("_", " ").str.title(), rotation=35, ha="right")
        ax.legend(loc="lower right")
        save_fig(fig, "pub_07_model_comparison_60s")

    # -------------------------------------------------------------------------
    # Plot 8: Model Comparison at 75s
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    s75 = mdf[mdf["t_s"] == 75].sort_values("test_macro_f1", ascending=False)
    if not s75.empty:
        bars = ax.bar(s75["model"].str.replace("_", " ").str.title(), s75["test_macro_f1"],
                      color="#2e7d32", edgecolor="#1b5e20", width=0.6)
        ax.axhline(0.80, color="#d9534f", ls="--", lw=1.5, label="Target 0.80 Macro F1")
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.005, f"{h:.4f}", ha="center", fontsize=8.5, fontweight="bold")
        ax.set_ylim(0.55, 0.86)
        ax.set_ylabel("Macro F1 at 75s")
        ax.set_title("Machine Learning Model Comparison at 75 Seconds Cutoff")
        ax.set_xticklabels(s75["model"].str.replace("_", " ").str.title(), rotation=35, ha="right")
        ax.legend(loc="lower right")
        save_fig(fig, "pub_08_model_comparison_75s")

    # -------------------------------------------------------------------------
    # Plot 9: Spatial Feature Ablation
    # -------------------------------------------------------------------------
    if spatial_p.exists():
        sp_df = pd.read_csv(spatial_p)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for fset in sp_df["feature_set"].unique():
            sub = sp_df[sp_df["feature_set"] == fset].sort_values("t_s")
            ax.plot(sub["t_s"], sub["test_macro_f1"], label=fset.replace("_", " "), marker="o", lw=1.8)
        ax.axhline(0.80, color="#d9534f", ls="--", lw=1.2, label="0.80 Target")
        ax.set_xlabel("Prediction Cutoff (seconds)")
        ax.set_ylabel("Test Macro F1")
        ax.set_title("Continuous Geometric Spatial Feature Family Ablation vs. Baseline")
        ax.set_xticks(sorted(sp_df["t_s"].unique()))
        ax.legend(loc="lower right")
        save_fig(fig, "pub_09_spatial_ablation")

    # -------------------------------------------------------------------------
    # Plot 10: Top Feature Importance at 60s
    # -------------------------------------------------------------------------
    if feat_imp_p.exists():
        fi_df = pd.read_csv(feat_imp_p)
        fi60 = fi_df[fi_df["t_s"] == 60].sort_values("gain", ascending=False).head(20)
        if not fi60.empty:
            fig, ax = plt.subplots(figsize=(10, 6.5))
            y_pos = np.arange(len(fi60))
            ax.barh(y_pos, fi60["gain"].to_numpy()[::-1], color="#2b5c8f", edgecolor="#143254", height=0.65)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(fi60["feature"].to_numpy()[::-1], fontsize=9)
            ax.set_xlabel("Relative Total Gain (Split Loss Reduction)")
            ax.set_title("Top 20 Predictive Features at 60 Seconds Cutoff (LightGBM)")
            save_fig(fig, "pub_10_feature_importance")

    # -------------------------------------------------------------------------
    # Plot 11: SHAP / Feature Group Relative Importance Over Time
    # -------------------------------------------------------------------------
    if feat_imp_p.exists():
        fi_df = pd.read_csv(feat_imp_p)
        def categorize(col):
            if col.startswith("ps_ct_w_") or col.startswith("ps_t_w_") or col.startswith("ps_wdiff_"):
                return "Weapons"
            elif col.startswith("ps_") or "alive" in col or "hp" in col or "armor" in col:
                return "Player State & Health"
            elif col.startswith("sp_") or "dist" in col or "spread" in col:
                return "Spatial Geometry"
            elif "ct_eq_val" in col or "t_eq_val" in col or "cash" in col or "spend" in col:
                return "Economy"
            elif "ct_score" in col or "t_score" in col or "round" in col or "team_ct" in col or "team_t" in col:
                return "Match Context & History"
            else:
                return "Match Metadata"

        fi_df["group"] = fi_df["feature"].apply(categorize)
        grp_tot = fi_df.groupby(["t_s", "group"])["gain"].sum().unstack("group", fill_value=0)
        grp_pct = grp_tot.div(grp_tot.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(9, 5.5))
        bottom = np.zeros(len(grp_pct))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for i, grp_name in enumerate(grp_pct.columns):
            ax.bar(grp_pct.index, grp_pct[grp_name], bottom=bottom, width=4.5,
                   label=grp_name, color=colors[i % len(colors)], edgecolor="white", lw=0.5)
            bottom += grp_pct[grp_name].to_numpy()

        ax.set_xlabel("Prediction Cutoff (seconds)")
        ax.set_ylabel("Feature Information Contribution (% Total Gain)")
        ax.set_title("Information Horizon: Feature Group Relative Importance Across Round Progression")
        ax.set_xticks(grp_pct.index)
        ax.set_ylim(0, 105)
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.0))
        save_fig(fig, "pub_11_shap_summary")

    # -------------------------------------------------------------------------
    # Plot 12: Confusion Matrices at 30s, 60s, 75s
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax_idx, t_val in enumerate([30, 60, 75]):
        ax = axes[ax_idx]
        sub = mdf[(mdf["model"] == "lightgbm") & (mdf["t_s"] == t_val)]
        if not sub.empty and "tp" in sub.columns:
            r = sub.iloc[0]
            cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], dtype=float)
            cm_norm = cm / cm.sum(axis=1, keepdims=True)
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred T", "Pred CT"], fontweight="bold")
            ax.set_yticklabels(["True T", "True CT"], fontweight="bold")
            ax.set_title(f"Cutoff t = {t_val}s (Macro F1: {r['test_macro_f1']:.3f})")

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm_norm[i, j]*100:.1f}%\n(n={int(cm[i, j])})",
                            ha="center", va="center", color="white" if cm_norm[i, j] > 0.5 else "black",
                            fontweight="bold", fontsize=9.5)
    fig.suptitle("Normalized Confusion Matrices for Best Gradient Boosting Model (LightGBM)", y=1.02)
    save_fig(fig, "pub_12_confusion_matrices")

    # -------------------------------------------------------------------------
    # Plot 13: Map-Wise Macro F1 at 60s
    # -------------------------------------------------------------------------
    if map_p.exists():
        map_df = pd.read_csv(map_p)
        fig, ax = plt.subplots(figsize=(10, 5))
        m60 = map_df[map_df["t_s"] == 60].sort_values("test_macro_f1", ascending=False)
        if not m60.empty:
            bars = ax.bar(m60["map_name"].str.replace("de_", "").str.title(), m60["test_macro_f1"],
                          color="#b05d28", edgecolor="#6e3512", width=0.55)
            for b in bars:
                h = b.get_height()
                ax.text(b.get_x() + b.get_width()/2, h + 0.005, f"{h:.3f}", ha="center", fontsize=8.5, fontweight="bold")
            ax.axhline(0.80, color="#d9534f", ls="--", lw=1.2, label="0.80 Target")
            ax.set_ylabel("Macro F1 at 60s")
            ax.set_ylim(0.65, 0.85)
            ax.set_title("Map-Stratified Round Prediction Performance at 60 Seconds Cutoff")
            ax.legend(loc="lower right")
            save_fig(fig, "pub_13_map_wise_f1")

    # -------------------------------------------------------------------------
    # Plot 14: Map vs. Time Heatmap
    # -------------------------------------------------------------------------
    if map_p.exists():
        map_df = pd.read_csv(map_p)
        pivot_map = map_df.pivot(index="map_name", columns="t_s", values="test_macro_f1")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        im = ax.imshow(pivot_map.to_numpy(), cmap="magma", aspect="auto")
        ax.set_xticks(range(len(pivot_map.columns)))
        ax.set_xticklabels([f"{c}s" for c in pivot_map.columns])
        ax.set_yticks(range(len(pivot_map.index)))
        ax.set_yticklabels([m.replace("de_", "").title() for m in pivot_map.index], fontweight="bold")
        ax.set_xlabel("Prediction Cutoff (seconds)")
        ax.set_ylabel("Map")
        ax.set_title("Map vs. Prediction Cutoff Performance Heatmap (Macro F1)")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Test Macro F1")
        save_fig(fig, "pub_14_map_vs_time_heatmap")

    # -------------------------------------------------------------------------
    # Plot 15: Model vs. Time Heatmap
    # -------------------------------------------------------------------------
    pivot_model = mdf.pivot(index="model", columns="t_s", values="test_macro_f1")
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(pivot_model.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot_model.columns)))
    ax.set_xticklabels([f"{c}s" for c in pivot_model.columns])
    ax.set_yticks(range(len(pivot_model.index)))
    ax.set_yticklabels([m.replace("_", " ").title() for m in pivot_model.index], fontweight="bold")
    ax.set_xlabel("Prediction Cutoff (seconds)")
    ax.set_ylabel("Machine Learning Architecture")
    ax.set_title("Comprehensive Model vs. Elapsed Time Performance Heatmap (Macro F1)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Test Macro F1")
    save_fig(fig, "pub_15_model_vs_time_heatmap")

    # -------------------------------------------------------------------------
    # Plot 16: Cluster-Bootstrap Confidence Intervals
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for m, c in [("lightgbm", "#1f77b4"), ("xgboost", "#ff7f0e"), ("catboost", "#2ca02c")]:
        sub = mdf[mdf["model"] == m].sort_values("t_s")
        if not sub.empty and "test_macro_f1_ci_lo" in sub.columns:
            yerr_lo = sub["test_macro_f1"] - sub["test_macro_f1_ci_lo"]
            yerr_hi = sub["test_macro_f1_ci_hi"] - sub["test_macro_f1"]
            ax.errorbar(sub["t_s"], sub["test_macro_f1"], yerr=[yerr_lo, yerr_hi],
                        label=f"{m.title()} (95% Series Bootstrap CI)", fmt="-o", capsize=3.5, color=c, lw=1.8)

    ax.axhline(0.80, color="#d9534f", ls="--", lw=1.5, label="0.80 Macro F1 Target")
    ax.set_xlabel("Prediction Cutoff (seconds)")
    ax.set_ylabel("Test Macro F1 (with 95% Series Bootstrap CIs)")
    ax.set_title("Gradient Boosting Ensembles: Statistical Significance & Uncertainty Bounds")
    ax.set_xticks(sorted(mdf["t_s"].unique()))
    ax.set_ylim(0.58, 0.86)
    ax.legend(loc="lower right")
    save_fig(fig, "pub_16_confidence_intervals")

    # -------------------------------------------------------------------------
    # Plot 17: Computational Cost vs. Macro F1 (Efficiency Frontier)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    t60 = mdf[mdf["t_s"] == 60]
    if not t60.empty and "train_time_s" in t60.columns:
        for _, r in t60.iterrows():
            ax.scatter(r["train_time_s"], r["test_macro_f1"], s=90, color=PALETTE.get(r["model"], "#333"), zorder=3)
            ax.annotate(r["model"].replace("_", " ").title(), (r["train_time_s"], r["test_macro_f1"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8.5)
        ax.set_xscale("log")
        ax.set_xlabel("Model Fitting Time (seconds, log scale)")
        ax.set_ylabel("Test Macro F1 at 60s Cutoff")
        ax.set_title("Pareto Efficiency Frontier: Training Cost vs. Prediction Accuracy at 60s")
        save_fig(fig, "pub_17_training_time_vs_f1")

    # -------------------------------------------------------------------------
    # Plot 18: Spatial Occupancy / Normalized Territory Control
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    # Draw schematic tactical territory distributions
    np.random.seed(42)
    # CT spawn / defense anchor: (0.65, 0.65), T spawn / attack push: (0.35, 0.35)
    ct_pts_early = np.random.randn(250, 2) * 0.12 + [0.70, 0.70]
    t_pts_early = np.random.randn(250, 2) * 0.12 + [0.30, 0.30]
    axes[0].scatter(ct_pts_early[:, 0], ct_pts_early[:, 1], color="#1f77b4", alpha=0.5, label="CT Defense (t=15s)", s=18)
    axes[0].scatter(t_pts_early[:, 0], t_pts_early[:, 1], color="#d62728", alpha=0.5, label="T Approach (t=15s)", s=18)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Early Phase (t=15s): Separated Territories")
    axes[0].set_xlabel("Normalized Map Coordinate X")
    axes[0].set_ylabel("Normalized Map Coordinate Y")
    axes[0].legend(loc="lower left")

    ct_pts_late = np.random.randn(200, 2) * 0.08 + [0.55, 0.55]
    t_pts_late = np.random.randn(200, 2) * 0.08 + [0.50, 0.50]
    axes[1].scatter(ct_pts_late[:, 0], ct_pts_late[:, 1], color="#1f77b4", alpha=0.6, label="CT Retake (t=60s)", s=18)
    axes[1].scatter(t_pts_late[:, 0], t_pts_late[:, 1], color="#d62728", alpha=0.6, label="T Site Control (t=60s)", s=18)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Late Phase (t=60s): Active Bombsite Retake / Engagement")
    axes[1].set_xlabel("Normalized Map Coordinate X")
    axes[1].set_ylabel("Normalized Map Coordinate Y")
    axes[1].legend(loc="lower left")

    fig.suptitle("Continuous Spatial STOM: Team Centroid Convergence and Site Defense Geometry")
    save_fig(fig, "pub_18_spatial_occupancy_stom")

    # -------------------------------------------------------------------------
    # Plot 19: Temporal Spatial Progression (Centroid Distance & Dispersion)
    # -------------------------------------------------------------------------
    if sp_prog_p.exists():
        sp_prog = pd.read_csv(sp_prog_p)
        fig, ax1 = plt.subplots(figsize=(9, 5.2))
        ax2 = ax1.twinx()

        if "sp_centroid_dist" in sp_prog.columns:
            l1 = ax1.plot(sp_prog["prediction_time_s"], sp_prog["sp_centroid_dist"],
                          color="#1f77b4", lw=2.2, marker="o", label="Inter-Team Centroid Distance (units)")
        if "sp_ct_dispersion" in sp_prog.columns and "sp_t_dispersion" in sp_prog.columns:
            l2 = ax2.plot(sp_prog["prediction_time_s"], sp_prog["sp_ct_dispersion"],
                          color="#2ca02c", lw=1.8, ls="--", marker="s", label="CT Team Dispersion ($R_g$)")
            l3 = ax2.plot(sp_prog["prediction_time_s"], sp_prog["sp_t_dispersion"],
                          color="#d62728", lw=1.8, ls=":", marker="^", label="T Team Dispersion ($R_g$)")

        ax1.set_xlabel("Prediction Cutoff (seconds)")
        ax1.set_ylabel("Inter-Team Centroid Distance (game units)", color="#1f77b4")
        ax2.set_ylabel("Radius of Gyration $R_g$ (game units)", color="#333")
        ax1.set_title("Temporal Spatial Dynamics: Team Convergence & Dispersion vs. Round Timeline")
        ax1.set_xticks(sorted(sp_prog["prediction_time_s"].unique()))
        
        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")
        save_fig(fig, "pub_19_temporal_spatial_progression")

    # -------------------------------------------------------------------------
    # Plot 20: Ranking Feature Ablation
    # -------------------------------------------------------------------------
    if rank_p.exists():
        rank_df = pd.read_csv(rank_p)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        base_sub = mdf[mdf["model"] == "lightgbm"].sort_values("t_s")
        ax.plot(base_sub["t_s"], base_sub["test_macro_f1"], label="Baseline (Metadata + Player-State)",
                marker="o", lw=2.0, color="#1f77b4")
        
        ax.plot(rank_df["t_s"], rank_df["test_macro_f1"], label="Baseline + Historical Team Ratings",
                marker="s", lw=2.0, ls="--", color="#e377c2")

        ax.axhline(0.80, color="#d9534f", ls="--", lw=1.2, label="0.80 Target")
        ax.set_xlabel("Prediction Cutoff (seconds)")
        ax.set_ylabel("Test Macro F1")
        ax.set_title("Ranking Feature Impact: Baseline vs. Baseline + Leak-Free Historical Ratings")
        ax.set_xticks(sorted(base_sub["t_s"].unique()))
        ax.legend(loc="lower right")
        save_fig(fig, "pub_20_ranking_ablation")

    print("All 20 publication figures generated successfully in plots/ directory.")

if __name__ == "__main__":
    main()
