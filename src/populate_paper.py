"""Populate paper/main.md and FINAL_RESEARCH_REPORT.md with empirical results and figure links."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB_DIR = ROOT / "tables"
RES_DIR = ROOT / "results"
PAPER_DIR = ROOT / "paper"

def build_condensed_model_table():
    """Create a concise 12-model summary table across key cutoffs: 10s, 30s, 50s, 60s, 75s."""
    mdf = pd.read_csv(RES_DIR / "model_comparison.csv")
    key_times = [10, 30, 50, 60, 75]
    sub = mdf[mdf["t_s"].isin(key_times)].copy()
    
    # Pivot to Model vs Time Macro F1
    piv = sub.pivot(index="model", columns="t_s", values="test_macro_f1").round(4)
    piv.columns = [f"t={c}s" for c in piv.columns]
    piv = piv.sort_values("t=60s", ascending=False).reset_index()
    piv["model"] = piv["model"].str.replace("_", " ").str.title()
    piv.rename(columns={"model": "Model Architecture"}, inplace=True)
    return piv.to_markdown(index=False)

def format_temporal_table():
    """Format PyTorch temporal sequence models table."""
    tdf = pd.read_csv(RES_DIR / "temporal_models.csv")
    cols = ["model", "t_s", "seq_len", "test_macro_f1", "test_accuracy", "train_time_s"]
    sub = tdf[cols].copy().round(4)
    sub.columns = ["Sequence Model", "Time (s)", "Seq Length", "Test Macro F1", "Accuracy", "Training Time (s)"]
    sub["Sequence Model"] = sub["Sequence Model"].str.upper()
    return sub.to_markdown(index=False)

def main():
    print("Populating paper/main.md with empirical tables and figures...")
    paper_path = PAPER_DIR / "main.md"
    paper_text = paper_path.read_text(encoding="utf-8")

    # 1. Prepare tables
    condensed_model_tbl = build_condensed_model_table()
    feature_ablation_tbl = (TAB_DIR / "feature_ablation_table.md").read_text(encoding="utf-8")
    temporal_tbl = format_temporal_table()
    early_pred_tbl = (TAB_DIR / "early_prediction_table.md").read_text(encoding="utf-8")
    stats_tbl = (TAB_DIR / "statistical_comparison_table.md").read_text(encoding="utf-8")
    map_tbl = (TAB_DIR / "map_wise_table.md").read_text(encoding="utf-8")

    # Limit stats_tbl to top 25 rows for paper readability
    stats_lines = stats_tbl.splitlines()
    if len(stats_lines) > 30:
        stats_condensed = "\n".join(stats_lines[:28]) + f"\n\n*(Full 118-row pairwise test table available in [tables/statistical_comparison_table.md](../tables/statistical_comparison_table.md))*"
    else:
        stats_condensed = stats_tbl

    # Replace placeholders
    paper_text = paper_text.replace(
        "<!-- INCLUDE_TABLE: tables/model_comparison_table.md -->",
        f"{condensed_model_tbl}\n\n*(Full 144-row evaluation matrix across all 12 cutoffs available in [tables/model_comparison_table.md](../tables/model_comparison_table.md))*\n\n"
        f"![Figure 1: Macro F1 vs Time for All Models](../plots/pub_01_macro_f1_vs_time_all_models.png)\n\n"
        f"![Figure 15: Model Performance Heatmap](../plots/pub_15_model_vs_time_heatmap.png)"
    )

    paper_text = paper_text.replace(
        "<!-- INCLUDE_TABLE: tables/feature_ablation_table.md -->",
        f"{feature_ablation_tbl}\n\n"
        f"![Figure 9: Spatial Feature Family Ablation](../plots/pub_09_spatial_ablation.png)\n\n"
        f"![Figure 18: Continuous Spatial STOM](../plots/pub_18_spatial_occupancy_stom.png)\n\n"
        f"![Figure 19: Temporal Spatial Progression](../plots/pub_19_temporal_spatial_progression.png)"
    )

    paper_text = paper_text.replace(
        "<!-- INCLUDE_TABLE: results/temporal_models.csv -->",
        f"{temporal_tbl}\n\n"
        f"![Figure 10: Feature Importance at 60s](../plots/pub_10_feature_importance.png)\n\n"
        f"![Figure 11: Feature Group Information Horizon](../plots/pub_11_shap_summary.png)"
    )

    paper_text = paper_text.replace(
        "<!-- INCLUDE_TABLE: tables/early_prediction_table.md -->",
        f"{early_pred_tbl}\n\n"
        f"![Figure 6: Matched Cohort vs Natural Coverage](../plots/pub_06_matched_cohort_vs_natural.png)\n\n"
        f"![Figure 7: Model Comparison at 60s](../plots/pub_07_model_comparison_60s.png)\n\n"
        f"![Figure 8: Model Comparison at 75s](../plots/pub_08_model_comparison_75s.png)"
    )

    paper_text = paper_text.replace(
        "<!-- INCLUDE_TABLE: tables/statistical_comparison_table.md -->",
        f"{stats_condensed}\n\n"
        f"![Figure 16: 95% Series Bootstrap Confidence Intervals](../plots/pub_16_confidence_intervals.png)\n\n"
        f"![Figure 12: Normalized Confusion Matrices](../plots/pub_12_confusion_matrices.png)\n\n"
        f"![Figure 17: Pareto Efficiency Frontier](../plots/pub_17_training_time_vs_f1)"
    )

    paper_text = paper_text.replace(
        "<!-- INCLUDE_TABLE: tables/map_wise_table.md -->",
        f"{map_tbl}\n\n"
        f"![Figure 13: Map-Stratified Macro F1 at 60s](../plots/pub_13_map_wise_f1.png)\n\n"
        f"![Figure 14: Map vs Prediction Cutoff Heatmap](../plots/pub_14_map_vs_time_heatmap.png)\n\n"
        f"![Figure 20: Ranking Feature Ablation](../plots/pub_20_ranking_ablation.png)"
    )

    paper_path.write_text(paper_text, encoding="utf-8")
    print("paper/main.md populated successfully.")

if __name__ == "__main__":
    main()
