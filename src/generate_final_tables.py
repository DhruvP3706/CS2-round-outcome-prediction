"""Generate all research tables required in Section 14 as CSV and Markdown documents."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"
TAB_DIR = ROOT / "tables"
TAB_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Generating comprehensive research tables...")
    
    # 1. Model comparison table
    mcmp_p = RES_DIR / "model_comparison.csv"
    if mcmp_p.exists():
        mdf = pd.read_csv(mcmp_p)
        cols = ["model", "t_s", "test_macro_f1", "test_ct_f1", "test_t_f1", "test_accuracy", "test_precision", "test_recall", "test_auc", "coverage_test"]
        sub = mdf[cols].copy()
        sub.columns = ["Model", "Time (s)", "Macro F1", "CT F1", "T F1", "Accuracy", "Precision", "Recall", "ROC-AUC", "Coverage"]
        sub = sub.sort_values(["Time (s)", "Macro F1"], ascending=[True, False]).round(4)
        sub.to_csv(TAB_DIR / "model_comparison_table.csv", index=False)
        (TAB_DIR / "model_comparison_table.md").write_text(sub.to_markdown(index=False))
        print("Model comparison table written.")

    # 2. Feature ablation table
    sp_p = RES_DIR / "spatial_ablation.csv"
    if sp_p.exists():
        sp_df = pd.read_csv(sp_p)
        base = sp_df[sp_df["feature_set"] == "Baseline_97"].set_index("t_s")["test_macro_f1"]
        rows = []
        for _, r in sp_df.iterrows():
            t = r["t_s"]
            b_val = base.get(t, float("nan"))
            delta = r["test_macro_f1"] - b_val
            ci_str = f"[{r.get('test_macro_f1_ci_lo', 0):.3f}, {r.get('test_macro_f1_ci_hi', 0):.3f}]"
            rows.append({
                "Feature Configuration": r["feature_set"],
                "Time (s)": t,
                "Macro F1": round(r["test_macro_f1"], 4),
                "Diff from Baseline": round(delta, 4),
                "95% Confidence Interval": ci_str
            })
        ab_df = pd.DataFrame(rows)
        ab_df.to_csv(TAB_DIR / "feature_ablation_table.csv", index=False)
        (TAB_DIR / "feature_ablation_table.md").write_text(ab_df.to_markdown(index=False))
        print("Feature ablation table written.")

    # 3. Map-wise table
    map_p = RES_DIR / "map_wise_results.csv"
    if map_p.exists():
        map_df = pd.read_csv(map_p)
        sub = map_df[["map_name", "n_test", "t_s", "test_macro_f1", "test_ct_f1", "test_t_f1", "test_accuracy"]].copy()
        sub.columns = ["Map", "Test Samples", "Time (s)", "Macro F1", "CT F1", "T F1", "Accuracy"]
        sub = sub.round(4)
        sub.to_csv(TAB_DIR / "map_wise_table.csv", index=False)
        (TAB_DIR / "map_wise_table.md").write_text(sub.to_markdown(index=False))
        print("Map-wise table written.")

    # 4. Statistical comparison table
    stat_p = RES_DIR / "statistical_tests.csv"
    if stat_p.exists():
        st_df = pd.read_csv(stat_p)
        # Apply Bonferroni / Benjamini-Hochberg correction
        n_tests = len(st_df)
        st_df["corrected_p_value"] = np.minimum(st_df["p_value"] * n_tests, 1.0)
        st_df["significant_corrected"] = st_df["corrected_p_value"] < 0.05
        cols = ["baseline", "proposed", "t_s", "delta_accuracy", "p_value", "corrected_p_value", "significant_corrected"]
        sub = st_df[cols].copy().round(4)
        sub.columns = ["Baseline", "Proposed Model", "Time (s)", "Paired Delta Accuracy", "p-value", "Corrected p-value", "Significant?"]
        sub.to_csv(TAB_DIR / "statistical_comparison_table.csv", index=False)
        (TAB_DIR / "statistical_comparison_table.md").write_text(sub.to_markdown(index=False))
        print("Statistical comparison table written.")

    # 5. Early prediction table
    if mcmp_p.exists():
        mdf = pd.read_csv(mcmp_p)
        best_rows = []
        for t in sorted(mdf["t_s"].unique()):
            sub_t = mdf[mdf["t_s"] == t].sort_values("test_macro_f1", ascending=False).iloc[0]
            ci_str = f"[{sub_t.get('test_macro_f1_ci_lo', 0):.3f}, {sub_t.get('test_macro_f1_ci_hi', 0):.3f}]"
            best_rows.append({
                "Prediction Time (s)": int(t),
                "Best Model": sub_t["model"],
                "Macro F1": round(sub_t["test_macro_f1"], 4),
                "Test Coverage": f"{sub_t['coverage_test']*100:.1f}%",
                "95% Confidence Interval": ci_str,
                "Meets 0.80 Target?": "YES" if sub_t["test_macro_f1"] >= 0.80 else "NO"
            })
        early_df = pd.DataFrame(best_rows)
        early_df.to_csv(TAB_DIR / "early_prediction_table.csv", index=False)
        (TAB_DIR / "early_prediction_table.md").write_text(early_df.to_markdown(index=False))
        print("Early prediction table written.")

if __name__ == "__main__":
    main()
