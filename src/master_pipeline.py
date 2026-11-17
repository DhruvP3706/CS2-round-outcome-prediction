"""Master Research Pipeline: CS2 Early Round Outcome Prediction.

Orchestrates:
1. Feature extraction (Metadata 63 + Player-state 34 + Novel Continuous Spatial 16 + Ranking 8 = 121 features)
2. Stage 4: Alternative STOM & Spatial Feature Ablation
3. Stage 5: Classical ML Model Benchmark (12 algorithms)
4. Stage 6: Temporal Sequence Models (LSTM, GRU, TCN)
5. Stage 7: Ranking Features & Map-Wise Analysis
6. Stage 8: Explainability, Statistical Significance (Wilcoxon, McNemar), Bootstrap CIs,
   and Dual-Cohort (Natural Coverage vs. Matched Cohort)
7. Stage 9: Generate publication tables and plots
"""
from __future__ import annotations

import sys
import os
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cs2pred.config import (
    LABEL_COL,
    PREDICTION_TIMES_S,
    PROCESSED_DIR,
    ROUND_KEY,
    TICK_RATE,
)
from cs2pred.splits import series_split
from cs2pred.ranking import compute_historical_rankings
from cs2pred.novel_spatial import extract_continuous_spatial_features
from cs2pred.modeling_extended import (
    fit_predict,
    tune_threshold,
    evaluate_metrics,
    cluster_bootstrap_ci,
)
from run_temporal_models import train_temporal_model

SNAP_DIR = ROOT / "data" / "features" / "step4" / "snapshots"
RES_DIR = ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

CUTOFFS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75]

def load_canonical():
    """Load canonical round table, series split, and metadata baseline features."""
    table = pd.read_parquet(PROCESSED_DIR / "round_table.parquet")
    split = series_split(table)
    table["split"] = table["match_id"].map(split)
    
    meta = pd.read_parquet(ROOT / "data" / "features" / "step2_metadata_features.parquet")
    meta_cols = [c for c in meta.columns if c not in ROUND_KEY + ["prediction_time_s", "prediction_tick", LABEL_COL, "split"]]
    return table, split, meta, meta_cols

def build_snapshot_features(table, split):
    """Vectorized aggregation of snapshot features: 34 player-state features + 16 novel continuous spatial features."""
    out_dir = ROOT / "data" / "features" / "step4_full"
    ps_path = out_dir / "player_state_features.parquet"
    sp_path = out_dir / "novel_spatial_features.parquet"
    if ps_path.exists() and sp_path.exists():
        print(f"Loading cached snapshot features from {out_dir}...")
        return pd.read_parquet(ps_path), pd.read_parquet(sp_path)

    snap_files = sorted(SNAP_DIR.glob("*.parquet"))
    print(f"Loading {len(snap_files)} snapshot files from {SNAP_DIR}...")
    
    t0 = time.time()
    dfs = [pd.read_parquet(f) for f in snap_files]
    all_snaps = pd.concat(dfs, ignore_index=True)
    all_snaps = all_snaps[all_snaps["team"].isin([2, 3])].copy()
    all_snaps["is_ct"] = (all_snaps["team"] == 3)
    print(f"Loaded {len(all_snaps)} snapshot rows in {time.time() - t0:.1f}s.")

    # 1. Player-state aggregations (vectorized)
    KEY = ["match_id", "map_name", "round", "t_s"]
    g = all_snaps.groupby(KEY + ["is_ct"])
    
    all_snaps["alive_flag"] = (all_snaps["hp"] > 0).astype(int)
    agg_df = g.agg(
        alive=("alive_flag", "sum"),
        hp=("hp", "sum"),
        armor=("armor", "sum"),
    ).unstack("is_ct", fill_value=0)

    ps_df = pd.DataFrame(index=agg_df.index)
    ps_df["ps_ct_alive"] = agg_df["alive"][True] if True in agg_df["alive"] else 0
    ps_df["ps_t_alive"] = agg_df["alive"][False] if False in agg_df["alive"] else 0
    ps_df["ps_alive_diff"] = ps_df["ps_ct_alive"] - ps_df["ps_t_alive"]

    ps_df["ps_ct_hp"] = agg_df["hp"][True] if True in agg_df["hp"] else 0
    ps_df["ps_t_hp"] = agg_df["hp"][False] if False in agg_df["hp"] else 0
    ps_df["ps_hp_diff"] = ps_df["ps_ct_hp"] - ps_df["ps_t_hp"]

    ps_df["ps_ct_avg_hp"] = ps_df["ps_ct_hp"] / ps_df["ps_ct_alive"].replace(0, np.nan)
    ps_df["ps_t_avg_hp"] = ps_df["ps_t_hp"] / ps_df["ps_t_alive"].replace(0, np.nan)

    ps_df["ps_ct_armor"] = agg_df["armor"][True] if True in agg_df["armor"] else 0
    ps_df["ps_t_armor"] = agg_df["armor"][False] if False in agg_df["armor"] else 0
    ps_df["ps_armor_diff"] = ps_df["ps_ct_armor"] - ps_df["ps_t_armor"]

    ps_df["ps_ct_avg_armor"] = ps_df["ps_ct_armor"] / ps_df["ps_ct_alive"].replace(0, np.nan)
    ps_df["ps_t_avg_armor"] = ps_df["ps_t_armor"] / ps_df["ps_t_alive"].replace(0, np.nan)

    # Weapon classes
    wclasses = ["rifle", "sniper", "smg", "heavy", "pistol", "knife", "utility"]
    wcnt = all_snaps.groupby(KEY + ["is_ct", "wclass"]).size().unstack(["is_ct", "wclass"], fill_value=0)
    for is_ct, sd in [(True, "ct"), (False, "t")]:
        for code, name in enumerate(wclasses):
            col_key = (is_ct, code)
            ps_df[f"ps_{sd}_w_{name}"] = wcnt[col_key] if col_key in wcnt.columns else 0

    for name in wclasses:
        ps_df[f"ps_wdiff_{name}"] = ps_df[f"ps_ct_w_{name}"] - ps_df[f"ps_t_w_{name}"]

    full_ps = ps_df.reset_index().rename(columns={"t_s": "prediction_time_s"})

    # 2. Novel continuous spatial features
    bounds_df = pd.read_csv(ROOT / "results" / "step4_full" / "map_bounds.csv")
    full_sp = extract_continuous_spatial_features(all_snaps, bounds_df)

    out_dir = ROOT / "data" / "features" / "step4_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_ps.to_parquet(out_dir / "player_state_features.parquet", index=False)
    full_sp.to_parquet(out_dir / "novel_spatial_features.parquet", index=False)
    print("Snapshot feature generation complete.")
    return full_ps, full_sp

def main():
    print("=== Master Research Pipeline Started ===")
    t_start = time.time()
    
    # 1. Canonical data and baseline
    table, split, meta, meta_cols = load_canonical()
    print(f"Canonical round table: {table.shape}, meta features: {len(meta_cols)}")

    # 2. Rankings
    rank_df = compute_historical_rankings()
    rank_cols = [c for c in rank_df.columns if c not in ROUND_KEY]
    print(f"Historical ranking features computed: {len(rank_cols)}")

    # 3. Snapshot features (PS and Novel Spatial)
    full_ps, full_sp = build_snapshot_features(table, split)
    ps_cols = [c for c in full_ps.columns if c.startswith("ps_")]
    sp_cols = [c for c in full_sp.columns if c.startswith("sp_")]
    print(f"Features: Meta={len(meta_cols)}, PS={len(ps_cols)}, Novel Spatial={len(sp_cols)}, Ranking={len(rank_cols)}")

    # Assemble master table per prediction cutoff
    join_keys = ROUND_KEY + ["prediction_time_s"]
    master = meta.merge(full_ps, on=join_keys, how="inner")
    master = master.merge(full_sp, on=join_keys, how="left")
    master = master.merge(rank_df, on=ROUND_KEY, how="left")
    master["split"] = master["match_id"].map(split)
    
    for col in sp_cols:
        master[col] = master[col].fillna(0.0)
    for col in rank_cols:
        master[col] = master[col].fillna(0.50 if "winrate" in col else 0.0)
    master = master.fillna(0.0)

    # Save summary of spatial progression across cutoffs
    sp_prog = master.groupby("prediction_time_s")[sp_cols].mean().reset_index()
    sp_prog.to_csv(RES_DIR / "spatial_progression_summary.csv", index=False)

    # -------------------------------------------------------------------------
    # STAGE 4: Alternative STOM & Spatial Feature Ablation
    # -------------------------------------------------------------------------
    print("\n--- STAGE 4: Novel Spatial Feature Ablation ---")
    feat_sets = {
        "Baseline_97": meta_cols + ps_cols,
        "Baseline_plus_Geometry": meta_cols + ps_cols + [c for c in sp_cols if "centroid" in c or "nearest" in c or "avg_pair" in c],
        "Baseline_plus_Dispersion": meta_cols + ps_cols + [c for c in sp_cols if "dispersion" in c or "internal_dist" in c],
        "Baseline_plus_Dynamics": meta_cols + ps_cols + [c for c in sp_cols if "shift" in c or "delta" in c],
        "Baseline_plus_AllSpatial": meta_cols + ps_cols + sp_cols,
    }

    sp_out_p = RES_DIR / "spatial_ablation.csv"
    if sp_out_p.exists() and len(pd.read_csv(sp_out_p)) == 60:
        print("Stage 4 already complete. Loaded results/spatial_ablation.csv")
    else:
        spatial_results = []
        for fset_name, fcols in feat_sets.items():
            print(f"Evaluating feature set: {fset_name} ({len(fcols)} features)...")
            for t in CUTOFFS:
                d = master[master["prediction_time_s"] == t]
                tr = d[d["split"] == "train"]
                va = d[d["split"] == "val"]
                te = d[d["split"] == "test"]

                Xtr, ytr = tr[fcols].astype(float), tr[LABEL_COL].astype(int).to_numpy()
                Xva, yva = va[fcols].astype(float), va[LABEL_COL].astype(int).to_numpy()
                Xte, yte = te[fcols].astype(float), te[LABEL_COL].astype(int).to_numpy()

                m, pva, pte, tr_time = fit_predict("lightgbm", Xtr, ytr, Xva, yva, Xte, seed=42)
                thr, val_mf1 = tune_threshold(yva, pva)
                met = evaluate_metrics(yte, pte, thr=thr)
                ci_lo, ci_hi = cluster_bootstrap_ci(yte, pte, thr, te["match_id"].to_numpy())

                spatial_results.append({
                    "feature_set": fset_name,
                    "n_features": len(fcols),
                    "t_s": t,
                    "val_threshold": thr,
                    "val_macro_f1": val_mf1,
                    "test_macro_f1": met["macro_f1"],
                    "test_macro_f1_ci_lo": ci_lo,
                    "test_macro_f1_ci_hi": ci_hi,
                    "test_ct_f1": met["ct_f1"],
                    "test_t_f1": met["t_f1"],
                    "test_accuracy": met["accuracy"],
                    "test_auc": met["roc_auc"],
                    "n_test": len(te),
                    "coverage_test": len(te) / int((table["split"] == "test").sum()),
                    "train_time_s": tr_time,
                })
        spatial_df = pd.DataFrame(spatial_results)
        spatial_df.to_csv(sp_out_p, index=False)
        print("Stage 4 complete. Saved results/spatial_ablation.csv")

    # -------------------------------------------------------------------------
    # STAGE 5: Comprehensive Classical ML Model Comparison
    # -------------------------------------------------------------------------
    print("\n--- STAGE 5: Classical ML Model Comparison ---")
    models_to_test = [
        "logreg", "decision_tree", "random_forest", "extra_trees",
        "gradient_boosting", "hist_gradient_boosting", "xgboost",
        "lightgbm", "catboost", "svm", "naive_bayes", "knn"
    ]
    base_feats = meta_cols + ps_cols
    model_cmp_results = []
    test_predictions = {}
    feat_imp_rows = []
    
    for mod_name in models_to_test:
        print(f"Training and evaluating model: {mod_name} across all 12 cutoffs...")
        for t in CUTOFFS:
            d = master[master["prediction_time_s"] == t]
            tr = d[d["split"] == "train"]
            va = d[d["split"] == "val"]
            te = d[d["split"] == "test"]

            Xtr, ytr = tr[base_feats].astype(float), tr[LABEL_COL].astype(int).to_numpy()
            Xva, yva = va[base_feats].astype(float), va[LABEL_COL].astype(int).to_numpy()
            Xte, yte = te[base_feats].astype(float), te[LABEL_COL].astype(int).to_numpy()

            try:
                m, pva, pte, tr_time = fit_predict(mod_name, Xtr, ytr, Xva, yva, Xte, seed=42)
                thr, val_mf1 = tune_threshold(yva, pva)
                met = evaluate_metrics(yte, pte, thr=thr)
                ci_lo, ci_hi = cluster_bootstrap_ci(yte, pte, thr, te["match_id"].to_numpy())
                test_predictions[(mod_name, t)] = pte

                model_cmp_results.append({
                    "model": mod_name,
                    "t_s": t,
                    "val_threshold": thr,
                    "val_macro_f1": val_mf1,
                    "test_macro_f1": met["macro_f1"],
                    "test_macro_f1_ci_lo": ci_lo,
                    "test_macro_f1_ci_hi": ci_hi,
                    "test_ct_f1": met["ct_f1"],
                    "test_t_f1": met["t_f1"],
                    "test_accuracy": met["accuracy"],
                    "test_precision": met["precision"],
                    "test_recall": met["recall"],
                    "test_auc": met["roc_auc"],
                    "test_prauc": met["pr_auc"],
                    "tp": met["tp"],
                    "fp": met["fp"],
                    "tn": met["tn"],
                    "fn": met["fn"],
                    "n_test": len(te),
                    "coverage_test": len(te) / int((table["split"] == "test").sum()),
                    "train_time_s": tr_time,
                })
                
                # Save feature importance for tree models at key cutoffs
                if mod_name == "lightgbm" and t in [10, 30, 45, 60, 75]:
                    imp = m.booster_.feature_importance(importance_type="gain")
                    for fn, iv in zip(base_feats, imp):
                        feat_imp_rows.append({"model": "lightgbm", "t_s": t, "feature": fn, "gain": float(iv)})

            except Exception as e:
                print(f"Error evaluating {mod_name} at t={t}: {e}")

    model_df = pd.DataFrame(model_cmp_results)
    model_df.to_csv(RES_DIR / "model_comparison.csv", index=False)
    if feat_imp_rows:
        pd.DataFrame(feat_imp_rows).to_csv(RES_DIR / "feature_importance.csv", index=False)
    print("Stage 5 complete. Saved results/model_comparison.csv and results/feature_importance.csv")

    # -------------------------------------------------------------------------
    # STAGE 6: Temporal Sequence Models (LSTM, GRU, TCN)
    # -------------------------------------------------------------------------
    print("\n--- STAGE 6: Temporal Sequence Models ---")
    temporal_times = [10, 20, 30, 40, 50, 60, 75]
    temporal_models = ["lstm", "gru", "tcn"]
    temporal_results = []

    for t in temporal_times:
        active_steps = [s for s in CUTOFFS if s <= t]
        d_t = master[master["prediction_time_s"] == t]
        eligible_keys = set(zip(d_t["match_id"], d_t["map_name"], d_t["round"]))

        sub = master[master["prediction_time_s"].isin(active_steps)].copy()
        sub["is_eligible"] = [k in eligible_keys for k in zip(sub["match_id"], sub["map_name"], sub["round"])]
        sub = sub[sub["is_eligible"]].sort_values(ROUND_KEY + ["prediction_time_s"]).reset_index(drop=True)

        seq_feature_cols = base_feats
        scaler = StandardScaler()
        tr_mask = (sub["split"] == "train")
        scaler.fit(sub.loc[tr_mask, seq_feature_cols].fillna(0.0))
        scaled_mat = scaler.transform(sub[seq_feature_cols].fillna(0.0))
        scaled_mat = np.nan_to_num(scaled_mat, nan=0.0, posinf=0.0, neginf=0.0)
        
        n_steps = len(active_steps)
        sub["round_idx"] = sub.groupby(ROUND_KEY).ngroup()
        n_rounds = sub["round_idx"].nunique()

        feat_dim = len(seq_feature_cols)
        tensor_data = np.zeros((n_rounds, n_steps, feat_dim), dtype=np.float32)
        
        for step_idx, step_val in enumerate(active_steps):
            step_rows = sub[sub["prediction_time_s"] == step_val]
            r_idx = step_rows["round_idx"].to_numpy()
            s_idx = step_rows.index.to_numpy()
            tensor_data[r_idx, step_idx, :] = scaled_mat[s_idx]

        round_meta = sub.groupby("round_idx").first()
        y_all = round_meta[LABEL_COL].astype(int).to_numpy()
        split_all = round_meta["split"].to_numpy()

        tr_idx = np.flatnonzero(split_all == "train")
        va_idx = np.flatnonzero(split_all == "val")
        te_idx = np.flatnonzero(split_all == "test")

        Xtr_seq, ytr_seq = tensor_data[tr_idx], y_all[tr_idx]
        Xva_seq, yva_seq = tensor_data[va_idx], y_all[va_idx]
        Xte_seq, yte_seq = tensor_data[te_idx], y_all[te_idx]

        for tmod in temporal_models:
            print(f"Training {tmod.upper()} at cutoff {t}s (seq_len={n_steps})...")
            try:
                met, pte = train_temporal_model(tmod, Xtr_seq, ytr_seq, Xva_seq, yva_seq, Xte_seq, yte_seq, seed=42)
                ci_lo, ci_hi = cluster_bootstrap_ci(yte_seq, pte, met["val_threshold"], round_meta.iloc[te_idx]["match_id"].to_numpy())
                test_predictions[(tmod, t)] = pte

                temporal_results.append({
                    "model": tmod,
                    "t_s": t,
                    "seq_len": n_steps,
                    "val_threshold": met["val_threshold"],
                    "val_macro_f1": met["val_macro_f1"],
                    "test_macro_f1": met["macro_f1"],
                    "test_macro_f1_ci_lo": ci_lo,
                    "test_macro_f1_ci_hi": ci_hi,
                    "test_ct_f1": met["ct_f1"],
                    "test_t_f1": met["t_f1"],
                    "test_accuracy": met["accuracy"],
                    "test_auc": met["roc_auc"],
                    "train_time_s": met["train_time_s"],
                    "n_test": len(te_idx),
                })
            except Exception as e:
                print(f"Error training {tmod} at t={t}: {e}")

    temporal_df = pd.DataFrame(temporal_results)
    temporal_df.to_csv(RES_DIR / "temporal_models.csv", index=False)
    print("Stage 6 complete. Saved results/temporal_models.csv")

    # -------------------------------------------------------------------------
    # STAGE 7: Ranking Features & Map-Wise Analysis
    # -------------------------------------------------------------------------
    print("\n--- STAGE 7: Ranking Features & Map-Wise Analysis ---")
    rank_results = []
    base_plus_rank = base_feats + rank_cols
    for t in CUTOFFS:
        d = master[master["prediction_time_s"] == t]
        tr = d[d["split"] == "train"]
        va = d[d["split"] == "val"]
        te = d[d["split"] == "test"]

        Xtr, ytr = tr[base_plus_rank].astype(float), tr[LABEL_COL].astype(int).to_numpy()
        Xva, yva = va[base_plus_rank].astype(float), va[LABEL_COL].astype(int).to_numpy()
        Xte, yte = te[base_plus_rank].astype(float), te[LABEL_COL].astype(int).to_numpy()

        m, pva, pte, tr_time = fit_predict("lightgbm", Xtr, ytr, Xva, yva, Xte, seed=42)
        thr, val_mf1 = tune_threshold(yva, pva)
        met = evaluate_metrics(yte, pte, thr=thr)
        ci_lo, ci_hi = cluster_bootstrap_ci(yte, pte, thr, te["match_id"].to_numpy())

        rank_results.append({
            "model": "lightgbm_plus_ranking",
            "t_s": t,
            "n_features": len(base_plus_rank),
            "val_threshold": thr,
            "val_macro_f1": val_mf1,
            "test_macro_f1": met["macro_f1"],
            "test_macro_f1_ci_lo": ci_lo,
            "test_macro_f1_ci_hi": ci_hi,
            "test_ct_f1": met["ct_f1"],
            "test_t_f1": met["t_f1"],
            "test_accuracy": met["accuracy"],
            "test_auc": met["roc_auc"],
            "train_time_s": tr_time,
        })
    rank_df_res = pd.DataFrame(rank_results)
    rank_df_res.to_csv(RES_DIR / "ranking_results.csv", index=False)

    # Map-wise analysis
    maps = sorted(master["map_name"].unique())
    map_results = []
    for mp in maps:
        for t in [30, 40, 50, 60, 75]:
            d = master[(master["prediction_time_s"] == t) & (master["map_name"] == mp)]
            tr = d[d["split"] == "train"]
            va = d[d["split"] == "val"]
            te = d[d["split"] == "test"]
            if len(te) < 10 or len(tr) < 30:
                continue

            Xtr, ytr = tr[base_feats].astype(float), tr[LABEL_COL].astype(int).to_numpy()
            Xva, yva = va[base_feats].astype(float), va[LABEL_COL].astype(int).to_numpy()
            Xte, yte = te[base_feats].astype(float), te[LABEL_COL].astype(int).to_numpy()

            m, pva, pte, _ = fit_predict("lightgbm", Xtr, ytr, Xva, yva, Xte, seed=42)
            thr, val_mf1 = tune_threshold(yva, pva)
            met = evaluate_metrics(yte, pte, thr=thr)

            map_results.append({
                "map_name": mp,
                "t_s": t,
                "n_train": len(tr),
                "n_test": len(te),
                "ct_win_rate_test": float(np.mean(yte)),
                "val_macro_f1": val_mf1,
                "test_macro_f1": met["macro_f1"],
                "test_ct_f1": met["ct_f1"],
                "test_t_f1": met["t_f1"],
                "test_accuracy": met["accuracy"],
            })
    map_df = pd.DataFrame(map_results)
    map_df.to_csv(RES_DIR / "map_wise_results.csv", index=False)
    print("Stage 7 complete. Saved ranking and map-wise results.")

    # -------------------------------------------------------------------------
    # STAGE 8: Statistical Significance & Dual Cohort
    # -------------------------------------------------------------------------
    print("\n--- STAGE 8: Statistical Tests & Dual-Cohort Evaluation ---")
    paired_rows = []
    for (mod_name, t), pte in test_predictions.items():
        if mod_name == "lightgbm":
            continue
        base_pte = test_predictions.get(("lightgbm", t))
        if base_pte is not None and len(base_pte) == len(pte):
            d = master[master["prediction_time_s"] == t]
            te = d[d["split"] == "test"]
            yte = te[LABEL_COL].astype(int).to_numpy()
            
            b_rows = model_df[(model_df["model"] == "lightgbm") & (model_df["t_s"] == t)]["val_threshold"].values
            base_thr = b_rows[0] if len(b_rows) > 0 else 0.50
            p_rows = model_df[(model_df["model"] == mod_name) & (model_df["t_s"] == t)]["val_threshold"].values
            prop_thr = p_rows[0] if len(p_rows) > 0 else 0.50

            base_pred = (base_pte >= base_thr).astype(int)
            prop_pred = (pte >= prop_thr).astype(int)

            base_correct = (base_pred == yte).astype(int)
            prop_correct = (prop_pred == yte).astype(int)

            delta_acc = float(np.mean(prop_correct) - np.mean(base_correct))
            
            conf_diff = np.abs(pte - 0.5) - np.abs(base_pte - 0.5)
            try:
                stat, p_val = wilcoxon(conf_diff)
            except Exception:
                stat, p_val = 0.0, 1.0

            paired_rows.append({
                "baseline": "lightgbm",
                "proposed": mod_name,
                "t_s": t,
                "delta_accuracy": delta_acc,
                "wilcoxon_stat": float(stat),
                "p_value": float(p_val),
                "significant_05": p_val < 0.05,
            })
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(RES_DIR / "statistical_tests.csv", index=False)

    # Matched-Cohort Analysis
    matched_rows = []
    for cohort_t in [60, 75]:
        d_cohort = master[master["prediction_time_s"] == cohort_t]
        cohort_rounds = set(zip(d_cohort["match_id"], d_cohort["map_name"], d_cohort["round"]))
        
        for t in [s for s in CUTOFFS if s <= cohort_t]:
            d = master[master["prediction_time_s"] == t]
            d_sub = d[d["split"] == "test"].copy()
            d_sub["in_cohort"] = [k in cohort_rounds for k in zip(d_sub["match_id"], d_sub["map_name"], d_sub["round"])]
            te_c = d_sub[d_sub["in_cohort"]]

            pte = test_predictions.get(("lightgbm", t))
            if pte is not None:
                c_mask = d_sub["in_cohort"].to_numpy()
                pte_c = pte[c_mask]
                yte_c = te_c[LABEL_COL].astype(int).to_numpy()
                thr_rows = model_df[(model_df["model"] == "lightgbm") & (model_df["t_s"] == t)]["val_threshold"].values
                thr = thr_rows[0] if len(thr_rows) > 0 else 0.50

                met = evaluate_metrics(yte_c, pte_c, thr=thr)
                matched_rows.append({
                    "cohort": f"live_at_{cohort_t}s",
                    "t_s": t,
                    "n_eval": len(te_c),
                    "macro_f1": met["macro_f1"],
                    "ct_f1": met["ct_f1"],
                    "t_f1": met["t_f1"],
                    "accuracy": met["accuracy"],
                })
    matched_df = pd.DataFrame(matched_rows)
    matched_df.to_csv(RES_DIR / "matched_cohort.csv", index=False)
    print("Stage 8 complete. Statistical tests and matched-cohort data saved.")

    # -------------------------------------------------------------------------
    # STAGE 9: Generate Figures & Tables
    # -------------------------------------------------------------------------
    print("\n--- STAGE 9: Generating Figures & Tables ---")
    import generate_publication_figures
    import generate_final_tables
    generate_publication_figures.main()
    generate_final_tables.main()
    print("Figures and tables generated successfully.")

    print(f"\nMaster pipeline complete in {(time.time() - t_start)/60:.1f} minutes.")

if __name__ == "__main__":
    main()
