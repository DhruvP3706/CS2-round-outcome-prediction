# Repository Audit: CS2 Round Outcome Prediction

**Document Type:** Publication-Quality System Audit  
**Author:** AI Research Specialist & Co-Author (Antigravity)  
**Date:** November 2026  
**Repository:** `CS2-round-outcome-prediction`  
**Current Baseline Macro F1:** ~0.795 at 75 s (LightGBM, 97 features, test coverage 62.6%)  
**Primary Research Goal:** Investigate whether Macro F1 > 0.80 can be achieved at 60 s or earlier (e.g., 50, 40, 30, 20, 10, 5 s) through novel spatial-temporal representations, ranking features, temporal sequence modeling, and comprehensive model comparisons, under strict leakage-free experimental validity.

---

## 1. Executive Summary & Audit Overview

This audit comprehensively inspects the entire codebase, directory structure, data processing pipelines, legacy implementations, and experiment logs in `CS2-round-outcome-prediction`. 

The repository contains two distinct generations of work:
1. **Legacy Exploratory Notebooks (`ML_STOMs` and `WP_DAC`):** Map-wise experiments using Spatio-Temporal Occupancy Maps (STOMs) derived exclusively from `weapon_fire` events (e.g., 10x10 spatial grids, epoch bins at 0, 10, 20, 30, 40, 50, 60 s, and round end). These were trained using Random Forest classifiers independently per map.
2. **Current Structured Pipeline (`src/`, `src/cs2pred/`, `results/step1` to `step7`, and `results/final`):** An end-to-end, reproducible modular pipeline using the OpenCS2 dataset (`blanchon/opencs2_dataset` on Hugging Face). The pipeline establishes:
   - A canonical round table of 16,527 rounds across 794 maps and 359 match series.
   - Leakage-safe metadata features (63 features: kill feed, bomb plant status, weapon types, opening duel timings, trade kills).
   - Player-state features (34 features: HP, armor, active weapon categories, alive player counts per team) -> 97 baseline features.
   - Controlled series-level train/validation/test splits (70% train: 251 series / 11,576 rounds; 15% val: 54 series / 2,448 rounds; 15% test: 54 series / 2,503 rounds; random seed 42).
   - Validation-only threshold selection (0.05 to 0.95 grid with 0.005 steps) to maximize validation Macro F1, followed by a single frozen test evaluation per cutoff.
   - Official final model: LightGBM (`c02_nl4_md3_mcs80`: 4 leaves, depth 3, min child samples 80, lr 0.03, colsample 0.8, subsample 0.8, L2 reg 2.0).
   - Step 4 (spatial grid occupancy), Step 5 (cumulative ablation across 7 feature families), and Step 7 (sub-second dynamics) were piloted and observed to provide no statistically significant test gain over the 97-feature baseline.

---

## 2. Repository Directory Structure & File Map

```
CS2-round-outcome-prediction/
├── .gitignore                                 # Ignores data/, venvs, caches
├── ML_STOMs-20260929T074442Z-1-001/           # Legacy: Machine Learning on STOMs
│   └── ML_STOMs/
│       ├── Ancient/                           # Notebooks for Ancient (RF on firing STOMs)
│       ├── Anubis/                            # Notebooks for Anubis
│       ├── Dust2/                             # Notebooks for Dust2
│       ├── Inferno/                           # Notebooks for Inferno
│       ├── Mirage/                            # Notebooks for Mirage
│       └── Vertigo/                           # Notebooks for Vertigo
├── WP_DAC-20260929T074452Z-1-001/             # Legacy: Win Prediction - Divide & Conquer
│   └── WP_DAC/                                # Map-wise feature extraction & prediction
│       ├── Ancient/, Anubis/, Dust2/, Inferno/, Mirage/, Vertigo/
├── logs/                                      # Execution logs from previous runs
│   ├── step4_compare.log                      # Pilot model comparisons (Step 4)
│   ├── step4_extract.log                      # Pilot snapshot extraction
│   ├── step4_full_compare.log                 # Full 794 map comparison
│   ├── step4_full_extract.log                 # Full snapshot extraction
│   ├── step5_ablation.log                     # Step 5 family ablation log
│   ├── step6.log                              # Step 6 LightGBM tuning log
│   ├── step7_extract.log                      # Step 7 dynamics extraction log
│   └── step7_pilot.log                        # Step 7 dynamics pilot evaluation
├── plots/                                     # Publication & diagnostic figures
│   ├── step2_f1_vs_time.png                   # Metadata baseline F1 curve
│   ├── step4_f1_vs_time.png                   # Step 4 pilot F1 curve
│   ├── step4_full_f1_vs_time.png              # Step 4 full data F1 curve
│   ├── step5_f1_vs_time.png                   # Step 5 ablation F1 curve
│   ├── step6_f1_vs_time.png                   # Step 6 tuning F1 curve
│   └── step7_dynamics_pilot.png               # Step 7 sub-second dynamics pilot
├── results/                                   # Empirical results & reports
│   ├── final/                                 # Official final deliverables
│   │   ├── CS2_Round_Outcome_Prediction_Final_Report.pdf
│   │   ├── f1_vs_time.csv                     # Official test curve (15 cutoffs)
│   │   ├── f1_vs_time.png                     # Official Macro F1 vs time plot
│   │   └── final_results_summary.md           # Executive numbers & explanations
│   ├── step1/                                 # Round table, validation report, eligibility
│   ├── step2/                                 # Metadata baseline, leakage checks, splits
│   ├── step3/                                 # Tick data reconnaissance probe
│   ├── step4/                                 # Pilot snapshot features & map bounds
│   ├── step4_full/                            # Full-scale snapshot feature comparison
│   ├── step5/                                 # 7-family cumulative ablation results
│   ├── step6/                                 # LightGBM parameter tuning & seed robustness
│   └── step7/                                 # Sub-second temporal dynamics pilot
├── src/                                       # Core executable scripts
│   ├── make_final_package.py                  # Aggregates results into final/
│   ├── step1_build_round_table.py             # Canonical round table builder
│   ├── step2_metadata_baseline.py             # Metadata baseline pipeline
│   ├── step3_tick_probe.py                    # Hugging Face remote stream probe
│   ├── step4_build_features.py                # Snapshot & spatial feature builder
│   ├── step4_compare_models.py                # Model comparison on pilot
│   ├── step4_extract_snapshots.py             # Parquet snapshot extractor
│   ├── step4_full_compare_models.py           # Full-scale model evaluator
│   ├── step5_ablation.py                      # Cumulative ablation runner
│   ├── step5_build_features.py                # Step 5 feature engineering
│   ├── step6_tune.py                          # Hyperparameter tuning runner
│   ├── step7_extract.py                       # High-frequency tick extractor
│   ├── step7_pilot.py                         # Dynamics model evaluator
│   └── cs2pred/                               # Shared reusable Python module
│       ├── __init__.py
│       ├── config.py                          # Constants, paths, prediction clocks
│       ├── features.py                        # Leakage-safe feature extractors
│       ├── modeling.py                        # Training, tuning, bootstrap CIs
│       ├── round_table.py                     # Canonical table & eligibility logic
│       ├── splits.py                          # Series-level split generator
│       ├── step5_features.py                  # Step 5 feature transformations
│       ├── step7_features.py                  # Step 7 dynamics feature logic
│       └── validation.py                      # Data validation test suite
```

---

## 3. Data Ingestion & Storage Architecture

### 3.1 OpenCS2 Dataset
The primary dataset is the **OpenCS2** competitive esports dataset:
- Hosted on Hugging Face: `blanchon/opencs2_dataset` and `blanchon/opencs2_dataset_wds`.
- The dataset captures professional and high-level competitive CS2 match demos parsed into tabular Parquet formats.
- Data categories:
  - **Metadata / Index:** `index/matches.parquet` (series/map metadata), `index/rounds.parquet` (round timestamps, winners, bomb events), `index/pov_rounds.parquet` (per-round player summaries).
  - **Events:** `events/kills.parquet` (event-level kill feed with tick, attacker, victim, weapon, headshot, smoke, blind, wallbang flags), `events/duels.parquet`, `events/round_player.parquet`.
  - **Full High-Rate Ticks:** Map-level tick streams containing per-player 64 Hz positions $(x, y, z)$, velocities $(v_x, v_y, v_z)$, view angles $(\text{pitch}, \text{yaw})$, health, armor, alive flags, and active weapon.

### 3.2 Local Caching vs. Remote Streaming
- Due to the large volume of full 64 Hz tick streams (~18+ GB across 794 maps), the pipeline was designed with **column projection and snapshot extraction**:
  - The script `step4_extract_snapshots.py` streams tick files from the Hub using `pyarrow.parquet.ParquetFile` and `HfFileSystem()`, extracts only rows matching prediction ticks ($t \in \{5, 10, \dots, 120\}$ s) plus a 0.5 s backward window, and discards raw tick frames.
  - The raw dataset is ignored by git (`.gitignore: data/`).

---

## 4. Audit of Previous Steps & Existing Results

| Step | Scope | Models | Features | Key Findings |
|---|---|---|---|---|
| **Step 1** | Canonical round table & validation | N/A | Keys & timing only | 16,527 rounds across 794 maps (359 series). Validated tick ordering and eligibility. |
| **Step 2** | Metadata-only baseline | LogReg, LightGBM | 63 features | Macro F1 rises from 0.622 at 5 s to 0.793 at 75 s, 0.842 at 90 s. LightGBM outperforms LogReg by +0.02 to +0.06 F1. |
| **Step 3** | Tick data probe | N/A | N/A | Verified 64 ticks/s rate, snapshot extraction viability, and column projections. |
| **Step 4** | Player state, spatial grid & movement | LightGBM | 63 (A) vs 97 (B) vs 306 (C) vs 314 (D) | Model B (+player state: HP, armor, weapon classes) gives consistent gain (+0.007 F1). Spatial grid (10x10 occupancy) and movement speed added no statistically significant gain (bootstrap CIs overlapped 0). |
| **Step 5** | Cumulative family ablation | LightGBM | 97 to 202 features (7 families) | Tested combat momentum (`cm`), player advantage (`pa`), firepower (`fw`), round state (`rs`), bomb timer (`bb`), pre-round interactions (`pr`), state changes (`sc`). No family reliably beat the 97-feature baseline (changes were within seed noise $\pm 0.005$). |
| **Step 6** | LightGBM hyperparameter tuning | LightGBM (13 configs) | 97 features | Best config `c02` (shallow trees: 4 leaves, depth 3, min child 80) achieved mean validation F1 of 0.7812 (+0.0036 over baseline). Stopped further tuning to prevent overfitting. |
| **Step 7** | Sub-second dynamics pilot | LightGBM | 97 to 205 features (windows: 0.5, 1, 2 s) | Short-term velocity changes and micro-engagements added no statistically significant benefit over static snapshot state. |
| **Final** | Benchmark consolidation | LightGBM `c02` | 97 features | Macro F1 reaches 0.7801 at 60 s and 0.7949 at 75 s on test. 0.80 target not reached $\le 60$ s. |

---

## 5. Audit of Professor's 15 Additional Suggestions

| Suggestion | Historical Status in Repo | Required Implementation / Audit Verdict |
|---|---|---|
| **AP_nextEvaluation** | Partially understood/unaddressed. | Clarified: represents the scheduled academic/project evaluation checkpoint for "Applied Programming / Applied Machine Learning". Requires a complete, self-contained, reproducible pipeline and defensible research deliverables. |
| **Different features in STOM except firing** | Addressed partially in Step 4/7 (occupancy counts, velocity), but failed. | Needs genuinely new spatial representations: alive-player spatial density, team centroid distance/dispersion, Voronoi/territory control, proximity to bomb sites, and spatial territory deltas over time. |
| **Rank features** | Unaddressed. | Must extract historical team/player performance (win rates, Elo/Glicko ratings, historical kill ratings) strictly prior to match timestamp without lookahead leakage. |
| **Time prediction** | Addressed across 15 discrete time cutoffs (5 s to 120 s). | Needs deep early-prediction analysis (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60 s), including matched-cohort analysis. |
| **Football paper** | Unaddressed in existing scripts. | Conduct academic literature comparison with sports outcome forecasting (pitch control, expected possession value, spatial grid models in football/soccer). |
| **Build pipeline** | Built in `src/`, but fragmented into separate scripts. | Needs an end-to-end master orchestrator running all evaluations, generating plots, and compiling reports seamlessly. |
| **Map analysis** | Addressed in legacy notebooks, but NOT in the modern 97-feature pipeline. | Must conduct map-wise evaluations across all competitive maps (Ancient, Anubis, Dust2, Inferno, Mirage, Nuke, Overpass/Vertigo) and compare map-specific vs. global models. |
| **Same analysis for result-test data** | Done in Step 6, but lacked full metrics (precision, recall, ROC-AUC, confusion matrices). | Expand test evaluation to all standard classification metrics and report confidence intervals on test data. |
| **Questions / statistical questions** | Partially addressed via cluster bootstrap CIs in Steps 2/4. | Formulate formal hypothesis tests (paired Wilcoxon signed-rank tests, McNemar's tests, bootstrap differences) answering RQ1-RQ10. |
| **Temporal models** | Unaddressed. Only tree models and LogReg were tested. | Implement and benchmark sequence models (LSTM, GRU, 1D-CNN/TCN) over the temporal progression leading to cutoff $t$. |
| **Classical models** | Only LogReg and LightGBM were systematically compared. | Implement and compare all minimum required classical models: Decision Tree, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, XGBoost, CatBoost, SVM, Naive Bayes, KNN. |
| **Rapid Fire paper comparison** | Unaddressed in existing documentation. | Identify the exact "Rapid Fire" CS:GO/CS2 research paper, audit its methodology, and perform a rigorous comparative discussion. |
| **Function of time / epochs** | Addressed via discrete cutoffs (5 s - 120 s). | Provide systematic rate-of-information analysis (information gain as a function of time). |
| **F1-score vs time graph** | Created in `plots/` and `results/final/f1_vs_time.png`. | Upgrade to publication-quality plots: multi-model comparison curves, uncertainty bands (95% CI), CT/T curves, coverage plots. |
| **Weather prediction / forecasting papers** | Unaddressed. | Review spatial-temporal forecasting literature (ConvLSTM, MetNet, GraphCast, radar nowcasting) and analyze transferability to CS2 round outcome modeling. |

---

## 6. Experimental Validity & Leakage Audit

A rigorous check of the current code confirms sound experimental validity:
1. **Clock & Freeze Time:**
   - Prediction clock starts ($t = 0$) strictly at `freeze_end_tick`, which marks the end of buy/freeze time and the start of playable action.
   - For prediction cutoff $t$, maximum allowed tick is $\text{prediction\_tick} = \text{freeze\_end\_tick} + \text{round}(t \times 64)$.
2. **Tick Rate:**
   - Confirmed at 64 ticks per second (CS2 standard sub-tick server snapshot tick rate).
3. **Data Leakage Safeguards:**
   - Functions `assert_no_outcome_columns()` and `lookahead_test()` explicitly verify that no post-cutoff events (kills, bomb plants, damage, survivor counts, round reasons) enter feature matrices.
   - Leakage tests in Steps 2, 5, and 7 confirmed that corrupting future data leaves features at $t \le T$ completely invariant.
4. **Data Splits:**
   - Series-level splitting (`splits.py`, seed 42) ensures all maps and rounds of a best-of-3 or best-of-1 series remain strictly isolated in either train, validation, or test.
   - Validation set is used solely for model selection and threshold tuning; the test set is evaluated once with frozen thresholds.
5. **Coverage Awareness:**
   - Live round eligibility requires $\text{playable\_duration\_s} > t$.
   - Test coverage drops with time: 100% at 5-10 s, 86.1% at 50 s, 78.4% at 60 s, 62.6% at 75 s, down to 12.3% at 120 s.
   - Any comparison across time cutoffs must account for this changing population via both natural-coverage and matched-cohort evaluations.

---

## 7. Audit Conclusion & Next Steps

The existing repository establishes a solid, leak-free baseline (Macro F1 = 0.7801 at 60 s, 0.7949 at 75 s with LightGBM and 97 features). However, key experimental gaps remain:
1. Lack of a multi-model benchmark (Random Forest, Extra Trees, XGBoost, CatBoost, SVM, KNN, HistGradientBoosting).
2. Absence of temporal sequence models (LSTM, GRU, TCN).
3. Under-explored spatial representations beyond simple 10x10 binary firing grids (e.g., density, team dispersion, territory control).
4. Lack of historical ranking/strength features.
5. Absence of map-wise modeling and matched-cohort statistical comparisons.

Stages 2 through 10 of the research plan directly address every gap and professor suggestion to produce a defensible, publication-grade study.
