# Experiment Gap Analysis: CS2 Round Outcome Prediction

**Document Type:** Strategic Research Gap Analysis & Action Plan  
**Repository:** `CS2-round-outcome-prediction`  
**Date:** November 2026  
**Author:** AI Research Specialist & Co-Author (Antigravity)  

---

## 1. Executive Summary

This document establishes a rigorous gap analysis comparing what has been thoroughly explored in the repository versus what critical scientific questions remain unanswered. By identifying these gaps, we define a targeted roadmap for Stages 2 through 10 of this research.

The primary research objective is:
> **How accurately and how early can we predict the winning team of a CS2 round using spatial, temporal, player-state, match-context, and available game information? Can Macro F1 exceed 0.80 at 60 seconds or earlier?**

---

## 2. Comprehensive Matrix of Completed vs. Missing Research

| Research Dimension | Current State in Repository | Critical Scientific Gap | Planned Stage & Action |
|---|---|---|---|
| **Classical ML Models** | Only LightGBM and Logistic Regression tested. | No comparison against Decision Trees, Random Forests, Extra Trees, Gradient Boosting, HistGradientBoosting, XGBoost, CatBoost, SVM, Naive Bayes, KNN. | **Stage 5:** Standardized benchmark of 10+ classical algorithms under identical features, splits, and validation thresholding. |
| **Temporal Deep Learning** | Zero sequence models evaluated. | CS2 rounds are dynamic sequential processes. No testing of recurrent (LSTM, GRU) or temporal convolutional (TCN) architectures. | **Stage 6:** Build sequential event/tick sequences up to cutoff $t$; evaluate LSTM, GRU, and 1D-CNN temporal models. |
| **Alternative Spatial STOMs** | Coarse 10x10 binary weapon fire (legacy) and 10x10 binary snapshot occupancy (Step 4). | Continuous geometry, team dispersion, centroid distances, Voronoi territory control, and bomb-site proximity were never tested. | **Stage 4:** Engineer continuous geometric spatial representations and conduct rigorous ablation. |
| **Ranking & Historical Strength** | Entirely absent. Only in-round and pre-round map index used. | No historical team ratings, pre-match win rates, or relative Elo/rating differentials evaluated. | **Stage 7:** Build leak-free historical team strength metrics strictly from prior matches; test incremental predictive value. |
| **Map-Specific Analysis** | Explored in legacy notebooks only; absent from modern pipeline. | Unknown how accuracy varies across maps (Dust2, Mirage, Inferno, Ancient, Anubis, Nuke, Overpass) and whether map-specific models beat global models. | **Stage 7 & 9:** Map-stratified evaluation, class balance analysis, and map vs. time heatmaps. |
| **Cohort Evaluation Rigor** | Only natural-coverage evaluation reported in `results/final/`. | Natural coverage falls from 100% at 5 s to 78.4% at 60 s and 12.3% at 120 s. Information gain is confounded with survival selection bias. | **Stage 6 & 9:** Dual evaluation: (A) Natural-coverage curve and (B) Matched-cohort curve (evaluating identical round subsets). |
| **Statistical Defensibility** | Bootstrap CIs were computed in Steps 2 & 4, but final curve (`results/final/f1_vs_time.csv`) lacks CIs. | No paired hypothesis tests (Wilcoxon signed-rank), no multiple-comparison corrections, no formal significance testing of deltas. | **Stage 8 & 10:** Multi-seed evaluation, series cluster bootstrap CIs, paired significance tests, and effect sizes. |
| **Literature Context** | Minimal academic references in repository. | Missing systematic comparison with "Rapid Fire" CS:GO research, soccer/football sports outcome models, and weather forecasting spatial analogies. | **Stage 9 & 10:** Formal literature review with verified academic citations and literature matrix. |
| **Pipeline Integration** | Pipeline fragmented into independent `step*.py` scripts. | Requires a unified end-to-end execution pipeline and publication-ready automated reporting. | **Stage 2 & Master Script:** Reusable orchestrator running experiments, generating figures, and compiling papers. |

---

## 3. Deep Dive into Key Research Gaps

### 3.1 Gap 1: The Model Architecture Spectrum
In Step 6, LightGBM hyperparameter tuning reached a plateau: reducing tree depth to 3 and leaves to 4 marginally improved validation F1 (+0.0036), but test Macro F1 remained at 0.7801 at 60 s.
- **Hypothesis to Test:** Tree-based ensembles with different inductive biases (Random Forest, Extra Trees, CatBoost, XGBoost) or linear regularized models with kernel approximations (SVM) may handle the correlation structure of 97 features differently.
- **Action:** Train and evaluate all 12 minimum classical models across all prediction cutoffs using the exact same series-level split and validation thresholding rule.

### 3.2 Gap 2: Temporal Sequence Modeling vs. Static Snapshots
The current 97-feature baseline summarizes history up to cutoff $t$ as static cumulative sums and single-tick snapshots.
- **Scientific Question:** Does the *temporal sequence of events* (the order and timing of kills, trades, damage, and position updates) carry predictive information that static aggregates lose?
- **Action:** Construct temporal sequence tensors containing event histories over time intervals $\tau \in [0, t]$. Train LSTM and GRU architectures, ensuring no information past cutoff $t$ enters the hidden states.

### 3.3 Gap 3: Continuous Geometric Spatial Representations
The professor specifically suggested: *"Different features in STOM except firing"*.
Previous spatial attempts failed because a 10x10 discrete grid creates 200 sparse binary features (most of which are 0 at any given tick), which trees penalize or ignore.
- **Alternative Formulation:** Continuous spatial descriptors:
  1. **Team Centroids:** $(\bar{x}_{CT}, \bar{y}_{CT})$ and $(\bar{x}_{T}, \bar{y}_{T})$.
  2. **Team Spatial Dispersion:** Radius of gyration $R_g = \sqrt{\frac{1}{N} \sum (x_i - \bar{x})^2 + (y_i - \bar{y})^2}$, measuring whether a team is tightly grouped or executing a split/spread default.
  3. **Inter-Team Engagement Distance:** Euclidean distance between CT and T centroids $d(\text{Centroid}_{CT}, \text{Centroid}_T)$.
  4. **Bomb-Site Proximity:** Distances to Bomb Site A and Bomb Site B centroids.
- **Action:** Extract these continuous geometric features, verify zero lookahead leakage, and perform feature-family ablation.

### 3.4 Gap 4: Player and Team Ranking Signals
In competitive Counter-Strike, team quality differentials are major predictors of round conversion (e.g., a top-5 team converting 4v5 rounds significantly more often than a lower-tier team).
- **Scientific Question:** Does incorporating pre-match historical team win rates and map proficiency boost prediction accuracy early in the round (5–30 s) when in-round kill events are sparse?
- **Action:** Chronologically compute historical team win rates and map-specific win rates strictly using matches completed before each match date. Test baseline vs. baseline + ranking features.

### 3.5 Gap 5: Disentangling Time Progression vs. Selection Bias
At 5 seconds, 100% of rounds are live (2,503 test rounds). At 60 seconds, 78.4% of rounds are live (1,962 test rounds). At 120 seconds, only 12.3% of rounds are live (309 test rounds).
- **The Confound:** A model predicting at 90 seconds (Macro F1 = 0.8436) is evaluated on a completely different, surviving round cohort (predominantly late bomb-defusal scenarios).
- **Action:** Perform a **Matched-Cohort Analysis**: evaluate all prediction times on the subset of rounds that survive to at least 60 s (or 75 s). This directly isolates the true contribution of elapsed time and game information.

---

## 4. Execution Plan to Address All Gaps

1. **Stage 1 (Complete):** Repository audit, feature inventory, and gap analysis.
2. **Stage 2:** Verify data pipeline, split integrity, tick rates, and evaluation framework.
3. **Stage 3:** Baseline reproduction (confirm existing Step 6 results down to exact decimal places).
4. **Stage 4:** Novel spatial representations (continuous geometry, dispersion, site proximity) & ablation.
5. **Stage 5:** Classical ML model comparison (LR, DT, RF, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost, HistGradientBoosting, SVM, Naive Bayes, KNN).
6. **Stage 6:** Temporal sequence models (LSTM, GRU, TCN).
7. **Stage 7:** Ranking features and comprehensive map-wise analysis.
8. **Stage 8:** Feature importance (SHAP, permutation), paired statistical significance tests, and bootstrap CIs.
9. **Stage 9:** Publication-quality figures (20 high-res plots), summary tables, and academic literature review.
10. **Stage 10:** Academic research paper (`paper/main.md`, `paper/references.bib`) and final research report (`FINAL_RESEARCH_REPORT.md`).
