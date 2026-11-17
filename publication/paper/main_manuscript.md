# Auditable Early Round-Outcome Prediction in Counter-Strike 2

## Abstract

This study evaluates Counter-Strike 2 round-winner prediction at fixed live
cutoffs after freeze time. A match-grouped split prevents match overlap across
training, validation, and test partitions. Classical models, compact causal
sequence models, exact planar geometry, occupancy summaries, and feature
ablations are compared with validation-only model and threshold selection. The
main test partition contains 2,503 rounds from 54 matches. At 60 seconds, the
validation-selected convex ensemble obtains accuracy 0.7808 and Macro F1
0.7795 on 1,962 eligible rounds, with coverage 0.7839. The 95% match-cluster
bootstrap interval for Macro F1 is [0.7614, 0.7965] for the selected
convex-ensemble-at-60s row; the paired selected-system comparison with the original
baseline has adjusted p=1.0. These results are exploratory because the test
partition was previously inspected. The requested fifth-sample holdout and
90/105/120-second horizons are absent and are not inferred.

## 1. Introduction

Within-round prediction is a changing-information problem: later cutoffs
contain more state but apply to fewer surviving rounds. This paper therefore
reports both natural coverage and matched-cohort comparisons. The goal is not
to establish an intrinsic accuracy limit, but to characterize the evidence
supported by the available CS2 telemetry.

## 2. Research questions and contributions

We address: (RQ1) how predictive quality changes over executed cutoffs,
(RQ2) how model families trade off accuracy and cost, (RQ3) whether tested
feature families add reliable gains, and (RQ4) how performance varies by map.
The contributions are an auditable grouped benchmark, an explicit
quality/coverage analysis, and a conservative feature/model comparison with
cluster-aware uncertainty. These are empirical contributions, not a claim of
new universal STOM theory.

## 3. Dataset and formulation

The canonical table contains 16,527 rounds from 359 matches across seven maps.
The positive class is CT win (`y_ct_win=1`); T win is zero. The split is
11,576/2,448/2,503 rounds from 251/54/54 matches for train/validation/test.
Each match occurs in exactly one partition. A round is eligible at time `t`
only when it remains live beyond `freeze_end_tick + 64*t`.

## 4. Features and models

The baseline combines metadata, cumulative causal event features, player-state
snapshots, and prior-round context. Additional families include combat removal,
movement and temporal dynamics, exact cross-team planar distances, team
centroids, within-team dispersion, and occupancy summaries. These descriptors
are not claimed to reproduce the historical firing-based STOM notebooks.

The benchmark includes logistic regression, decision tree, random forest,
extra trees, gradient boosting, histogram gradient boosting, XGBoost,
LightGBM, CatBoost, linear SVM, KNN, Gaussian Naive Bayes, a convex ensemble,
and compact LSTM/GRU/causal-convolution candidates. Choices and thresholds are
selected on validation data only. Test values are descriptive.

## 5. Evaluation

Macro F1 is the mean of CT and T F1 with both classes included. We also report
accuracy, per-class precision/recall/F1, weighted F1, balanced accuracy,
ROC-AUC, average precision, confusion counts, sample count, and coverage.
Uncertainty resamples complete matches (2,000 draws). Paired tests swap
predictions at match level (5,000 draws), with Holm correction across the
reported cutoffs.

## 6. Results

The complete validation-selected table is
[`performance_by_time.csv`](../tables/performance_by_time.csv). Selected
Macro F1 rises from 0.6224 at 5 seconds to 0.7795 at 60 seconds and 0.7974
at 75 seconds, while coverage falls from 1.0000 to 0.6256. The 0.80 point
estimate is not reached by the selected system in the executed horizons.
This is not evidence of an information-theoretic limit.

Natural and matched-cohort results are separated in
[`matched_cohort.csv`](../tables/matched_cohort.csv). Map-wise results are
descriptive because independent match counts and class balance vary.
Feature-family results are in [`feature_ablation.csv`](../tables/feature_ablation.csv);
the paired statistical output is in
[`paired_tests.csv`](../statistics/paired_tests.csv).

## 7. Leakage and reproducibility

Split isolation, label alignment, validation threshold recomputation, and
post-cutoff event invariance checks pass for saved artifacts. The
implementation remains limited by missing raw tick provenance in cached
snapshots and the absence of the original combined tick cache. The full
reproduction commands are in
[`README.md`](../reproducibility/README.md).

## 8. Related work

Prior CS:GO work has studied round outcomes from frame information and
telemetry; other work targets match outcomes or player deaths. These tasks,
cutoffs, and metrics are not interchangeable. The verified bibliography and
comparison matrix are in [`literature_review.md`](../literature/literature_review.md)
and [`literature_matrix.csv`](../literature/literature_matrix.csv).

## 9. Threats to validity and limitations

The random grouped split is not chronological or team-held-out. The test
partition was previously inspected, so this is an exploratory re-evaluation.
The fifth-sample experiment is unavailable, and 90/105/120-second cutoffs were
not executed. Planar geometry omits walls, levels, visibility, and navigation.
The limited number of independent matches constrains small-stratum inference.

## 10. Conclusion

The available evidence supports a reproducible description of within-round
CS2 prediction quality and coverage under the stated grouped protocol. It does
not support claims of prospective generalization, causal feature value, or a
fundamental accuracy ceiling. A fresh chronological/team-held-out evaluation,
raw-tick re-extraction, and the missing fifth-sample experiment are required
before a confirmatory publication claim.
