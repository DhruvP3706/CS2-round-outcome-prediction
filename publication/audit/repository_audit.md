# Repository audit

## Scope and provenance

This audit was prepared from the working tree on branch
`research/publication-readiness`. The source baseline is commit
`bb1c7328d272a0a6a6efc75f82a84506f133cfec`, whose parent is
`d286cdd5050d7c5cbfc1458ecaf1ec0a31408ad3`. The worktree contained
pre-existing uncommitted experiment scripts and generated outputs; they were
not discarded or overwritten.

The repository has two relevant historical lines: `main` contains the initial
project, while `sanidhya_end_eval` contains the earlier end-to-end evaluation.
The current grouped benchmark is based on the later audit/evaluation commits.
Legacy STOM notebooks are retained, but their external split provenance and
dataset cannot be established from the notebooks alone. They are therefore not
pooled with the grouped benchmark.

## Structure

- `src/`: round-table construction, feature extraction, models, validation,
  experiments, reporting, and verification.
- `data/`: local OpenCS2 metadata, processed round tables, and feature caches.
  Raw/local data are not treated as redistributable.
- `results/main_experiment/`: split manifest, checkpoints, predictions, metrics,
  ablations, maps, matched cohorts, uncertainty, and plots.
- `research/`: audit, literature, reproducibility, and methodological notes.
- `paper/`, `plots/`, and `tables/`: earlier research presentation outputs.
- `ML_STOMs-*/` and `WP_DAC-*/`: historical notebooks and STOM-related work.
- `publication/`: this publication-readiness package.

## Dataset and task

The evaluated canonical table contains 16,527 rounds from 359 matches across
7 maps. The primary split contains 11,576 training rounds from 251 matches,
2,448 validation rounds from 54 matches, and 2,503 test rounds from 54
matches. Match groups are disjoint across partitions. `y_ct_win=1` denotes a
CT win and zero denotes a T win.

The task is to estimate the round winner from information available at fixed
cutoffs after freeze time. Eligibility is determined by the round remaining
live beyond the cutoff; later-round outcomes are not predictors.

## Experiment inventory

The main experiment evaluates classical models, a validation-selected
LightGBM baseline and feature ablations, compact causal sequence models,
exact planar geometry, occupancy summaries, map strata, matched cohorts, and
series-level uncertainty. Thresholds, model choices, and feature choices are
selected on validation data. Saved compressed predictions and checkpoint
metadata support independent recomputation.

The requested fifth-sample holdout is **not present on this branch**. No
fifth-sample metrics are reported here. The legacy notebooks are not a
substitute because their sample definition and split provenance differ.

## Results currently available

The validation-selected main benchmark provides results at 5, 10, 15, 20, 25,
30, 35, 40, 45, 50, 60, and 75 seconds. It includes accuracy, Macro F1, CT/T
F1, precision/recall, weighted F1, balanced accuracy, ROC-AUC, PR-AUC,
confusion counts, sample count, and coverage where defined. Cluster bootstrap
intervals and paired series-swap tests are saved separately.

## Missing or incomplete evidence

1. The requested 90, 105, and 120 second cutoffs are not in the executed
   benchmark.
2. The fifth-sample experiment is absent.
3. The original combined tick feature cache is absent; historical reproduction
   is therefore a diagnostic using the available player-state cache.
4. Cached snapshots do not retain all raw tick provenance. Timeline checks
   establish consistency with the dataset clock, not independent wall-clock
   validation.
5. Full methods and numerical results for inaccessible literature sources could
   not be independently verified.
6. Prospective, chronological, team-held-out, and unseen-map generalization
   evaluations were not completed.

These limitations prevent a claim of confirmatory generalization or complete
publication readiness.
