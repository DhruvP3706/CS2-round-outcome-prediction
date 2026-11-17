# Leakage and validation audit

## Findings

**No issue found for tested grouped split isolation.** The saved manifest
assigns each match to one partition only; the maximum number of partitions per
match is one. Prediction keys and labels are checked against the canonical
manifest.

**No issue found for tested future-event invariance.** The executed checks
corrupted post-cutoff kills and plants at 10, 30, 60, and 75 seconds without
changing earlier feature values. The checks are recorded in
`leakage_checks.csv`.

**No issue found for validation threshold selection.** Thresholds are
recomputed from validation predictions and compared with each frozen threshold
in the prediction artifacts. Hyperparameter and feature choices are recorded
as validation-selected.

**Major limitation: cached provenance.** Snapshot caches do not preserve all
raw tick fields. Source inspection and timeline consistency checks support the
cutoff implementation, but they are not a complete independent raw-demo
re-extraction.

**Minor limitation: temporal generalization.** The grouped split is random by
match/series rather than chronological. It prevents within-match partition
leakage but does not estimate future-date deployment performance.

The implemented clock relation is `freeze_end_tick + 64 * t` and is checked
against the available event timeline. This establishes internal consistency
with the dataset's tick-derived seconds; it does not independently prove a
constant physical tick rate or resolve sub-tick parser behavior.

## Verdict

The tested pipeline supports the narrower claim that the saved grouped
benchmark is internally auditable against the available cached data. It does
not support the stronger claim that the entire historical or raw-demo pipeline
is leakage-free under all extraction paths.
