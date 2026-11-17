# Abstract

This study evaluates Counter-Strike 2 round-winner prediction at fixed live
cutoffs after freeze time. A match-grouped split prevents match overlap across
training, validation, and test partitions, while natural and matched cohorts
separate predictive quality from declining late-round coverage. Classical
models, compact causal sequence models, exact planar geometry, occupancy
summaries, and feature ablations are compared using validation-only selection.
The main test partition contains 2,503 rounds from 54 matches. At 60 seconds,
the validation-selected system reaches 0.7808 accuracy and 0.7795 Macro F1 on
1,962 eligible rounds, with 0.7839 coverage. Cluster bootstrap intervals and
paired series-level tests show no reliable adjusted improvement over the
original baseline at that cutoff. The evaluation is exploratory because the
test partition was previously inspected, and the requested fifth-sample
holdout and later 90/105/120-second cutoffs are not available on this branch.
