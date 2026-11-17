# Literature review and novelty assessment

The verified literature is summarized in `literature_matrix.csv`. Primary
links and verification notes are preserved in [`research/literature_review.md`](../../research/literature_review.md).
The closest precedents study CS:GO round outcomes from frame or telemetry
information. Other works predict match outcomes or player deaths and should
not be numerically compared with this fixed-cutoff round Macro F1.

## Novelty assessment

Already established: round-outcome prediction from health, equipment,
economic, spatial, and event telemetry; frame-based and real-time CS:GO
prediction; classical model comparisons; spatial-temporal forecasting
representations in adjacent domains.

Different in this repository: a CS2 multi-map benchmark with explicit live
cutoffs, grouped match splitting, natural and matched coverage, validation-only
selection, exact planar geometry/occupancy summaries, saved per-round
predictions, and cluster-aware paired comparisons.

Supported contribution: an auditable evaluation protocol and an empirical
description of the quality/coverage trade-off under the available data.
Not supported: a claim that the ordinary occupancy features reproduce the
historical firing-based STOM representation, a causal claim for feature
importance, an information-theoretic 0.80 limit, or broad external
generalization.

The strongest material improvement would be a fresh chronological and
team-held-out evaluation, plus a verified fifth-sample holdout and independent
raw-tick re-extraction. A contribution claim should remain exploratory until
those tests are completed.
