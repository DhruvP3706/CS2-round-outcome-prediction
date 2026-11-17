# Final skeptical review

## Assessment

- **Originality:** potentially useful evaluation protocol; algorithmic novelty
  is limited and STOM novelty is not established.
- **Technical correctness:** saved-artifact checks pass for the tested grouped
  benchmark; raw-tick provenance remains incomplete.
- **Experimental design:** grouped and validation-aware, but random rather than
  chronological and not team-held-out.
- **Dataset quality:** sufficient for an exploratory benchmark; selection and
  access limitations remain.
- **Leakage prevention:** tested split, label, threshold, and future-event
  invariants pass; full raw extraction is not independently re-run.
- **Statistical validity:** match-level bootstrap and paired tests are
  appropriate; adjusted improvements are not reliable.
- **Baseline quality:** broad classical coverage is useful, but search remains
  intentionally limited.
- **Reproducibility:** saved predictions and manifests are strong; missing
  caches and missing holdout reduce completeness.
- **Generalization:** not established beyond the random grouped test cohort.
- **Clarity:** substantially documented in the package.
- **Significance:** exploratory evidence, not yet a definitive CS2 result.

## Blockers and revisions

**Critical blockers:** recover and independently execute the fifth-sample
holdout; add a fresh chronological or prospective test; verify raw tick and
snapshot provenance.

**Major revisions:** execute 90/105/120 seconds or narrow the claimed horizon;
report the exact STOM representation separately from ordinary occupancy; expand
independent literature verification; freeze a confirmatory protocol before a
new test collection.

**Minor revisions:** add PDF exports for all final figures, pin the complete
environment, and document licensing and dataset acquisition in more detail.

## Verdict

**B -- Promising, but requires additional experiments.** The main grouped
benchmark is internally auditable and the uncertainty analysis is
appropriately conservative. It is not suitable for a strong peer-reviewed
claim of generalization or novelty until the blockers above are addressed.
