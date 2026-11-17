# Independent metric verification

`metric_verification.csv` is copied from the saved test predictions report.
The independent verifier in `src/research_verify_artifacts.py` checks every
saved checkpoint against its prediction file and the split manifest.

Verified invariants:

- prediction keys are unique;
- labels agree with the canonical manifest;
- each prediction file contains exactly the eligible rows for its split and
  cutoff;
- thresholds equal the threshold recomputed from validation predictions;
- accuracy and two-class Macro F1 agree with scikit-learn recomputation;
- test rows remain separate from train and validation rows;
- all match IDs occur in exactly one partition.

The verifier recorded 296 global run artifacts, with split isolation, cohort
equality, label equality, validation-threshold recalculation, accuracy, and
Macro F1 checks passing. The CSV retains the broader metric columns generated
by the reporting pipeline, including confusion counts and coverage.

This is an artifact-level verification, not a fresh raw-demo extraction.
Consequently, it does not eliminate the documented limitations of cached
snapshot provenance or prior inspection of the test partition.
