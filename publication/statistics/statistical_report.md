# Statistical reliability report

The primary comparison treats the 54-match test partition as clustered data.
Bootstrap resamples complete matches, not rounds, with 2,000 repetitions.
Paired model comparisons use identical prediction keys and a match-level
prediction-swap permutation test with 5,000 draws. Holm correction is applied
over the reported cutoff comparisons.

The saved `paired_tests.csv` and `confidence_intervals.csv` contain the
recomputed estimates. At 60 seconds, the validation-selected convex ensemble
versus the original baseline has Macro F1 difference 0.00205, 95% interval
[-0.00676, 0.01018], and Holm-adjusted p=1.0. The interval includes zero.
The only unadjusted cutoff with a positive interval in the saved comparison is
45 seconds, but its corrected p-value is 0.08398 and therefore does not meet a
0.05 family-wise threshold.

These intervals condition on fitted models and the exploratory test reuse.
They do not include uncertainty from model search, data collection, or
prospective deployment. The absent fifth-sample experiment has no population
cluster inference here.
