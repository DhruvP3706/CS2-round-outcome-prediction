# Executed evaluation

All model/feature choices below maximize validation Macro F1; test scores are descriptive.

```
 t_s                  model      feature_set  accuracy  macro_f1    n  coverage
   5 hist_gradient_boosting         baseline  0.622453  0.622406 2503  1.000000
  10        convex_ensemble         baseline  0.631243  0.630986 2503  1.000000
  15        convex_ensemble         baseline  0.640800  0.640656 2500  0.998801
  20               catboost         baseline  0.669345  0.669260 2489  0.994407
  25        convex_ensemble         baseline  0.693117  0.693110 2470  0.986816
  30               catboost         baseline  0.700041  0.700030 2427  0.969636
  35      original_baseline         baseline  0.712784  0.712467 2378  0.950060
  40               catboost         baseline  0.729892  0.727692 2325  0.928885
  45               lightgbm   without_combat  0.751669  0.751261 2247  0.897723
  50               lightgbm temporal_spatial  0.759629  0.758843 2155  0.860967
  60        convex_ensemble         baseline  0.780836  0.779547 1962  0.783859
  75        convex_ensemble         baseline  0.797573  0.797435 1566  0.625649
```

Macro F1 always averages CT and T F1, including single-class samples. Precision, recall and PR-AUC use CT as positive. PR-AUC is average precision. Linear SVM outputs sigmoid-transformed margins, not calibrated probabilities.

Confidence intervals resample whole series (2,000 replicates). Paired comparisons swap model predictions at series level (5,000 permutations); Holm correction covers the reported cutoff comparisons. These intervals condition on the fitted models; they do not include model-selection or data-collection uncertainty. A single-series holdout has no population cluster interval or inferential p-value.

The historical test set has already been inspected by earlier research. These results are an exploratory re-evaluation, not a new untouched confirmatory test. A random grouped split does not establish future-date generalization. Rankings are excluded because team identity alignment is unverified. Transformer and broad Optuna searches are deferred; two prespecified candidates per classical model are compared on the fixed grouped validation set. Temporal models use a 20-epoch cap.
