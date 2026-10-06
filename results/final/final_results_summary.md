# CS2 Round Outcome Prediction — Final Results Summary

## Final model (official)
- **Feature set (97):** Step 2 metadata features (63) + Step 4 player-state features (34). No Step 5, Step 7, spatial-occupancy or movement features.
- **Model:** LightGBM (`c02_nl4_md3_mcs80`, chosen in Step 6 by mean *validation* macro F1 over 30/40/50/60/75 s; test never used for selection).
- **Parameters (from `results/step6/selected_config.json`):** num_leaves=4, max_depth=3, min_child_samples=80, learning_rate=0.03, feature_fraction=0.8, bagging_fraction=0.8 (bagging every iteration), lambda_l1=0.0, lambda_l2=2.0; n_estimators cap 5000 with early stopping (100 rounds) on validation log-loss; seed 42.
- **Evaluation:** one separate model per prediction time; decision threshold chosen on validation to maximise macro F1 and frozen; test evaluated once. Positive class = CT win.
- **Split (series-level, seed 42, 70/15/15):** train 251 series / 555 maps / 11,576 rounds; validation 54 / 119 / 2,448; test 54 / 120 / 2,503 (16,527 rounds, 794 maps, 359 series in total).

## Results (test, all 15 prediction times)
| time (s) | macro F1 | CT F1 | T F1 | eligible test rounds | coverage | val. threshold |
|---|---|---|---|---|---|---|
| 5 | 0.6228 | 0.6197 | 0.6260 | 2503 | 1.000 | 0.515 |
| 10 | 0.6290 | 0.6487 | 0.6093 | 2503 | 1.000 | 0.495 |
| 15 | 0.6412 | 0.6376 | 0.6448 | 2500 | 0.999 | 0.510 |
| 20 | 0.6724 | 0.6661 | 0.6788 | 2489 | 0.994 | 0.515 |
| 25 | 0.6882 | 0.6849 | 0.6915 | 2470 | 0.987 | 0.510 |
| 30 | 0.7096 | 0.6997 | 0.7195 | 2427 | 0.970 | 0.520 |
| 35 | 0.7169 | 0.7055 | 0.7284 | 2378 | 0.950 | 0.520 |
| 40 | 0.7287 | 0.7050 | 0.7524 | 2325 | 0.929 | 0.540 |
| 45 | 0.7500 | 0.7416 | 0.7585 | 2247 | 0.898 | 0.490 |
| 50 | 0.7591 | 0.7437 | 0.7745 | 2155 | 0.861 | 0.520 |
| 60 | 0.7801 | 0.7634 | 0.7968 | 1962 | 0.784 | 0.515 |
| 75 | 0.7949 | 0.7912 | 0.7987 | 1566 | 0.626 | 0.445 |
| 90 | 0.8436 | 0.8234 | 0.8637 | 1157 | 0.462 | 0.565 |
| 105 | 0.8998 | 0.8807 | 0.9189 | 704 | 0.281 | 0.580 |
| 120 | 0.8423 | 0.7586 | 0.9260 | 309 | 0.123 | 0.645 |

## Key numbers
- Best final macro F1 **below 60 s:** 0.7591 (at 50 s, coverage 86.1%).
- F1 at 60 s: **0.7801** (coverage 78.4%); 75 s: **0.7949** (62.6%); 90 s: **0.8436** (46.2%); 105 s: **0.8998** (28.1%); 120 s: **0.8423** (12.3%).
- **First time macro F1 >= 0.80: 90 s** (0.8436; coverage 46.2%). 0.80 is not reached at 60 s (0.780) or 75 s (0.795).
- Highest value on the curve: 0.8998 at 105 s, on only 704 test rounds (28.1% coverage) — not comparable to early-time results.
- Coverage at important times: 30 s 97.0%, 40 s 92.9%, 50 s 86.1%, 60 s 78.4%, 75 s 62.6%, 90 s 46.2%, 105 s 28.1%, 120 s 12.3%.

## Why late-time F1 must be read cautiously
- Coverage falls steeply (46% of test rounds at 90 s, 28% at 105 s, 12% at 120 s): only rounds that are still live at time t are eligible, so the evaluated population changes with t and is not a random sample of rounds.
- Class balance shifts: in the test split the CT win rate of eligible rounds falls from 0.47 at 60 s to 0.44 (90 s), 0.40 (105 s) and 0.26 (120 s).
- Small samples (1,157 / 704 / 309 rounds at 90 / 105 / 120 s; training sets of 4,963 / 3,101 / 1,321 rounds) make F1 volatile; the Step 2 bootstrap interval of the (different) Step 2 LightGBM at 120 s spans 0.79–0.89. The 105 s > 120 s non-monotonicity is within that noise.
- The curve has no confidence intervals (none were computed for the final configuration). Seed-to-seed spread of ~0.003–0.008 was observed at 30–75 s in Steps 5–6.

## Exploratory results (NOT the final model)
Steps 5 and 7 produced isolated test values slightly above the final curve at single times (e.g. Step 5 set C at 60 s: 0.787; Step 7 set C at 40 s: 0.736 vs 0.729). These were within seed noise, were not validated, and do not replace the Step 6 configuration.
