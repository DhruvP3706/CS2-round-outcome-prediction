# Experiment gap analysis

Previously completed: grouped metadata and state baselines; occupancy/movement ablation; 13-config LightGBM tuning; three-seed short-window feature tests; classical and temporal benchmark CSVs. These are historical evidence, not newly executed results.

New priorities: exact manifests; retain missing-snapshot rounds; freeze tuning decisions before test prediction; persist predictions for both threshold policies; same cohort across algorithms; correct feature names; actual SHAP and observed occupancy; consistent two-class metrics; paired cluster inference; independent fifth-series methodology on Branch B.

Resource-conscious search: two prespecified configurations per classical model/time, with LightGBM/CatBoost validation early stopping. This directly respects the existing grouped validation partition and avoids row-level CV. RandomizedSearchCV and Optuna are optional search mechanisms, not requirements for valid tuning; broader group-aware nested searches remain unexecuted. Multiple seeds are limited to LightGBM baseline/combined at 30/40/50/60/75 seconds. Transformer is deferred because independent series count is modest relative to model capacity and the CPU budget prioritizes LSTM/GRU/TCN coverage.

Rankings cannot be defensibly evaluated until team identity and historical match completion metadata are verified. Navigation-aware spatial control, bomb-site distance and region-level utility would require verified map annotations or more raw fields. No synthetic substitute is used.

A missing plot or failed experiment must be reported explicitly. Generated tables and checkpoint counts are the execution record. Do not infer completion from an old report or a planned model list.
