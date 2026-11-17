# Repository audit, 17 November 2026

Baseline provenance: `sanidhya_end_eval`, `27845f6b5f8ebd0c8702633fb825ceb9dd66db62`, clean working tree. Source and notebook inventories are in `source_inventory.csv` and `notebook_inventory.csv`; the latter scans all 65 notebooks, including split and evaluation code. The older uppercase audit files are preserved under research/archive as historical documents, not current verification.

## Existing implementation

Steps 1–2 build a round table, label CT=1/T=0, cumulative/recent kill features, bomb timing, prior-round context, and a seed-42 split by `match_id`. Step 3 probes tick files. Step 4 extracts alive-player snapshots at exact prediction ticks, movement in a backward half-second window, and training-bound spatial grids. Steps 5 and 7 already tested combat, state, firepower, movement and short-window dynamics. Step 6 performed a 13-configuration LightGBM search and seed checks. The master pipeline subsequently added 12 classical models, LSTM/GRU/TCN, geometry and ranking code. Legacy STOM notebooks include Random Forest experiments using externally assigned Training/Testing directories on D: and a distinct dataset. Their split provenance cannot be verified from the notebook alone; they must not be pooled with the grouped benchmark.

Existing output summaries report no robust improvement from Step 5 or Step 7 families. Those results and their code remain intact. The new suite is not a claim that positions, movement or kills were absent.

## Findings affecting validity

- Series splitting is deterministic and groups all maps. The audit executes an exact comparison to `results/step2/series_split.csv`.
- Existing inner joins silently remove eligible metadata rows without snapshots. New results retain them, with train-fitted median imputation and missingness indicators. Therefore new and old cohorts/preprocessing are not automatically identical.
- The original `data/features/step4_full/step4_tick_features.parquet` is absent. The newer cached player-state table is present. New reproduction uses that available cache; discrepancies must be reported, not hidden.
- `novel_spatial.py` names a centroid-distance proxy `sp_nearest_enemy_dist`, and `sp_avg_pair_dist` duplicates centroid distance. New extraction computes actual cross-team pair distances and team pair means.
- The previous figure called `pub_11_shap_summary` aggregates gain, not SHAP. `pub_18_spatial_occupancy_stom` draws random Gaussian points. New plots use native LightGBM TreeSHAP contributions and observed snapshots.
- `generate_final_tables.py` picks the largest test F1 as the best model. New choices use validation F1, including feature configuration. Descriptive test maxima must never be presented as validation selection.
- `ranking.py` assumes a player-slot/team mapping and overtime side switching without adequate verification. Dates alone do not prove a prior match had completed. Rankings are excluded pending identity and completion-time evidence; no new rating benefit is claimed.
- Existing calibrated SVM uses ordinary row folds internally, and histogram boosting can use an internal random validation split. New SVM uses fixed margin transformation; new histogram boosting disables internal early stopping.
- Existing macro averaging can omit absent classes; new Macro F1 always averages the specified two classes. Undefined ranking metrics remain missing on single-class cohorts.
- Existing Wilcoxon comparisons on row-level confidence differences do not establish clustered F1 improvement. New paired series bootstrap and series-level prediction-swap tests address the declared estimand.
- The old final report's assertion that 0.80 before 60 seconds is theoretically impossible is unsupported. A failed experiment establishes no information-theoretic bound. Several summary values also disagree across sections; new prose is generated from saved metrics.

## Data and timeline checks

`research_audit.py` executes label/key/timeline/relational checks, verifies kill `event_seconds` against `(tick-freeze_end)/64`, and corrupts post-cutoff events at four cutoffs. This verifies consistency with the dataset clock, not an independent physical tick-rate measurement. Snapshot caches omit raw tick provenance, so the extractor's exact-tick mask is audited in source; complete independent raw-tick re-extraction remains outside this local rerun.

Resource check before training: 20 logical CPUs, approximately 31.8 GiB RAM (17.5 GiB initially free), approximately 268 GiB disk free, NVIDIA T400 4 GB. Temporal runs use CPU with four Torch threads. No external datasets are downloaded for training.

## Remaining threats

The historical test results have already informed earlier project discussion. Reuse is exploratory, even with validation-only choices now. The random grouped split is not chronological deployment validation. Distances are planar Euclidean distances, not navigation or line-of-sight distances; vertical levels and map topology are not modeled. Short sequences and a limited search budget may underfit temporal models. Team-held-out and future-date evaluation are future work.
