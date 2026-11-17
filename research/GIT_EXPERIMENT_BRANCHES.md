# Independent experimental branches

Original branch: `sanidhya_end_eval`. Original commit: `27845f6b5f8ebd0c8702633fb825ceb9dd66db62`.
The initial working tree was clean. No stash, reset, clean, or overwrite of user changes was needed.

Branch A, `research/main-model-comparison`, starts at the original commit. It preserves the original seed-42 series assignments and stores new outputs under `results/main_experiment/`.

Branch B, `research/fifth-sample-holdout`, is created from the completed Branch A commit. Only B adds deterministic fifth-series selection and results under `results/fifth_sample_experiment/`. Its inherited main results remain separate. This is an independent split case study, not a statistically independent new dataset: some of its observations were used by the earlier experiment.

Switch with `git switch research/main-model-comparison` or `git switch research/fifth-sample-holdout`. Return to the unchanged original with `git switch sanidhya_end_eval`. Commit or otherwise preserve any subsequent working changes before switching. Inspect final branch tips with `git log -1 <branch>`.

All commits remain local. No push or merge of B into A is authorized or performed.
