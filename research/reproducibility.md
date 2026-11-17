# Reproducibility

Run from the repository root with the available Python environment. Dependencies and versions are captured in each experiment's `environment.json`. Local data remain Git-ignored; source hashes and exact round split manifests accompany results.

```
python -m unittest discover -s tests -v
python src/research_audit.py
python src/research_experiment.py --stage classical
python src/research_experiment.py --stage ablation
python src/research_experiment.py --stage temporal
python src/research_experiment.py --stage report
```

Run stages sequentially when reproducing. Per-model/time/seed checkpoints support resumption of identical code/data/configuration. Do not reuse an output directory after changing those inputs; choose a fresh `--out` directory. Do not invoke two workers on the same stage and output. `selection/` captures all candidates, thresholds and model parameters; compressed predictions retain identifiers, labels, probabilities or transformed SVM margins, frozen thresholds and split. Training and validation predictions are saved separately by the split field. Tree models use 6–8 CPU threads; Torch uses four.

An `original_baseline` run repeats the historical LightGBM configuration on the available player-state cache. It is not automatically a bit-for-bit reproduction because the old combined tick cache is absent and the new cohort retains missing snapshots. See `baseline_reproduction.csv` for measured differences.

All times are seconds after freeze end. Eligibility requires the round to remain live strictly beyond the cutoff. Natural coverage is eligible rounds divided by all canonical rounds in that split. Matched coverage uses the same test rounds eligible at 75 seconds at every earlier cutoff. No early-model probabilities are carried forward to later cutoffs.

Test inference is not a basis for changing candidates or selecting features. The entire study remains exploratory because earlier project work already inspected this test partition. No results were pushed to a remote.
