# Reproducibility package

Run commands from the repository root with the local dataset available:

```text
python -m unittest discover -s tests -v
python src/research_audit.py
python src/research_verify_artifacts.py --out results/main_experiment
python src/research_experiment.py --stage report
```

The full staged reproduction is documented in [`research/reproducibility.md`](../../research/reproducibility.md).
The data are local and may be subject to source licensing or access
restrictions; they are not redistributed by this package.

The exact seed-42 split is saved in
`results/main_experiment/split_manifest.csv`. Saved prediction files preserve
round keys, labels, probabilities or transformed margins, thresholds, and
partition labels. Environment metadata are stored with the experiment output.

Do not reuse an output directory after changing source, data, or configuration.
The fifth-sample experiment and 90/105/120-second horizons are not reproducible
from this branch because their artifacts are absent.
