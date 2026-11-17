"""Generate a conservative manuscript from completed result files."""
from pathlib import Path
import argparse
import shutil
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--include-fifth',action='store_true'); args=p.parse_args()
    out=ROOT/'results/main_experiment'; paper=ROOT/'paper'
    for sub in ['figures','tables']: (paper/sub).mkdir(exist_ok=True)
    best=pd.read_csv(out/'best_model_by_time.csv'); r=best[best.t_s==60].iloc[0]
    crossing=best[best.macro_f1>.8]
    earliest='not reached at any evaluated cutoff' if crossing.empty else f'{int(crossing.t_s.min())} seconds'
    manifest=pd.read_csv(out/'split_manifest.csv')
    summary=manifest.groupby('split').agg(series=('match_id','nunique'),rounds=('round','size'),ct=('y_ct_win','sum'))
    evidence=(out/'tables/best_model_by_time.md').read_text(encoding='utf-8')
    # Compact paper table: complete details remain in the result directory.
    cols=['t_s','model','feature_set','accuracy','macro_f1','ct_f1','t_f1','n','coverage']
    from research_report import table
    table(out,'paper_selected_results',best[cols]); evidence=(out/'tables/paper_selected_results.md').read_text(encoding='utf-8')
    stat=pd.read_csv(out/'statistical_tests.csv'); s60=stat[stat.t_s==60].iloc[0]
    fifth='The separate fifth-series case study is implemented only on Branch B. No case-study result is included on this main-experiment branch.'
    if args.include_fifth:
        b=ROOT/'results/fifth_sample_experiment'; bb=pd.read_csv(b/'best_model_by_time.csv'); br=bb[bb.t_s==60].iloc[0]
        import json
        sel=json.loads((b/'selected_fifth_sample.json').read_text())
        fifth=f"The deterministic fifth numeric series ID is {sel['selected_match_id']}. Its {sel['n_rounds']} rounds contain {sel['ct_rounds']} CT and {sel['t_rounds']} T wins. At 60 seconds, validation-selected {br['model']} ({br['feature_set']}) obtains accuracy {br['accuracy']:.4f} and Macro F1 {br['macro_f1']:.4f} on {int(br['n'])} live rounds. This single-series case study permits no population-level cluster confidence interval or significance claim. Its identity was not selected using its outcomes. It overlaps observations previously used by the main experiment, so it is not an untouched independent dataset. See the separate tables and figures in results/fifth_sample_experiment."
    text=f'''# Time-Spatial Machine Learning for Early Round Outcome Prediction in Counter-Strike 2

## Abstract

We study live-round outcome prediction at twelve cutoffs from 5 to 75 seconds after freeze time. A series-grouped evaluation preserves the existing seed-42 split and compares classical estimators, causal sequence models, exact spatial geometry and occupancy summaries. Under validation-only model and feature selection, the selected 60-second system is {r['model']} with {r['feature_set']} features: test accuracy {r['accuracy']:.4f}, Macro F1 {r['macro_f1']:.4f}, {int(r['n'])} rounds and coverage {r['coverage']:.3f}. The first selected-model Macro F1 above 0.80 is {earliest}. These results are exploratory because earlier project work had already inspected the same test partition. We publish identifiers, predictions, selection records and paired series-level inference rather than infer a fundamental prediction limit from these experiments.

## 1. Introduction and motivation

Round-winner probabilities can support retrospective tactical analysis and broadcast context. Prediction accuracy alone is insufficient: a model evaluated late in a round sees more information and a different surviving cohort. We examine both information timing and coverage while keeping related maps and rounds together.

## 2. Research questions and contributions

We ask whether Macro F1 exceeds 0.80 by 60 or 50 seconds, which validation-selected estimator performs best, whether spatial/temporal additions contribute, and whether differences survive series-level uncertainty. Contributions are an auditable rerun of the existing grouped benchmark, corrected exact-distance and occupancy representations, matched-cohort evaluation, and a separately documented fifth-series case study. Existing movement, combat and temporal features are acknowledged rather than presented as newly discovered data.

## 3. Related work

Xia, Salinas and Morstatter (2025) study round outcomes and player deaths using professional Dust2 telemetry and classical estimators. Hasegawa and Hirota (2024) use ESTA frame information and XGBoost to examine health, equipment and spatial context. Marshall, Mavromoustakos Blom and Spronck (2022) predict deaths over a three-second horizon, which differs from our round-level target. Liuhanen (2024) predicts match outcomes, not early round outcomes. Bibliographic identities of Xu Huang et al. (2022) and Paul et al. (2024) are verified, but inaccessible primary methods prevent a complete numerical comparison. Spearman (2018) and Shi et al. (2015) motivate spatial influence and temporal representations; neither establishes their usefulness in CS2. Primary links and verification limits appear in research/literature_review.md.

## 4. Dataset and problem formulation

The local OpenCS2 tables contain {len(manifest)} rounds in {manifest.match_id.nunique()} series and {manifest.map_id.nunique()} maps. A series is identified by match_id, while map_name and round complete the round key. The positive label y_ct_win=1 means CT wins; zero means T wins. We estimate P(CT wins | observations at or before t). A round is eligible only when round_end_tick exceeds freeze_end_tick + 64t. End-time information is used for evaluation eligibility, never as a predictor.

```
{summary.to_string()}
```

## 5. Feature engineering and spatial representation

The established baseline contains 63 metadata/combat/context and 34 player-state features. Event counts and observed bomb state are masked by the cutoff. Snapshots contain alive players at the exact cutoff; missing snapshots are retained and imputed from training statistics. Exact spatial features include 2D cross-team minimum and mean distances, within-team pair means, centroids and speed. Fixed 1024-unit world-coordinate cells yield occupancy entropy, occupied-cell counts and cross-team shared-cell fractions. These are planar geometry descriptors, not visibility, navigable distance or semantic map control. Temporal differences are divided by elapsed time and never reference a future snapshot.

## 6. Temporal and machine-learning methods

We compare Logistic Regression, Decision Tree, Random Forest, Extra Trees, linear SVM, KNN, Gaussian Naive Bayes, Gradient Boosting, histogram gradient boosting, XGBoost, LightGBM and CatBoost. Two prespecified configurations per algorithm are chosen on validation Macro F1. An additional original-configuration LightGBM serves as baseline. SVM margins are sigmoid-transformed for thresholding and ranking; they are not calibrated probabilities. LSTM, GRU and a causal dilated convolutional network consume key-aligned snapshot sequences ending at t. Their epoch cap is 20 with patience four on validation BCE. The convolutional model is a compact causal CNN without residual blocks; its implementation is not claimed to reproduce every TCN architecture. Transformer and larger nested searches remain unexecuted.

## 7. Experimental setup and splitting

The primary split uses the original shuffled series IDs and approximately 70/15/15 proportions, seed 42. All model families share exact assignments and eligible test rounds. Preprocessing is fitted on training only; hyperparameters, thresholds and early stopping use validation only. Each selected model is evaluated with threshold 0.5 and its frozen validation threshold. Selection files are written before test prediction. Comparisons across feature configurations include retuning, so their differences measure a configured system effect, not a pure fixed-parameter feature effect.

## 8. Evaluation metrics

Primary metrics are accuracy and Macro F1 = (CT F1 + T F1)/2, with both labels always included. Additional metrics include class F1, CT precision/recall, weighted F1, balanced accuracy, ROC-AUC, CT average precision, confusion counts and coverage. AUC measures are undefined for single-class cohorts and reported missing. Training, validation and test metrics remain separate. Natural coverage includes all live rounds at each cutoff; matched evaluation uses exactly the test rounds live at 75 seconds for every cutoff.

## 9. Main results and early prediction

The following table uses validation-selected models and features. Complete model comparisons and default-threshold results are in results/main_experiment.

{evidence}

![Test Macro F1](figures/main_macro_f1_vs_time.svg)

![Matched cohort](figures/main_matched_macro_f1.svg)

The selected-model 0.80 crossing is {earliest}. A point estimate crossing a target does not establish that its population value exceeds that target. Later natural-cohort scores also reflect which rounds survive; matched-cohort curves help describe this selection effect but do not remove all tactical heterogeneity.

## 10. Ablation, rankings and explainability

Feature ablation compares metadata, state, baseline, exact geometry, causal spatial changes, combined features and combat removal. Actual results are in tables/main_feature_ablation.md. Three training seeds are tested for LightGBM baseline and combined systems at 30/40/50/60/75 seconds; mean and standard deviation quantify only that limited seed sensitivity. Validation permutation importance and native LightGBM TreeSHAP are computed at these same cutoffs. Neither is causal evidence. Historical rankings are excluded because team-to-slot alignment and prior-match completion are insufficiently verified; no ranking benefit is claimed.

## 11. Map-wise analysis

Every prediction is stratified by map, preserving sample size and class balance. Separate LightGBM map-specific models are compared at selected cutoffs when training and validation have sufficient samples. Small map test cohorts should be treated descriptively. Pooled map-level tables and map-specific comparison files preserve separate training methodology. The planar distance representation cannot distinguish floors on Nuke or account for walls.

## 12. Statistical analysis

The primary paired comparison uses the original-configuration LightGBM on identical test keys. At 60 seconds the selected system's Macro F1 difference is {s60['difference']:.4f}, with 95% paired cluster bootstrap interval [{s60['ci_low']:.4f}, {s60['ci_high']:.4f}] and Holm-corrected permutation p={s60['corrected_p_value']:.4f}. Bootstrap uses 2,000 whole-series resamples; the permutation null swaps paired predictions by series using 5,000 draws. Holm adjustment covers the twelve selected-model cutoff comparisons. The intervals condition on fitted models and do not include selection uncertainty. See tables/main_statistical_tests.md.

## 13. Separate fifth-series holdout

{fifth}

## 14. Discussion and limitations

The results characterize these features, data and searches; they do not prove an intrinsic accuracy ceiling. Missing snapshot handling differs from older inner-join experiments, so historical numerical differences cannot all be attributed to model quality. The original combined tick feature cache is absent; baseline_reproduction.csv records measured differences using the available state cache. Timeline consistency supports 64 ticks per second but does not independently verify wall-clock sampling. Snapshot provenance relies partly on audited extraction code because cached rows omit source tick fields.

## 15. Threats to validity

Earlier researchers already examined this test set, making this a reproducible exploratory study rather than a fresh confirmatory evaluation. Grouping series prevents within-series cross-partition leakage but does not remove repeated-team dependencies or establish future-date generalization. Prior-round scores legitimately use observations available earlier in the same map. Date-based and team-held-out tests would strengthen external validity. Limited tuning, short temporal sequences, incomplete utility/economy fields and unverified historical identity mapping restrict the claims. There is no claim of publication acceptance.

## 16. Conclusion

The executed benchmark reports accuracy, class-balanced F1 and coverage under explicit grouped evaluation. Its saved predictions make model differences and uncertainty auditable. Additional data and independent prospective evaluation are needed before claiming reliable generalization or a fundamental limit on early round prediction.

## References

See references.bib and the source-linked verification matrix in research/literature_review.md. Only verified bibliographic identities are included in the new bibliography; missing full-text methods are explicitly acknowledged.
'''
    (paper/'main.md').write_text(text,encoding='utf-8')
    for name in ['macro_f1_vs_time','accuracy_vs_time','matched_macro_f1','ablation','confidence_intervals','observed_occupancy']:
        for ext in ['png','svg']: shutil.copyfile(out/'plots'/f'{name}.{ext}',paper/'figures'/f'main_{name}.{ext}')
    for name in ['paper_selected_results','feature_ablation','map_analysis','statistical_tests']:
        for ext in ['csv','md']:
            src=out/f'{name}.csv' if ext=='csv' else out/'tables'/f'{name}.md'
            shutil.copyfile(src,paper/'tables'/f'main_{name}.{ext}')
    (ROOT/'research/model_comparison.md').write_text((out/'evaluation_report.md').read_text(encoding='utf-8'),encoding='utf-8')
    (ROOT/'research/statistical_analysis.md').write_text('# Series-level statistical analysis\n\n'+(out/'tables/statistical_tests.md').read_text(encoding='utf-8')+'\n\nPaired series bootstrap, 2,000 draws; series-swap permutation, 5,000 draws; Holm adjustment across cutoff comparisons. Conditional on fitted models and exploratory due to prior test inspection. Feature-ablation comparisons are descriptive unless separately tested. No population inference from the single-series case study.\n',encoding='utf-8')

if __name__=='__main__': main()
