# Feature inventory

The exact columns used in each new experiment are saved in `results/main_experiment/feature_sets.json`; missingness and near-duplicate correlations are computed only on training observations.

| Family | Existing implementation | New investigation |
|---|---|---|
| Metadata/context | Map indicators, map index, prior scores, loss streaks, pistol/half/overtime | Retained; metadata-only ablation |
| Combat | Cumulative kills/deaths, opening kill, trades, recent kills, special kills, clutch states | Retained; combat-removal sensitivity |
| Player state | 34 alive/health/armor/weapon-class summaries | Retained; state-only ablation |
| Bomb | Observed plant state and elapsed plant time | Retained; final plant tick never directly enters model |
| Spatial | Step 4 exact pair distances and training-bound occupancy; later master code approximates proximity | Exact opposing/team pair distances; fixed 1024-unit cell occupancy entropy and cell count; opposing-player shared-cell fraction |
| Movement | Backward half-second speed and previous short-window experiments | Current alive-player mean speed; causal per-second changes |
| Temporal | Previous snapshots and sub-second windows already investigated | Per-second changes of exact geometry and state; key-aligned LSTM/GRU/causal CNN sequences |
| Utility | Smoke/blind kill context and active utility weapon class | No reliable full grenade trajectory or map-region utility event data in the local feature inputs |
| Ranking | Historical map-win-rate code | Excluded: team identity and completion timing not adequately established |

Fixed world-coordinate cells require no learned map bounds and are safe under both splits. They do not represent semantic sites or territory control. Pairwise geometry is 2D; occupied cells reflect observed alive players. Missing teams/snapshots remain missing, with train-only imputation. Feature importance is association, not causation.

New `geometry`, `temporal_spatial`, and `combined` configurations are evaluated alongside the 97-column baseline at all 12 cutoffs. Previous failed short-window families are not rerun wholesale; the reason for this investigation is corrected exact distance semantics, occupancy summaries, cohort retention and explicit selection auditability.

Exact pair distances already existed in Step 4. Their reuse corrects the later master implementation, not a novelty claim. Fixed-world cell entropy/shared-cell fractions and per-second changes are the new representation tested here.
