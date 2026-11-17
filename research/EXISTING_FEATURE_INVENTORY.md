# Existing Feature Inventory: CS2 Round Outcome Prediction

**Document Type:** Technical Feature Catalog & Representation Analysis  
**Repository:** `CS2-round-outcome-prediction`  
**Date:** November 2026  
**Author:** AI Research Specialist & Co-Author (Antigravity)  

---

## 1. Overview of Feature Generations

Across the lifecycle of the repository, multiple feature representations have been engineered and evaluated. This document provides a complete, authoritative inventory of all existing features, their exact mathematical formulations, input data sources, and empirical status.

| Generation / Stage | Count | Source | Primary Focus | Empirical Verdict |
|---|---|---|---|---|
| **Step 2 (Metadata Baseline)** | 63 | `kills.parquet`, `rounds.parquet`, `matches.parquet` | Pre-round context, cumulative kills, deaths, weapons, trades, opening duels, bomb plant | **Retained in Baseline** (Strongest signal) |
| **Step 4 (Player-State)** | 34 | Player snapshot at prediction tick | Team total HP, armor, active weapon classes, alive player counts | **Retained in Baseline** (+0.007 Macro F1 gain) |
| **Combined Baseline (Step 6)** | **97** | Steps 2 + 4 | Core benchmark model representation | **Official Benchmark Standard** |
| **Step 4 (Spatial Occupancy & Speed)** | 217 | 10x10 map grid & velocity | 100 CT + 100 T grid cells, alive density, horizontal speed | *Dropped* (Overlapped 0 in bootstrap CIs) |
| **Step 5 (Multi-Family Expansion)** | 105 | Snapshots + Kill feed | 7 families: Combat momentum, player advantage, firepower, bomb clock, etc. | *Dropped* (Within seed noise $\pm 0.005$) |
| **Step 7 (Sub-Second Dynamics)** | 108 | High-frequency ticks (0.5–2 s backward windows) | Micro-accelerations, state deltas, engagement indicators | *Dropped* (No measurable gain over static state) |
| **Legacy STOMs (`WP_DAC` / `ML_STOMs`)** | 200+ | `weapon_fire.parquet` | Map-wise 10x10 binary weapon fire occupancy per epoch | *Legacy* (Subsumed by modern pipeline) |

---

## 2. Core 97-Feature Baseline Inventory

The official benchmark model relies on 97 features strictly evaluated at prediction cutoff $t$:

### 2.1 Pre-Round Context (Step 2 — 3 Features)
Features known prior to round start; completely immune to intra-round leakage:
1. `map_index`: Sequential integer index of the map in the match series ($1, 2, 3$).
2. `match_date`: Encoded calendar/epoch timestamp of the match.
3. `event`: Tournament / competition event identifier.

### 2.2 Bomb Plant Features (Step 2 — 2 Features)
Computed using `rounds.parquet` masked by `prediction_tick = freeze_end_tick + round(t * 64)`:
4. `bomb_planted`: Binary indicator $\mathbb{I}(\text{bomb\_plant\_tick} \le \text{prediction\_tick})$.
5. `seconds_since_plant`: Elapsed time $(t - \text{bomb\_plant\_tick} / 64)$ if planted; else $0.0$.

### 2.3 Cumulative Kill & Combat Feed Features (Step 2 — 36 Features)
Aggregated over all kills occurring in $\text{rel\_tick} \le t \times 64$:
- **Kill & Death Counts:** `ct_kill`, `t_kill`, `ct_death`, `t_death`, `n_deaths`, `kill_diff` $(\text{ct\_kill} - \text{t\_kill})$.
- **Inferred Alive Counts:** `ct_alive` $(5 - \text{ct\_death})$, `t_alive` $(5 - \text{t\_death})$, `alive_diff` $(\text{ct\_alive} - \text{t\_alive})$.
- **Weapon-Specific Kills:**
  - Headshot kills: `ct_hs`, `t_hs`, `hs_total`.
  - AWP sniper kills: `ct_awp`, `t_awp`, `awp_total`.
  - Tactical kills: `ct_smoke`, `t_smoke`, `smoke_total` (through smoke).
  - Blind kills: `ct_blind`, `t_blind`, `blind_total` (attacker flashed).
  - Wallbang kills: `ct_wall`, `t_wall`, `wall_total` (penetration kills).
  - No-scope kills: `ct_noscope`, `t_noscope`.
- **Trade Kills:** `ct_trade`, `t_trade`, `trade_total`. A kill is classified as a trade if the killer's teammate was eliminated by the victim within a strict 5.0-second backward window ($320$ ticks).
- **Extreme Alive Advantage:** `max_alive_adv_ct` ($\max_{\tau \le t} (\text{alive\_diff}_\tau, 0)$), `max_alive_adv_t` ($\max_{\tau \le t} (-\text{alive\_diff}_\tau, 0)$).

### 2.4 Opening Duel Dynamics (Step 2 — 3 Features)
First kill event of the round (if occurred before cutoff $t$):
- `opening_occurred`: Binary indicator $\mathbb{I}(\exists \text{ kill with rel\_tick} \le t \times 64)$.
- `opening_side`: $+1$ if CT secured opening kill, $-1$ if T secured opening kill, $0$ if no kill yet.
- `opening_time_s`: Exact elapsed seconds $(\text{rel\_tick} / 64)$ of the opening kill ($t$ if none).

### 2.5 Clutch & Situational State (Step 2 — 3 Features)
Derived sequentially from the kill feed:
- `ever_1v1`: $\mathbb{I}(\text{ct\_alive} = 1 \land \text{t\_alive} = 1)$ at any point up to $t$.
- `ever_ct_clutch`: $\mathbb{I}(\text{ct\_alive} = 1 \land \text{t\_alive} \ge 2)$ at any point up to $t$.
- `ever_t_clutch`: $\mathbb{I}(\text{t\_alive} = 1 \land \text{ct\_alive} \ge 2)$ at any point up to $t$.

### 2.6 Map Identity One-Hot Encodings (Step 2 — 7 Features)
Indicator columns for the 7 active duty maps in the dataset:
- `map_de_ancient`, `map_de_anubis`, `map_de_dust2`, `map_de_inferno`, `map_de_mirage`, `map_de_nuke`, `map_de_overpass`.

### 2.7 Inferred Live State (Step 2 — 12 Features)
Cross-interaction and ratio terms:
- Proportions of team alive, normalized alive ratios, time elapsed since last observed kill event.

### 2.8 Player-State Snapshot Features (Step 4 — 34 Features)
Extracted directly at the exact prediction tick from player snapshot rows:
- **Health (HP) Aggregates:**
  - `ps_ct_health_total`: Sum of current HP across all alive CT players ($\in [0, 500]$).
  - `ps_t_health_total`: Sum of current HP across all alive T players ($\in [0, 500]$).
  - `ps_health_diff`: $\text{ps\_ct\_health\_total} - \text{ps\_t\_health\_total}$.
  - `ps_ct_health_mean`, `ps_t_health_mean`: Mean HP per alive player.
  - `ps_ct_health_min`, `ps_t_health_min`: Minimum HP among alive players.
- **Armor Aggregates:**
  - `ps_ct_armor_total`, `ps_t_armor_total`: Total armor value per team ($\in [0, 500]$).
  - `ps_armor_diff`: $\text{ps\_ct\_armor\_total} - \text{ps\_t\_armor\_total}$.
  - `ps_ct_has_armor_count`, `ps_t_has_armor_count`: Count of players with armor $> 0$.
  - `ps_ct_has_helmet_count`, `ps_t_has_helmet_count`: Count of players with helmet equipped.
- **Direct Alive Counts:**
  - `ps_ct_alive_count`, `ps_t_alive_count`: Direct alive count from tick snapshot.
  - `ps_alive_diff`: Direct snapshot alive count difference.
- **Active Weapon Class Counts per Side:**
  - Weapon categories: Rifle, Sniper, SMG, Heavy, Pistol, Knife, Utility.
  - Features: `ps_ct_wclass_rifle`, `ps_t_wclass_rifle`, `ps_ct_wclass_sniper`, `ps_t_wclass_sniper`, `ps_ct_wclass_smg`, `ps_t_wclass_smg`, `ps_ct_wclass_pistol`, `ps_t_wclass_pistol`, `ps_ct_wclass_utility`, `ps_t_wclass_utility`.
  - Differentials: `ps_wclass_rifle_diff`, `ps_wclass_sniper_diff`.

---

## 3. Previously Explored & Discarded Feature Families

### 3.1 Step 4 Spatial Occupancy Grid & Speed (217 Features)
- **Spatial Grid:** 10x10 uniform Cartesian grid per map based on $[q_{0.5\%}, q_{99.5\%}]$ coordinate bounds.
  - 100 binary occupancy indicators for CT: $\mathbb{I}(\text{at least 1 CT in cell }(i, j))$.
  - 100 binary occupancy indicators for T: $\mathbb{I}(\text{at least 1 T in cell }(i, j))$.
  - 8 regional density counts.
- **Movement / Speed:** Window-mean horizontal velocity $\sqrt{v_x^2 + v_y^2}$ over a 0.5 s backward window:
  - Moving players ($> 30\text{ units/s}$), stationary players ($< 5\text{ units/s}$), mean team speeds.
- **Ablation Finding:** Adding spatial occupancy to the 97-feature baseline shifted Macro F1 by $+0.0026$ at $t \le 60$ s, with 95% bootstrap confidence intervals consistently crossing zero. Spatial features in this raw grid form added no distinct predictive signal beyond what is already captured by player alive counts and health.

### 3.2 Step 5 Multi-Family Cumulative Expansion (105 Features)
Step 5 evaluated 7 distinct feature families added cumulatively on top of the 97 baseline features:
1. **Combat Momentum (`cm_`, 26 features):** Kill streak lengths, side of current streak, time since last kill per team, kill rate differential in the last 10 s vs. round average.
2. **Player Advantage (`pa_`, 16 features):** Normalized HP advantage, player survival probabilities, health-armor weighted score.
3. **Weapon & Firepower (`fw_`, 18 features):** A-priori weapon tier weights (AWP=4.5, Rifle=4.0, SMG=2.5, Pistol=1.0), team firepower sum and diff.
4. **Round State (`rs_`, 12 features):** Ratio of round duration elapsed, phase indicators.
5. **Bomb / Objective Timer (`bb_`, 10 features):** Exact bomb countdown timer, defuse window feasibility.
6. **Pre-round Live Interactions (`pr_`, 11 features):** Map-specific interaction terms with current alive differentials.
7. **State Changes (`sc_`, 12 features):** Snapshot-to-snapshot deltas ($\Delta\text{HP}_{t - \Delta t}$, $\Delta\text{Alive}_{t - \Delta t}$).
- **Ablation Finding:** Macro F1 deltas vs. baseline across all families were between $-0.002$ and $+0.001$, which was within the measured random seed noise floor of $\pm 0.005$. None of the 7 families demonstrated a statistically defensible improvement.

### 3.3 Step 7 Sub-Second Dynamics (108 Features)
- High-frequency backward look-back windows (0.5 s, 1.0 s, 2.0 s) capturing micro-acceleration, directional heading variance, and engagement proximity.
- **Ablation Finding:** 3-seed mean of Macro F1 difference vs. baseline:
  - Player state changes: $+0.0002$
  - Movement dynamics: $-0.0013$
  - Micro-spatial: $+0.0006$
  - Micro-engagements: $-0.0007$
  - All combined: $-0.0010$
- Concluded that sub-second noise does not generalize to macro round outcomes.

---

## 4. Legacy STOM Features (`WP_DAC` & `ML_STOMs`)

The legacy notebooks in `ML_STOMs` and `WP_DAC` generated features exclusively from `weapon_fire.parquet`:
- **Relative Time:** $t=0$ was set to the *first weapon fire event of the round*, rather than `freeze_end_tick`.
- **Epoch Bins:** $0, 10, 20, 30, 40, 50, 60$ s and `Round End`.
- **Occupancy Mapping:** 10x10 binary matrices of weapon fire origin coordinates $(x, y)$, tracking cumulative fired cells and newly fired cells per epoch.
- **Limitation Identified:** Firing only occurs when players engage or spam choke points; passive holding, rotators, and lurking players generate zero signal. This directly explains why the professor recommended: *"Different features in STOM except firing"*.

---

## 5. Identified Opportunities for Genuinely New Features

Based on this audit, previous feature engineering stalled because it relied on either:
1. Coarse 10x10 binary grids that lost geometric precision, or
2. High-frequency sub-second noise that overfitted.

The genuinely promising, untested feature directions are:
1. **Continuous Spatial Density & Clustering:**
   - Alive player spatial centroids: $(\bar{x}_{CT}, \bar{y}_{CT})$, $(\bar{x}_{T}, \bar{y}_{T})$.
   - Spatial dispersion / spread: team standard deviations $(\sigma_{x}, \sigma_{y})$ measuring team cohesion vs. split setups.
   - Inter-team Euclidean distance: distance between CT centroid and T centroid.
   - Minimum inter-team player distance (proximity to closest enemy).
2. **Bomb Site Territory Control:**
   - Distance of team centroids to designated Bomb Site A and Bomb Site B centroids.
   - Number of alive CTs / Ts within defensive radii of sites A and B.
3. **Historical Player / Team Strength Ratings:**
   - Pre-match team win rates and map-specific win rates strictly computed from previous matches (chronological, no lookahead).
4. **Utility & Grenade Statistics:**
   - OpenCS2 contains `index/smoke_flash_stats.parquet` (smoke/flash/HE grenade events) and weapon metadata.
