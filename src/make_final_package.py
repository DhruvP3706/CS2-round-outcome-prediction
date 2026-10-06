"""Format ALREADY-COMPLETED results (Steps 1-7) into results/final/. Runs no models and downloads nothing."""
import base64
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "final"
OUT.mkdir(exist_ok=True)
HIGH = {30, 40, 50, 60, 75, 90, 105, 120}
TIMES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 90, 105, 120]

# ------------------------------------------------------------------ PART 1: final curve (Step 6 selected config + 97 features, seed 42)
cfg = json.load(open(R / "step6/selected_config.json"))
P = cfg["params"]
assert (P["num_leaves"], P["max_depth"], P["min_child_samples"], P["learning_rate"], P["feature_fraction"],
        P["bagging_fraction"], P["lambda_l2"], P["lambda_l1"]) == (4, 3, 80, 0.03, 0.8, 0.8, 2.0, 0.0)
s6 = pd.read_csv(R / "step6/final_results.csv")
assert (s6.config == cfg["selected"]).all() and (s6.seed == 42).all() and list(s6.t_s) == TIMES
F = pd.DataFrame({"time_s": s6.t_s, "macro_f1": s6.test_macro_f1, "ct_f1": s6.test_ct_f1, "t_f1": s6.test_t_f1,
                  "eligible_test_rounds": s6.n_test, "coverage": s6.coverage_test, "validation_threshold": s6.val_threshold})
F.to_csv(OUT / "f1_vs_time.csv", index=False)

fig, (a, b) = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True, gridspec_kw={"height_ratios": [3.2, 1.1], "hspace": 0.08})
a.plot(F.time_s, F.ct_f1, "--", color="#9aa4b2", lw=1, marker="^", ms=4, label="CT-class F1")
a.plot(F.time_s, F.t_f1, "--", color="#c9a26b", lw=1, marker="v", ms=4, label="T-class F1")
a.plot(F.time_s, F.macro_f1, "-", color="#1f4e9c", lw=2.4, marker="o", ms=7, mec="white", mew=1, label="Macro F1 (primary)", zorder=3)
for x, y in zip(F.time_s, F.macro_f1):
    a.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=7.5, color="#1f4e9c")
a.axhline(0.80, color="#c0392b", ls=":", lw=1.5); a.text(5, 0.806, "F1 = 0.80", color="#c0392b", fontsize=9)
a.set_ylim(0.55, 0.96); a.set_ylabel("F1 (test)"); a.grid(alpha=0.25); a.legend(loc="lower right", fontsize=8.5, frameon=False)
a.set_title("CS2 Round Outcome Prediction — Macro F1 vs Prediction Time", fontsize=13, fontweight="bold", pad=22)
a.text(0.5, 1.02, "Test macro F1; thresholds selected on validation", transform=a.transAxes, ha="center", fontsize=9.5, color="#444")
b.bar(F.time_s, F.eligible_test_rounds, width=3.4, color="#8fa6c9")
for x, n, c in zip(F.time_s, F.eligible_test_rounds, F.coverage):
    b.text(x, n + 40, f"{c:.0%}", ha="center", fontsize=7)
b.set_ylabel("eligible\ntest rounds"); b.set_ylim(0, 3100); b.grid(alpha=0.25, axis="y")
b.set_xlabel("Prediction time (seconds after freeze end)  — labels: test coverage"); b.set_xticks(TIMES)
fig.text(0.01, 0.005, "LightGBM (4 leaves, depth 3, min_child 80), 97 features (Step 2 metadata + Step 4 player state), seed 42. Single series-level split; no confidence intervals computed.", fontsize=7, color="#555")
fig.savefig(OUT / "f1_vs_time.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ supporting tables
s2 = pd.read_csv(R / "step2/results.csv"); s2l = s2[s2.model == "lightgbm"].set_index("t_s"); s2r = s2[s2.model == "logreg"].set_index("t_s")
s4f = pd.read_csv(R / "step4_full/results.csv"); s4p = pd.read_csv(R / "step4/results.csv")
s5 = pd.read_csv(R / "step5/results.csv"); s5n = pd.read_csv(R / "step5/seed_noise.csv")
s6t = pd.read_csv(R / "step6/tuning_summary.csv"); s6tt = pd.read_csv(R / "step6/tuning_results.csv"); s6s = pd.read_csv(R / "step6/seed_robustness.csv")
s7 = pd.read_csv(R / "step7/ablation_results.csv"); s7s = pd.read_csv(R / "step7/ablation_seed_summary.csv")
elig = pd.read_csv(R / "step1/eligibility_summary.csv").set_index("t_s"); cb = pd.read_csv(R / "step1/class_balance_by_map.csv")
vr = pd.read_csv(R / "step1/validation_report.csv"); split = pd.read_csv(R / "step2/split_sizes.csv")
leak2 = pd.read_csv(R / "step2/leakage_checks.csv"); leak5 = pd.read_csv(R / "step5/leakage_checks.csv"); leak7 = pd.read_csv(R / "step7/leakage_checks.csv")
meta_dir = ROOT / "data/opencs2/metadata"
cnt = {n: len(pd.read_parquet(meta_dir / f"{n}.parquet", columns=[pd.read_parquet(meta_dir / f"{n}.parquet").columns[0]])) for n in ["matches", "rounds", "round_player", "kills", "pov_rounds"]}

best_lt60 = F[F.time_s < 60].sort_values("macro_f1").iloc[-1]
f_at = F.set_index("time_s")
cross = F[F.macro_f1 >= 0.80].iloc[0]
top = F.sort_values("macro_f1").iloc[-1]

def fmt(x, d=3):
    return f"{x:.{d}f}"

md = f"""# CS2 Round Outcome Prediction — Final Results Summary

## Final model (official)
- **Feature set (97):** Step 2 metadata features (63) + Step 4 player-state features (34). No Step 5, Step 7, spatial-occupancy or movement features.
- **Model:** LightGBM (`{cfg['selected']}`, chosen in Step 6 by mean *validation* macro F1 over 30/40/50/60/75 s; test never used for selection).
- **Parameters (from `results/step6/selected_config.json`):** num_leaves={P['num_leaves']}, max_depth={P['max_depth']}, min_child_samples={P['min_child_samples']}, learning_rate={P['learning_rate']}, feature_fraction={P['feature_fraction']}, bagging_fraction={P['bagging_fraction']} (bagging every iteration), lambda_l1={P['lambda_l1']}, lambda_l2={P['lambda_l2']}; n_estimators cap {cfg['n_estimators_cap']} with early stopping (100 rounds) on validation log-loss; seed 42.
- **Evaluation:** one separate model per prediction time; decision threshold chosen on validation to maximise macro F1 and frozen; test evaluated once. Positive class = CT win.
- **Split (series-level, seed 42, 70/15/15):** train 251 series / 555 maps / 11,576 rounds; validation 54 / 119 / 2,448; test 54 / 120 / 2,503 (16,527 rounds, 794 maps, 359 series in total).

## Results (test, all 15 prediction times)
| time (s) | macro F1 | CT F1 | T F1 | eligible test rounds | coverage | val. threshold |
|---|---|---|---|---|---|---|
""" + "\n".join(f"| {int(r.time_s)} | {r.macro_f1:.4f} | {r.ct_f1:.4f} | {r.t_f1:.4f} | {int(r.eligible_test_rounds)} | {r.coverage:.3f} | {r.validation_threshold:.3f} |" for r in F.itertuples()) + f"""

## Key numbers
- Best final macro F1 **below 60 s:** {best_lt60.macro_f1:.4f} (at {int(best_lt60.time_s)} s, coverage {best_lt60.coverage:.1%}).
- F1 at 60 s: **{f_at.loc[60,'macro_f1']:.4f}** (coverage {f_at.loc[60,'coverage']:.1%}); 75 s: **{f_at.loc[75,'macro_f1']:.4f}** ({f_at.loc[75,'coverage']:.1%}); 90 s: **{f_at.loc[90,'macro_f1']:.4f}** ({f_at.loc[90,'coverage']:.1%}); 105 s: **{f_at.loc[105,'macro_f1']:.4f}** ({f_at.loc[105,'coverage']:.1%}); 120 s: **{f_at.loc[120,'macro_f1']:.4f}** ({f_at.loc[120,'coverage']:.1%}).
- **First time macro F1 >= 0.80: {int(cross.time_s)} s** ({cross.macro_f1:.4f}; coverage {cross.coverage:.1%}). 0.80 is not reached at 60 s ({f_at.loc[60,'macro_f1']:.3f}) or 75 s ({f_at.loc[75,'macro_f1']:.3f}).
- Highest value on the curve: {top.macro_f1:.4f} at {int(top.time_s)} s, on only {int(top.eligible_test_rounds)} test rounds ({top.coverage:.1%} coverage) — not comparable to early-time results.
- Coverage at important times: 30 s {f_at.loc[30,'coverage']:.1%}, 40 s {f_at.loc[40,'coverage']:.1%}, 50 s {f_at.loc[50,'coverage']:.1%}, 60 s {f_at.loc[60,'coverage']:.1%}, 75 s {f_at.loc[75,'coverage']:.1%}, 90 s {f_at.loc[90,'coverage']:.1%}, 105 s {f_at.loc[105,'coverage']:.1%}, 120 s {f_at.loc[120,'coverage']:.1%}.

## Why late-time F1 must be read cautiously
- Coverage falls steeply (46% of test rounds at 90 s, 28% at 105 s, 12% at 120 s): only rounds that are still live at time t are eligible, so the evaluated population changes with t and is not a random sample of rounds.
- Class balance shifts: in the test split the CT win rate of eligible rounds falls from {s2l.loc[60,'test_ct_rate']:.2f} at 60 s to {s2l.loc[90,'test_ct_rate']:.2f} (90 s), {s2l.loc[105,'test_ct_rate']:.2f} (105 s) and {s2l.loc[120,'test_ct_rate']:.2f} (120 s).
- Small samples (1,157 / 704 / 309 rounds at 90 / 105 / 120 s; training sets of 4,963 / 3,101 / 1,321 rounds) make F1 volatile; the Step 2 bootstrap interval of the (different) Step 2 LightGBM at 120 s spans {s2l.loc[120,'test_macro_f1_ci_lo']:.2f}–{s2l.loc[120,'test_macro_f1_ci_hi']:.2f}. The 105 s > 120 s non-monotonicity is within that noise.
- The curve has no confidence intervals (none were computed for the final configuration). Seed-to-seed spread of ~0.003–0.008 was observed at 30–75 s in Steps 5–6.

## Exploratory results (NOT the final model)
Steps 5 and 7 produced isolated test values slightly above the final curve at single times (e.g. Step 5 set C at 60 s: 0.787; Step 7 set C at 40 s: 0.736 vs 0.729). These were within seed noise, were not validated, and do not replace the Step 6 configuration.
"""
(OUT / "final_results_summary.md").write_text(md)

# ------------------------------------------------------------------ PART 2: PDF (HTML -> headless Chrome)
def b64(p):
    return "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()

def table(df, hl=None, cls="", colfmt=None):
    colfmt = colfmt or {}
    h = "<table class='%s'><thead><tr>%s</tr></thead><tbody>" % (cls, "".join(f"<th>{c}</th>" for c in df.columns))
    for _, r in df.iterrows():
        k = " class='hl'" if hl and hl(r) else ""
        h += f"<tr{k}>" + "".join(f"<td>{colfmt.get(c, lambda v: v)(r[c])}</td>" for c in df.columns) + "</tr>"
    return h + "</tbody></table>"

f3 = lambda v: f"{v:.3f}"
f1 = lambda v: f"{v:.1%}"
def fig(path, cap, w="100%"):
    return f"<figure><img src='{b64(path)}' style='width:{w}'><figcaption>{cap}</figcaption></figure>"

# tables
T_final = pd.DataFrame({"Time (s)": F.time_s, "Macro F1": F.macro_f1, "CT F1": F.ct_f1, "T F1": F.t_f1, "Eligible test rounds": F.eligible_test_rounds,
                        "Coverage": F.coverage, "Val. threshold": F.validation_threshold})
tfinal = table(T_final, hl=lambda r: int(r["Time (s)"]) in HIGH, colfmt={"Time (s)": lambda v: f"{int(v)}", "Macro F1": f3, "CT F1": f3, "T F1": f3, "Coverage": f1, "Val. threshold": f3, "Eligible test rounds": lambda v: f"{int(v):,}"})
T_s2 = pd.DataFrame({"Time (s)": TIMES, "Logistic reg.": [s2r.loc[t, "test_macro_f1"] for t in TIMES], "LightGBM": [s2l.loc[t, "test_macro_f1"] for t in TIMES],
                     "LightGBM 95% CI": [f"{s2l.loc[t,'test_macro_f1_ci_lo']:.3f}–{s2l.loc[t,'test_macro_f1_ci_hi']:.3f}" for t in TIMES],
                     "Val. thr. (LGBM)": [s2l.loc[t, "val_threshold"] for t in TIMES], "Test CT rate": [s2l.loc[t, "test_ct_rate"] for t in TIMES]})
tS2 = table(T_s2, hl=lambda r: int(r["Time (s)"]) in {30, 40, 50, 60, 75, 90}, colfmt={"Time (s)": lambda v: f"{int(v)}", "Logistic reg.": f3, "LightGBM": f3, "Val. thr. (LGBM)": f3, "Test CT rate": f3})
T_el = pd.DataFrame({"Time (s)": TIMES, "Eligible rounds (all splits)": [int(elig.loc[t, "n_eligible"]) for t in TIMES], "Coverage": [elig.loc[t, "coverage"] for t in TIMES],
                     "CT win rate": [elig.loc[t, "ct_win_rate"] for t in TIMES], "Maps with eligible rounds": [int(elig.loc[t, "n_maps_eligible"]) for t in TIMES]})
tEl = table(T_el, colfmt={"Time (s)": lambda v: f"{int(v)}", "Coverage": f1, "CT win rate": f3, "Eligible rounds (all splits)": lambda v: f"{v:,}"})
sp = split.copy(); sp["ct_win_rate"] = sp.ct_win_rate.map(f3); sp["split"] = sp.split.str.capitalize()
tsp = table(sp.rename(columns={"split": "Split", "n_series": "Series", "n_rounds": "Rounds", "ct_win_rate": "CT win rate", "n_maps": "Maps"})[["Split", "Series", "Maps", "Rounds", "CT win rate"]], colfmt={"Rounds": lambda v: f"{int(v):,}"})
tmap = table(cb.assign(map_name=cb.map_name.str[3:].str.capitalize(), mean=cb["mean"].map(f3)).rename(columns={"map_name": "Map", "size": "Rounds", "mean": "CT win rate"}), colfmt={"Rounds": lambda v: f"{int(v):,}"})
# Step 4 full
fo = ["A_meta", "B_meta+player", "C_meta+player+spatial", "D_meta+player+spatial+move"]
p4 = s4f.pivot(index="t_s", columns="model", values="test_macro_f1")[fo]; p4.insert(0, "Step 2 LightGBM", s2l.test_macro_f1)
p4 = p4.loc[[30, 40, 50, 60, 75, 90]].reset_index().rename(columns={"t_s": "Time (s)", "A_meta": "A: meta (63)", "B_meta+player": "B: +player state (97)", "C_meta+player+spatial": "C: +spatial (306)", "D_meta+player+spatial+move": "D: +movement (314)"})
t4 = table(p4, colfmt={**{c: f3 for c in p4.columns if c != "Time (s)"}, "Time (s)": lambda v: f"{int(v)}"})
cmp4 = pd.read_csv(R / "step4_full/comparison.csv"); m4 = cmp4.assign(le60=cmp4.t_s <= 60).groupby(["comparison", "le60"]).delta.mean().unstack()
# Step 5
p5 = s5.pivot(index="t_s", columns="set", values="test_macro_f1").loc[[20, 30, 40, 50, 60, 75]].reset_index().rename(columns={"t_s": "Time (s)"})
t5 = table(p5, colfmt={**{c: f3 for c in p5.columns if c != "Time (s)"}, "Time (s)": lambda v: f"{int(v)}"})
a5 = s5[(s5.set != "A") & (s5.t_s <= 75)].groupby("set").delta_vs_A.mean()
# Step 6
t6 = table(s6t.assign(config=s6t.config).rename(columns={"config": "Configuration", "mean_val_macro_f1": "Mean val. macro F1", "mean_val_logloss": "Mean val. log-loss", "mean_val_auc": "Mean val. AUC"}), colfmt={c: (lambda v: f"{v:.4f}") for c in ["Mean val. macro F1", "Mean val. log-loss", "Mean val. AUC"]}, hl=lambda r: r["Configuration"] == cfg["selected"])
sr = s6s.groupby(["config", "t_s"]).test_macro_f1.agg(["mean", "min", "max"]).unstack(0)
t6b = pd.DataFrame({"Time (s)": [30, 40, 50, 60, 75], "Step 4 B baseline cfg (seed 42)": [s6.baseline_test_macro_f1[s6.t_s == t].iloc[0] for t in [30, 40, 50, 60, 75]],
                    "Selected cfg (seed 42)": [f_at.loc[t, "macro_f1"] for t in [30, 40, 50, 60, 75]],
                    "Δ seed 42": [s6.delta_vs_baseline[s6.t_s == t].iloc[0] for t in [30, 40, 50, 60, 75]],
                    "Δ 95% CI": [f"{s6.delta_lo[s6.t_s==t].iloc[0]:+.3f} … {s6.delta_hi[s6.t_s==t].iloc[0]:+.3f}" for t in [30, 40, 50, 60, 75]],
                    "Baseline, 3 seeds mean (min–max)": [f"{sr[('mean','c00_baseline')][t]:.3f} ({sr[('min','c00_baseline')][t]:.3f}–{sr[('max','c00_baseline')][t]:.3f})" for t in [30, 40, 50, 60, 75]],
                    "Selected, 3 seeds mean (min–max)": [f"{sr[('mean',cfg['selected'])][t]:.3f} ({sr[('min',cfg['selected'])][t]:.3f}–{sr[('max',cfg['selected'])][t]:.3f})" for t in [30, 40, 50, 60, 75]]})
t6c = table(t6b, colfmt={"Time (s)": lambda v: f"{int(v)}", "Step 4 B baseline cfg (seed 42)": f3, "Selected cfg (seed 42)": f3, "Δ seed 42": lambda v: f"{v:+.3f}"})
# Step 7
q = s7[s7.seed == 42].pivot(index="t_s", columns="set", values="test_macro_f1")
d7 = s7[s7.seed == 42].pivot(index="t_s", columns="set", values="test_delta_vs_A").drop(columns="A")
v7 = s7[s7.seed == 42].pivot(index="t_s", columns="set", values="val_delta_vs_A").drop(columns="A")
t7 = table(q.reset_index().rename(columns={"t_s": "Time (s)"}), colfmt={**{c: f3 for c in q.columns}, "Time (s)": lambda v: f"{int(v)}"})
t7d = table(d7.reset_index().rename(columns={"t_s": "Time (s)"}), colfmt={**{c: (lambda v: f"{v:+.3f}") for c in d7.columns}, "Time (s)": lambda v: f"{int(v)}"})
t7v = table(v7.reset_index().rename(columns={"t_s": "Time (s)"}), colfmt={**{c: (lambda v: f"{v:+.3f}") for c in v7.columns}, "Time (s)": lambda v: f"{int(v)}"})
noise5 = s5n.groupby(["set", "t_s"]).test_macro_f1.agg(["min", "max"])
# progression table
prog = pd.DataFrame({"Time (s)": [30, 40, 50, 60, 75, 90], "Step 2 LightGBM (63 feat., 15 leaves)": [s2l.loc[t, "test_macro_f1"] for t in [30, 40, 50, 60, 75, 90]],
                     "Step 4 B (97 feat., 15 leaves)": [s4f[s4f.model == "B_meta+player"].set_index("t_s").test_macro_f1[t] for t in [30, 40, 50, 60, 75, 90]],
                     "FINAL: Step 6 (97 feat., 4 leaves)": [f_at.loc[t, "macro_f1"] for t in [30, 40, 50, 60, 75, 90]]})
tprog = table(prog, colfmt={**{c: f3 for c in prog.columns if c != "Time (s)"}, "Time (s)": lambda v: f"{int(v)}"})
nwarn = int((vr.status == "WARN").sum()); npass = int((vr.status == "PASS").sum())
leak_tbl = lambda d, c: table(d[c])
step3 = (R / "step3/probe_results.md").read_text()

css = """
@page { size: A4; margin: 22mm 18mm 22mm 18mm; @bottom-center { content: counter(page); font: 9pt Helvetica, Arial, sans-serif; color:#666 } }
@page :first { @bottom-center { content: '' } }
body { font: 10.3pt/1.42 'Helvetica Neue', Helvetica, Arial, sans-serif; color:#1b1f24; }
h1 { font-size: 17pt; color:#17407f; border-bottom: 2px solid #17407f; padding-bottom:3pt; margin-top: 0; page-break-before: always; }
h1.nb { page-break-before: avoid }
h2 { font-size: 12pt; color:#17407f; margin: 14pt 0 4pt }
p { margin: 5pt 0; text-align: justify } li { margin: 2pt 0 }
table { border-collapse: collapse; margin: 8pt 0; font-size: 8.8pt; width: 100%; page-break-inside: avoid }
th { background:#17407f; color:white; padding: 3.5pt 5pt; text-align:center; font-weight:600 }
td { padding: 2.8pt 5pt; border-bottom: 1px solid #d8dde4; text-align:center } tr:nth-child(even) td { background:#f5f7fa }
tr.hl td { background:#fff3cd !important; font-weight:600 }
figure { margin: 8pt 0; text-align:center; page-break-inside: avoid } figcaption { font-size:8.5pt; color:#555; margin-top:3pt }
.title { text-align:center; padding-top: 55mm } .title h0 { display:block; font-size: 30pt; font-weight:700; color:#17407f }
.sub { font-size: 15pt; color:#444; margin: 10pt 0 26pt } .meta { display:inline-block; text-align:left; font-size: 10.5pt; margin-top: 12pt; line-height:1.7; border-top: 2px solid #17407f; padding-top: 12pt }
.box { border-left: 4px solid #17407f; background:#eef3fb; padding: 6pt 10pt; margin: 8pt 0 } .warn { border-left-color:#c0392b; background:#fdf0ee }
.pipe { display:flex; flex-direction:column; align-items:center; margin: 10pt 0 } .pipe div.b { border:1.5px solid #17407f; border-radius:6px; padding:5pt 18pt; background:#eef3fb; min-width: 230pt; text-align:center; font-weight:600; font-size:9.5pt }
.pipe div.b.f { background:#17407f; color:white } .pipe div.a { color:#17407f; font-size:14pt; line-height:12pt }
code { font: 9pt Menlo, Consolas, monospace; background:#f0f2f5; padding: 0 2pt } .toc li { list-style:none; margin:1.5pt 0 } .toc ul { padding-left: 4pt }
"""
steps = ["Executive Summary", "Problem Definition", "Dataset", "Data Validation and Temporal Alignment", "Train/Validation/Test Split", "Step 2 — Metadata Baseline",
         "Step 3 — Tick Data Reconnaissance", "Step 4 — Player-State and Spatial Features", "Step 5 — Feature Engineering and Ablation", "Step 6 — LightGBM Tuning",
         "Step 7 — Sub-Second Dynamics", "Final Pipeline", "Final Model", "Final Results", "Interpretation of the Curve", "What Worked", "What Did Not Work",
         "Limitations", "Conclusion", "Future Work"]
toc = "".join(f"<li>{i + 2}. {s}</li>" for i, s in enumerate(steps))
pipe = "".join(f"<div class='b{' f' if i in (9, 10) else ''}'>{x}</div>" + ("<div class='a'>↓</div>" if i < 10 else "") for i, x in enumerate(
    ["OpenCS2 dataset (metadata + map-level ticks)", "Dataset validation (Step 1)", "Round table + eligibility", "Temporal alignment: t = 0 at freeze_end_tick, 64 Hz",
     "Series-level train / validation / test split", "Metadata features (63) — kill feed, bomb, pre-round", "Player-state features (34) from ticks at the prediction tick",
     "97-feature final representation", "Selected LightGBM (Step 6: 4 leaves, depth 3, min_child 80)", "Validation threshold tuning (per prediction time)", "Frozen test evaluation → final F1-vs-time curve"]))

html = f"""<html><head><meta charset='utf-8'><style>{css}</style></head><body>
<div class='title'><h0>Counter-Strike 2 Round Outcome Prediction</h0><div class='sub'>Temporal Round-Win Prediction Using Classical Machine Learning</div>
<div class='meta'><b>Dataset:</b> OpenCS2 (metadata + map-level tick files, 7 maps, 16,527 rounds)<br><b>Task:</b> predict the round winner (CT vs T) at 15 prediction times, 5–120 s after freeze end<br>
<b>Primary metric:</b> test macro F1 (thresholds tuned on validation only)<br><b>Final model:</b> LightGBM (4 leaves, depth 3, min_child_samples 80) on 97 features (Step 2 metadata + Step 4 player state)<br>
<b>Scope:</b> classical machine learning only; no deep learning, video or audio</div></div>

<h1>Contents</h1><ul class='toc'>{toc}</ul>
<div class='box'><b>Reading guide.</b> Results are kept in three distinct classes: <b>exploratory/pilot</b> (Step 4 pilot, subsets of training data), <b>ablation</b> (Steps 4–5 and 7, feature-family comparisons) and the <b>final defensible result</b> (Step 6 selected configuration on the 97-feature set, Section 15). Only the last is the official F1-vs-time curve.</div>

<h1>2. Executive Summary</h1>
<p><b>Problem.</b> Given what is observable up to <i>t</i> seconds after the freeze period ends, predict which side (CT or T) will win a Counter-Strike 2 round. Because information accumulates as the round plays out, performance is reported as a function of prediction time at 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 90, 105 and 120 seconds.</p>
<p><b>Data and approach.</b> The OpenCS2 dataset (16,527 rounds, 794 map instances, 359 series, 7 maps) was used with a strictly series-level train/validation/test split. Features come from the kill feed, bomb events and pre-round context (63 metadata features) and from the 64 Hz map-level tick data (34 player-state features: alive count, HP, armor and active-weapon class per side). Models are classical (logistic regression, LightGBM); the decision threshold is tuned on validation only and frozen before a single test evaluation.</p>
<p><b>Stages.</b> (1) dataset validation and temporal alignment; (2) metadata baseline; (3) tick-data reconnaissance; (4) player-state, spatial and movement features; (5) seven families of hand-crafted features with cumulative ablation; (6) small validation-driven LightGBM tuning; (7) sub-second temporal/spatial dynamics. Steps 5 and 7 produced no reliable improvement and Step 6 only a marginal one.</p>
<div class='box'><b>Final model.</b> LightGBM (num_leaves 4, max_depth 3, min_child_samples 80, seed 42) on the 97-feature representation (Step 2 metadata + Step 4 player state), selected by validation performance.<br>
<b>Final curve.</b> Test macro F1 rises from {f_at.loc[5,'macro_f1']:.3f} at 5 s to {f_at.loc[30,'macro_f1']:.3f} at 30 s, {f_at.loc[50,'macro_f1']:.3f} at 50 s, {f_at.loc[60,'macro_f1']:.3f} at 60 s and {f_at.loc[75,'macro_f1']:.3f} at 75 s. <b>The 0.80 level is first reached at {int(cross.time_s)} s ({cross.macro_f1:.3f}, test coverage {cross.coverage:.0%}); it is not reached before 60 s or before 75 s.</b> The best F1 below 60 s is {best_lt60.macro_f1:.3f} (at {int(best_lt60.time_s)} s).</div>
<p><b>Conclusion.</b> Round-state information up to 75 s supports macro F1 of about 0.71–0.80 with this representation; richer hand-crafted or short-window features did not help beyond seed noise. Late-time values (90 s and later) are computed on a shrinking, differently balanced subset of rounds and should not be read as equivalent to early-round performance.</p>

<h1>3. Problem Definition</h1>
<p><b>Input.</b> For each round and prediction time <i>t</i>, only information available at <code>prediction_tick = freeze_end_tick + t·64</code> (the dataset runs at 64 ticks per second): kill-feed events with tick ≤ prediction_tick, the bomb-plant event if already occurred, pre-round context (map, round number, prior score, loss streaks) and, from Step 4 on, the player state at the prediction tick.</p>
<p><b>Target.</b> Binary round winner; the positive class is <b>CT win</b> (label <code>y_ct_win</code>). Overall CT win rate is {elig.loc[5,'ct_win_rate']:.3f}.</p>
<p><b>Metric.</b> Primary: <b>macro F1</b> (unweighted mean of CT-class and T-class F1) on the test split. CT F1 and T F1 are reported as secondary metrics. The decision threshold is selected on validation to maximise macro F1 and frozen before the test evaluation.</p>
<p><b>Why multiple prediction times.</b> A round's outcome is far more predictable late than early; a single time would hide this. Evaluating 15 times describes how quickly predictability grows, and which times are realistically useful. For each time only rounds still live at that time are eligible (<code>playable_duration &gt; t</code>), so the eligible set shrinks with t (Section 5).</p>

<h1>4. Dataset</h1>
<p>The OpenCS2 dataset provides, per round, match metadata, kill events, per-player round statistics and (optionally) per-tick state, video and audio. Used here:</p>
{table(pd.DataFrame({'Item':['Rounds','Map instances','Series (match_id)','Maps'],'Value':['16,527','794','359','7: Dust2, Mirage, Ancient, Nuke, Inferno, Overpass, Anubis']}))}
{table(pd.DataFrame({'File':['matches.parquet','rounds.parquet','round_player.parquet','kills.parquet','pov_rounds.parquet'],'Rows':[f"{cnt[k]:,}" for k in ['matches','rounds','round_player','kills','pov_rounds']],
 'Use':['one row per map instance: map, date, event, map index','one row per round: tick boundaries, winner, outcome fields (outcome fields never used as features)','per-player final round statistics (never used as features)','one row per kill: tick, sides, slots, weapon, flags (used after masking by tick ≤ t)','per-player-per-round POV index incl. tick-file paths (used for slot→side and size/probe planning)']}))}
<p><b>Canonical key.</b> <code>(match_id, map_name, round)</code> identifies a round. A series (<code>match_id</code>) can contain several maps.</p>
<p>Rounds per map and CT win rate:</p>{tmap}
<p><b>Excluded modalities.</b> Video and audio were deliberately excluded (scope: classical ML on structured data). Per-POV tick files were probed only (Step 3); all tick-derived features come from the map-level tick files.</p>

<h1>5. Data Validation and Temporal Alignment</h1>
<p><b>Step 1</b> built a canonical round table and ran {len(vr)} validation checks: {npass} PASS, {nwarn} WARN, 0 errors. The warnings were documented and handled rather than ignored:</p>
<ul><li><b>3,456 rounds</b> have a freeze period &gt; 25 s (timeouts/pauses). The prediction clock is anchored on freeze end, so these do not distort it.</li>
<li><b>703 kills occur after round end</b> (post-round C4/world deaths): all kill features are masked by tick ≤ prediction tick, so they never enter.</li>
<li><b>3,024 round_player rows</b> with <code>player_slot = -1</code> (pseudo-player) were dropped.</li></ul>
<p><b>Temporal alignment.</b> t = 0 is <code>freeze_end_tick</code>, the start of playable time, at 64 Hz. The playable duration is <code>(round_end_tick − freeze_end_tick)/64</code>. A round is eligible at t if <code>playable_duration &gt; t</code>, so the eligible sets are nested (checked). <code>prediction_tick = freeze_end_tick + t·64</code> is the last tick whose data may be used.</p>
<p><b>Why freeze_end_tick.</b> The playable clock is the only reference that is comparable across rounds with very different freeze/pause lengths, and the tick files (Step 3) are aligned to it (every player's rows start exactly at freeze_end_tick). The earlier STOM-based alignment is not documented in the project result files, so its specific shortcomings are <i>not recorded</i> here.</p>
<p><b>Leakage control.</b> Outcome columns (winner, reason, bomb outcome, kills totals, survivors, etc.) are listed in <code>OUTCOME_COLUMNS</code> and asserted absent from the feature frame; event-level data is used only after masking to tick ≤ prediction tick; pre-round score/streak use prior rounds only. A look-ahead test (Step 2) corrupted all post-cutoff kills and plant events and confirmed that features at t were unchanged at t = 10/30/60/90 s.</p>
<h2>Eligibility by prediction time (all splits)</h2>{tEl}

<h1>6. Train / Validation / Test Split</h1>
<p>Splitting is at the <b>series</b> level (<code>match_id</code>): series ids are sorted, shuffled with seed 42 and divided 70/15/15. All maps of a series stay in the same split, because maps from the same series share teams, players and context; splitting by round or map would leak this and inflate test scores.</p>{tsp}
<p>Total: 359 series, 794 maps, 16,527 rounds. The split is fixed for all experiments. There is a single primary split; no cross-validation was performed.</p>

<h1>7. Step 2 — Metadata Baseline</h1>
<p><b>63 features</b> available from the kill feed, bomb events and pre-round context: <b>44 kill-feed features</b> (kills/deaths/headshots/AWP/smoke/blind/wallbang/no-scope/trade kills per side, alive counts and difference, opening-kill side and time, recent 5 s/10 s windows, 1v1/clutch states, time since last kill), <b>3 bomb features</b> (planted, time since plant, plant time) and <b>16 pre-round features</b> (map one-hot, map index, prior score and difference, loss streaks, pistol/second-half/overtime flags). All event features use only events with tick ≤ prediction tick.</p>
<p><b>Models.</b> Standardised logistic regression and LightGBM (15 leaves, learning rate 0.03, min_child_samples 40, subsample/colsample 0.8, λ2 = 2, early stopping on validation log-loss). One model per prediction time; threshold from a 0.05–0.95 grid on validation (ties toward 0.5); test evaluated once; 95% intervals by series-level bootstrap.</p>
{tS2}
<p>Macro F1 rises from about 0.62 at 5 s to 0.70 at 30 s and 0.77 at 60 s (LightGBM). LightGBM is slightly better than logistic regression at most times. {fig(ROOT/'plots/step2_f1_vs_time.png','Figure 7.1 — Step 2 metadata baseline (test macro F1 and eligible test rounds).','82%')}</p>

<h1>8. Step 3 — Tick Data Reconnaissance</h1>
<p>Step 3 probed two tick routes without bulk download:</p>
<ul><li><b>Per-POV tick files:</b> 165,270 files, ≈26.6 GB (sidecar sizes); each holds one player's POV (tick, buttons, view angles pitch/yaw, input weapon, team, alive, health, armor, x/y/z, velocity x/y/z).</li>
<li><b>Map-level tick files</b> (blanchon/opencs2_dataset_wds): <b>794 files, 18.09 GB</b> (median 22.4 MB), ≈<b>629 M tick rows</b> (estimated from a 975,502-row probe file); each file holds <b>all 10 players</b> with a <code>player_slot</code> column, team (2 = T, 3 = CT), alive state, health, armor, active weapon, position, velocity and view angles, at 64 Hz, 1-tick spacing, aligned to freeze_end_tick.</li></ul>
<p><b>Why map-level files.</b> They contain all 10 players in one file (no per-player joins), are smaller in total than the per-POV route, and allow column projection by Parquet. Verified in the probe: each (round, player) starts at freeze_end_tick, spacing is contiguous, the first alive=False tick equals the player's kill tick (100% of 162 deaths), and the number of alive players at prediction ticks equals 10 − kills(tick ≤ t) in all 67/67 checks. Raw ticks were processed map by map and never retained; only compact snapshots were kept.</p>

<h1>9. Step 4 — Player-State and Spatial Features</h1>
<p>Feature groups built from the map-level ticks at the prediction tick (nothing after it):</p>
<ul><li><b>Player state (34):</b> per side alive count, total and mean HP, total and mean armor, differences; counts of players by coarse active-weapon class (rifle, sniper, SMG, heavy, pistol, knife, utility) per side and their differences.</li>
<li><b>Spatial (209):</b> 10×10 occupancy grids of alive players for CT and for T (200 count features) plus 9 summaries (centroids, centroid distance, spreads, nearest and mean CT–T distance).</li>
<li><b>Movement (8):</b> mean/max speed, moving and stationary player counts per side over a 0.5 s backward window.</li></ul>
<p><b>Map-specific bounds.</b> For each map, x and y bounds are the 0.5th and 99.5th percentiles of alive-player positions sampled once per second in <i>training</i> files, padded by 2% of the range; positions outside are clipped. The same bounds apply to all rounds of a map (the bounds table is in <code>results/step4_full/map_bounds.csv</code>).</p>
<p><b>Pilot (exploratory).</b> 421 of 794 map files (9.59 GB: all validation and test series + one third of training series) were processed first. Comparisons used a one-third training set and gave small, mostly non-significant gains (player state ≈ +0.01 on average for t ≤ 60 s); spatial features alone were slightly negative. These pilot numbers are exploratory.</p>
<p><b>Full data (ablation).</b> All 794 files (18.09 GB) were processed; tick-feature coverage of eligible rounds was 100% (train/test) and 99.99% (validation), and tick-derived alive counts matched the kill feed in 100% of rounds. Test macro F1, LightGBM (15 leaves), full training set:</p>{t4}
<p>Mean paired ΔF1: player state (B − A) {m4.loc['B_vs_A', True]:+.4f} for t ≤ 60 s and {m4.loc['B_vs_A', False]:+.4f} for t &gt; 60 s; spatial over player state (C − B) {m4.loc['C_vs_B', True]:+.4f} (t ≤ 60 s); movement over spatial (D − C) {m4.loc['D_vs_C', True]:+.4f} (t ≤ 60 s). Only player state showed a fairly consistent (still small) gain, significant at 20 s and 25 s; the 200-cell occupancy grid and movement features added no reliable signal and were dropped. {fig(ROOT/'plots/step4_full_f1_vs_time.png','Figure 9.1 — Step 4 full-data comparison (test macro F1; lower panel: ΔF1 vs A with 95% series-bootstrap CI).','80%')}</p>

<h1>10. Step 5 — Feature Engineering and Ablation</h1>
<p>Seven hand-crafted families (105 new features) were built from the kill feed and Step 4 snapshots, all backward-looking:</p>
<ul><li><b>Combat momentum (36):</b> kill/headshot/AWP/trade/special-kill differentials; 5/10/15/20 s window kills and deaths; kill-rate momentum; time since last CT/T kill; streak side/length/start.</li>
<li><b>Player advantage (14):</b> absolute alive advantage, HP/armor ratios, low-/high-/full-HP counts, alive×HP and alive×armor interactions.</li>
<li><b>Weapon/firepower (7):</b> fixed a-priori firepower score per weapon class (rifle 4, sniper 4.5, SMG/heavy 2.5, pistol 1, knife 0, utility 0.5), differences, averages.</li>
<li><b>Round state (13):</b> opening-kill timing × alive advantage, state codes (3v3, 2v2, 1v1, clutch-like), min/total alive.</li>
<li><b>Bomb/time (6):</b> estimated time to explosion (40 s timer assumed), planted × alive/HP/firepower interactions.</li>
<li><b>Pre-round × live (13):</b> score, loss streak, round number, pistol and half indicators and map × alive difference.</li>
<li><b>State changes (16):</b> changes vs the previous prediction snapshot and deaths in the last 1/2 s.</li></ul>
<p><b>Ablation.</b> Cumulative sets A (97 features) → B (+combat momentum) → C (+player advantage) → D (+weapon) → E (+round state) → F (+bomb/time) → G (+pre-round × live) → H (+state changes; 202 features), same LightGBM configuration, same split.</p>{t5}
<p>Mean ΔF1 vs A for t ≤ 75 s: {', '.join(f'{k} {v:+.3f}' for k, v in a5.items())}. Re-running A with two other seeds changed test F1 by ±0.003–0.007 at 20–75 s, as large as any observed difference. Only isolated increments had bootstrap intervals excluding zero (e.g. player advantage at 20 s +0.011) and others were significantly <i>negative</i> (e.g. weapon/firepower at 20 s −0.010).</p>
<p><b>Redundancy.</b> 34 near-duplicate pairs (|Spearman| &gt; 0.98 at 50 s) were found; many new features restate existing ones (for example cm_kill_diff = kill_diff, the 5/10 s kill windows = recent5/10 kills, total alive = n_deaths). The snapshots only exist at the prediction ticks, so the requested 0.5/1/2 s state-change features could not be built in this step (done in Step 7).</p>
<p><b>Leakage test.</b> Post-cutoff kills, plants and snapshots were scrambled for T = 10, 30, 60, 90 s; all features at t ≤ T were unchanged. Because no family improved results beyond noise, <b>all seven were dropped</b>, and nothing from Step 5 is in the final model. {fig(ROOT/'plots/step5_f1_vs_time.png','Figure 10.1 — Step 5 cumulative ablation (exploratory; heat map shows ΔF1×100 vs A).','82%')}</p>

<h1>11. Step 6 — LightGBM Tuning</h1>
<p>The question was whether the configuration limited performance. Using exactly the 97 features, 13 configurations (baseline plus 12 variations of num_leaves, max_depth, min_child_samples, learning rate, feature/bagging fraction, λ1, λ2) were trained at 30, 40, 50, 60 and 75 s on the training set and scored on <b>validation only</b>. <b>Selection rule (fixed in advance): highest mean validation macro F1 over the five times, seed 42.</b> Test data was not used in any model call during the search. The baseline configuration was first reproduced exactly (maximum |difference| to Step 4 = 0.0000 at all 15 times).</p>{t6}
<p>The whole range of mean validation F1 is 0.0060; the selected configuration (num_leaves 4, max_depth 3, min_child_samples 80) scores {cfg['mean_val_macro_f1']:.4f} against {cfg['baseline_mean_val_macro_f1']:.4f} for the baseline (+{cfg['mean_val_macro_f1']-cfg['baseline_mean_val_macro_f1']:.4f}).</p>
<div class='box'><b>Selected configuration (results/step6/selected_config.json):</b> num_leaves = {P['num_leaves']}, max_depth = {P['max_depth']}, min_child_samples = {P['min_child_samples']}, learning_rate = {P['learning_rate']}, feature_fraction = {P['feature_fraction']}, bagging_fraction = {P['bagging_fraction']}, lambda_l1 = {P['lambda_l1']}, lambda_l2 = {P['lambda_l2']}, seed = 42; n_estimators cap {cfg['n_estimators_cap']}, early stopping 100 rounds on validation log-loss.</div>
<p>Test comparison against the previous Step 4 configuration (three seeds at the focus times):</p>{t6c}
<p>Gains are +0.000 to +0.012 (seed 42) and +0.001 to +0.007 (seed means) — at or below the baseline's own seed spread (0.003–0.008); only 30 s has a bootstrap interval excluding zero. <b>Tuning therefore produced only marginal gains, and the model configuration was not the main bottleneck.</b> The lower-capacity configuration is, if anything, slightly more stable.</p>

<h1>12. Step 7 — Sub-Second Dynamics</h1>
<p>Because earlier snapshots exist only at the prediction ticks, Step 7 re-streamed the 794 map-level tick files (column projection, nothing stored) and extracted backward-only windows of <b>0.5 s, 1 s and 2 s</b> ending exactly at the prediction tick, at 30/40/50/60/75 s only (67,739 round-time rows). Four groups were added to the 97-feature baseline A:</p>
<ul><li><b>B — player-state changes (30):</b> HP lost, players hurt, max HP loss, deaths, armor lost per side and window.</li>
<li><b>C — movement dynamics (42):</b> mean/max speed, speed variance, moving players, displacement, centroid shift.</li>
<li><b>D — spatial dynamics (22):</b> spread change, nearest-enemy and pair-distance changes, minimum CT–T distance, pair counts within 500/1000/2000 units (no occupancy grid).</li>
<li><b>E — engagement dynamics (14):</b> kills per side and kill differential in the windows, engagement flags, time since last kill and last damage.</li></ul>
<p>Fixed Step 6 configuration, seed 42 (plus seeds 1 and 2 for noise); F = A + B + C + D + E (205 features). Test macro F1 and ΔF1 vs A:</p>{t7}
<p><b>Test ΔF1 vs A</b></p>{t7d}<p><b>Validation ΔF1 vs A</b></p>{t7v}
<p>No family reaches +0.01 at any time (largest: C +0.008 at 40 s, reversing to −0.006 at 30 s and −0.005 at 60 s); no test bootstrap interval excludes zero; validation does not support the apparent gains. <b>The direction was abandoned</b>: in this representation, short-term dynamics add no information beyond the 97-feature state. Leakage test: scrambling all tick rows and kill events after T = 30/50/60 s left every feature at t ≤ T unchanged and windows always end exactly at the prediction tick. {fig(ROOT/'plots/step7_dynamics_pilot.png','Figure 12.1 — Step 7 pilot (exploratory).','92%')}</p>

<h1>13. Final Pipeline</h1>
<div class='pipe'>{pipe}</div>

<h1>14. Final Model</h1>
<div class='box'><b>The final reported model uses the 97-feature representation consisting of Step 2 metadata features plus Step 4 player-state features, together with the LightGBM configuration selected in Step 6.</b></div>
<ul><li><b>Features (97):</b> 63 Step 2 metadata features (44 kill-feed, 3 bomb, 16 pre-round) + 34 Step 4 player-state features.</li>
<li><b>Not included:</b> the 200-cell occupancy grid and spatial summaries, movement features (Step 4), all Step 5 families, all Step 7 dynamics.</li>
<li><b>LightGBM:</b> objective binary; num_leaves {P['num_leaves']}; max_depth {P['max_depth']}; min_child_samples {P['min_child_samples']}; learning_rate {P['learning_rate']}; feature_fraction {P['feature_fraction']}; bagging_fraction {P['bagging_fraction']} (bagging_freq 1); lambda_l1 {P['lambda_l1']}; lambda_l2 {P['lambda_l2']}; n_estimators cap {cfg['n_estimators_cap']} with early stopping (100 rounds, validation log-loss); random_state 42.</li>
<li><b>Training/evaluation:</b> a separate model for each of the 15 prediction times, trained on eligible training rounds; threshold from validation macro-F1 maximisation, frozen; one test evaluation.</li></ul>
<p>Steps 5 and 7 were exploratory: they did not improve the representation beyond seed noise and therefore did not replace it. The final model was chosen by validation performance (Step 6) — not by any test score and not by the highest score at any single time.</p>

<h1>15. Final Results</h1>
<p><i>Official results: Step 6 selected configuration + 97 features, seed 42.</i></p>
{fig(OUT/'f1_vs_time.png','Figure 15.1 — Final test macro F1 vs prediction time (thresholds selected on validation). Lower panel: eligible test rounds and test coverage.','100%')}
{tfinal}
<p>Highlighted rows: 30, 40, 50, 60, 75, 90, 105 and 120 s.</p>
<ul><li><b>Best final F1 below 60 s:</b> {best_lt60.macro_f1:.3f} (at {int(best_lt60.time_s)} s, coverage {best_lt60.coverage:.0%}).</li>
<li><b>F1 at 60 s:</b> {f_at.loc[60,'macro_f1']:.3f} (coverage {f_at.loc[60,'coverage']:.0%}). <b>F1 at 75 s:</b> {f_at.loc[75,'macro_f1']:.3f} (coverage {f_at.loc[75,'coverage']:.0%}).</li>
<li><b>First crossing of 0.80:</b> {int(cross.time_s)} s ({cross.macro_f1:.3f}; coverage {cross.coverage:.0%}).</li>
<li><b>Highest reported F1:</b> {top.macro_f1:.3f} at {int(top.time_s)} s (only {int(top.eligible_test_rounds)} test rounds, coverage {top.coverage:.0%}).</li></ul>
<h2>Progression across steps (test macro F1)</h2>{tprog}
<p>The Step 6 curve improves on the Step 2 and Step 4 B configurations by ≤ 0.012 at these times. Step 2 and Step 4 B figures are reproduced from their own result files.</p>
<p>No confidence intervals were computed for the final configuration. As an indication of uncertainty, the Step 2 LightGBM (a different model) had series-bootstrap 95% intervals about ±0.015–0.02 wide at 30–75 s, and seed-to-seed spread was 0.003–0.008.</p>

<h1>16. Interpretation of the Curve</h1>
<ul><li><b>Early prediction is hard.</b> At 5–15 s macro F1 is only {f_at.loc[5,'macro_f1']:.2f}–{f_at.loc[15,'macro_f1']:.2f}: few events have happened and the pre-round context carries little information.</li>
<li><b>Information accumulates.</b> F1 rises steadily as kills, HP/armor/weapon state and bomb events accumulate: {f_at.loc[30,'macro_f1']:.3f} at 30 s, {f_at.loc[50,'macro_f1']:.3f} at 50 s, {f_at.loc[75,'macro_f1']:.3f} at 75 s.</li>
<li><b>Late-time results rest on fewer rounds.</b> Coverage is {f_at.loc[60,'coverage']:.0%} at 60 s, {f_at.loc[75,'coverage']:.0%} at 75 s, {f_at.loc[90,'coverage']:.0%} at 90 s, {f_at.loc[105,'coverage']:.0%} at 105 s and {f_at.loc[120,'coverage']:.0%} at 120 s (309 test rounds).</li>
<li><b>Class balance shifts.</b> The eligible CT win rate (all splits) falls from {elig.loc[60,'ct_win_rate']:.2f} at 60 s to {elig.loc[90,'ct_win_rate']:.2f}, {elig.loc[105,'ct_win_rate']:.2f} and {elig.loc[120,'ct_win_rate']:.2f} at 90, 105 and 120 s (test split: {s2l.loc[90,'test_ct_rate']:.2f}, {s2l.loc[105,'test_ct_rate']:.2f}, {s2l.loc[120,'test_ct_rate']:.2f}).</li>
<li><b>Caution at 90 s and later.</b> The subsets of long rounds are selected populations with different class balance and small training sets (4,963 / 3,101 / 1,321 rounds at 90 / 105 / 120 s); F1 there should not be treated as directly equivalent to early-round performance. The drop from 105 s ({f_at.loc[105,'macro_f1']:.3f}) to 120 s ({f_at.loc[120,'macro_f1']:.3f}) is within the sampling noise at n = 309. Why late rounds differ (e.g. role of the bomb state) was not analysed.</li></ul>

<h1>17. What Worked</h1>
<ul><li><b>Temporal alignment</b> on freeze_end_tick with strict tick ≤ t masking; the nested eligibility sets.</li>
<li><b>Leakage controls:</b> outcome-column exclusion and look-ahead corruption tests (Steps 2, 5, 7) — all passed.</li>
<li><b>Series-level split</b>, preventing same-series leakage.</li>
<li><b>Metadata features</b> deliver the bulk of the signal (Step 2: 0.62 → 0.77 at 5 → 60 s).</li>
<li><b>Player state</b> (alive/HP/armor/weapon class) gave the most consistent additional gain (≈ +0.008 mean for t ≤ 60 s, significant at 20–25 s).</li>
<li><b>LightGBM</b> with small trees, with validation-only early stopping and thresholds; the shallow Step 6 configuration was marginally better and a little more stable.</li>
<li><b>Validation threshold selection</b> kept thresholds near 0.5 (0.49–0.54 at 30–60 s).</li></ul>

<h1>18. What Did Not Work</h1>
<ul><li><b>200-cell spatial occupancy grid</b> — Step 4 full data: spatial over player state {m4.loc['C_vs_B', True]:+.4f} mean ΔF1 (t ≤ 60 s), no consistently significant time; likely too many sparse features for ~4k–11k rounds.</li>
<li><b>Movement features</b> — D − C {m4.loc['D_vs_C', True]:+.4f} (t ≤ 60 s).</li>
<li><b>Hand-crafted aggregate families</b> (Step 5) — mean ΔF1 vs A of −0.002…+0.001 for t ≤ 75 s, below the seed noise; many features duplicated existing ones.</li>
<li><b>Sub-second dynamics</b> (Step 7) — test ΔF1 between −0.006 and +0.008, none reaching +0.01 and none supported by validation.</li>
<li><b>Extensive tuning</b> — 13 configurations spanned only 0.006 in validation F1; test gains were within noise.</li></ul>
<p>All these conclusions rest on one series-level split and single-seed or three-seed runs, so they are evidence of "no reliable gain", not proof of absence of any effect.</p>

<h1>19. Limitations</h1>
<ul><li><b>Single primary series-level split;</b> no repeated splits or cross-validation. Test uncertainty is reflected only in bootstrap intervals computed for earlier experiments, not for the final curve.</li>
<li><b>Seed noise</b> of ≈ ±0.003–0.008 limits the resolution of all comparisons.</li>
<li><b>Late-time coverage</b> is low (46% / 28% / 12% at 90 / 105 / 120 s) and sample sizes small.</li>
<li><b>Class-balance shift</b> across times; F1 is not comparable across times without this in mind.</li>
<li><b>Intrinsic round randomness:</b> a part of the outcome is determined by events after the prediction time; how much was not quantified.</li>
<li><b>Classical ML scope:</b> no sequence or deep-learning models; weapon information is coarse (7 active-weapon classes, no AWP/shotgun/C4 split); positions are 2-D only.</li>
<li><b>No oracle / predictability-ceiling experiment</b> was run, so the 0.80 target is not shown to be unreachable and nothing in this report claims it is.</li>
<li><b>Minor row-count differences:</b> a few rounds (≤ 2 validation rounds at some times, 1 training round at some times) lacked tick snapshots and were dropped in the 97-feature experiments; split sizes quoted in Section 6 are those of the full round table.</li>
<li>The Step 6 results contain no confidence intervals for absolute F1; the final curve has none.</li></ul>

<h1>20. Conclusion</h1>
<p>With ONE consistent configuration — LightGBM (4 leaves, depth 3, min_child_samples 80) on the 97-feature representation of Step 2 metadata plus Step 4 player state — test macro F1 grows from {f_at.loc[5,'macro_f1']:.3f} at 5 s to {f_at.loc[30,'macro_f1']:.3f} at 30 s, {f_at.loc[50,'macro_f1']:.3f} at 50 s, {f_at.loc[60,'macro_f1']:.3f} at 60 s and {f_at.loc[75,'macro_f1']:.3f} at 75 s, and first reaches 0.80 at {int(cross.time_s)} s ({cross.macro_f1:.3f}, {cross.coverage:.0%} of test rounds eligible).</p>
<ul><li><b>Was 0.80 achieved before 60 s?</b> No (best below 60 s: {best_lt60.macro_f1:.3f}, at {int(best_lt60.time_s)} s).</li>
<li><b>Before 75 s?</b> No (60 s: {f_at.loc[60,'macro_f1']:.3f}; 75 s: {f_at.loc[75,'macro_f1']:.3f}, just under).</li>
<li><b>Useful representation:</b> kill-feed/bomb/pre-round metadata plus a compact per-side player-state summary. Large spatial grids, movement features, hand-crafted aggregates and sub-second windows did not help beyond noise.</li>
<li><b>What the experiments suggest:</b> with these feature sources and classical models, predictability seems to saturate near 0.76 at 50 s and 0.78–0.80 at 60–75 s; remaining gains are unlikely to come from more of the same features or from model tuning. Whether the limit lies in the information or in the model class was not determined.</li></ul>

<h1>21. Future Work</h1>
<p><i>None of the following was performed in this project.</i></p>
<ul><li><b>Predictability ceiling / oracle analysis</b> to estimate how much of the remaining error is intrinsic round randomness.</li>
<li><b>Richer tactical representations</b> (e.g. bomb-site and zone control, utility/grenade state, economy and buy information, finer weapon identity).</li>
<li><b>Sequence models</b> over the player-state trajectory.</li>
<li><b>Stronger spatial representations</b> (map-aware zones/graphs rather than raw occupancy grids).</li>
<li><b>Alternative temporal architectures</b> and repeated/cross-validated series-level evaluation with confidence intervals for the final curve.</li></ul>
</body></html>"""

tmp = Path(tempfile.mkdtemp(prefix="cs2report_", dir="/tmp/claude-1005/-home-dnp-Desktop-Additional/a4b172b4-eee7-4c8d-b840-5a32bd050c42/scratchpad"))
(tmp / "r.html").write_text(html)
subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={OUT/'CS2_Round_Outcome_Prediction_Final_Report.pdf'}", f"file://{tmp/'r.html'}"], check=True, capture_output=True)
print("done", sorted(p.name for p in OUT.iterdir()))
