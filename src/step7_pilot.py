"""Step 7b: leakage checks + pilot ablation A..F at 30/40/50/60/75 s with the fixed Step 6 LightGBM config (c02), seed 42 (+ seeds 1,2 for noise)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import f1_score, log_loss, roc_auc_score  # noqa: E402

from cs2pred.config import LABEL_COL, PROCESSED_DIR, ROUND_KEY, TICK_RATE  # noqa: E402
from cs2pred.features import kill_flags, load_kills  # noqa: E402
from cs2pred.modeling import tune_threshold  # noqa: E402
from cs2pred.step7_features import GROUP_PREFIX, RAW_COLS, engagement_features, extract_dynamics  # noqa: E402
from step6_tune import CONFIGS, fit, load, mf1  # noqa: E402
from step7_extract import FOCUS, jobs, read_ticks  # noqa: E402

RES = ROOT / "results" / "step7"
CFG = CONFIGS["c02_nl4_md3_mcs80"]
SETS = {"A": [], "B": ["B"], "C": ["C"], "D": ["D"], "E": ["E"], "F": ["B", "C", "D", "E"]}
DESC = {"A": "baseline 97", "B": "+player-state changes", "C": "+movement dynamics", "D": "+spatial dynamics", "E": "+engagement dynamics", "F": "+B+C+D+E"}


def leakage(table, kills):
    flags = kill_flags(kills, table)
    js, _ = jobs()
    pick = [js[i] for i in (0, 150, 300, 450, 600, 750)]
    rows = []
    rng = np.random.default_rng(0)
    raws = []
    for mid, mp, path in pick:
        raws.append((mid, mp, read_ticks(path), table[(table.match_id == mid) & (table.map_name == mp)].set_index("round")))
    for T in (30, 50, 60):
        times = [t for t in FOCUS if t <= T]
        n_cmp, n_bad, nscr, bad_cols = 0, 0, 0, set()
        for mid, mp, tb, rt in raws:
            fe = rt.freeze_end_tick.reindex(tb["round"]).to_numpy()
            fut = (tb.tick.to_numpy() > fe + T * TICK_RATE)
            tc = tb.copy()
            for c in ["team_num", "is_alive", "health", "armor_value", "x", "y", "velocity_x", "velocity_y", "player_slot"]:
                tc.loc[fut, c] = rng.permutation(tc.loc[fut, c].to_numpy())
            nscr += int(fut.sum())
            a, b = extract_dynamics(tb, rt, times), extract_dynamics(tc, rt, times)
            cols = [c for c in a.columns if c not in ("round", "t_s")]
            ok = len(a) == len(b) and (a[["round", "t_s"]].values == b[["round", "t_s"]].values).all()
            same = ok and np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True)
            n_cmp += len(a); n_bad += 0 if same else len(a)
            if not same:
                bad_cols |= {c for c in cols if not ok or not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True)}
            assert a._win_last_rel.max() == 0 and a._win_first_rel.min() >= -128     # windows end exactly at pred, never cross it
        # kill-feed engagement features
        k = kills.copy()
        fe = table.set_index(ROUND_KEY).freeze_end_tick.reindex(pd.MultiIndex.from_frame(k[ROUND_KEY])).to_numpy()
        fk = k.tick.to_numpy() > fe + T * TICK_RATE
        for c in ["attacker_side", "victim_side", "weapon", "headshot", "through_smoke", "wallbang", "attacker_blind", "noscope"]:
            k.loc[fk, c] = rng.permutation(k.loc[fk, c].to_numpy())
        f2 = kill_flags(k, table)
        eng_ok = all(np.allclose(*[x.sort_values(ROUND_KEY).drop(columns=ROUND_KEY).to_numpy(float) for x in (engagement_features(flags, t), engagement_features(f2, t))], equal_nan=True) for t in times)
        rows.append({"T_s": T, "times_checked": str(times), "tick_rows_scrambled": nscr, "kill_events_scrambled": int(fk.sum()),
                     "files_checked": len(raws), "round_time_rows_compared": n_cmp, "tick_features_unchanged": n_bad == 0,
                     "engagement_features_unchanged": bool(eng_ok), "windows_end_at_pred": True, "changed_cols": ";".join(sorted(bad_cols)[:5])})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main():
    RES.mkdir(parents=True, exist_ok=True); (ROOT / "plots").mkdir(exist_ok=True)
    table = pd.read_parquet(PROCESSED_DIR / "round_table.parquet")
    kills = load_kills()
    leak = leakage(table, kills); leak.to_csv(RES / "leakage_checks.csv", index=False)
    assert leak.tick_features_unchanged.all() and leak.engagement_features_unchanged.all()

    df, feats, n_test_total = load()
    df = df[df.prediction_time_s.isin(FOCUS)]
    dyn = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "data/features/step7/tickdyn").glob("*.parquet"))], ignore_index=True)
    dyn = dyn.drop(columns=["_win_last_rel", "_win_first_rel"]).rename(columns={"t_s": "prediction_time_s"})
    flags = kill_flags(kills, table)
    eng = pd.concat([engagement_features(flags, t).assign(prediction_time_s=t) for t in FOCUS])
    n0 = len(df)
    df = df.merge(dyn, on=ROUND_KEY + ["prediction_time_s"], how="inner").merge(eng, on=ROUND_KEY + ["prediction_time_s"], how="left")
    ec = [c for c in eng.columns if c.startswith("dyn_e_") and "since" not in c]
    df[ec] = df[ec].fillna(0)
    df["dyn_e_since_last_kill_s"] = df["dyn_e_since_last_kill_s"].fillna(df.prediction_time_s)
    print(f"rows with step7 features {len(df)}/{n0}", flush=True)
    groups = {g: [c for c in df.columns if c.startswith(p)] for g, p in GROUP_PREFIX.items()}
    sets = {k: feats + [c for g in v for c in groups[g]] for k, v in SETS.items()}
    q = pd.DataFrame([{"group": g, "feature": c, "nan_frac": df[c].isna().mean(), "n_unique": df[c].nunique(), "n_inf": int(np.isinf(df[c].astype(float)).sum())} for g, cs in groups.items() for c in cs])
    q.to_csv(RES / "feature_quality.csv", index=False)
    (RES / "feature_groups.txt").write_text("\n".join(f"[{g}] ({len(c)})\n" + "\n".join(c) for g, c in groups.items()) +
                                             "\n\nSets: " + ", ".join(f"{k}={len(v)}" for k, v in sets.items()) +
                                             "\nWindows: w05=32 ticks, w10=64, w20=128, all ending exactly at pred tick; survivors = alive at window start and at pred.\n")
    print({g: len(c) for g, c in groups.items()}, {k: len(v) for k, v in sets.items()}, flush=True)

    rows, rng = [], np.random.default_rng(0)
    for t in FOCUS:
        d = df[df.prediction_time_s == t]
        P = {s: d[d.split == s] for s in ("train", "val", "test")}
        y = {s: P[s][LABEL_COL].astype(int).to_numpy() for s in P}
        res = {}
        for k, cols in sets.items():
            for seed in (42, 1, 2):
                X = {s: P[s][cols].astype(float) for s in P}
                m = fit(CFG, seed, X["train"], y["train"], X["val"], y["val"])
                pva, pte = m.predict_proba(X["val"])[:, 1], m.predict_proba(X["test"])[:, 1]
                thr, vf1 = tune_threshold(y["val"], pva)
                pred = (pte >= thr).astype(int)
                res[(k, seed)] = (pva, pte, thr)
                rows.append({"set": k, "seed": seed, "t_s": t, "n_features": len(cols), "n_test": len(y["test"]), "coverage_test": len(y["test"]) / n_test_total,
                             "val_threshold": thr, "val_macro_f1": vf1, "val_logloss": log_loss(y["val"], pva), "val_auc": roc_auc_score(y["val"], pva),
                             "test_macro_f1": mf1(y["test"], pred), "test_ct_f1": f1_score(y["test"], pred, pos_label=1, zero_division=0),
                             "test_t_f1": f1_score(y["test"], pred, pos_label=0, zero_division=0), "test_auc": roc_auc_score(y["test"], pte)})
        # paired series bootstrap (seed 42) for val and test deltas vs A
        for part_, yy, gi in (("val", y["val"], P["val"].match_id.to_numpy()), ("test", y["test"], P["test"].match_id.to_numpy())):
            ug = np.unique(gi); idx = {u: np.flatnonzero(gi == u) for u in ug}
            B = np.empty((500, len(SETS)))
            for b in range(500):
                kk = np.concatenate([idx[u] for u in rng.choice(ug, len(ug))])
                for j, k in enumerate(SETS):
                    pv = res[(k, 42)][0 if part_ == "val" else 1]
                    B[b, j] = mf1(yy[kk], (pv[kk] >= res[(k, 42)][2]).astype(int))
            for j, k in enumerate(SETS):
                r = [x for x in rows if x["set"] == k and x["seed"] == 42 and x["t_s"] == t][0]
                d_ = B[:, j] - B[:, 0]
                r[f"{part_}_delta_lo"], r[f"{part_}_delta_hi"] = (np.percentile(d_, 2.5), np.percentile(d_, 97.5)) if k != "A" else (0.0, 0.0)
        print(t, {k: round([x for x in rows if x["set"] == k and x["seed"] == 42 and x["t_s"] == t][0]["test_macro_f1"], 3) for k in SETS}, flush=True)
    R = pd.DataFrame(rows)
    for part_ in ("val", "test"):
        base = R[R.set == "A"].set_index(["seed", "t_s"])[f"{part_}_macro_f1"]
        R[f"{part_}_delta_vs_A"] = [r_[f"{part_}_macro_f1"] - base[(r_.seed, r_.t_s)] for _, r_ in R.iterrows()]
    R.to_csv(RES / "ablation_results.csv", index=False)
    S = R.groupby(["set", "t_s"]).agg(val_f1_mean=("val_macro_f1", "mean"), test_f1_mean=("test_macro_f1", "mean"),
                                       val_delta_mean=("val_delta_vs_A", "mean"), test_delta_mean=("test_delta_vs_A", "mean"),
                                       test_f1_min=("test_macro_f1", "min"), test_f1_max=("test_macro_f1", "max")).reset_index()
    S.to_csv(RES / "ablation_seed_summary.csv", index=False)
    # plot
    s42 = R[R.seed == 42]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8))
    for i, k in enumerate(SETS):
        d_ = s42[s42.set == k]
        ax.plot(d_.t_s, d_.test_macro_f1, marker="o", color=plt.get_cmap("tab10")(i), label=f"{k} {DESC[k]} ({d_.n_features.iloc[0]})")
    ax.set_xlabel("prediction time (s)"); ax.set_ylabel("test macro F1 (seed 42)"); ax.grid(alpha=.25); ax.legend(fontsize=7)
    w = 0.13
    for i, k in enumerate(list(SETS)[1:]):
        d_ = S[S.set == k]
        bx.bar(np.arange(5) + (i - 2) * w, d_.test_delta_mean, w, color=plt.get_cmap("tab10")(i + 1), label=k)
        bx.scatter(np.arange(5) + (i - 2) * w, d_.val_delta_mean, color="k", s=8, zorder=3, label="val Δ" if i == 0 else None)
    bx.axhline(0.01, ls=":", color="#c0392b"); bx.axhline(0, color="k", lw=.7); bx.set_xticks(range(5)); bx.set_xticklabels(FOCUS)
    bx.set_title("ΔF1 vs A, mean of 3 seeds (bars=test, dots=val); dotted = +0.01 criterion"); bx.legend(fontsize=7); bx.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(ROOT / "plots" / "step7_dynamics_pilot.png", dpi=150)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(s42[["set", "t_s", "n_features", "val_threshold", "val_macro_f1", "val_delta_vs_A", "val_delta_lo", "val_delta_hi", "test_macro_f1", "test_ct_f1", "test_t_f1", "test_delta_vs_A", "test_delta_lo", "test_delta_hi", "n_test", "coverage_test"]].round(3).to_string(index=False))
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
