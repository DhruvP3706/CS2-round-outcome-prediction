"""Per-time models, validation-only threshold tuning, single test evaluation."""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import LABEL_COL

THRESHOLD_GRID = np.round(np.arange(0.05, 0.9501, 0.005), 3)


def macro_f1(y, p, thr):
    return f1_score(y, (p >= thr).astype(int), average="macro", labels=[0, 1], zero_division=0)


def tune_threshold(y_val, p_val):
    """Threshold maximising macro F1 on validation; ties resolved toward 0.5."""
    scores = np.array([macro_f1(y_val, p_val, t) for t in THRESHOLD_GRID])
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    i = best[np.argmin(np.abs(THRESHOLD_GRID[best] - 0.5))]
    return float(THRESHOLD_GRID[i]), float(scores[i])


def make_model(name: str):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            objective="binary", n_estimators=2000, learning_rate=0.03, num_leaves=15,
            min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=2.0, random_state=42, n_jobs=8, verbose=-1)
    raise ValueError(name)


def fit_predict(name, Xtr, ytr, Xva, yva, Xte):
    m = make_model(name)
    best_iter = None
    if name == "lightgbm":  # early stopping on VALIDATION logloss only
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        best_iter = int(m.best_iteration_ or m.n_estimators)
    else:
        m.fit(Xtr, ytr)
    return m, m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], best_iter


def cluster_bootstrap_ci(y, p, thr, groups, n_boot=300, seed=0):
    """95% CI of test macro F1, resampling whole series (frozen threshold)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        vals.append(macro_f1(y[pick], p[pick], thr))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate_time(feat: pd.DataFrame, split_by_series: pd.Series, cols: list[str], t: int,
                  n_total_by_split: dict, model_names=("logreg", "lightgbm")):
    d = feat.copy()
    d["split"] = d["match_id"].map(split_by_series)
    parts = {s: d[d.split == s] for s in ("train", "val", "test")}
    y = {s: parts[s][LABEL_COL].astype(int).to_numpy() for s in parts}
    X = {s: parts[s][cols].astype(float) for s in parts}
    rows = []
    for name in model_names:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, pva, pte, best_iter = fit_predict(name, X["train"], y["train"], X["val"], y["val"], X["test"])
        thr, val_f1 = tune_threshold(y["val"], pva)      # validation only
        yt = y["test"]
        pred = (pte >= thr).astype(int)                    # frozen threshold, test evaluated once
        lo, hi = cluster_bootstrap_ci(yt, pte, thr, parts["test"]["match_id"].to_numpy())
        rows.append({
            "model": name, "t_s": t,
            "n_train": len(y["train"]), "n_val": len(y["val"]), "n_test": len(yt),
            "coverage_test": len(yt) / n_total_by_split["test"],
            "val_threshold": thr, "val_macro_f1": val_f1,
            "test_macro_f1": f1_score(yt, pred, average="macro", labels=[0, 1], zero_division=0),
            "test_macro_f1_ci_lo": lo, "test_macro_f1_ci_hi": hi,
            "test_ct_f1": f1_score(yt, pred, pos_label=1, zero_division=0),
            "test_t_f1": f1_score(yt, pred, pos_label=0, zero_division=0),
            "test_macro_f1_at_0.5": macro_f1(yt, pte, 0.5),
            "test_auc": roc_auc_score(yt, pte) if len(np.unique(yt)) > 1 else np.nan,
            "test_ct_rate": float(yt.mean()), "test_pred_ct_rate": float(pred.mean()),
            "lgbm_best_iter": best_iter,
        })
    return rows
