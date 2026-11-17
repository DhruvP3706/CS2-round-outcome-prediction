"""Comprehensive ML modeling module supporting 12 classical algorithms, validation threshold tuning,
cluster bootstrap confidence intervals, and full classification performance metrics.
"""
from __future__ import annotations

import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
)
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

THRESHOLD_GRID = np.round(np.arange(0.05, 0.9501, 0.005), 3)

def macro_f1(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)

def tune_threshold(y_val, p_val):
    """Find threshold that maximizes Macro F1 on validation; ties resolved toward 0.5."""
    scores = np.array([macro_f1(y_val, p_val, t) for t in THRESHOLD_GRID])
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    i = best[np.argmin(np.abs(THRESHOLD_GRID[best] - 0.5))]
    return float(THRESHOLD_GRID[i]), float(scores[i])

def make_model(name: str, seed: int = 42):
    """Instantiate a classical ML model with standardized hyperparameters."""
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2500, random_state=seed))
    elif name == "decision_tree":
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=30, random_state=seed)
    elif name == "random_forest":
        return RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=15, random_state=seed, n_jobs=8)
    elif name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=300, max_depth=12, min_samples_leaf=15, random_state=seed, n_jobs=8)
    elif name == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=200, learning_rate=0.04, max_depth=4, random_state=seed)
    elif name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40, random_state=seed)
    elif name == "xgboost":
        return xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, max_depth=4, min_child_weight=40,
                                 subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=8, eval_metric="logloss")
    elif name == "lightgbm":
        return lgb.LGBMClassifier(objective="binary", n_estimators=2000, learning_rate=0.03, num_leaves=4,
                                  max_depth=3, min_child_samples=80, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, reg_lambda=2.0, random_state=seed, n_jobs=8, verbose=-1)
    elif name == "catboost":
        return CatBoostClassifier(iterations=500, learning_rate=0.03, depth=4, l2_leaf_reg=3.0,
                                  random_seed=seed, thread_count=8, verbose=0)
    elif name == "svm":
        return make_pipeline(StandardScaler(), CalibratedClassifierCV(LinearSVC(C=0.1, random_state=seed, max_iter=2500), cv=3))
    elif name == "naive_bayes":
        return make_pipeline(StandardScaler(), GaussianNB())
    elif name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=8))
    else:
        raise ValueError(f"Unknown model architecture: {name}")

def fit_predict(name: str, Xtr, ytr, Xva, yva, Xte, seed: int = 42):
    """Fit model on train data (with early stopping on validation if supported), return validation & test probs."""
    m = make_model(name, seed=seed)
    t0 = time.time()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "lightgbm":
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="binary_logloss",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        elif name == "xgboost":
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        elif name == "catboost":
            m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=100, verbose=0)
        else:
            m.fit(Xtr, ytr)
            
    train_time = time.time() - t0
    
    pva = m.predict_proba(Xva)[:, 1]
    pte = m.predict_proba(Xte)[:, 1]
    return m, pva, pte, train_time

def evaluate_metrics(y_true, p_pred, thr=0.5):
    """Compute comprehensive classification metrics."""
    pred = (p_pred >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    
    mf1 = f1_score(y_true, pred, average="macro", zero_division=0)
    f1_ct = f1_score(y_true, pred, pos_label=1, zero_division=0)
    f1_t = f1_score(y_true, pred, pos_label=0, zero_division=0)
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, p_pred)
    except Exception:
        auc = float("nan")
        
    try:
        prauc = average_precision_score(y_true, p_pred)
    except Exception:
        prauc = float("nan")

    return {
        "macro_f1": float(mf1),
        "ct_f1": float(f1_ct),
        "t_f1": float(f1_t),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(auc),
        "pr_auc": float(prauc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }

def cluster_bootstrap_ci(y, p, thr, groups, n_boot=500, seed=42):
    """95% Cluster Bootstrap Confidence Interval for test Macro F1 by series."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(n_boot):
        sample_g = rng.choice(ug, size=len(ug), replace=True)
        pick = np.concatenate([idx[g] for g in sample_g])
        vals.append(macro_f1(y[pick], p[pick], thr))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
