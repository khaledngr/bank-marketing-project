"""Shared evaluation: out-of-fold scores, threshold-free metrics and the cost-based decision.

Every model in the project is judged by the same functions, so the numbers are comparable.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from src.config import CONFIG
from src.data import cv_splitter

COST = CONFIG["business"]["cost_per_call"]
REVENUE = CONFIG["business"]["revenue_per_subscription"]
THRESHOLDS = np.round(np.arange(0.01, 1.0, 0.01), 2)


def out_of_fold(pipe, X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Each client is scored by a model that never saw them: a fresh copy of the whole pipeline
    is fitted on the other 4 folds. Returns the scores and the AP of each fold."""
    oof = np.zeros(len(y))
    fold_ap = []
    for train_idx, val_idx in cv_splitter().split(X, y):
        model = clone(pipe).fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        fold_ap.append(average_precision_score(y.iloc[val_idx], oof[val_idx]))
    return oof, np.array(fold_ap)


def profit(y: pd.Series, call: np.ndarray) -> float:
    """EUR earned by calling the selected clients: margin on each subscriber minus every call."""
    return REVENUE * np.sum(call & (y == 1)) - COST * np.sum(call)


def profit_curve(y: pd.Series, scores: np.ndarray) -> pd.Series:
    """Profit per 1,000 clients if we call everyone scoring above each threshold."""
    return pd.Series([profit(y, scores >= t) * 1000 / len(y) for t in THRESHOLDS], index=THRESHOLDS)


def best_threshold(y: pd.Series, scores: np.ndarray) -> float:
    """The threshold that maximises profit. Only ever called on out-of-fold scores."""
    return float(profit_curve(y, scores).idxmax())


def expected_calibration_error(y: pd.Series, scores: np.ndarray, n_bins: int = 10) -> float:
    """ECE (S7): gap between mean predicted and observed rate in each of `n_bins` equal-width
    bins, weighted by the share of clients in the bin."""
    bins = np.minimum((scores * n_bins).astype(int), n_bins - 1)
    frame = pd.DataFrame({"bin": bins, "p": scores, "y": np.asarray(y)})
    per_bin = frame.groupby("bin").agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return float((per_bin["n"] / len(frame) * (per_bin["observed"] - per_bin["predicted"]).abs()).sum())


def summarize(y: pd.Series, scores: np.ndarray, threshold: float) -> dict:
    """Threshold-free ranking metrics plus what happens if we call above `threshold`."""
    call = scores >= threshold
    return {
        "AP": average_precision_score(y, scores),
        "ROC_AUC": roc_auc_score(y, scores),
        "threshold": threshold,
        "precision": precision_score(y, call, zero_division=0),
        "recall": recall_score(y, call),
        "share_called": call.mean(),
        "profit_per_1000": profit(y, call) * 1000 / len(y),
    }
