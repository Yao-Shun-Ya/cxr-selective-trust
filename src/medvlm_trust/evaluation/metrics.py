"""Evaluation metrics for discrimination, calibration, and selective prediction."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def _finite_pair(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    return y_true[mask], y_score[mask]


def auroc(y_true, y_score) -> float:
    y_true, y_score = _finite_pair(y_true, y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def auprc(y_true, y_score) -> float:
    y_true, y_score = _finite_pair(y_true, y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def binary_f1(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(f1_score(y_true[mask], y_pred[mask]))


def sensitivity_specificity(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return float(sens), float(spec)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Standard equal-width ECE for binary outcomes."""
    y_true, y_prob = _finite_pair(y_true, y_prob)
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += (mask.sum() / len(y_true)) * abs(acc - conf)
    return float(ece)


def brier(y_true, y_prob) -> float:
    y_true, y_prob = _finite_pair(y_true, y_prob)
    if len(y_true) == 0:
        return float("nan")
    return float(brier_score_loss(y_true, y_prob))


def high_confidence_error_rate(y_true, y_prob, threshold: float = 0.9):
    y_true, y_prob = _finite_pair(y_true, y_prob)
    mask = y_prob >= threshold
    if mask.sum() == 0:
        return float("nan"), 0
    # For probabilities p >= 0.5, positive prediction; otherwise negative.
    pred = (y_prob[mask] >= 0.5).astype(int)
    error = np.mean(pred != y_true[mask])
    return float(error), int(mask.sum())


def risk_coverage_curve(error: Iterable[float], risk_score: Iterable[float]):
    """Return coverage, risk arrays under the rule 'highest risk abstains first'."""
    error = np.asarray(error, dtype=float)
    risk_score = np.asarray(risk_score, dtype=float)
    mask = np.isfinite(error) & np.isfinite(risk_score)
    error, risk_score = error[mask], risk_score[mask]
    if len(error) == 0:
        return np.array([]), np.array([])
    order = np.argsort(risk_score)  # low risk accepted first
    sorted_error = error[order]
    coverage = np.arange(1, len(error) + 1) / len(error)
    risk = np.cumsum(sorted_error) / np.arange(1, len(error) + 1)
    return coverage, risk


def aurc(error: Iterable[float], risk_score: Iterable[float]) -> float:
    coverage, risk = risk_coverage_curve(error, risk_score)
    if len(coverage) == 0:
        return float("nan")
    return float(np.trapz(risk, coverage))


def coverage_at_risk(error: Iterable[float], risk_score: Iterable[float], max_risk: float) -> float:
    coverage, risk = risk_coverage_curve(error, risk_score)
    if len(coverage) == 0:
        return float("nan")
    valid = np.where(risk <= max_risk)[0]
    if len(valid) == 0:
        return 0.0
    return float(coverage[valid[-1]])


def selective_accuracy(error: Iterable[float], risk_score: Iterable[float], coverage: float) -> float:
    coverage_arr, _ = risk_coverage_curve(error, risk_score)
    if len(coverage_arr) == 0:
        return float("nan")
    n_accept = max(1, int(round(coverage * len(error))))
    # Recompute using sorted order to keep this simple and explicit.
    error = np.asarray(error, dtype=float)
    risk_score = np.asarray(risk_score, dtype=float)
    mask = np.isfinite(error) & np.isfinite(risk_score)
    error, risk_score = error[mask], risk_score[mask]
    order = np.argsort(risk_score)
    accepted = error[order][:n_accept]
    return float(1.0 - np.mean(accepted))


def error_detection_auroc(correct: Iterable[int], risk_score: Iterable[float]) -> float:
    """AUROC of risk score for detecting incorrect predictions.

    correct = 1 for correct, 0 for incorrect.
    Risk score should be higher for more likely incorrect.
    """
    correct = np.asarray(correct, dtype=float)
    risk_score = np.asarray(risk_score, dtype=float)
    mask = np.isfinite(correct) & np.isfinite(risk_score)
    correct, risk_score = correct[mask], risk_score[mask]
    if len(np.unique(correct)) < 2:
        return float("nan")
    # We want high risk -> incorrect, so use -risk as the score for "correct".
    return float(roc_auc_score(correct, -risk_score))


def reliability_curve(y_true, y_prob, n_bins: int = 10):
    y_true, y_prob = _finite_pair(y_true, y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=True)
    centers, accs, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        centers.append(float(np.mean(y_prob[mask])))
        accs.append(float(np.mean(y_true[mask])))
        counts.append(int(mask.sum()))
    return np.array(centers), np.array(accs), np.array(counts)
