"""Selective-prediction audit: confidence, entropy and margin baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _coverage_at(error: np.ndarray, score: np.ndarray, budget: float) -> float:
    """Coverage at a risk budget; stable sorting matches the paper's pipeline."""
    order = np.argsort(-score, kind="stable")
    err = error[order]
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    ok = np.where(risk <= budget + 1e-12)[0]
    return float((ok[-1] + 1) / len(err)) if len(ok) else 0.0


def _metrics(error: np.ndarray, score: np.ndarray) -> dict:
    return {
        "error_detection_auroc": float(roc_auc_score(error, -score)),
        "coverage_at_5pct_risk": _coverage_at(error, score, 0.05),
        "coverage_at_10pct_risk": _coverage_at(error, score, 0.10),
    }


def selective_audit(frame: pd.DataFrame) -> dict:
    """Compare status-aware confidence with probability-only baselines.

    Required columns: ``error`` and either ``p_calibrated`` or ``p``.
    ``conf`` is used when present; otherwise it is derived from ``pred``:
    ``p`` for a positive call and ``1-p`` for a negative call.
    """
    if "error" not in frame:
        raise ValueError("selective CSV needs an 'error' column")
    p_col = "p_calibrated" if "p_calibrated" in frame else "p"
    if p_col not in frame:
        raise ValueError("selective CSV needs 'p_calibrated' or 'p'")
    p = frame[p_col].astype(float).clip(1e-6, 1 - 1e-6).to_numpy()
    error = frame["error"].astype(float).to_numpy()
    if "conf" in frame:
        conf = frame["conf"].astype(float).to_numpy()
    elif "pred" in frame:
        pred = frame["pred"].astype(float).to_numpy()
        conf = np.where(pred >= 0.5, p, 1 - p)
    else:
        raise ValueError("selective CSV needs 'conf' or 'pred'")

    entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    margin = np.abs(p - 0.5)
    return {
        "n_items": int(len(frame)),
        "prevalence_error": float(np.mean(error)) if len(error) else None,
        "signals": {
            "confidence": _metrics(error, conf),
            "entropy": _metrics(error, -entropy),
            "margin": _metrics(error, margin),
        },
    }
