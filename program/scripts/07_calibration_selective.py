#!/usr/bin/env python
"""Calibration + pooled selective prediction on existing predictions.jsonl.

Usage:
  python scripts/07_calibration_selective.py \
    --predictions outputs/overnight/medgemma_4b_it_1000/predictions.jsonl \
    --output-dir outputs/overnight/medgemma_4b_it_1000/calibration_selective
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.evaluation.metrics import (  # noqa: E402
    aurc, auroc, brier, coverage_at_risk, expected_calibration_error,
    selective_accuracy,
)


def load_predictions(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        gt = obj.get("ground_truth", {}) or {}
        for label, fp in (obj.get("findings") or {}).items():
            gt_val = gt.get(f"label_{label}")
            if gt_val not in {"positive", "negative"}:
                continue
            y_true = 1.0 if gt_val == "positive" else 0.0
            p = fp.get("probability_positive")
            if p is None:
                continue
            status = fp.get("status")
            rows.append({
                "image_id": obj.get("image_id"),
                "split": obj.get("split"),
                "label": label,
                "y_true": y_true,
                "p": float(p),
                "status": status,
            })
    return pd.DataFrame(rows)


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(y_true, p) -> float:
    if len(np.unique(y_true)) < 2:
        return 1.0
    z = _logit(p)

    def nll(log_t):
        t = np.exp(log_t)
        q = _sigmoid(z / t)
        q = np.clip(q, 1e-6, 1 - 1e-6)
        return -np.mean(y_true * np.log(q) + (1 - y_true) * np.log(1 - q))

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x)) if res.success else 1.0


def apply_temperature(p, t):
    return _sigmoid(_logit(p) / t)


def macro_metrics(df: pd.DataFrame, p_col: str) -> pd.DataFrame:
    rows = []
    for label, g in df.groupby("label"):
        y = g["y_true"].to_numpy()
        p = g[p_col].to_numpy()
        pred = (p >= 0.5).astype(float)
        rows.append({
            "label": label,
            "n": len(g),
            "auroc": auroc(y, p),
            "ece": expected_calibration_error(y, p),
            "brier": brier(y, p),
            "error_detection_auroc": np.nan,
            "risk": float(np.mean(pred != y)),
        })
    return pd.DataFrame(rows)


def risk_coverage(df: pd.DataFrame, score_col: str):
    # confidence in the model's selected status
    y = df["y_true"].to_numpy()
    p = df[score_col].to_numpy()
    status = df["status"].fillna("uncertain").to_numpy()
    pred = np.where(status == "positive", 1.0, np.where(status == "negative", 0.0, np.nan))
    conf = np.where(status == "positive", p, np.where(status == "negative", 1 - p, 0.5))
    error = np.where(np.isnan(pred), 1.0, (pred != y).astype(float))
    risk_score = 1.0 - conf
    order = np.argsort(risk_score)
    y_sorted = error[order]
    n = len(y_sorted)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(y_sorted) / np.arange(1, n + 1)
    return coverage, risk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-val-split", default="source_val")
    parser.add_argument("--test-split", default="internal_test")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_predictions(Path(args.predictions))
    df.to_csv(out_dir / "flat_predictions.csv", index=False)

    cal_rows = []
    for label, g in df.groupby("label"):
        src = g[g["split"] == args.source_val_split]
        test = g[g["split"] == args.test_split]
        if len(src) == 0 or len(test) == 0:
            continue
        t = fit_temperature(src["y_true"].to_numpy(), src["p"].to_numpy())
        cal_rows.append({
            "label": label,
            "temperature": t,
            "n_source_val": len(src),
            "n_test": len(test),
            "auroc_before": auroc(test["y_true"].to_numpy(), test["p"].to_numpy()),
            "auroc_after": auroc(test["y_true"].to_numpy(), apply_temperature(test["p"].to_numpy(), t)),
            "ece_before": expected_calibration_error(test["y_true"].to_numpy(), test["p"].to_numpy()),
            "ece_after": expected_calibration_error(test["y_true"].to_numpy(), apply_temperature(test["p"].to_numpy(), t)),
            "brier_before": brier(test["y_true"].to_numpy(), test["p"].to_numpy()),
            "brier_after": brier(test["y_true"].to_numpy(), apply_temperature(test["p"].to_numpy(), t)),
        })
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(out_dir / "calibration_by_label.csv", index=False)

    test = df[df["split"] == args.test_split].copy()
    # fit one global temperature on source_val for pooled selective analysis
    src = df[df["split"] == args.source_val_split]
    t_global = fit_temperature(src["y_true"].to_numpy(), src["p"].to_numpy())
    test["p_calibrated"] = apply_temperature(test["p"].to_numpy(), t_global)
    coverage, risk = risk_coverage(test, "p_calibrated")
    pd.DataFrame({"coverage": coverage, "risk": risk}).to_csv(out_dir / "risk_coverage_curve.csv", index=False)
    y = test["y_true"].to_numpy()
    status = test["status"].fillna("uncertain").to_numpy()
    pred = np.where(status == "positive", 1.0, np.where(status == "negative", 0.0, np.nan))
    error = np.where(np.isnan(pred), 1.0, (pred != y).astype(float))
    risk_score = 1.0 - np.where(
        status == "positive",
        test["p_calibrated"],
        np.where(status == "negative", 1 - test["p_calibrated"], 0.5),
    )
    summary = {
        "global_temperature": t_global,
        "n_test": int(len(test)),
        "aurc": float(aurc(error, risk_score)),
        "coverage_at_5pct_risk": float(coverage_at_risk(error, risk_score, 0.05)),
        "coverage_at_10pct_risk": float(coverage_at_risk(error, risk_score, 0.10)),
        "selective_accuracy_at_80pct_coverage": float(selective_accuracy(error, risk_score, 0.80)),
    }
    (out_dir / "selective_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
