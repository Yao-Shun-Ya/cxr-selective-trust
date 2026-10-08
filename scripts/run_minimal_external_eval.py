#!/usr/bin/env python
"""Minimal external evaluation for Gate 1/2.

Input: CSV or JSONL with one row per model x dataset x image x finding.
Required columns: model, dataset, label, y_true, y_prob, y_pred
Optional columns: confidence, risk_score, correct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medvlm_trust.evaluation.metrics import (  # noqa: E402
    aurc,
    auroc,
    auprc,
    brier,
    coverage_at_risk,
    error_detection_auroc,
    expected_calibration_error,
    high_confidence_error_rate,
    selective_accuracy,
)


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def compute_group(df: pd.DataFrame) -> dict:
    y_true = df["y_true"].astype(float).to_numpy()
    y_prob = df["y_prob"].astype(float).to_numpy()
    y_pred = df["y_pred"].astype(float).to_numpy()
    confidence = df["confidence"].astype(float).to_numpy() if "confidence" in df.columns else y_prob
    risk_score = df["risk_score"].astype(float).to_numpy() if "risk_score" in df.columns else (1.0 - confidence)
    correct = df["correct"].astype(float).to_numpy() if "correct" in df.columns else (y_pred == y_true).astype(float)
    error = (y_pred != y_true).astype(float)

    hce, hce_n = high_confidence_error_rate(y_true, y_prob, threshold=0.9)
    return {
        "n": int(len(df)),
        "auroc": auroc(y_true, y_prob),
        "auprc": auprc(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
        "brier": brier(y_true, y_prob),
        "high_confidence_error": hce,
        "high_confidence_n": hce_n,
        "error_detection_auroc": error_detection_auroc(correct, risk_score),
        "aurc": aurc(error, risk_score),
        "coverage_at_5pct_risk": coverage_at_risk(error, risk_score, 0.05),
        "coverage_at_10pct_risk": coverage_at_risk(error, risk_score, 0.10),
        "selective_accuracy_at_80pct_coverage": selective_accuracy(error, risk_score, 0.80),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    path = Path(args.predictions)
    df = load_table(path)
    required = {"model", "dataset", "label", "y_true", "y_prob", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing required columns: {sorted(missing)}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (model, dataset, label), group in df.groupby(["model", "dataset", "label"], dropna=False):
        metrics = compute_group(group)
        metrics.update({"model": model, "dataset": dataset, "label": label})
        rows.append(metrics)

    by_label = pd.DataFrame(rows)
    by_label.to_csv(out_dir / "metrics_by_label.csv", index=False)

    summary = {}
    for (model, dataset), group in df.groupby(["model", "dataset"], dropna=False):
        summary[f"{model}|{dataset}"] = compute_group(group)
    (out_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'metrics_by_label.csv'}")
    print(f"wrote {out_dir / 'metrics_summary.json'}")


if __name__ == "__main__":
    main()
