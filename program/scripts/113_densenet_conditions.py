#!/usr/bin/env python
"""113 - DenseNet-121 over the 19-cell grid and PadChest: operating-point analysis.

Reads outputs/densenet_baseline/grid_predictions.csv and
padchest_predictions.csv, applies a fixed 0.5 rule (the CNN has no verbalised
status), and reports per cell:

  * macro AUROC over the seven pathology labels,
  * pooled recall, positive rate and no_finding positive rate,
  * the clean -> severe change (the Reversal-3 question for a supervised model).

Output:
  outputs/densenet_baseline/condition_metrics.csv

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\113_densenet_conditions.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "densenet_baseline"
NIH_MAN = ROOT / "data" / "manifests" / "nih_eval_1506.csv"
PC_MAN = ROOT / "data" / "manifests" / "padchest_shift.csv"

LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax", "no_finding"]
PATHOLOGY = [l for l in LABELS if l != "no_finding"]


def truth_frame(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False)
    return d[["image_id"] + [f"label_{l}" for l in LABELS]].copy()


def attach(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    d = pred.merge(truth, on="image_id", how="inner")
    for l in LABELS:
        d[f"y_{l}"] = (d[f"label_{l}"].astype(str).str.lower() == "positive").astype(int)
    return d


def cell_metrics(d: pd.DataFrame, condition: str, severity: str) -> dict:
    aus, recs, pos = [], [], []
    for l in PATHOLOGY:
        y = d[f"y_{l}"].to_numpy()
        p = d[f"p_{l}"].to_numpy()
        if len(np.unique(y)) > 1:
            aus.append(roc_auc_score(y, p))
        pred = (p >= 0.5).astype(int)
        if y.sum():
            recs.append(float((pred * y).sum() / y.sum()))
        pos.append(float(pred.mean()))
    nf = d["y_no_finding"].to_numpy()
    nf_pred = (d["p_no_finding"].to_numpy() >= 0.5).astype(int)
    return {
        "condition": condition, "severity": severity, "n": len(d),
        "macro_auroc": round(float(np.mean(aus)), 4) if aus else None,
        "recall": round(float(np.mean(recs)), 4) if recs else None,
        "positive_rate": round(float(np.mean(pos)), 4) if pos else None,
        "no_finding_rate": round(float(nf_pred.mean()), 4) if len(nf) else None,
        "no_finding_accuracy": round(float((nf_pred == nf).mean()), 4) if len(nf) else None,
    }


def main() -> int:
    grid = pd.read_csv(OUT / "grid_predictions.csv")
    d = attach(grid, truth_frame(NIH_MAN))
    rows = []
    for (cond, sev), g in d.groupby(["condition", "severity"]):
        rows.append(cell_metrics(g, cond, sev))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "condition_metrics.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))

    pc_path = OUT / "padchest_predictions.csv"
    if pc_path.exists():
        pc = attach(pd.read_csv(pc_path), truth_frame(PC_MAN))
        aus, recs = [], []
        for l in PATHOLOGY:
            y = pc[f"y_{l}"].to_numpy()
            p = pc[f"p_{l}"].to_numpy()
            if len(np.unique(y)) > 1:
                aus.append(roc_auc_score(y, p))
            if y.sum():
                recs.append(float(((p >= 0.5).astype(int) * y).sum() / y.sum()))
        summary = pd.DataFrame([{
            "layer": "padchest", "n": len(pc),
            "macro_auroc": round(float(np.mean(aus)), 4) if aus else None,
            "recall": round(float(np.mean(recs)), 4) if recs else None,
        }])
        summary.to_csv(OUT / "padchest_metrics.csv", index=False, encoding="utf-8-sig")
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
