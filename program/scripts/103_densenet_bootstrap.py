#!/usr/bin/env python
"""103 - Patient-cluster bootstrap CI for the DenseNet-121 supervised baseline.

The baseline is the supervised anchor of the paper: 0.7329 macro pathology AUROC
against 0.50--0.63 for six VLMs.  A point estimate without an interval is a weak
anchor, so this script resamples *patients* (not images) 1,000 times and reports
a percentile interval for the macro AUROC on the 1,506-study evaluation set.

Output: outputs/densenet_baseline/bootstrap_ci.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/densenet_baseline"
MANIFEST = ROOT / "data/manifests/nih_full.csv"

LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax"]
N_BOOT = 1000
SEED = 0


def main() -> int:
    pred = pd.read_csv(OUT / "test_predictions.csv")
    man = pd.read_csv(MANIFEST, low_memory=False)
    man = man[["image_id", "patient_id"]].drop_duplicates("image_id")
    d = pred.merge(man, on="image_id", how="left")
    missing = int(d["patient_id"].isna().sum())
    if missing:
        raise SystemExit(f"{missing} predictions have no patient_id")

    # Ground truth comes from the manifest, not from the prediction file.
    gt = pd.read_csv(MANIFEST, low_memory=False)
    gt = gt[gt.image_id.isin(d.image_id)][["image_id"] + [f"label_{l}" for l in LABELS]]
    d = d.merge(gt, on="image_id", how="left")
    y = np.column_stack([
        (d[f"label_{l}"].astype(str).str.lower() == "positive").to_numpy()
        for l in LABELS
    ]).astype(int)
    p = d[LABELS].to_numpy()

    def macro(idx: np.ndarray) -> float:
        vals = []
        for j in range(len(LABELS)):
            yy = y[idx, j]
            if len(np.unique(yy)) == 2:
                vals.append(roc_auc_score(yy, p[idx, j]))
        return float(np.mean(vals))

    point = macro(np.arange(len(d)))
    patients = d["patient_id"].to_numpy()
    groups = pd.Series(np.arange(len(d))).groupby(patients).apply(lambda s: s.to_numpy()).to_numpy()
    rng = np.random.default_rng(SEED)
    stats = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        stats.append(macro(idx))
    lo, hi = np.percentile(stats, [2.5, 97.5])

    rows = [{
        "n_images": len(d),
        "n_patients": int(d.patient_id.nunique()),
        "macro_pathology_auroc": round(point, 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "n_boot": N_BOOT,
        "seed": SEED,
    }]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
