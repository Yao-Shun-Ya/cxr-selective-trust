#!/usr/bin/env python
"""107 - PadChest per-label AUROC and patient-cluster CI (label hierarchy, out of domain).

The label-level hierarchy in the paper is an NIH result.  This script asks the
same question on PadChest: for each model and each of the seven pathology
labels, what is the native-resolution AUROC, and is it estimable at all?  A
model that emits a (near-)constant answer has AUROC 0.500 by construction, so
the table is also a responsiveness diagnostic at the label level.

Output: outputs/analysis_sensitivity/padchest_label_hierarchy.csv

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\107_padchest_label_hierarchy.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data/manifests/padchest_shift.csv"
MATRIX = ROOT / "outputs/matrix/padchest"
OUT = ROOT / "outputs/analysis_sensitivity/padchest_label_hierarchy.csv"

MODELS = {
    "medgemma_4b_it": "MedGemma-4B",
    "medgemma_15_4b_it": "MedGemma-1.5",
    "qwen25_vl_7b_instruct": "Qwen2.5-VL",
    "qwen3_vl_8b_instruct": "Qwen3-VL",
    "chexagent_8b": "CheXagent",
    "internvl3_5_8b": "InternVL3.5",
}
LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax"]
N_BOOT = 1000
SEED = 0


def prob(rec: dict, label: str) -> float:
    f = (rec.get("parsed") or {}).get("findings") or {}
    v = (f.get(label) or {}).get("probability_positive")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return 0.0
    v = float(v)
    return v if 0.0 <= v <= 1.0 else 0.0


def main() -> int:
    man = pd.read_csv(MAN, low_memory=False)
    truth = {}
    patient = {}
    for _, r in man.iterrows():
        iid = str(r["image_id"])
        truth[iid] = {l: str(r.get(f"label_{l}") or "").lower() == "positive" for l in LABELS}
        patient[iid] = str(r.get("patient_id", ""))

    rows = []
    for slug, name in MODELS.items():
        path = MATRIX / slug / "shift" / "image_reliance_predictions.jsonl"
        if not path.exists():
            continue
        data = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if str(r.get("condition")) != "original":
                    continue
                iid = str(r.get("image_id"))
                if iid not in truth or not (r.get("parsed") or {}).get("parse_success"):
                    continue
                data.append((iid, [1 if truth[iid][l] else 0 for l in LABELS],
                             [prob(r, l) for l in LABELS]))
        if not data:
            continue
        y = np.array([d[1] for d in data], dtype=int)
        p = np.array([d[2] for d in data], dtype=float)
        pats = np.array([patient[d[0]] for d in data])
        groups = pd.Series(np.arange(len(data))).groupby(pats).apply(lambda s: s.to_numpy()).to_numpy()
        rng = np.random.default_rng(SEED)
        boots = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(groups), len(groups))
            boots.append(np.concatenate([groups[i] for i in pick]))

        for j, label in enumerate(LABELS):
            yy = y[:, j]
            if len(np.unique(yy)) < 2:
                rows.append({"model": name, "label": label, "n": len(data),
                             "n_positive": int(yy.sum()), "auroc": None,
                             "ci_lo": None, "ci_hi": None})
                continue
            point = roc_auc_score(yy, p[:, j])
            stats = []
            for idx in boots:
                yb = y[idx, j]
                if len(np.unique(yb)) == 2:
                    stats.append(roc_auc_score(yb, p[idx, j]))
            lo, hi = np.percentile(stats, [2.5, 97.5])
            rows.append({"model": name, "label": label, "n": len(data),
                         "n_positive": int(yy.sum()), "auroc": round(float(point), 4),
                         "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4)})
        print(f"{name}: {len(data)} studies")

    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
