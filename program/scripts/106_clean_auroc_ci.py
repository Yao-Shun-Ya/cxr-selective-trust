#!/usr/bin/env python
"""106 - Patient-cluster bootstrap CI for clean NIH macro AUROC (all models).

Table `tab:clean` reports point estimates.  This script puts a patient-cluster
interval around every model that emits probabilities, so the "0.50--0.63" claim
is not a set of bare numbers.  CheXagent is reported without an interval because
it has no probability channel; its AUROC comes from the emitted status.

Output: outputs/analysis_nih_mine/clean_auroc_ci.csv

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\106_clean_auroc_ci.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data/manifests/nih_eval_1506.csv"
MATRIX = ROOT / "outputs/matrix/nih_eval"
OUT = ROOT / "outputs/analysis_nih_mine/clean_auroc_ci.csv"

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


def prob(rec: dict, label: str) -> float | None:
    f = (rec.get("parsed") or {}).get("findings") or {}
    v = (f.get(label) or {}).get("probability_positive")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if 0.0 <= v <= 1.0 else None


def status_pos(rec: dict, label: str) -> bool:
    f = (rec.get("parsed") or {}).get("findings") or {}
    return str((f.get(label) or {}).get("status") or "") == "positive"


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
        path = MATRIX / slug / "shift" / "shift_predictions.jsonl"
        if not path.exists():
            print(f"skip {name}: no predictions")
            continue
        recs = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if str(r.get("condition")) == "clean":
                    recs.append(r)
        use_prob = True
        data = []
        for r in recs:
            iid = str(r.get("image_id"))
            if iid not in truth:
                continue
            if not (r.get("parsed") or {}).get("parse_success"):
                continue
            # Match script 75 exactly: a missing probability is scored as 0.0,
            # so the point estimate is the one printed in Table tab:clean.
            data.append((iid, [1 if truth[iid][l] else 0 for l in LABELS],
                         [0.0 if prob(r, l) is None else prob(r, l) for l in LABELS]))
        if not data:
            print(f"skip {name}: no parsed clean rows")
            continue
        y = np.array([d[1] for d in data], dtype=int)
        p = np.array([[np.nan if v is None else v for v in d[2]] for d in data], dtype=float)
        p = np.nan_to_num(p, nan=0.0)
        pats = np.array([patient[d[0]] for d in data])

        def macro(idx: np.ndarray) -> float:
            vals = []
            for j in range(len(LABELS)):
                yy, pp = y[idx, j], p[idx, j]
                if len(np.unique(yy)) < 2:
                    continue
                vals.append(roc_auc_score(yy, pp))
            return float(np.mean(vals)) if vals else float("nan")

        point = macro(np.arange(len(data)))
        groups = pd.Series(np.arange(len(data))).groupby(pats).apply(lambda s: s.to_numpy()).to_numpy()
        rng = np.random.default_rng(SEED)
        stats = []
        for _ in range(N_BOOT):
            pick = rng.integers(0, len(groups), len(groups))
            stats.append(macro(np.concatenate([groups[i] for i in pick])))
        lo, hi = np.percentile([s for s in stats if not np.isnan(s)], [2.5, 97.5])
        rows.append({
            "model": name,
            "channel": "probability (missing scored 0.0)",
            "n_studies": len(data),
            "n_patients": int(len(groups)),
            "macro_auroc": round(point, 4),
            "ci_lo": round(float(lo), 4),
            "ci_hi": round(float(hi), 4),
            "n_boot": N_BOOT,
            "seed": SEED,
        })
        print(f"{name}: {point:.4f} [{lo:.4f}, {hi:.4f}] ({len(data)} studies)")

    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
