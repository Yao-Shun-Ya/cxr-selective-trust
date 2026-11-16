#!/usr/bin/env python
"""88 - Per-label patient-cluster bootstrap intervals and BH-FDR.

CPU-only, read-only.  The manuscript reports a pooled severe-degradation
recall loss and a label-level retention ranking.  This script adds the two
statistics a reviewer asks for at the label level:

* a patient-cluster bootstrap interval for the clean -> severe recall change
  (all six severe degradation types pooled, so the unit is the patient, not
  the cell);
* a two-sided bootstrap p-value and Benjamini-Hochberg q-value across the
  seven pathology labels, separately per model.

Inputs (frozen):
  outputs/matrix/nih_eval/<model>/shift/shift_predictions.jsonl
  data/manifests/nih_eval_1732.csv

Output:
  outputs/analysis_sensitivity/per_label_bootstrap_fdr.csv

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\88_per_label_ci_fdr.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis_sensitivity"
MATRIX = ROOT / "outputs" / "matrix" / "nih_eval"
MAN = ROOT / "data" / "manifests" / "nih_eval_1732.csv"

PATHOLOGY = ["atelectasis", "cardiomegaly", "consolidation", "edema",
             "pleural_effusion", "pneumonia", "pneumothorax"]
MODELS = ["internvl3_5_8b", "qwen3_vl_8b_instruct",
          "medgemma_15_4b_it", "medgemma_4b_it"]
N_BOOT = 1000
SEED = 0


def load_patients() -> dict[str, str]:
    out = {}
    for r in csv.DictReader(open(MAN, encoding="utf-8-sig")):
        out[r["image_id"]] = str(r["patient_id"])
    return out


def bh(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order]
    n = len(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = min(prev, ranked[i] * n / (i + 1))
        q[i] = val
        prev = val
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def main() -> int:
    patients = load_patients()
    rows = []
    for model in MODELS:
        path = MATRIX / model / "shift" / "shift_predictions.jsonl"
        # Per image: clean (tp, gt) and severe (tp, gt) for each label.
        clean: dict[str, dict[str, list[int]]] = {}
        severe: dict[str, dict[str, list[int]]] = {}
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                cond, sev = r.get("condition"), r.get("severity")
                if cond == "clean":
                    bucket = clean
                elif sev == "severe":
                    bucket = severe
                else:
                    continue
                parsed = r.get("parsed") or {}
                if not parsed.get("parse_success"):
                    continue
                iid = r["image_id"]
                bucket.setdefault(iid, {l: [0, 0] for l in PATHOLOGY})
                gt = r.get("ground_truth") or {}
                findings = parsed.get("findings") or {}
                for label in PATHOLOGY:
                    y = str(gt.get(f"label_{label}") or "").lower() == "positive"
                    p = str((findings.get(label) or {}).get("status") or "") == "positive"
                    bucket[iid][label][0] += int(y and p)
                    bucket[iid][label][1] += int(y)

        ids = sorted(set(clean) & set(severe))
        pts = np.array([patients.get(i, i) for i in ids])
        uniq = sorted(set(pts.tolist()))
        by_patient = {q: np.where(pts == q)[0] for q in uniq}
        rng = np.random.default_rng(SEED)

        for label in PATHOLOGY:
            c_tp = np.array([clean[i][label][0] for i in ids], dtype=float)
            c_gt = np.array([clean[i][label][1] for i in ids], dtype=float)
            s_tp = np.array([severe[i][label][0] for i in ids], dtype=float)
            s_gt = np.array([severe[i][label][1] for i in ids], dtype=float)
            if c_gt.sum() == 0 or s_gt.sum() == 0:
                continue
            rc = c_tp.sum() / c_gt.sum()
            rs = s_tp.sum() / s_gt.sum()
            delta = rs - rc
            boots = np.empty(N_BOOT, dtype=float)
            for b in range(N_BOOT):
                pick = rng.choice(uniq, size=len(uniq), replace=True)
                idx = np.concatenate([by_patient[q] for q in pick])
                cg, sg = c_gt[idx].sum(), s_gt[idx].sum()
                if cg == 0 or sg == 0:
                    boots[b] = np.nan
                else:
                    boots[b] = s_tp[idx].sum() / sg - c_tp[idx].sum() / cg
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            # Two-sided bootstrap p against delta = 0.
            p_boot = 2.0 * min(np.mean(boots <= 0), np.mean(boots >= 0))
            rows.append({
                "model": model, "label": label,
                "n_images": len(ids), "n_patients": len(uniq),
                "recall_clean": round(float(rc), 6),
                "recall_severe": round(float(rs), 6),
                "delta_recall": round(float(delta), 6),
                "ci_lo": round(float(lo), 6), "ci_hi": round(float(hi), 6),
                "p_boot": round(float(min(1.0, max(0.0, p_boot))), 6),
            })
        print(f"done {model}: {len(ids)} images, {len(uniq)} patients")

    tab = pd.DataFrame(rows)
    tab["q_bh"] = np.nan
    for model, g in tab.groupby("model"):
        q = bh(g["p_boot"].to_numpy(dtype=float))
        tab.loc[g.index, "q_bh"] = np.round(q, 6)
    tab.to_csv(OUT / "per_label_bootstrap_fdr.csv", index=False,
               encoding="utf-8-sig")
    printable = tab.copy()
    for c in ["recall_clean", "recall_severe", "delta_recall", "ci_lo", "ci_hi"]:
        printable[c] = (printable[c].astype(float) * 100).round(2)
    print(printable.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
