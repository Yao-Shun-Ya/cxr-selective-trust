#!/usr/bin/env python
"""Patient-cluster bootstrap for the operating-point cells.

Why this script exists
----------------------
Table 5 reports severe-vs-clean changes as point estimates.  A reviewer can
ask for intervals.  This script resamples patients (all images of a patient
move together) and recomputes the paired change for the three cells labelled
``operating-point shift'': InternVL3.5 contrast, InternVL3.5 brightness, and
Qwen3-VL contrast.  Clean and degraded rows are joined on image_id.

Output: outputs/analysis_sensitivity/operating_point_ci.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis_sensitivity"
MATRIX = ROOT / "outputs" / "matrix" / "nih_eval"
MAN = ROOT / "data" / "manifests" / "nih_eval_1732.csv"

PATHOLOGY = ["atelectasis", "cardiomegaly", "consolidation", "edema",
             "pleural_effusion", "pneumonia", "pneumothorax"]

CELLS = [("internvl3_5_8b", "contrast"), ("internvl3_5_8b", "brightness"),
         ("qwen3_vl_8b_instruct", "contrast")]
N_BOOT = 1000
SEED = 0


def load_manifest() -> dict[str, str]:
    out = {}
    for r in csv.DictReader(open(MAN, encoding="utf-8-sig")):
        out[r["image_id"]] = str(r["patient_id"])
    return out


def image_counts(rec: dict) -> tuple[int, int, int]:
    """(tp, gt_pos, pred_pos) over the seven pathology labels."""
    gt = rec.get("ground_truth") or {}
    findings = (rec.get("parsed") or {}).get("findings") or {}
    tp = gt_pos = pred_pos = 0
    for label in PATHOLOGY:
        y = str(gt.get(f"label_{label}") or "").lower() == "positive"
        p = str((findings.get(label) or {}).get("status") or "") == "positive"
        gt_pos += int(y)
        pred_pos += int(p)
        tp += int(y and p)
    return tp, gt_pos, pred_pos


def pooled(rows: list[tuple[int, int, int]]) -> tuple[float, float, float]:
    tp = sum(r[0] for r in rows)
    gt = sum(r[1] for r in rows)
    pp = sum(r[2] for r in rows)
    recall = tp / gt if gt else float("nan")
    precision = tp / pp if pp else float("nan")
    ppr = pp / (len(rows) * len(PATHOLOGY)) if rows else float("nan")
    return recall, precision, ppr


def main() -> int:
    patients = load_manifest()
    rows = []
    for model, condition in CELLS:
        p = MATRIX / model / "shift" / "shift_predictions.jsonl"
        clean, sev = {}, {}
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("condition") == "clean":
                clean[r["image_id"]] = image_counts(r)
            elif (r.get("condition") == condition and r.get("severity") == "severe"):
                sev[r["image_id"]] = image_counts(r)
        ids = sorted(set(clean) & set(sev))
        if not ids:
            continue
        pts = {i: patients.get(i, i) for i in ids}
        uniq = sorted(set(pts.values()))
        by_patient = {q: [i for i in ids if pts[i] == q] for q in uniq}

        c_all = [clean[i] for i in ids]
        s_all = [sev[i] for i in ids]
        rc, pc, qc = pooled(c_all)
        rs, ps, qs = pooled(s_all)
        point = (rs - rc, ps - pc, qs - qc)

        rng = np.random.default_rng(SEED)
        boots = []
        for _ in range(N_BOOT):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = [i for q in pick for i in by_patient[q]]
            rc_b, pc_b, qc_b = pooled([clean[i] for i in idx])
            rs_b, ps_b, qs_b = pooled([sev[i] for i in idx])
            boots.append((rs_b - rc_b, ps_b - pc_b, qs_b - qc_b))

        def ci(k: int) -> tuple[float, float]:
            v = np.asarray([b[k] for b in boots], dtype=float)
            v = v[np.isfinite(v)]
            return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

        r_ci, p_ci, q_ci = ci(0), ci(1), ci(2)
        rows.append({
            "model": model, "condition": condition, "n_images": len(ids),
            "n_patients": len(uniq),
            "d_recall": round(point[0], 6),
            "d_recall_lo": round(r_ci[0], 6), "d_recall_hi": round(r_ci[1], 6),
            "d_precision": round(point[1], 6),
            "d_precision_lo": round(p_ci[0], 6), "d_precision_hi": round(p_ci[1], 6),
            "d_ppr": round(point[2], 6),
            "d_ppr_lo": round(q_ci[0], 6), "d_ppr_hi": round(q_ci[1], 6),
        })
        print(f"{model} {condition}: d_recall={point[0]*100:+.2f} "
              f"[{r_ci[0]*100:+.2f},{r_ci[1]*100:+.2f}] pp; "
              f"d_precision={point[1]*100:+.2f} [{p_ci[0]*100:+.2f},{p_ci[1]*100:+.2f}]; "
              f"d_ppr={point[2]*100:+.2f} [{q_ci[0]*100:+.2f},{q_ci[1]*100:+.2f}]")

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "operating_point_ci.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
