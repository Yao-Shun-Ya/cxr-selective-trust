#!/usr/bin/env python
"""89 - Post-hoc fixed-threshold check for the operating-point cells.

CPU-only, read-only.  Table 6 labels three cells an *operating-point shift*
rather than a proven threshold change.  The frozen predictions carry a
verbalised ``probability_positive`` per finding, so we can ask a cheaper
question without re-running inference:

* does a fixed 0.5 cut on the verbalised probability reproduce the status-based
  recall/precision/PPR change?
* does a cut chosen on clean images to match the model's clean positive count
  reproduce it?

If neither reproduces the status-based change, the shift is not a simple
post-hoc threshold on the verbalised probability; a true rerun with a fixed
decision rule is still required.  We report this as a negative/ambiguous
result, not as a mechanism proof.

Inputs:
  outputs/matrix/nih_eval/<model>/shift/shift_predictions.jsonl

Output:
  outputs/analysis_sensitivity/fixed_threshold_posthoc.csv

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\89_fixed_threshold_posthoc.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis_sensitivity"
MATRIX = ROOT / "outputs" / "matrix" / "nih_eval"

PATHOLOGY = ["atelectasis", "cardiomegaly", "consolidation", "edema",
             "pleural_effusion", "pneumonia", "pneumothorax"]
CELLS = [("internvl3_5_8b", "contrast"), ("internvl3_5_8b", "brightness"),
         ("qwen3_vl_8b_instruct", "contrast")]


def load_cell(model: str, condition: str):
    path = MATRIX / model / "shift" / "shift_predictions.jsonl"
    clean, severe = {}, {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("condition") == "clean":
                clean[r["image_id"]] = r
            elif (r.get("condition") == condition
                  and r.get("severity") == "severe"):
                severe[r["image_id"]] = r
    return clean, severe


def counts(rec: dict, rule: str, thresholds: dict[str, float] | None = None):
    gt = rec.get("ground_truth") or {}
    findings = (rec.get("parsed") or {}).get("findings") or {}
    tp = gtp = pp = 0
    for label in PATHOLOGY:
        y = str(gt.get(f"label_{label}") or "").lower() == "positive"
        fp = findings.get(label) or {}
        if rule == "status":
            pred = str(fp.get("status") or "") == "positive"
        else:
            p = fp.get("probability_positive")
            t = 0.5 if thresholds is None else thresholds.get(label, 0.5)
            pred = isinstance(p, (int, float)) and float(p) >= t
        gtp += int(y)
        pp += int(pred)
        tp += int(y and pred)
    return tp, gtp, pp


def pooled(records: list[dict], rule: str, thresholds=None):
    tp = gtp = pp = 0
    for r in records:
        a, b, c = counts(r, rule, thresholds)
        tp += a
        gtp += b
        pp += c
    rec = tp / gtp if gtp else float("nan")
    prec = tp / pp if pp else float("nan")
    ppr = pp / (len(records) * len(PATHOLOGY)) if records else float("nan")
    return rec, prec, ppr


def clean_matched_thresholds(clean: list[dict]) -> dict[str, float]:
    """Per label, the threshold that best matches the clean status-positive count."""
    out = {}
    for label in PATHOLOGY:
        wanted = sum(
            1 for r in clean
            if str(((r.get("parsed") or {}).get("findings", {}).get(label, {})
                    or {}).get("status") or "") == "positive"
        )
        vals = []
        for r in clean:
            p = ((r.get("parsed") or {}).get("findings", {}).get(label, {})
                 or {}).get("probability_positive")
            if isinstance(p, (int, float)):
                vals.append(float(p))
        if not vals:
            out[label] = 0.5
            continue
        best, best_gap = 0.5, 10**9
        for t in sorted(set(vals)):
            n = sum(1 for v in vals if v >= t)
            gap = abs(n - wanted)
            if gap < best_gap:
                best, best_gap = t, gap
        out[label] = best
    return out


def main() -> int:
    rows = []
    for model, condition in CELLS:
        clean, severe = load_cell(model, condition)
        ids = sorted(set(clean) & set(severe))
        c = [clean[i] for i in ids]
        s = [severe[i] for i in ids]
        th = clean_matched_thresholds(c)

        for rule, thresholds in [("status", None), ("p>=0.5", None),
                                 ("p>=clean_matched", th)]:
            rc, pc, qc = pooled(c, "p" if rule != "status" else "status",
                                thresholds if rule == "p>=clean_matched" else None)
            rs, ps, qs = pooled(s, "p" if rule != "status" else "status",
                                thresholds if rule == "p>=clean_matched" else None)
            rows.append({
                "model": model, "condition": condition, "rule": rule,
                "n_images": len(ids),
                "recall_clean": round(rc, 6), "recall_severe": round(rs, 6),
                "d_recall": round(rs - rc, 6),
                "precision_clean": round(pc, 6), "precision_severe": round(ps, 6),
                "d_precision": round(ps - pc, 6),
                "ppr_clean": round(qc, 6), "ppr_severe": round(qs, 6),
                "d_ppr": round(qs - qc, 6),
            })
        print(f"{model} {condition}: " + ", ".join(
            f"{r['rule']} dR={r['d_recall']*100:+.2f}" for r in rows[-3:]))

    tab = list(rows)
    with open(OUT / "fixed_threshold_posthoc.csv", "w", newline="",
              encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tab[0]))
        w.writeheader()
        w.writerows(tab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
