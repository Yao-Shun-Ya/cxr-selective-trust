#!/usr/bin/env python
"""112 - Cross-family schema and paraphrase analysis for the Qwen3-VL sweep.

Reads outputs/schema_xfamily/qwen3_vl_8b_instruct/<variant>/<condition>/
image_reliance_predictions.jsonl and reports, per cell:

  * parse rate and contract-violation rate,
  * pooled pathology recall and no_finding positive rate on real images,
  * pooled pathology recall and no_finding positive rate on blank images,
  * probability completeness.

Variants v0/v1/v4 answer the cross-family schema question; p1/p2 answer the
paraphrase question on a second model.

Output:
  outputs/analysis_round2/xfamily_schema.csv

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\112_analyze_xfamily_schema.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "schema_xfamily" / "qwen3_vl_8b_instruct"
MAN = ROOT / "data" / "manifests" / "nih_eval_1506.csv"
OUT = ROOT / "outputs" / "analysis_round2" / "xfamily_schema.csv"

PATHOLOGY = ["atelectasis", "cardiomegaly", "consolidation", "edema",
             "pleural_effusion", "pneumonia", "pneumothorax"]
LABELS = PATHOLOGY + ["no_finding"]


def load_truth() -> dict[str, dict]:
    d = pd.read_csv(MAN, low_memory=False)
    return {str(r["image_id"]): {l: str(r.get(f"label_{l}") or "").lower() == "positive"
                                 for l in LABELS}
            for _, r in d.iterrows()}


def status(rec: dict, label: str) -> str:
    f = (rec.get("parsed") or {}).get("findings") or {}
    return str((f.get(label) or {}).get("status") or "")


def prob(rec: dict, label: str) -> float | None:
    f = (rec.get("parsed") or {}).get("findings") or {}
    v = (f.get(label) or {}).get("probability_positive")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if 0.0 <= v <= 1.0 else None


def cell(path: Path, truth: dict) -> dict:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in recs if (r.get("parsed") or {}).get("parse_success")]
    path_pairs = [(truth.get(str(r.get("image_id")), {}).get(l, False), status(r, l))
                  for r in ok for l in PATHOLOGY]
    den = sum(1 for y, _ in path_pairs if y)
    rec = sum(1 for y, s in path_pairs if y and s == "positive") / den if den else None
    return {
        "n": len(recs),
        "parse_rate": round(len(ok) / len(recs), 4) if recs else None,
        "contract_violation_rate": round(
            sum(1 for r in recs if (r.get("contract") or {}).get("contract_violation")) / len(recs), 4)
        if recs else None,
        "pathology_recall": round(rec, 4) if rec is not None else None,
        "no_finding_positive_rate": round(
            sum(1 for r in ok if status(r, "no_finding") == "positive") / len(ok), 4) if ok else None,
        "probability_completeness": round(
            sum(1 for r in ok for l in LABELS if prob(r, l) is not None) / (len(ok) * len(LABELS)), 4)
        if ok else None,
    }


def main() -> int:
    truth = load_truth()
    rows = []
    for path in sorted(RUN.glob("*/*/image_reliance_predictions.jsonl")):
        variant, condition = path.parent.parent.name, path.parent.name
        row = {"model": "qwen3_vl_8b_instruct", "variant": variant, "condition": condition}
        row.update(cell(path, truth))
        rows.append(row)
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
