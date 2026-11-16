#!/usr/bin/env python
"""116 - Paraphrase agreement on Qwen3-VL against its own v0 baseline (083).

Mirrors the MedGemma-4B paraphrase table: two paraphrases of v0 are compared
with v0 on the same 1,506 studies, per label, on real and blank images.

Output:
  outputs/analysis_round2/xfamily_paraphrase.csv

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\116_paraphrase_xfamily.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "schema_xfamily" / "qwen3_vl_8b_instruct"
OUT = ROOT / "outputs" / "analysis_round2" / "xfamily_paraphrase.csv"

LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax", "no_finding"]


def read(variant: str, condition: str) -> dict[str, dict]:
    p = RUN / variant / condition / "image_reliance_predictions.jsonl"
    out = {}
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[str(r.get("image_id"))] = r
    return out


def status(rec: dict, label: str) -> str:
    f = (rec.get("parsed") or {}).get("findings") or {}
    return str((f.get(label) or {}).get("status") or "")


def main() -> int:
    v0 = {"original": read("v0_simple", "original"),
          "blank": read("v0_simple", "blank")}
    rows = []
    for variant in ["p1_paraphrase_short", "p2_paraphrase_second_reader"]:
        for condition in ["original", "blank"]:
            other = read(variant, condition)
            base = v0[condition]
            common = sorted(set(base) & set(other))
            if not common:
                continue
            per = []
            for label in LABELS:
                a = [status(base[i], label) for i in common]
                b = [status(other[i], label) for i in common]
                agree = float(np.mean([x == y for x, y in zip(a, b)]))
                try:
                    kappa = float(cohen_kappa_score(a, b))
                except ValueError:
                    kappa = float("nan")
                per.append((label, agree, kappa))
            rows.append({
                "variant": variant, "condition": condition, "n": len(common),
                "mean_agreement": round(float(np.mean([p[1] for p in per])), 4),
                "mean_kappa": round(float(np.nanmean([p[2] for p in per])), 4),
                "no_finding_agreement": round(
                    float([p[1] for p in per if p[0] == "no_finding"][0]), 4),
                "pathology_agreement_min": round(
                    float(min(p[1] for p in per if p[0] != "no_finding")), 4),
            })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
