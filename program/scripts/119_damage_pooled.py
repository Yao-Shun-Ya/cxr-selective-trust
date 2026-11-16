#!/usr/bin/env python
"""119 - Pooled recall change for the two-axis abstract sentence.

The abstract compares the damage of noise with that of resolution loss.  The
number must be reproducible from the frozen predictions, so this script pools
the seven pathology slots over the four readable models (InternVL3.5,
MedGemma-1.5, MedGemma-4B, Qwen3-VL), pairing each degraded cell with the clean
cell of the same model.

Output:
  outputs/analysis_sensitivity/damage_pooled.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data" / "manifests" / "nih_eval_1506.csv"
LAB = ["atelectasis", "cardiomegaly", "consolidation", "edema",
       "pleural_effusion", "pneumonia", "pneumothorax"]
MODELS = ["internvl3_5_8b", "medgemma_15_4b_it", "medgemma_4b_it",
          "qwen3_vl_8b_instruct"]


def pooled(pred: Path, truth: dict, condition: str, severity: str | None) -> tuple[int, int]:
    tp = pos = 0
    with pred.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("condition") != condition:
                continue
            if severity is not None and r.get("severity") != severity:
                continue
            if not (r.get("parsed") or {}).get("parse_success"):
                continue
            f = (r.get("parsed") or {}).get("findings") or {}
            row = truth[r["image_id"]]
            for lab in LAB:
                if row[lab]:
                    pos += 1
                    if (f.get(lab) or {}).get("status") == "positive":
                        tp += 1
    return tp, pos


def main() -> int:
    man = pd.read_csv(MAN, low_memory=False)
    truth = {
        str(r["image_id"]): {l: str(r.get(f"label_{l}") or "").lower() == "positive"
                             for l in LAB}
        for _, r in man.iterrows()
    }
    rows = []
    for cond in ("noise", "resolution"):
        tp_c = pos_c = tp_s = pos_s = 0
        for model in MODELS:
            p = ROOT / "outputs/matrix/nih_eval" / model / "shift" / "shift_predictions.jsonl"
            a, b = pooled(p, truth, "clean", None)
            c, d = pooled(p, truth, cond, "severe")
            tp_c += a; pos_c += b; tp_s += c; pos_s += d
        rows.append({
            "condition": cond,
            "models": ";".join(MODELS),
            "clean_recall": round(tp_c / pos_c, 6),
            "severe_recall": round(tp_s / pos_s, 6),
            "delta": round(tp_s / pos_s - tp_c / pos_c, 6),
        })
    out = ROOT / "outputs/analysis_sensitivity/damage_pooled.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
