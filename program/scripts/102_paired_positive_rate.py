#!/usr/bin/env python
"""102 - Is the operating-point move of the second reasoning model an artefact
of coverage?

The paper reports that Qwen3-VL-8B-Thinking calls fewer pathology slots positive
under damage (0.250 -> 0.125).  Because coverage also falls (92.6% -> 80.9%), an
unpaired comparison could be driven by *which* studies the model finished.  This
script recomputes the positive-call rate on the paired subset only: studies that
produced a usable answer in both the clean cell and the severe cell.

Output: outputs/analysis_round2/paired_positive_rate.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GRID = (ROOT / "outputs/matrix/nih_eval/qwen3_vl_8b_thinking/shift"
        / "shift_predictions.jsonl")
OUT = ROOT / "outputs/analysis_round2"

LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax"]


def positive_rate(rows: list[dict]) -> float:
    hits = 0
    for r in rows:
        findings = (r.get("parsed") or {}).get("findings") or {}
        hits += sum(1 for l in LABELS
                    if str((findings.get(l) or {}).get("status")) == "positive")
    return hits / (len(rows) * len(LABELS)) if rows else float("nan")


def main() -> int:
    rows = [json.loads(line) for line in GRID.open(encoding="utf-8") if line.strip()]
    cells: dict[str, dict[str, dict]] = {}
    for r in rows:
        cells.setdefault(f"{r['condition']}/{r['severity']}", {})[str(r["image_id"])] = r

    clean = cells["clean/none"]
    out = []
    for cell, imgs in sorted(cells.items()):
        if cell == "clean/none":
            continue
        common = sorted(set(imgs) & set(clean))
        rc = positive_rate([clean[i] for i in common])
        rs = positive_rate([imgs[i] for i in common])
        out.append({
            "cell": cell,
            "n_paired": len(common),
            "pos_rate_clean_paired": round(rc, 4),
            "pos_rate_cell_paired": round(rs, 4),
            "delta_pos_rate_paired": round(rs - rc, 4),
            "pos_rate_clean_unpaired": round(positive_rate(list(clean.values())), 4),
            "pos_rate_cell_unpaired": round(positive_rate(list(imgs.values())), 4),
            "delta_pos_rate_unpaired": round(
                positive_rate(list(imgs.values())) - positive_rate(list(clean.values())), 4),
        })
    df = pd.DataFrame(out)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "paired_positive_rate.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
