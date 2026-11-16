#!/usr/bin/env python
"""Rule-based audit of explicit image-quality language in model replies.

Why this script exists
----------------------
Section 5.3 measures *marker-based* noticeability (the reasoning-channel
opener).  A reviewer can ask whether the model says the image is degraded in
words instead of via that marker.  This script answers that with a transparent
rule: count replies whose raw text contains an explicit quality term.  It is a
lower bound (a paraphrase without these stems is missed), and it is reported as
such.

Conditions: clean, noise/severe, resolution/severe on the NIH grid.
Output: outputs/analysis_sensitivity/quality_language_audit.csv
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis_sensitivity"
MATRIX = ROOT / "outputs" / "matrix" / "nih_eval"

MODELS = [
    "internvl3_5_8b", "medgemma_15_4b_it", "medgemma_4b_it",
    "qwen3_vl_8b_instruct", "qwen25_vl_7b_instruct", "chexagent_8b",
]
CONDITIONS = [("clean", None), ("noise", "severe"), ("resolution", "severe")]

QUALITY = re.compile(
    r"\b(noise|noisy|degrad\w*|quality|artifact\w*|artefact\w*|blur\w*|corrupt\w*|grain\w*)\b",
    re.IGNORECASE,
)
EXPLICIT = re.compile(
    r"(image|scan|radiograph|study)\s+(is|appears|looks|was)\s+[^.]{0,40}"
    r"(noisy|degraded|poor|low.quality|blurry|corrupt)",
    re.IGNORECASE,
)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODELS:
        p = MATRIX / model / "shift" / "shift_predictions.jsonl"
        if not p.exists():
            continue
        counts = {(c, s): {"n": 0, "kw": 0, "explicit": 0} for c, s in CONDITIONS}
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            key = (r.get("condition"), r.get("severity"))
            if key not in counts:
                continue
            text = (r.get("raw_output") or "").strip()
            counts[key]["n"] += 1
            if QUALITY.search(text):
                counts[key]["kw"] += 1
            if EXPLICIT.search(text):
                counts[key]["explicit"] += 1
        for (cond, sev), v in counts.items():
            if v["n"] == 0:
                continue
            rows.append({
                "model": model,
                "condition": cond,
                "severity": sev or "none",
                "n": v["n"],
                "quality_keyword_rate": round(v["kw"] / v["n"], 6),
                "explicit_quality_sentence_rate": round(v["explicit"] / v["n"], 6),
            })
    with open(OUT / "quality_language_audit.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['model']:26s} {r['condition']:10s} n={r['n']:4d} "
              f"kw={r['quality_keyword_rate']:.4f} explicit={r['explicit_quality_sentence_rate']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
