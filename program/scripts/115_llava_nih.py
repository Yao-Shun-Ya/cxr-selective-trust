#!/usr/bin/env python
"""115 - LLaVA-Med on a NIH subset: is it still a two-template model in-domain?

The paper reports LLaVA-Med as a measured exclusion on PadChest and Open-i but
did not run the NIH grid.  This script summarises a 200-study NIH subset
(original + blank) and reports the same responsiveness statistics as script 78,
so the exclusion can be stated as measured in-domain too.

Input:
  outputs/llava_nih/llava_med_v1_5_7b/image_reliance_predictions.jsonl

Output:
  outputs/analysis_responsiveness_llava/nih_subset.csv

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\115_llava_nih.py
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "llava_nih" / "llava_med_v1_5_7b" / "image_reliance_predictions.jsonl"
OUT = ROOT / "outputs" / "analysis_responsiveness_llava" / "nih_subset.csv"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def main() -> int:
    if not RUN.exists():
        raise SystemExit(f"missing {RUN}")
    rows = [json.loads(l) for l in RUN.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    out = []
    for cond, g in df.groupby("condition"):
        replies = [norm(x) for x in g["raw_output"].fillna("")]
        nonempty = [x for x in replies if x]
        c = Counter(nonempty)
        out.append({
            "condition": cond, "n": len(g),
            "nonempty": len(nonempty),
            "distinct": len(c),
            "top_share": round(c.most_common(1)[0][1] / len(nonempty), 4) if c else None,
            "top_reply": (c.most_common(1)[0][0][:160] if c else ""),
            "parse_rate": round(float(g["parsed"].apply(
                lambda x: bool((x or {}).get("parse_success"))).mean()), 4),
            "error_rate": round(float(g["error"].notna().mean()), 4),
        })
    res = pd.DataFrame(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(res.to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
