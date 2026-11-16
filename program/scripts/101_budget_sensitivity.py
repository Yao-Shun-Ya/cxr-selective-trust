#!/usr/bin/env python
"""101 - Budget sensitivity of the reasoning model (CPU only, read-only).

The grid needed three token budgets before most replies could finish:
2,560 -> 3,584 -> 5,120. That is not just plumbing: the share of replies that
fail to close a reasoning block, and the length of the trace that does close,
are both measurements of how much "thinking" the model spends on an image.

This script reads the archive written by `filter_strict_valid_rows.py`
(every reply that was NOT a usable answer, tagged with the budget that produced
it) and reports, per degradation cell and per budget:

  * attempts, close rate, mean/median characters;
  * how many replies still had no usable answer at each budget.

Output: outputs/analysis_round2/budget_sensitivity.csv
Usage:  .\\.venv-vl\\Scripts\\python.exe scripts\\101_budget_sensitivity.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs" / "analysis_round2" / "thinking_invalid_archive.jsonl"
OUT = ROOT / "outputs" / "analysis_round2"


def main() -> int:
    if not ARCHIVE.exists():
        print(f"[skip] no archive at {ARCHIVE}")
        return 0
    rows = [json.loads(l) for l in ARCHIVE.open(encoding="utf-8") if l.strip()]
    print(f"archived invalid replies: {len(rows)}")

    by = defaultdict(list)
    for r in rows:
        by[(r.get("pass"), f"{r.get('condition')}/{r.get('severity')}")].append(r)

    out = []
    for (pass_label, cell), rr in sorted(by.items()):
        chars = [r.get("chars") or 0 for r in rr]
        out.append({
            "pass": pass_label, "cell": cell, "n_unusable": len(rr),
            "close_rate": round(sum(1 for r in rr if r.get("has_close")) / len(rr), 4),
            "mean_chars": round(float(np.mean(chars)), 1),
            "median_chars": round(float(np.median(chars)), 1),
        })
    tab = pd.DataFrame(out)
    tab.to_csv(OUT / "budget_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(tab.to_string(index=False))

    print("\nper cell, unusable replies by budget pass:")
    piv = tab.pivot_table(index="cell", columns="pass", values="n_unusable", aggfunc="sum").fillna(0)
    print(piv.astype(int).to_string())
    print(f"\nwrote {OUT / 'budget_sensitivity.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
