#!/usr/bin/env python
"""99 - How much budget do the truncated Thinking replies actually need?

CPU-only, read-only. The grid ran at 2,560 tokens and 4,280 of 10,542 replies
were cut off inside the reasoning block (no closing tag, no JSON). Re-running
those cells is only worth the GPU hours if the extra budget is likely to
finish them, so this script compares:

* the character-length distribution of replies that DID close (parsed rows);
* the length of the replies that were cut off (they all stop at the cap).

A closed reply's distribution tells us where the cap should sit. If the 99th
percentile of closed replies is well under the cap, the truncated ones are
likely just above it and a modest increase (3,584-4,096 tokens) should rescue
most of them; if the capped ones are far beyond that, the model genuinely rambles
and the non-termination is itself the measurement.

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\99_truncation_budget.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID = (ROOT / "outputs" / "matrix" / "nih_eval" / "qwen3_vl_8b_thinking"
        / "shift" / "shift_predictions.jsonl")
# The runner's tokenizer is Qwen-style; ~3.6-4.0 characters per token on this
# model's English prose. Both bounds are reported so the choice of cap is not
# an artefact of one guess.
CHARS_PER_TOKEN = (3.6, 4.0)


def main() -> int:
    rows = []
    with GRID.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    closed = [len(r.get("raw_output") or "") for r in rows
              if (r.get("parsed") or {}).get("parse_success")]
    cut = [len(r.get("raw_output") or "") for r in rows
           if not (r.get("parsed") or {}).get("parse_success")]
    print(f"rows {len(rows)}  closed {len(closed)}  cut {len(cut)}")
    for name, arr in (("closed", closed), ("cut", cut)):
        if not arr:
            print(f"{name}: none (every reply is strictly valid)")
            continue
        a = np.array(arr, dtype=float)
        print(f"{name}: p50={np.percentile(a,50):.0f} p90={np.percentile(a,90):.0f} "
              f"p95={np.percentile(a,95):.0f} p99={np.percentile(a,99):.0f} max={a.max():.0f}")
    lo, hi = CHARS_PER_TOKEN
    print("\nimplied tokens (chars / chars-per-token):")
    for name, arr, pct in (("closed p99", closed, 99), ("cut p50", cut, 50),
                           ("cut p90", cut, 90)):
        if not arr:
            continue
        v = float(np.percentile(np.array(arr, dtype=float), pct))
        print(f"  {name}: {v/lo:.0f} - {v/hi:.0f} tokens")
    # A cap that would have covered X % of the closed replies.
    for cap_tokens in (2816, 3072, 3584, 4096, 5120, 6144):
        need = cap_tokens * hi  # conservative: most characters per token
        share = float(np.mean(np.array(closed) <= need)) if closed else 0.0
        print(f"  cap {cap_tokens}: covers {share:.1%} of closed replies at {hi} chars/token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
