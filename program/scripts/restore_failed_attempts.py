#!/usr/bin/env python
"""Re-insert already-attempted-but-failed rows so `--resume` does not redo them.

Why this is needed
------------------
`DurableJsonl` treats a key as done when a row exists for it, regardless of
whether that row holds a usable answer. The strict filter therefore removes
failed rows so they get retried at a bigger budget -- correct when the budget
changed, wrong when it did not: a greedy model given the same prompt and the
same cap produces the same non-terminating trace, so re-running it is pure
waste.

This script restores rows from a backup that are missing from the live grid, so
the runner skips them. Use it when the remaining failures already failed at the
*current* budget.

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\restore_failed_attempts.py \
      --backup outputs/matrix/nih_eval/qwen3_vl_8b_thinking/shift/shift_predictions.jsonl.bak_20261110_140813 \
      --grid   outputs/matrix/nih_eval/qwen3_vl_8b_thinking/shift/shift_predictions.jsonl \
      --cap 5120
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def key(rec: dict) -> tuple:
    return (str(rec.get("image_id")), str(rec.get("condition")), str(rec.get("severity")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backup", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--cap", type=int, required=True,
                    help="token budget the restored rows already failed at")
    args = ap.parse_args()

    backup = Path(args.backup)
    grid = Path(args.grid)
    for p in (backup, grid):
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise SystemExit(f"missing {p}")
    backup = backup if backup.is_absolute() else ROOT / backup
    grid = grid if grid.is_absolute() else ROOT / grid

    have = set()
    with grid.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                have.add(key(json.loads(line)))

    add = []
    with backup.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            k = key(rec)
            if k in have:
                continue
            rec["attempted_cap"] = args.cap
            add.append(json.dumps(rec, ensure_ascii=False) + "\n")
            have.add(k)

    if not add:
        print("[restore] nothing missing; grid unchanged")
        return 0

    with grid.open("a", encoding="utf-8") as fh:
        fh.write("".join(add))
        fh.flush()
        os.fsync(fh.fileno())
    sidecar = grid.with_suffix(grid.suffix + ".progress.json")
    if sidecar.exists():
        sidecar.unlink()
    print(f"[restore] appended {len(add)} previously-attempted rows "
          f"(cap {args.cap}); they will be skipped by the next run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
