#!/usr/bin/env python
"""Prepare `outputs/pretest` for re-running the model/stage combinations that failed.

Why a preparation step is needed at all
---------------------------------------
Re-running the same driver command does NOT redo them, for two independent reasons:

1. **Stage level.** `outputs/state/pretest_state.json` marks every stage of the
   pre-run as complete -- including the three that produced no usable data,
   because the run happened before `Test-StageOutput` was strengthened. The driver
   skips completed stages (`[skip] ... (already complete)`), so those stages would
   never start. The retry therefore uses a SEPARATE state file; the original stays
   untouched as the record of what the pre-run actually did.

2. **Cell level.** Every runner writes through `DurableJsonl`, whose "already done"
   set is keyed on rows whose `error` field is empty -- NOT on whether the reply
   parsed. An empty reply from llava and a truncated `<unused94>thought` trace from
   MedGemma-1.5 both have `error: ""`, so both are treated as finished work and
   skipped, no matter what the new token budget or config would produce.

So the cells have to be released first. This script does that per cell, and
**backs up every file it touches** before touching it:

* `clear` -- the whole file moves aside (its rows are unusable, or were produced by
  a configuration that must not be mixed with the retry);
* `topup` -- only the rows that failed to parse are dropped; the rows that parsed
  are kept, because they are valid data. A token cap that was not hit does not
  change a reply, so the kept rows remain comparable with the retried ones.

Backups land in `outputs/pretest_retry_backup/<model>/<stage>/`, so the truncated
thought traces (evidence for docs/50 section 2.8) and the empty llava replies
(evidence for section 2.7) survive the cleanup.

Usage::

    .venv-vl\\Scripts\\python.exe scripts\\65_prepare_prerun_retry.py            # dry run
    .venv-vl\\Scripts\\python.exe scripts\\65_prepare_prerun_retry.py --apply
    .venv-vl\\Scripts\\python.exe scripts\\65_prepare_prerun_retry.py --apply --mode full
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRETEST = ROOT / "outputs" / "pretest" / "nih_eval"
BACKUP = ROOT / "outputs" / "pretest_retry_backup"

# (model, stage dir, file name, policy)
#
# POLICY, and the reason for each one -- this is the part worth reading:
#
#  clear  llava_med shift/reliance : every row is an empty string AND was produced
#                                    with the broken `rope_theta` (docs/50 2.7).
#                                    Nothing in them is usable or comparable.
#  clear  medgemma_15 precision    : 0 of 400 rows parsed (all truncated thought
#                                    traces) -- see docs/50 2.8.
#  clear  chexagent shift/reliance : the 8 existing shift rows were produced while
#                                    the model was paging through the Windows
#                                    driver in **fp32**; the retry runs the fp16
#                                    checkpoint. One comparison, one precision.
#  topup  medgemma_15 shift/reliance: 89.9% / 63.5% of the rows parsed at the old
#                                    256-token budget. Those replies are complete
#                                    and unaffected by raising the cap; only the
#                                    rows that ran out of budget are dropped and
#                                    retried at 1024.
CELLS = [
    ("llava_med_v1_5_7b", "shift", "shift_predictions.jsonl", "clear"),
    ("llava_med_v1_5_7b", "reliance", "image_reliance_predictions.jsonl", "clear"),
    ("medgemma_15_4b_it", "precision", "image_reliance_predictions.jsonl", "clear"),
    ("medgemma_15_4b_it", "shift", "shift_predictions.jsonl", "topup"),
    ("medgemma_15_4b_it", "reliance", "image_reliance_predictions.jsonl", "topup"),
    ("chexagent_8b", "shift", "shift_predictions.jsonl", "clear"),
    ("chexagent_8b", "reliance", "image_reliance_predictions.jsonl", "clear"),
]

STAGE_DIR = {"precision": "reliance_bf16"}   # the bf16 control writes to reliance_bf16


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="do it; without this flag nothing is written")
    ap.add_argument("--mode", choices=["policy", "full"], default="policy",
                    help="policy = the table in this file (clear the unusable cells, "
                         "keep the parsed rows elsewhere); full = clear every cell")
    ap.add_argument("--pretest-root", default=str(PRETEST))
    ap.add_argument("--backup-root", default=str(BACKUP))
    args = ap.parse_args()

    pretest = Path(args.pretest_root)
    backup = Path(args.backup_root)
    if not pretest.exists():
        print(f"ERROR: {pretest} does not exist")
        return 1

    print(f"pretest root : {pretest}")
    print(f"backup root  : {backup}")
    print(f"mode         : {args.mode}{'  (DRY RUN -- nothing will be written)' if not args.apply else ''}")
    print()

    plan: list[tuple[str, str, int, str]] = []
    for model, stage, fname, policy in CELLS:
        if args.mode == "full":
            policy = "clear"
        cell = pretest / model / STAGE_DIR.get(stage, stage)
        path = cell / fname
        if not path.exists():
            print(f"  [--] {model}/{stage}: no {fname} (nothing to prepare)")
            continue
        rows = read_rows(path)
        parsed = sum(1 for r in rows if (r.get("parsed") or {}).get("parse_success"))
        if policy == "clear":
            action = f"clear: move all {len(rows)} rows aside"
            remaining = 0
        else:
            action = f"topup: drop {len(rows) - parsed} unparsed, keep {parsed}"
            remaining = parsed
        plan.append((model, stage, remaining, policy))
        print(f"  [{policy:5s}] {model}/{stage}")
        print(f"          file     : {path.relative_to(ROOT)}")
        print(f"          rows     : {len(rows)} ({parsed} parsed, {len(rows) - parsed} not)")
        print(f"          action   : {action}")

    if not plan:
        print("\nnothing to do")
        return 0

    total_expected = sum(r for _, _, r, _ in plan)
    print(f"\nAfter this, {len(plan)} cell(s) will be released for re-running, "
          f"{total_expected} existing rows kept in place.")

    if not args.apply:
        print("\nDRY RUN: re-run with --apply to do it. Every touched file is copied to")
        print(f"          {backup.relative_to(ROOT)} first.")
        return 0

    for model, stage, fname, policy in CELLS:
        policy = "clear" if args.mode == "full" else policy
        cell = pretest / model / STAGE_DIR.get(stage, stage)
        path = cell / fname
        if not path.exists():
            continue
        dest = backup / model / stage
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest / fname)
        sidecar = Path(str(path) + ".progress.json")
        if sidecar.exists():
            shutil.copy2(sidecar, dest / sidecar.name)
        if policy == "clear":
            path.unlink()
            if sidecar.exists():
                sidecar.unlink()
            print(f"  cleared  {model}/{stage}  (backup: {dest.relative_to(ROOT)})")
        else:
            kept = [r for r in read_rows(path)
                    if (r.get("parsed") or {}).get("parse_success")]
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                            encoding="utf-8")
            # The sidecar records a byte offset into the OLD file. Leaving it would
            # make the durable reader truncate the rewritten file to a stale offset
            # (`bytes` past the new size is handled, but a *smaller* offset is not),
            # so it moves aside with the backup and is rebuilt on the next run.
            if sidecar.exists():
                sidecar.unlink()
            print(f"  topped up {model}/{stage}: {len(kept)} rows kept "
                  f"(backup: {dest.relative_to(ROOT)})")

    print("\nprepared. Backups are under "
          f"{backup.relative_to(ROOT)}; the original state file is untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
