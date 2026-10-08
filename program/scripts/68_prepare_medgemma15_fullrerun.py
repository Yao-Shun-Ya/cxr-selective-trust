#!/usr/bin/env python
"""Prepare a FULL 2048-token re-run of medgemma_15_4b_it shift + reliance.

Why this exists
---------------
The 2026-10-08 retry *topped up* these two stages at the then-current budget
(1024): the pre-run's 256-token rows that parsed were kept, and only the rows
that had run out of budget were retried. That left 16 residual failures
(15 in shift, 1 in reliance) which are 1024-token truncations -- the reply is a
thought trace plus a JSON that gets cut mid-object. `$MaxNewTokensOverride` is
now 2048 (docs/50 section 2.11), so a full re-run at 2048 should clear them and
leave every medgemma_15 stage on one uniform budget.

What it touches (every file is copied aside FIRST)
--------------------------------------------------
* ``shift/shift_predictions.jsonl`` and ``reliance/image_reliance_predictions.jsonl``
  (plus their ``.progress.json`` sidecars) are MOVED into
  ``outputs/pretest_retry_backup/medgemma_15_4b_it_fullrerun_20261008/<stage>/``.
  That is a NEW directory on purpose: the earlier backup under
  ``medgemma_15_4b_it/`` holds the *pre-run 256-token original* and must survive.
* the two ``nih/medgemma_15_4b_it/{shift,reliance}`` entries are removed from
  ``outputs/state/pretest_retry_state.json``, so the driver no longer prints
  ``[skip] (already complete)`` for them.

Dry run by default; ``--apply`` executes. The state file carries a UTF-8 BOM
(the driver writes it with ``Set-Content -Encoding UTF8``), so it is read with
``utf-8-sig``; and the state edit is idempotent, so re-running repairs a crash
that landed between the file move and the state edit.

After this, run ``66_run_prerun_retry.ps1``: step 2 re-runs the two cleared
stages at 2048, while chexagent and medgemma precision stay ``[skip]`` (they are
already complete).

Usage::

    .venv-vl\\Scripts\\python.exe scripts\\68_prepare_medgemma15_fullrerun.py
    .venv-vl\\Scripts\\python.exe scripts\\68_prepare_medgemma15_fullrerun.py --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRETEST = ROOT / "outputs" / "pretest" / "nih_eval" / "medgemma_15_4b_it"
BACKUP = ROOT / "outputs" / "pretest_retry_backup" / "medgemma_15_4b_it_fullrerun_20261008"
STATE = ROOT / "outputs" / "state" / "pretest_retry_state.json"

# (stage dir, prediction file name, state key)
STAGES = [
    ("shift", "shift_predictions.jsonl", "nih/medgemma_15_4b_it/shift"),
    ("reliance", "image_reliance_predictions.jsonl", "nih/medgemma_15_4b_it/reliance"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually move the files and edit the state file")
    args = ap.parse_args()

    plan = []
    for stage, fname, key in STAGES:
        src = PRETEST / stage / fname
        side = src.with_suffix(src.suffix + ".progress.json")
        dst = BACKUP / stage
        plan.append((stage, key, src, side, dst))

    print("=" * 78)
    print("FULL 2048 RE-RUN PREP -- medgemma_15_4b_it shift + reliance")
    print("=" * 78)
    print(f"  mode     : {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"  backup to: {BACKUP.relative_to(ROOT)}")
    print(f"  state    : {STATE.relative_to(ROOT)}")
    print()

    keys = {k for _, k, _, _, _ in plan}
    # The state file is written by the driver's `Set-Content -Encoding UTF8`, which
    # emits a UTF-8 BOM on PowerShell 5 -- read it BOM-tolerantly or json.loads dies
    # on char 0 (the project's standing BOM trap). Write it back with the same BOM
    # so both tools keep reading it identically.
    st = None
    if STATE.exists():
        st = json.loads(STATE.read_text(encoding="utf-8-sig"))
    completed_before = list(st.get("completed", [])) if st else []
    keys_in_state = [k for k in completed_before if k in keys]

    n_present = 0
    for stage, key, src, side, dst in plan:
        rows = 0
        if src.exists():
            rows = sum(1 for _ in src.open(encoding="utf-8"))
            n_present += 1
        print(f"  {stage:9s} rows={rows:5d}  {'MOVE' if src.exists() else 'ABSENT'}")
        print(f"            {src.relative_to(ROOT)}")
        print(f"        -> {dst.relative_to(ROOT)}/{src.name}")

    if n_present == 0 and not keys_in_state:
        print()
        print("  Nothing to do: both prediction files are absent and the state file "
              "no longer lists either stage.")
        return 0

    if not args.apply:
        print()
        print("  DRY RUN -- nothing moved, state file untouched. Re-run with --apply.")
        return 0

    # --- 1. move files aside (skips any a previous attempt already moved) -----
    for stage, key, src, side, dst in plan:
        if not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst / src.name))
        if side.exists():
            shutil.move(str(side), str(dst / side.name))
        print(f"  moved {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

    # --- 2. drop the two state entries ---------------------------------------
    # Idempotent on purpose: this runs even when the files were already moved, so a
    # crash between the move and the state edit (which happened once -- the BOM) is
    # repaired by simply re-running instead of leaving the stages half-prepared.
    if st is not None:
        st["completed"] = [k for k in completed_before if k not in keys]
        tmp = STATE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(st, indent=4), encoding="utf-8-sig")
        tmp.replace(STATE)
        print(f"  state: removed {keys_in_state}")
        print(f"  state: completed now = {st['completed']}")
    else:
        print("  state: no state file to edit")

    print()
    print("  Done. Next: powershell -NoProfile -ExecutionPolicy Bypass "
          "-File scripts\\66_run_prerun_retry.ps1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())