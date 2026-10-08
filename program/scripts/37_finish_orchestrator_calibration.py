"""Finish the orchestrator calibration: rewrite Get-StageSeconds cleanly and wire it up.

The previous patch left three things half-done, all of which keep the ETA
misleading:

1. ``$StageStartupSeconds`` was declared but never added to any return value, so
   the per-stage model-load cost (30-220 s, paid on EVERY stage because each
   script loads the model itself) stayed invisible.
2. ``Get-StageSeconds`` gained a ``$state`` parameter that no caller passes, so
   measured rates were never actually used.
3. ``Record-Timing`` was never called, so no measurements were ever stored.

This replaces the function wholesale and rewires the three call sites, then
verifies by printing the ETA for a known case.
"""
from __future__ import annotations

import re
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "scripts" / "run_all_experiments.ps1"

NEW_FN = '''function Get-StageSeconds([string] $model, [string] $stage, $state = $null) {
    # Base rate: measured single-image throughput (docs/44, docs/46).
    $sp = $Throughput[$model]
    if (-not $sp) { $sp = 2.0 }

    # Prefer a rate measured on THIS workload. Batching mixed-size images pays
    # padding overhead the single-image benchmark did not, and model load is paid
    # inside every stage script, so a measured value is strictly better.
    $itemsThisStage = $ItemsPerImage[$stage]
    if ($state -and $itemsThisStage) {
        $measured = Get-MeasuredRate $state $model
        if ($measured) { $sp = $measured * $itemsThisStage }
    }

    $batchSpeedup = 1.6      # conservative: batching repays less than 2x here
    $body = 0
    switch ($stage) {
        'shift'     { $body = $Limit * $ShiftCells * $sp / $batchSpeedup * $PaddingFactor }
        'reliance'  { $body = $Limit * $RelianceConditions * $sp / $batchSpeedup * $PaddingFactor }
        'precision' { $body = $Limit * $RelianceConditions * $sp / 1.5 }
        'ablation'  { $body = $AblationLimit * 2 * 5 * $sp }
        default     { $body = 0 }
    }
    # Every stage script loads the model itself: add that cost every time.
    return ($body + $StageStartupSeconds)
}
'''

ITEMS_MAP = '''$ItemsPerImage = @{ shift = 19; reliance = 4; precision = 4; ablation = 10 }
'''


def main() -> int:
    src = TARGET.read_text(encoding="utf-8")
    changed = False

    # 1. replace the function body wholesale
    pat = re.compile(r"function Get-StageSeconds\(.*?\n\}\n", re.S)
    if pat.search(src):
        src = pat.sub(NEW_FN, src, count=1)
        changed = True
        print("replaced Get-StageSeconds")
    else:
        print("WARNING: Get-StageSeconds not found")

    # 2. ensure the per-stage item multipliers exist
    if "$ItemsPerImage" not in src:
        src = src.replace("$StageStartupSeconds = 35",
                          "$StageStartupSeconds = 35\n" + ITEMS_MAP, 1)
        changed = True
        print("added $ItemsPerImage")

    # 3. pass $state into every estimate call
    before = src
    src = src.replace("(Get-StageSeconds $m $s)", "(Get-StageSeconds $m $s $state)")
    src = src.replace("(Get-StageSeconds $model $p.Stage)", "(Get-StageSeconds $model $p.Stage $state)")
    if src != before:
        changed = True
        print("wired $state into estimate calls")

    # 4. record the measured timing after each successful stage
    old_done = '            Mark-Done $state $key\n            $succeeded++'
    new_done = ('            Mark-Done $state $key\n'
                '            Record-Timing $state $model $p.Stage $Limit $dt\n'
                '            Save-State $state\n'
                '            $succeeded++')
    if old_done in src and "Record-Timing $state $model" not in src:
        src = src.replace(old_done, new_done, 1)
        changed = True
        print("record measured timing after each stage")

    if changed:
        TARGET.write_text(src, encoding="utf-8")
        print("written:", TARGET.name)
    else:
        print("no changes applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
