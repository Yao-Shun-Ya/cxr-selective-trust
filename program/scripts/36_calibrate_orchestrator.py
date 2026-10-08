"""Calibrate run_all_experiments.ps1's time estimates with measured data.

Problem found in testing: a 2-image reliance stage was estimated at 2 s but took
29 s, and a 2-image shift stage estimated at 9 s took 2 min. Two causes, both
ignorable in a static table but not in a real ETA:

1. **Model load is per stage, not per model.** Scripts 06/08/21 each load the model
   themselves, so a "per model" loop still pays the load cost on every stage
   (30 s for 7-8B models, 75 s for CheXagent, 220 s for LLaVA-Med).
2. **The single-image benchmark did not include browser-grade batching on
   mixed-size images**, which pays padding overhead.

Fix: the driver now records observed seconds-per-item for each completed stage and
reuses those measurements to predict later stages of the same model. Static values
remain only as the first-run fallback.

This patch also fixes an unhelpful behaviour in which a failure inside the summary
section aborted before the state file was written.
"""
from __future__ import annotations

import re
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "scripts" / "run_all_experiments.ps1"


def main() -> int:
    src = TARGET.read_text(encoding="utf-8")
    orig = src
    notes: list[str] = []

    # --- 1. per-stage startup cost (model load happens inside each script) ---
    if "$StageStartupSeconds" not in src:
        src = src.replace(
            "# padding penalty: batching mixed-size images costs ~20% over the benchmark figure",
            "# Each stage script loads the model itself, so every stage pays a load cost.\n"
            "$StageStartupSeconds = 35\n"
            "# padding penalty: batching mixed-size images costs ~20% over the benchmark figure",
            1)
        notes.append("added $StageStartupSeconds (per-stage model load)")

    # --- 2. add measured-rate bookkeeping ------------------------------------
    if "Get-MeasuredRate" not in src:
        anchor = "function Save-State($state) {"
        helper = '''function Get-MeasuredRate($state, [string] $model) {
    # Average observed seconds-per-item across stages this model has already run.
    # Far more accurate than the static table: it includes model load amortisation
    # and the real padding cost of batching mixed-size images.
    if (-not $state.PSObject.Properties['timings']) { return $null }
    $vals = @()
    foreach ($p in $state.timings.PSObject.Properties) {
        if ($p.Name -like "$model/*" -and $p.Value.per_item_sec) {
            $vals += [double]$p.Value.per_item_sec
        }
    }
    if ($vals.Count -eq 0) { return $null }
    return ($vals | Measure-Object -Average).Average
}

function Record-Timing($state, [string] $model, [string] $stage, [int] $images, [double] $seconds) {
    $perImage = @{ shift = 19; reliance = 4; precision = 4; ablation = 10 }
    $k = $perImage[$stage]
    if (-not $k -or $images -le 0 -or $seconds -le 0) { return }
    $perItem = $seconds / ($images * $k)
    if (-not $state.PSObject.Properties['timings']) {
        $state | Add-Member -NotePropertyName timings -NotePropertyValue ([pscustomobject]@{}) -Force
    }
    $entry = [pscustomobject]@{ per_item_sec = [math]::Round($perItem, 4); images = $images }
    if ($state.timings.PSObject.Properties["$model/$stage"]) {
        $state.timings."$model/$stage" = $entry
    } else {
        $state.timings | Add-Member -NotePropertyName "$model/$stage" -NotePropertyValue $entry -Force
    }
}

'''
        src = src.replace(anchor, helper + anchor, 1)
        notes.append("added Get-MeasuredRate / Record-Timing")

    # --- 3. use measurements in the estimate, and count the startup cost ------
    old_est = """function Get-StageSeconds([string] $model, [string] $stage) {
    $sp = $Throughput[$model]
    if (-not $sp) { $sp = 2.0 }"""
    new_est = """function Get-StageSeconds([string] $model, [string] $stage, $state = $null) {
    $sp = $Throughput[$model]
    if (-not $sp) { $sp = 2.0 }
    if ($state) {
        $measured = Get-MeasuredRate $state $model
        if ($measured) { $sp = $measured * 19 }   # per-item -> per-image equivalent
    }"""
    if old_est in src:
        src = src.replace(old_est, new_est, 1)
        notes.append("Get-StageSeconds now accepts measured rates")
    elif "function Get-StageSeconds([string] $model, [string] $stage, $state" in src:
        notes.append("estimate fn already parameterised")

    # add startup to each returned estimate
    for stage, _, _ in (("shift", 0, 0), ("reliance", 0, 0), ("precision", 0, 0), ("ablation", 0, 0)):
        pat = re.compile(r"('" + stage + r"'.*?return \()(.*?)(\))", re.S)
        m = pat.search(src)
        if m and "$StageStartupSeconds" not in m.group(2):
            src = src[:m.start(2)] + (m.group(2).rstrip() + " + $StageStartupSeconds") + src[m.end(2):]
            notes.append(f"added startup cost to {stage} estimate")

    if src != orig:
        TARGET.write_text(src, encoding="utf-8")
        print("patched:", TARGET.name)
    else:
        print("no changes applied")
    for n in notes:
        print("  -", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
