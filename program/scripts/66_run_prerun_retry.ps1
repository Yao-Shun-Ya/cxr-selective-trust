<#
.SYNOPSIS
    One-click completion of the 100-image pre-run.

.DESCRIPTION
    Re-runs the model/stage combinations that produced no usable data in the
    2026-10-05 pre-run, into the SAME outputs\pretest tree, without touching the
    four models that already succeeded.

    The four steps, in the only order that works:

      1. scripts\65_prepare_prerun_retry.py --apply
         Releases the failed cells. Neither half can be skipped:
           * stage level -- pretest_state.json already marks those stages complete,
             so the driver would print `[skip] (already complete)`;
           * cell level -- DurableJsonl counts a row as done when its `error` field
             is empty, which every truncated MedGemma thought trace satisfies, so
             even a direct runner call would treat it as finished work.
         Every file touched is copied to outputs\pretest_retry_backup\ first.

      2. run_all_layers.ps1 for chexagent_8b + medgemma_15_4b_it (5 stages)
         A SEPARATE state file (pretest_retry_state.json) so the pre-run's own
         record stays intact, and the pretest output root so nothing lands in the
         production matrix. Both must be separated together or the driver refuses
         to start (exit 3).

      3. llava_med_v1_5_7b, run directly rather than through the driver.
         The driver's [BAD] guard deletes a stage that produces no parseable rows --
         which is exactly what llava does by design (prose, never JSON, 0/1900 after
         the rope_theta fix). Running it directly keeps that file on disk as the
         paper's "contract non-compliance" evidence.

      4. Self-check: rows and parsed counts for every pre-run cell, with explicit
         assertions for the cells that must change and the cells that must not.

    Resumable at step granularity. Steps 1 and 3 write a marker under
    outputs\state\; step 2 is recognised from its own state file. A step whose
    marker is present is skipped unless -Force is given. That guard matters most
    for step 1: re-running the prepare script after a successful retry would CLEAR
    the good chexagent rows it just produced.

    NOTE ON --seed: step 3 deliberately passes no --seed. The default seed=0 is
    what makes the retry select the same 100 images as the pre-run (shuffle then
    head(100)). Changing it would make the retry pick a different subset and grow
    the file to ~3600 rows -- which the driver's coverage guard would NOT catch,
    because it only checks that a stage wrote at least 95% of its expected rows.

.PARAMETER DryRun
    Print the prepare plan, the driver plan and the commands, then exit. Runs
    nothing and writes nothing.

.PARAMETER Force
    Redo steps whose marker already exists (including re-running the prepare step).

.EXAMPLE
    # look first -- CPU only, no GPU, writes nothing
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\66_run_prerun_retry.ps1 -DryRun

    # do it (~2 h of GPU time, serial)
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\66_run_prerun_retry.ps1
#>
[CmdletBinding()]
param(
    [switch] $DryRun,
    [switch] $Force
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Py         = Join-Path $Root '.venv-vl\Scripts\python.exe'
$Prepare    = Join-Path $Root 'scripts\65_prepare_prerun_retry.py'
$Driver     = Join-Path $Root 'scripts\run_all_layers.ps1'
$OutputRoot = 'outputs\pretest'
$RetryState = 'outputs\state\pretest_retry_state.json'
$StateDir   = Join-Path $Root 'outputs\state'
$MkPrepare  = Join-Path $StateDir 'prerun_retry.step1_prepared'
$MkLlava    = Join-Path $StateDir 'prerun_retry.step3_llava'

$RetryModels  = @('chexagent_8b', 'medgemma_15_4b_it')
$DriverStages = @(
    'nih/chexagent_8b/shift',
    'nih/chexagent_8b/reliance',
    'nih/medgemma_15_4b_it/precision',
    'nih/medgemma_15_4b_it/shift',
    'nih/medgemma_15_4b_it/reliance'
)

# No --seed on either command, on purpose. See the note in the header.
$LlavaShift = @(
    'scripts\08_shift_benchmark.py', '--model', 'llava_med_v1_5_7b',
    '--manifest', 'data\manifests\nih_eval_1732.csv',
    '--split', 'internal_test,source_val', '--images-root', 'data\nih\images',
    '--output-dir', 'outputs\pretest\nih_eval\llava_med_v1_5_7b\shift',
    '--limit', '100', '--batch-size', '1', '--max-new-tokens', '256'
)
$LlavaReliance = @(
    'scripts\06_image_reliance.py', '--model', 'llava_med_v1_5_7b',
    '--manifest', 'data\manifests\nih_eval_1732.csv',
    '--split', 'internal_test,source_val', '--images-root', 'data\nih\images',
    '--output-dir', 'outputs\pretest\nih_eval\llava_med_v1_5_7b\reliance',
    '--limit', '100', '--conditions', 'original', 'blank', 'blur', 'shuffle',
    '--batch-size', '1', '--max-new-tokens', '256'
)

function Write-Step([string] $msg) {
    Write-Host ""
    Write-Host ("=" * 82)
    Write-Host ("  $msg")
    Write-Host ("=" * 82)
}

function Test-DriverStagesDone {
    if (-not (Test-Path $RetryState)) { return $false }
    try { $st = Get-Content $RetryState -Raw | ConvertFrom-Json } catch { return $false }
    $done = @($st.completed)
    foreach ($k in $DriverStages) { if ($done -notcontains $k) { return $false } }
    return $true
}

# ------------------------------------------------------------------ preflight
Write-Step "PREFLIGHT"
$missing = @()
foreach ($p in @($Py, $Prepare, $Driver)) {
    if (-not (Test-Path $p)) { $missing += $p }
}
if ($missing.Count -gt 0) {
    Write-Host "ERROR: missing required file(s):" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    exit 2
}
Write-Host "  python : $Py"
Write-Host "  output : $OutputRoot   (the pre-run tree; outputs\matrix is untouched)"
Write-Host "  state  : $RetryState"
Write-Host "           (separate from pretest_state.json, which stays as the pre-run record)"
if ($DryRun) {
    Write-Host "  MODE   : DRY RUN -- nothing is run, nothing is written" -ForegroundColor Yellow
} else {
    Write-Host "  MODE   : RUN -- GPU, serial, ~2 h (medgemma_15 shift+reliance whole, at 2048)" -ForegroundColor Yellow
}

# ---------------------------------------------------- 1. release failed cells
Write-Step "STEP 1/4 -- release the failed cells (65_prepare_prerun_retry.py)"
if ((Test-Path $MkPrepare) -and -not $Force) {
    Write-Host "  [skip] step 1 already done (marker: $MkPrepare). Use -Force to redo."
    Write-Host "         Re-running it would CLEAR cells that are now good; that is why it is gated."
} else {
    if ($DryRun) {
        Write-Host "  DRY RUN: would run -> python scripts\65_prepare_prerun_retry.py --apply"
        Write-Host "           showing the plan it would apply:"
        Write-Host ""
        & $Py $Prepare
    } else {
        & $Py $Prepare --apply
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: prepare step exited $LASTEXITCODE -- nothing else was run." -ForegroundColor Red
            exit 1
        }
        Set-Content -Path $MkPrepare -Value ("prepared " + (Get-Date).ToString('s')) -Encoding ascii
    }
}

# ------------------------------------------------- 2. driver: 5 retry stages
Write-Step "STEP 2/4 -- driver: chexagent_8b + medgemma_15_4b_it (5 stages, 100 images)"
if ((Test-DriverStagesDone) -and -not $Force) {
    Write-Host "  [skip] all 5 driver stages already complete in $RetryState"
} else {
    # Invoke the driver in a SEPARATE powershell process via -File, exactly as the
    # documented command does. Calling it in-process (`& $Driver @dargs`) binds the
    # arguments wrong: the driver declares BOTH $Models and $Layers with
    # ValueFromRemainingArguments, and an in-process array splat lets -Models swallow
    # the following switches, so -Limit ends up receiving the string "-Models".
    $dargs = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Driver,
        '-Models', ($RetryModels -join ','),
        '-Layers', 'nih',
        '-Limit', '100',
        '-OutputRoot', $OutputRoot,
        '-StateFile', $RetryState
    )
    if ($DryRun) { $dargs += '-DryRun' }
    Write-Host ("  -> powershell " + ($dargs -join ' '))
    Write-Host ""
    & powershell @dargs
    $code = $LASTEXITCODE
    if ($DryRun) {
        # nothing ran
    } elseif ($code -eq 3) {
        Write-Host ""
        Write-Host "  NOTE: driver exit 3 = coverage assertion, NOT an inference failure." -ForegroundColor Yellow
        Write-Host "        Inference finished and every stage is marked done; some cell has" -ForegroundColor Yellow
        Write-Host "        < 95% label coverage. Report it next to the metric; do not drop it." -ForegroundColor Yellow
    } elseif ($code -ne 0) {
        Write-Host "ERROR: driver exited $code" -ForegroundColor Red
        exit 1
    }
}

# --------------------------------------------- 3. llava, direct (keep evidence)
Write-Step "STEP 3/4 -- llava_med_v1_5_7b, run directly (bypasses the driver's [BAD] delete)"
if ((Test-Path $MkLlava) -and -not $Force) {
    Write-Host "  [skip] step 3 already done (marker: $MkLlava). Use -Force to redo."
} else {
    foreach ($cmd in @($LlavaShift, $LlavaReliance)) {
        if ($DryRun) {
            Write-Host ("  DRY RUN: would run -> python " + ($cmd -join ' '))
            continue
        }
        Write-Host ""
        Write-Host ("  -> python " + ($cmd -join ' '))
        & $Py @cmd
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: llava command exited $LASTEXITCODE" -ForegroundColor Red
            exit 1
        }
    }
    if (-not $DryRun) {
        Set-Content -Path $MkLlava -Value ("llava done " + (Get-Date).ToString('s')) -Encoding ascii
    }
}

# ------------------------------------------------------------- 4. self-check
Write-Step "STEP 4/4 -- self-check (rows / parsed for every pre-run cell)"
if ($DryRun) {
    Write-Host "  DRY RUN: skipping the self-check"
} else {
    $checkPath = Join-Path $StateDir 'prerun_retry_selfcheck.py'
    $check = @'
import glob
import json
import os

# Rows every top-level cell must hold after the retry. The seven retried cells are
# listed alongside the ten that must be untouched: asserting the untouched ones is
# how this check proves the retry did not overwrite another model.
EXPECT_ROWS = {
    "chexagent_8b/shift": 1900,
    "chexagent_8b/reliance": 400,
    "internvl3_5_8b/shift": 1900,
    "internvl3_5_8b/reliance": 400,
    "llava_med_v1_5_7b/shift": 1900,
    "llava_med_v1_5_7b/reliance": 400,
    "medgemma_15_4b_it/shift": 1900,
    "medgemma_15_4b_it/reliance": 400,
    "medgemma_15_4b_it/reliance_bf16": 400,
    "medgemma_4b_it/shift": 1900,
    "medgemma_4b_it/reliance": 400,
    "medgemma_4b_it/reliance_bf16": 400,
    "qwen25_vl_7b_instruct/shift": 1900,
    "qwen25_vl_7b_instruct/reliance": 400,
    "qwen25_vl_7b_instruct/reliance_bf16": 400,
    "qwen3_vl_8b_instruct/shift": 1900,
    "qwen3_vl_8b_instruct/reliance": 400,
}

# Cells that were broken and must now parse. The number is the MEASURED,
# documented count -- NOT 100% -- because two deterministic model refusals survive
# every legitimate fix: chexagent answers a bare label ("no_finding") instead of the
# object on ~0.9% of cells, and medgemma_15 degenerates into an ABSORBING thought-
# channel repetition loop on 3 of 1900 shift cells (no '{' is ever emitted, the tail
# is exactly periodic, and greedy decoding makes the loop inescapable -- so no token
# budget can help; measured in scripts/70). Both reproduce exactly under greedy
# decoding, so re-running cannot change them, and asserting 100% would only force a
# fabricated number. A count BELOW the floor is a real finding: record the shortfall
# in docs/50 section 2.12 rather than editing this number to pass.
MUST_PARSE_AT_LEAST = {
    "chexagent_8b/shift": 1880,          # 1900 rows - 20 bare-label refusals
    "chexagent_8b/reliance": 399,        # 400 - 1 bare-label refusal
    "medgemma_15_4b_it/shift": 1897,     # 1900 - 3 absorbing thought-channel loops
    "medgemma_15_4b_it/reliance": 400,
    "medgemma_15_4b_it/reliance_bf16": 400,
}

# Cells whose correct result is prose that never parses. A non-zero count here would
# mean the JSON contract suddenly started being honoured, which is a finding, not a fix.
MUST_STAY_PROSE = {"llava_med_v1_5_7b/shift", "llava_med_v1_5_7b/reliance"}

failures = []
rows_seen = {}

for path in sorted(glob.glob("outputs/pretest/**/*_predictions.jsonl", recursive=True)):
    n = ok = 0
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        n += 1
        try:
            if (json.loads(line).get("parsed") or {}).get("parse_success"):
                ok += 1
        except Exception:
            pass
    parts = os.path.normpath(path).split(os.sep)
    key = parts[-3] + "/" + parts[-2]
    rows_seen[key] = n

    tags = []
    if key in EXPECT_ROWS and n != EXPECT_ROWS[key]:
        tags.append("expected %d rows" % EXPECT_ROWS[key])
        failures.append(key + ": rows %d != %d" % (n, EXPECT_ROWS[key]))
    if key in MUST_PARSE_AT_LEAST and ok < MUST_PARSE_AT_LEAST[key]:
        tags.append("expected >= %d parsed" % MUST_PARSE_AT_LEAST[key])
        failures.append(key + ": parsed %d < %d" % (ok, MUST_PARSE_AT_LEAST[key]))
    if key in MUST_STAY_PROSE and ok != 0:
        tags.append("expected prose (0 parsed)")
        failures.append(key + ": parsed %d, expected 0" % ok)

    print("%-46s rows=%5d parsed=%5d%s" % (key, n, ok, ("   <-- " + "; ".join(tags)) if tags else ""))

missing = sorted(set(EXPECT_ROWS) - set(rows_seen))
for key in missing:
    failures.append(key + ": no file")
    print("%-46s MISSING" % key)

print()
if failures:
    print("SELF-CHECK: PROBLEM (%d)" % len(failures))
    for f in failures:
        print("  - " + f)
    raise SystemExit(3)
print("SELF-CHECK: OK -- retried cells are complete, other models untouched")
'@
    # Write to a file and run it, rather than piping the text into `python -`:
    # Windows PowerShell 5.1 encodes a piped string with the OEM code page, which
    # is one more way for a check that is supposed to catch silent corruption to
    # itself fail silently.
    #
    # -Encoding ascii, NOT utf8: `Set-Content -Encoding UTF8` writes a BOM in
    # PowerShell 5.1, and a BOM is a trap this project has already been bitten by.
    # The check body is pure ASCII, so ascii is both BOM-free and lossless.
    Set-Content -Path $checkPath -Value $check -Encoding ascii
    & $Py $checkPath
    $checkCode = $LASTEXITCODE
    Remove-Item $checkPath -Force -ErrorAction SilentlyContinue
    if ($checkCode -ne 0) {
        Write-Host ""
        Write-Host "SELF-CHECK FAILED -- do not treat the pre-run as complete." -ForegroundColor Red
        Write-Host "Inspect outputs\pretest_retry_backup\ and re-run this script (steps are resumable)." -ForegroundColor Red
        exit 1
    }
}

Write-Step "DONE"
if ($DryRun) {
    Write-Host "  Dry run finished. Re-run without -DryRun to execute."
} else {
    Write-Host "  Pre-run completed. Backups of the released cells: outputs\pretest_retry_backup\"
    Write-Host "  medgemma_15 shift/reliance were re-run WHOLE at 2048 tokens -- a uniform budget,"
    Write-Host "  not a top-up -- so every row in those two stages came from one setting."
    Write-Host "  chexagent was re-run GREEDY (num_beams=1); every earlier chexagent row was"
    Write-Host "  beam search, which is what the guard rejected. Its residual failures are"
    Write-Host "  deterministic bare-label refusals and are reported, not hidden: docs/50 2.12."
}
