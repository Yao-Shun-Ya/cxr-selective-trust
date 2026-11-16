<#
  run_round2_all.ps1 -- one-command launcher for the four round-2 GPU experiments.

  Stage order (sequential; one 32 GB card, so never run two stages at once):
    1. DenseNet-121 supervised baseline                 ~5-10 min
    2. Reasoning-channel probe        Qwen3-VL-8B-Thinking, 16 calls, ~8 min
    3. Prompt paraphrase              MedGemma-4B, 2 wordings x 2 conditions, ~4 h
    4. MedGemma-1.5 schema ablation   all five schemas x 2 conditions, ~20-24 h
    5. Second-model grid              7 cells x 1,506 = 10,542 calls, ~75-78 h
                                      launched only if the probe passes

  Total ~100-104 h (~4.3 days), inside the 7-day ceiling with room for a re-run.

  Everything is resumable. Re-running the same command after an interruption
  continues from the durable JSONL byte offsets instead of redoing finished
  cells.

  Usage
  -----
      powershell -NoProfile -ExecutionPolicy Bypass -File program\scripts\run_round2_all.ps1

  Options
  -----
      -CoreAblation     only v0/v1 (4 cells, ~10-12 h) instead of all five
      -SkipSecondModel  skip stages 2 and 5 even if the checkpoint is present
      -SkipDenseNet     skip stage 1
      -PreflightOnly    run the checks and exit without starting anything
#>
param(
    [switch]$SkipSecondModel,
    [switch]$CoreAblation,
    [switch]$SkipDenseNet,
    [switch]$PreflightOnly
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$scriptDir = $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$runStart = Get-Date
function Stamp { param([string]$Text) Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Text) }

Write-Host ("=" * 78)
Write-Host "GPU round 2 -- sequential plan (fail-fast order)"
Write-Host ("=" * 78)
Write-Host "  1. DenseNet-121 baseline         (~5-10 min; cheapest result, finishes first)"
Write-Host "  2. Reasoning marker probe        (~8 min; decides whether stage 5 is possible)"
Write-Host "  3. Prompt paraphrase             (~4 h, MedGemma-4B)"
Write-Host "  4. MedGemma-1.5 schema ablation  (5 schemas ~20-24 h; -CoreAblation ~10-12 h)"
Write-Host "  5. Qwen3-VL-Thinking 7-cell grid (~75-78 h; only if the probe passes)"
Write-Host "  total ~100-104 h"
Write-Host ""

# --------------------------------------------------------------------------- #
# preflight: fail in the first minute, not after four hours
# --------------------------------------------------------------------------- #
$pyVl = Join-Path $root ".venv-vl\Scripts\python.exe"
$pyModern = Join-Path $root ".venv-modern\Scripts\python.exe"
foreach ($p in @($pyVl, $pyModern)) {
    if (-not (Test-Path $p)) { throw "missing interpreter: $p" }
}
$manifest = Join-Path $root "data\manifests\nih_eval_1732.csv"
if (-not (Test-Path $manifest)) { throw "missing manifest: $manifest" }

$gpuFree = $null
try {
    $line = (& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) | Select-Object -First 1
    $gpuFree = [int]$line.Trim()
} catch { Write-Host "[warn] nvidia-smi unavailable; continuing without the GPU-idle check" }
if ($null -ne $gpuFree) {
    Stamp ("GPU free: {0:N0} MiB" -f $gpuFree)
    if ($gpuFree -lt 20000) {
        throw "only $gpuFree MiB free; another process is using the card. Close it and re-run."
    }
}

# Thinking checkpoint: 4 shards, exact byte sizes. A short file would fail the
# loader after a 10 h wait, so it is checked here and the second-model stages are
# skipped (not failed) when incomplete.
$thinkSizes = @{
    "model-00001-of-00004.safetensors" = 4902275944
    "model-00002-of-00004.safetensors" = 4915962496
    "model-00003-of-00004.safetensors" = 4999831048
    "model-00004-of-00004.safetensors" = 2716270024
}
$thinkDir = Join-Path $root "models\Qwen3-VL-8B-Thinking"
$thinkReady = $true
foreach ($n in $thinkSizes.Keys) {
    $f = Join-Path $thinkDir $n
    if (-not (Test-Path $f) -or (Get-Item $f).Length -ne $thinkSizes[$n]) { $thinkReady = $false }
}
if ($thinkReady) { Stamp "Qwen3-VL-8B-Thinking: 4/4 shards complete" }
else {
    Stamp "Qwen3-VL-8B-Thinking: INCOMPLETE -- run program\scripts\run_round2_download_thinking.ps1"
    if (-not $SkipSecondModel) {
        Stamp "second-model stages disabled for this run"
        $SkipSecondModel = $true
    }
}
Write-Host ""

if ($PreflightOnly) {
    Stamp "preflight only: nothing launched"
    if (-not $thinkReady) { exit 3 }
    exit 0
}

$stageStart = Get-Date

# --------------------------------------------------------------------------- #
# 1. DenseNet baseline -- cheapest, so a result lands within ten minutes
# --------------------------------------------------------------------------- #
if (-not $SkipDenseNet) {
    Stamp "stage 1/5 DenseNet-121 baseline"
    & (Join-Path $scriptDir "run_round2_densenet.ps1")
    if ($LASTEXITCODE -ne 0) { throw "stage 1 failed" }
    Stamp ("stage 1 done in {0:N2} h" -f ((Get-Date) - $stageStart).TotalHours)
} else {
    Stamp "stage 1/5 skipped (-SkipDenseNet)"
}
$stageStart = Get-Date

# --------------------------------------------------------------------------- #
# 2. Reasoning probe -- five minutes that decide whether stage 5 is worth 10 h
# --------------------------------------------------------------------------- #
# The probe must run BEFORE the long stages: if the second model has no
# readable reasoning channel there is no grid to run, and finding that out
# after fourteen hours of ablation would waste the whole budget.
$gridReady = $false
if ($SkipSecondModel) {
    Stamp "stage 2/5 skipped (second model unavailable or explicitly skipped)"
} else {
    Stamp "stage 2/5 reasoning-channel probe"
    & (Join-Path $scriptDir "run_round2_reasoning_probe.ps1")
    $probeCode = $LASTEXITCODE
    if ($probeCode -eq 0) {
        $gridReady = $true
        Stamp "probe passed; the 7-cell grid is queued for stage 5"
    } else {
        Stamp "probe did not pass (exit $probeCode); the 7-cell grid will NOT run"
    }
}
$stageStart = Get-Date

# --------------------------------------------------------------------------- #
# 3. prompt paraphrase
# --------------------------------------------------------------------------- #
Stamp "stage 3/5 prompt paraphrase"
& (Join-Path $scriptDir "run_round2_paraphrase.ps1")
if ($LASTEXITCODE -ne 0) { throw "stage 3 failed" }
Stamp ("stage 3 done in {0:N1} h" -f ((Get-Date) - $stageStart).TotalHours)
$stageStart = Get-Date

# --------------------------------------------------------------------------- #
# 4. schema ablation
# --------------------------------------------------------------------------- #
Stamp "stage 4/5 MedGemma-1.5 schema ablation"
if ($CoreAblation) {
    & (Join-Path $scriptDir "run_round2_mg15_ablation.ps1")
} else {
    & (Join-Path $scriptDir "run_round2_mg15_ablation.ps1") -Full
}
if ($LASTEXITCODE -ne 0) { throw "stage 4 failed" }
Stamp ("stage 4 done in {0:N1} h" -f ((Get-Date) - $stageStart).TotalHours)
$stageStart = Get-Date

# --------------------------------------------------------------------------- #
# 5. second reasoning model grid
# --------------------------------------------------------------------------- #
if ($gridReady) {
    Stamp "stage 5/5 Qwen3-VL-Thinking 7-cell grid"
    & (Join-Path $scriptDir "run_round2_reasoning_grid.ps1")
    if ($LASTEXITCODE -ne 0) { throw "stage 5 failed" }
    Stamp ("stage 5 done in {0:N1} h" -f ((Get-Date) - $stageStart).TotalHours)
} else {
    Stamp "stage 5/5 not run"
}

$elapsed = (Get-Date) - $runStart
Write-Host ""
Write-Host ("ALL DONE in {0:N1} h" -f $elapsed.TotalHours)
Write-Host "Next: .\.venv-vl\Scripts\python.exe scripts\96_analyze_round2.py"
