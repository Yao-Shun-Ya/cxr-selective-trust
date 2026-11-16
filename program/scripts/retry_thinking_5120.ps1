<#
  retry_thinking_5120.ps1 -- final pass for the reasoning-model grid.

  Where this sits
  ---------------
  Pass 1 ran at 2,560 tokens (57.6 % usable). Pass 2 re-ran the failures at
  3,584 (about 50 % of them closed; the rest are 13-19k-character traces).
  This pass keeps only *strictly valid* replies -- a closing </think> plus a
  decodable JSON object after it -- and re-runs everything else at 5,120
  tokens, batch N, with expandable CUDA segments.

  Why 5,120 and expandable segments
  ---------------------------------
  A known-failing smoke at 5,120 closed 12/12 cells, with the longest reply at
  18,866 characters (~4,700 tokens), i.e. the cap is genuinely needed.
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` removes the allocator
  fragmentation that pushed earlier runs to 30 GB; with it, batch 4 / 5,120
  peaked at 22,749 MiB (10,011 MiB headroom) at ~51 s per call.

  Usage
  -----
      cd D:\Paper
      powershell -NoProfile -ExecutionPolicy Bypass -File program\scripts\retry_thinking_5120.ps1
#>
param(
    [int]$BatchSize = 4,
    [int]$MaxNewTokens = 5120,
    [switch]$NoAnalyses
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$pyVl = Join-Path $root ".venv-vl\Scripts\python.exe"
$pyModern = Join-Path $root ".venv-modern\Scripts\python.exe"
$gridRel = "outputs\matrix\nih_eval\qwen3_vl_8b_thinking\shift\shift_predictions.jsonl"
$archiveRel = "outputs\analysis_round2\thinking_invalid_archive.jsonl"

function Stamp([string]$t) { Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $t) }

if (-not (Test-Path (Join-Path $root $gridRel))) { throw "missing grid file" }
$before = (Get-Content (Join-Path $root $gridRel)).Count
Stamp ("grid rows before strict filter: {0}" -f $before)

Write-Host "== step 1: keep only strictly valid replies, archive the rest =="
& $pyVl "scripts\filter_strict_valid_rows.py" --path $gridRel --archive $archiveRel --pass-label "pre_5120"
if ($LASTEXITCODE -ne 0) { throw "strict filter failed" }
$kept = (Get-Content (Join-Path $root $gridRel)).Count
Stamp ("strictly valid rows kept: {0}; to regenerate: {1}" -f $kept, (10542 - $kept))
if ($kept -ge 10542) { Stamp "nothing to retry"; exit 0 }

$free = $null
try {
    $line = (& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) | Select-Object -First 1
    $free = [int]$line.Trim()
} catch { Write-Host "[warn] nvidia-smi unavailable" }
if ($null -ne $free) {
    Stamp ("GPU free {0:N0} MiB" -f $free)
    if ($free -lt 20000) { throw "less than 20 GiB free; close the other GPU process first" }
}

Write-Host ""
Write-Host ("== step 2: retry {0} cells at {1} tokens, batch {2} ==" -f (10542 - $kept), $MaxNewTokens, $BatchSize)
Stamp "generation starting (live progress bar in this window)"
& $pyModern "scripts\08_shift_benchmark.py" `
    --model qwen3_vl_8b_thinking `
    --manifest "data/manifests/nih_eval_1732.csv" `
    --images-root "data/nih/images" `
    --output-dir "outputs/matrix/nih_eval/qwen3_vl_8b_thinking/shift" `
    --split "internal_test,source_val" `
    --limit 1506 `
    --severities severe `
    --batch-size $BatchSize `
    --max-new-tokens $MaxNewTokens `
    --seed 0
Stamp ("generation exit {0}" -f $LASTEXITCODE)

$rows = (Get-Content (Join-Path $root $gridRel)).Count
Stamp ("grid rows now: {0}/10542" -f $rows)

Write-Host ""
Write-Host "== step 3: archive any remaining invalid replies =="
& $pyVl "scripts\filter_strict_valid_rows.py" --path $gridRel --archive $archiveRel --pass-label "post_5120"
$final = (Get-Content (Join-Path $root $gridRel)).Count
Stamp ("strictly valid coverage: {0}/10542 = {1:P1}" -f $final, ($final / 10542))

if ($NoAnalyses) { Stamp "analyses skipped (-NoAnalyses)"; exit 0 }

Write-Host ""
Write-Host "== step 4: analyses =="
& $pyVl "scripts\98_round2_audit.py"
& $pyVl "scripts\96_analyze_round2.py"
& $pyVl "scripts\97_round2_stats.py"
& $pyVl "scripts\101_budget_sensitivity.py"

Stamp "DONE"
