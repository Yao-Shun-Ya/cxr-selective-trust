<#
  finish_thinking_5120.ps1 -- run only the cells that have never been attempted
  at 5,120, then archive the failures and refresh the analyses.

  What went wrong with the previous version
  ----------------------------------------
  `retry_thinking_5120.ps1` removed every row without a usable answer and then
  re-ran the whole set. That is right when the token budget increased, and wrong
  when it did not: 890 cells had already failed *at 5,120*, so re-running them at
  5,120 regenerated the identical non-terminating traces. That is why the
  remaining count jumped from 914 to 1,804 and the ETA from ~17 h to ~30 h.

  What this script does instead
  -----------------------------
  1. Restores those already-failed-at-5,120 rows from the pre-filter backup, so
     the durable layer counts them as attempted and skips them.
  2. Runs the grid; only the ~914 never-attempted cells are generated.
  3. Archives every still-invalid row for the budget-sensitivity analysis, and
     prints the strict coverage.
  4. Runs the audits and the analyses.

  Usage
  -----
      cd D:\Paper
      powershell -NoProfile -ExecutionPolicy Bypass -File program\scripts\finish_thinking_5120.ps1

      # change the restore source if a newer backup exists
      ... -RestoreFrom outputs\matrix\nih_eval\qwen3_vl_8b_thinking\shift\shift_predictions.jsonl.bak_20261110_140813
#>
param(
    [int]$BatchSize = 4,
    [int]$MaxNewTokens = 5120,
    [string]$RestoreFrom = "outputs\matrix\nih_eval\qwen3_vl_8b_thinking\shift\shift_predictions.jsonl.bak_20261110_140813",
    [switch]$SkipRestore
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
function Rows([string]$p) { if (Test-Path $p) { return (Get-Content -LiteralPath $p).Count } else { return 0 } }

if (-not $SkipRestore) {
    if (-not (Test-Path (Join-Path $root $RestoreFrom))) {
        throw "restore source missing: $RestoreFrom (pass -SkipRestore to run without it)"
    }
    Write-Host "== step 1: keep known 5,120 failures out of the queue =="
    & $pyVl "scripts\restore_failed_attempts.py" --backup $RestoreFrom --grid $gridRel --cap $MaxNewTokens
    if ($LASTEXITCODE -ne 0) { throw "restore failed" }
    Stamp ("grid rows after restore: {0}" -f (Rows (Join-Path $root $gridRel)))
}

$pending = 10542 - (Rows (Join-Path $root $gridRel))
Stamp ("cells never attempted at {0} tokens: {1}" -f $MaxNewTokens, $pending)
if ($pending -le 0) { Stamp "nothing left to generate"; }

$free = $null
try {
    $line = (& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) | Select-Object -First 1
    $free = [int]$line.Trim()
} catch { Write-Host "[warn] nvidia-smi unavailable" }
if ($null -ne $free) {
    Stamp ("GPU free {0:N0} MiB" -f $free)
    if ($free -lt 20000) { throw "less than 20 GiB free; close the other GPU process first" }
}

if ($pending -gt 0) {
    Write-Host ""
    Write-Host ("== step 2: generate {0} cells at {1} tokens, batch {2} ==" -f $pending, $MaxNewTokens, $BatchSize)
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
}

Write-Host ""
Write-Host "== step 3: archive invalid rows, report strict coverage =="
& $pyVl "scripts\filter_strict_valid_rows.py" --path $gridRel --archive $archiveRel --pass-label "post_5120"
$final = Rows (Join-Path $root $gridRel)
Stamp ("strictly valid coverage: {0}/10542 = {1:P1}" -f $final, ($final / 10542))

Write-Host ""
Write-Host "== step 4: audits and analyses =="
& $pyVl "scripts\98_round2_audit.py"
& $pyVl "scripts\96_analyze_round2.py"
& $pyVl "scripts\97_round2_stats.py"
& $pyVl "scripts\101_budget_sensitivity.py"

Stamp "DONE"
