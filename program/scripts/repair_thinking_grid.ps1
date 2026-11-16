<#
  repair_thinking_grid.ps1 -- retry the truncated Thinking-grid replies with a
  larger token budget, then refresh the analyses.

  Why this is needed
  ------------------
  The grid finished all 10,542 cells, but the audit found 4,475 replies (42.5 %)
  were cut off inside the reasoning block: no </think>, no JSON, no `error`
  field -- i.e. a *successful* call with an unusable answer. The closed replies
  cluster at 7,090 chars (p50) and 9,747 (p99), while every cut reply sits at
  ~9,700-11,000 chars, right at the 2,560-token cap. The budget was simply a
  little too small, not the model unable to answer.

  What this script does
  ---------------------
  1. Backs up the grid JSONL, keeps only parse-successful rows, and removes the
     byte-offset sidecar (DurableJsonl counts a truncated row as done, so
     `--resume` would otherwise skip it forever).
  2. Re-runs only the missing cells at 3,584 tokens, batch 5, resuming
     automatically (08 has no --resume flag; DurableJsonl decides what is
     pending). Measured: peak 28,154 MiB, 4,606 MiB headroom, ~30 s/call.
  3. Re-runs the audit (98) and the two analyses (96, 97).

  The 2,560-token pass is not wasted: its non-termination rate is kept as a
  budget-sensitivity result (the share of replies that never close a reasoning
  block rises from clean to resolution/severe).

  Usage
  -----
      cd D:\Paper
      powershell -NoProfile -ExecutionPolicy Bypass -File program\scripts\repair_thinking_grid.ps1

      # only back up + filter, do not touch the GPU
      powershell -NoProfile -ExecutionPolicy Bypass -File program\scripts\repair_thinking_grid.ps1 -FilterOnly
#>
param(
    [int]$MaxNewTokens = 3584,
    [int]$BatchSize = 5,
    [switch]$FilterOnly
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"

$pyVl = Join-Path $root ".venv-vl\Scripts\python.exe"
$pyModern = Join-Path $root ".venv-modern\Scripts\python.exe"
$gridRel = "outputs\matrix\nih_eval\qwen3_vl_8b_thinking\shift\shift_predictions.jsonl"
$grid = Join-Path $root $gridRel

function Stamp([string]$t) { Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $t) }

if (-not (Test-Path $grid)) { throw "missing grid file: $grid" }
$before = (Get-Content -LiteralPath $grid).Count
Stamp ("grid rows before filter: {0}" -f $before)

Write-Host "== step 1: backup + drop unusable rows =="
& $pyVl "scripts\filter_durable_failed_rows.py" --path $gridRel
if ($LASTEXITCODE -ne 0) { throw "filter step failed" }

if ($FilterOnly) { Stamp "filter only: GPU not touched"; exit 0 }

$after = (Get-Content -LiteralPath $grid).Count
Stamp ("kept {0} rows; {1} cells will be re-generated" -f $after, (10542 - $after))

if ($after -ge 10542) { Stamp "nothing to retry"; exit 0 }

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
Write-Host ("== step 2: retry {0} cells at {1} tokens, batch {2} ==" -f (10542 - $after), $MaxNewTokens, $BatchSize)
Stamp "generation starting (progress bar is live in this window)"
$code = 0
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
$code = $LASTEXITCODE
Stamp ("generation exit {0}" -f $code)

$rows = (Get-Content -LiteralPath $grid).Count
Stamp ("grid rows now: {0}/10542" -f $rows)

Write-Host ""
Write-Host "== step 3: audit + analyses =="
& $pyVl "scripts\98_round2_audit.py"
& $pyVl "scripts\96_analyze_round2.py"
& $pyVl "scripts\97_round2_stats.py"

Stamp "DONE"
Write-Host "Audit  : outputs\analysis_round2\round2_audit.json"
Write-Host "Summary: outputs\analysis_round2\round2_summary.json"
Write-Host "Stats  : outputs\analysis_round2\round2_label_ci.csv, round2_contract.csv, round2_marker.csv"
