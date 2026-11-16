<#
  run_round2_resume.ps1 -- finish round 2 from a normal PowerShell window.

  Why this exists
  ---------------
  A stage started from inside the editor dies when the editor is closed. This
  script is meant to be launched from your own PowerShell window, so it runs
  independently of the editor and keeps going as long as that window stays open.

  What it does, in order (each step skipped if already complete):
    1. v3_report_first, both cells   -- the two cells a CUDA kernel bug killed
       (attention backend: memory-efficient SDPA disabled; batch 4, because
       batch 8 hit the VRAM ceiling and stalled). Falls back to eager attention
       automatically if the first attempt fails.
    2. Qwen3-VL-Thinking grid        -- 7 cells x 1,506 = 10,542 calls, ~55-62 h
       (batch 6 / 2,560 tokens: 18.6 s/call and 5.1 GiB VRAM headroom measured in
        the 4/5/6/7 sweep; batch 7 leaves only 1.6 GiB and was rejected)
    3. CPU analyses 96 and 97        -- summary + bootstrap/FDR/contract/marker

  Progress is visible live in the window (the stage writes its own tqdm bar).
  From any other window you can also run:
      powershell -NoProfile -ExecutionPolicy Bypass -File program\scripts\round2_status.ps1

  Usage
  -----
      cd D:\Paper
      powershell -NoProfile -ExecutionPolicy Bypass -File program\scripts\run_round2_resume.ps1

      # preflight only (checks GPU / interpreters / manifest, launches nothing)
      powershell -NoProfile -ExecutionPolicy Bypass -File program\scripts\run_round2_resume.ps1 -CheckOnly
#>
param([switch]$CheckOnly)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"

$pyVl = Join-Path $root ".venv-vl\Scripts\python.exe"
$pyModern = Join-Path $root ".venv-modern\Scripts\python.exe"
$ablRoot = Join-Path $root "outputs\ablation_mg15\medgemma_15_4b_it\v3_report_first"
$gridFile = Join-Path $root "outputs\matrix\nih_eval\qwen3_vl_8b_thinking\shift\shift_predictions.jsonl"

function Stamp([string]$t) { Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $t) }
function Rows([string]$p) { if (Test-Path $p) { return (Get-Content -LiteralPath $p).Count } else { return 0 } }
function Run-Py([string]$exe, [string[]]$argv, [string]$label) {
    Stamp "$label : starting"
    $p = Start-Process -FilePath $exe -ArgumentList $argv -NoNewWindow -Wait -PassThru
    Stamp ("$label : exit {0}" -f $p.ExitCode)
    return $p.ExitCode
}

# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #
$ok = $true
foreach ($p in @($pyVl, $pyModern)) {
    if (-not (Test-Path $p)) { Write-Host "[FAIL] missing interpreter $p"; $ok = $false }
}
$man = Join-Path $root "data\manifests\nih_eval_1732.csv"
if (-not (Test-Path $man)) { Write-Host "[FAIL] missing manifest $man"; $ok = $false }
$think = Join-Path $root "models\Qwen3-VL-8B-Thinking\model-00001-of-00004.safetensors"
if (-not (Test-Path $think)) { Write-Host "[FAIL] Thinking checkpoint missing"; $ok = $false }

$free = $null
try {
    $freeLine = (& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) | Select-Object -First 1
    $free = [int]$freeLine.Trim()
} catch { Write-Host "[warn] nvidia-smi unavailable" }
if ($null -ne $free) {
    Stamp ("GPU free {0:N0} MiB" -f $free)
    if ($free -lt 20000 -and -not $CheckOnly) {
        Write-Host "[FAIL] less than 20 GiB free; close the other GPU process first"
        $ok = $false
    }
}
if (-not $ok) { exit 1 }

$v3o = Rows (Join-Path $ablRoot "original\image_reliance_predictions.jsonl")
$v3b = Rows (Join-Path $ablRoot "blank\image_reliance_predictions.jsonl")
$grid = Rows $gridFile
Stamp ("v3_report_first  original {0}/1506   blank {1}/1506" -f $v3o, $v3b)
Stamp ("second-model grid  {0}/10542" -f $grid)

if ($CheckOnly) { Stamp "check only: nothing launched"; exit 0 }

# --------------------------------------------------------------------------- #
# 1. v3_report_first
# --------------------------------------------------------------------------- #
if ($v3o -lt 1506 -or $v3b -lt 1506) {
    $cellArgs = @(
        "scripts\21_prompt_ablation.py",
        "--model", "medgemma_15_4b_it",
        "--variant", "v3_report_first",
        "--condition", "original", "blank",
        "--manifest", "data/manifests/nih_eval_1732.csv",
        "--images-root", "data/nih/images",
        "--output-root", "outputs/ablation_mg15",
        "--split", "internal_test,source_val",
        "--limit", "1506",
        "--batch-size", "4",
        "--max-new-tokens", "2048",
        "--resume"
    )

    Write-Host ""
    Write-Host "== step 1: v3_report_first (attention backend: memory-efficient SDPA off) =="
    $env:CXR_DISABLE_MEM_EFFICIENT_SDPA = "1"
    $code = Run-Py $pyVl $cellArgs "v3 attempt A"
    Remove-Item Env:CXR_DISABLE_MEM_EFFICIENT_SDPA -ErrorAction SilentlyContinue

    if ($code -ne 0 -or (Rows (Join-Path $ablRoot "original\image_reliance_predictions.jsonl")) -lt 1506) {
        Write-Host ""
        Write-Host "== step 1b: retry with eager attention =="
        $eagerArgs = $cellArgs + @("--attn-implementation", "eager")
        $code = Run-Py $pyVl $eagerArgs "v3 attempt B"
    }

    $v3o = Rows (Join-Path $ablRoot "original\image_reliance_predictions.jsonl")
    $v3b = Rows (Join-Path $ablRoot "blank\image_reliance_predictions.jsonl")
    Stamp ("v3 after repair  original {0}/1506   blank {1}/1506" -f $v3o, $v3b)
    if ($code -ne 0 -or $v3o -lt 1506 -or $v3b -lt 1506) {
        Write-Host "[STOP] v3 did not complete. Logs: outputs\round2_logs"
        exit 1
    }
} else {
    Stamp "step 1 skipped: v3 already complete"
}

# --------------------------------------------------------------------------- #
# 2. second-model grid
# --------------------------------------------------------------------------- #
$grid = Rows $gridFile
if ($grid -lt 10542) {
    Write-Host ""
    Write-Host "== step 2: Qwen3-VL-Thinking grid (7 cells x 1,506, bs6, ~55-62 h) =="
    $gridArgs = @(
        "scripts\08_shift_benchmark.py",
        "--model", "qwen3_vl_8b_thinking",
        "--manifest", "data/manifests/nih_eval_1732.csv",
        "--images-root", "data/nih/images",
        "--output-dir", "outputs/matrix/nih_eval/qwen3_vl_8b_thinking/shift",
        "--split", "internal_test,source_val",
        "--limit", "1506",
        "--severities", "severe",
        "--batch-size", "6",
        "--max-new-tokens", "2560",
        "--seed", "0"
    )
    $code = Run-Py $pyModern $gridArgs "second-model grid"
    $grid = Rows $gridFile
    Stamp ("grid rows {0}/10542" -f $grid)
    if ($code -ne 0 -or $grid -lt 10542) {
        Write-Host "[STOP] grid did not complete. Re-run this script to resume."
        exit 1
    }
} else {
    Stamp "step 2 skipped: grid already complete"
}

# --------------------------------------------------------------------------- #
# 3. CPU analyses
# --------------------------------------------------------------------------- #
Write-Host ""
Write-Host "== step 3: CPU analyses =="
Run-Py $pyVl @("scripts\96_analyze_round2.py") "analysis 96" | Out-Null
Run-Py $pyVl @("scripts\97_round2_stats.py") "analysis 97" | Out-Null

Write-Host ""
Stamp "ALL DONE"
Write-Host "Summary : outputs\analysis_round2\round2_summary.json"
Write-Host "Stats   : outputs\analysis_round2\round2_label_ci.csv, round2_contract.csv, round2_marker.csv"
