<#
  run_round2_fix_v3.ps1 -- rerun the two v3_report_first cells after a CUDA
  kernel failure in the main ablation run.

  What happened
  -------------
  The ablation wrote all eight v0/v1/v2 cells, then v3_report_first failed on
  its first batch with

      RuntimeError: p.attn_bias_ptr is not correctly aligned

  and produced zero rows for both conditions.  That message comes from torch
  2.5.1's memory-efficient SDPA kernel, which requires the attention-bias
  pointer to be 16-byte aligned; the v3 prompt happens to create a prefill
  length where it is not.  v0/v1/v2 (shorter prompts) and v4 (longer prompt)
  do not hit that length, which is why only v3 is missing.

  A plain retry would rebuild the same batch and fail identically, so this
  script changes the kernel, not the prompt: it disables the memory-efficient
  backend (flash / math remain, and they implement the same attention maths).
  If that still fails, it falls back to `--attn-implementation eager`.

  Each candidate backend is screened on 32 images (four batches, the same
  first images and therefore the same padded lengths as the real run) before a
  five-hour full rerun is committed.

  Batch size 4, not 8
  -------------------
  With the memory-efficient kernel disabled, batch 8 reached 32.1 of 32.8 GiB
  and stopped completing batches (100 % GPU utilisation at 83 W — allocator
  thrash, not compute). Batch 4 leaves ~12 GiB head, and the 32 rows already
  written at batch 8 are reused by `--resume`.

  Usage
  -----
      powershell -NoProfile -ExecutionPolicy Bypass -File program\scripts\run_round2_fix_v3.ps1
#>
param([int]$SmokeLimit = 32, [int]$BatchSize = 4)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$logDir = Join-Path $root "outputs\round2_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$py = Join-Path $root ".venv-vl\Scripts\python.exe"
$smokeRoot = Join-Path $root "outputs\fix_v3_smoke"
if (Test-Path $smokeRoot) { Remove-Item -Recurse -Force $smokeRoot }

function Invoke-Cell {
    param([int]$Limit, [string]$OutRoot, [string]$Tag, [hashtable]$Env, [string[]]$Extra)
    $log = Join-Path $logDir ("fix_v3_{0}_{1}.log" -f $Tag, (Get-Date -Format "HHmmss"))
    foreach ($k in $Env.Keys) { Set-Item -Path "Env:$k" -Value $Env[$k] }
    $argv = @(
        "scripts\21_prompt_ablation.py",
        "--model", "medgemma_15_4b_it",
        "--variant", "v3_report_first",
        "--condition", "original", "blank",
        "--manifest", "data/manifests/nih_eval_1732.csv",
        "--images-root", "data/nih/images",
        "--output-root", $OutRoot,
        "--split", "internal_test,source_val",
        "--limit", "$Limit",
        "--batch-size", "$BatchSize",
        "--max-new-tokens", "2048",
        "--resume"
    ) + $Extra
    $proc = Start-Process -FilePath $py -ArgumentList $argv `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -NoNewWindow -PassThru -Wait
    foreach ($k in $Env.Keys) { Remove-Item -Path "Env:$k" -ErrorAction SilentlyContinue }
    return @{ Code = $proc.ExitCode; Log = $log }
}

Write-Host "v3_report_first repair"
Write-Host "  smoke: $SmokeLimit images per cell at bs$BatchSize"
Write-Host ""

$candidates = @(
    @{ Tag = "no_mem_efficient"; Env = @{ CXR_DISABLE_MEM_EFFICIENT_SDPA = "1" }; Extra = @() },
    @{ Tag = "eager";            Env = @{};                                    Extra = @("--attn-implementation", "eager") }
)

$chosen = $null
foreach ($c in $candidates) {
    Write-Host ("[smoke] backend candidate: {0}" -f $c.Tag)
    $r = Invoke-Cell -Limit $SmokeLimit -OutRoot $smokeRoot -Tag $c.Tag -Env $c.Env -Extra $c.Extra
    $pred = Join-Path $smokeRoot "medgemma_15_4b_it\v3_report_first\original\image_reliance_predictions.jsonl"
    $rows = if (Test-Path $pred) { (Get-Content $pred | Measure-Object -Line).Lines } else { 0 }
    if ($r.Code -eq 0 -and $rows -eq $SmokeLimit) {
        Write-Host ("[smoke] PASS {0}: {1}/{2} rows" -f $c.Tag, $rows, $SmokeLimit)
        $chosen = $c
        break
    }
    Write-Host ("[smoke] FAIL {0}: exit {1}, {2}/{3} rows; see {4}" -f $c.Tag, $r.Code, $rows, $SmokeLimit, $r.Log)
    Get-Content $r.Log -Tail 6
}

if (-not $chosen) { throw "no working attention backend found; inspect the smoke logs in outputs\round2_logs" }

Write-Host ""
Write-Host ("[full] rerunning v3_report_first with {0}" -f $chosen.Tag)
$full = Invoke-Cell -Limit 1506 -OutRoot (Join-Path $root "outputs\ablation_mg15") `
    -Tag ("full_" + $chosen.Tag) -Env $chosen.Env -Extra $chosen.Extra
Get-Content $full.Log -Tail 25
if (Test-Path "$($full.Log).err") { Get-Content "$($full.Log).err" -Tail 15 }
if ($full.Code -ne 0) { throw "v3 full rerun failed with exit $($full.Code); see $($full.Log)" }

Remove-Item -Recurse -Force $smokeRoot -ErrorAction SilentlyContinue
Write-Host "v3_report_first complete: outputs/ablation_mg15/medgemma_15_4b_it/v3_report_first"
Write-Host "Next: powershell -NoProfile -ExecutionPolicy Bypass -File program\scripts\run_round2_reasoning_grid.ps1"
