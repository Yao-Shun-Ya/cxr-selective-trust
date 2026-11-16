param([int]$Limit = 1506)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$logDir = Join-Path $root "outputs\round2_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("reasoning_grid_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

$py = Join-Path $root ".venv-modern\Scripts\python.exe"
Write-Host "Second-model grid: Qwen3-VL-8B-Thinking, clean + six severe cells"
Write-Host "  design: 7 cells x 1,506 = 10,542 calls, batch 6, 2,560 tokens"
Write-Host "  measured: 18.6 s/call at batch 6 (peak 27,663 MiB, 5,097 MiB headroom); batch 7 leaves 1,624 MiB and was rejected, batch 8 thrashed"
Write-Host "  note: batch 7 leaves 1,624 MiB and batch 8 thrashed at 32,121 MiB;"
Write-Host "        batch 6 is the largest measured size with >=3 GiB headroom."
Write-Host "  log: $log"

$argv = @(
    "scripts\08_shift_benchmark.py",
    "--model", "qwen3_vl_8b_thinking",
    "--manifest", "data/manifests/nih_eval_1732.csv",
    "--images-root", "data/nih/images",
    "--output-dir", "outputs/matrix/nih_eval/qwen3_vl_8b_thinking/shift",
    "--split", "internal_test,source_val",
    "--limit", "$Limit",
    "--severities", "severe",
    "--batch-size", "6",
    # 2,048 tokens truncated 2 of 14 smoke replies before they closed </think>,
    # which loses the JSON entirely. 2,560 keeps the extra head only for the
    # replies that need it (generation stops at EOS), so the cost is ~2-3 %.
    "--max-new-tokens", "2560",
    "--seed", "0"
)

$errLog = "$log.err"
$proc = Start-Process -FilePath $py -ArgumentList $argv -RedirectStandardOutput $log `
    -RedirectStandardError $errLog -NoNewWindow -PassThru -Wait
$code = $proc.ExitCode
Get-Content $log -Tail 40
if (Test-Path $errLog) { Get-Content $errLog -Tail 20 }
if ($code -ne 0) { throw "second-model grid failed with exit $code; see $log / $errLog" }
Write-Host "done: outputs/matrix/nih_eval/qwen3_vl_8b_thinking/shift"
