param([int]$Epochs = 8, [int]$BatchSize = 32, [switch]$NoPretrained)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$logDir = Join-Path $root "outputs\round2_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("densenet_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

$py = Join-Path $root ".venv-vl\Scripts\python.exe"
Write-Host "DenseNet-121 supervised baseline"
Write-Host "  train 3,871 / val 735 / eval 1,506; patient overlap 0"
Write-Host "  expected: $Epochs epochs, ~25 s/epoch + eval, 5-10 min total"
Write-Host "  log: $log"

$argv = @(
    "scripts\94_train_densenet_baseline.py",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--workers", "4"
)
if (-not $NoPretrained) { $argv += "--pretrained" }

$errLog = "$log.err"
$proc = Start-Process -FilePath $py -ArgumentList $argv -RedirectStandardOutput $log `
    -RedirectStandardError $errLog -NoNewWindow -PassThru -Wait
$code = $proc.ExitCode
Get-Content $log -Tail 60
if (Test-Path $errLog) { Get-Content $errLog -Tail 25 }
if ($code -ne 0) { throw "DenseNet baseline failed with exit $code; see $log / $errLog" }
Write-Host "done: outputs/densenet_baseline/metrics.json"
