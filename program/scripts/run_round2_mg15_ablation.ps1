param([switch]$Full, [string[]]$Variants, [int]$Limit = 1506)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$logDir = Join-Path $root "outputs\round2_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("mg15_ablation_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

$py = Join-Path $root ".venv-vl\Scripts\python.exe"
# -Variants additively extends the core set: a second invocation that names only
# v4 reuses the durable v0/v1 rows and runs the missing cells, so an extra
# schema can be bought with the slack left over after the first pass.
$variants = if ($Variants) {
    $Variants
} elseif ($Full) {
    @("v0_simple", "v1_per_label_binary", "v2_few_shot", "v3_report_first", "v4_normality_pressure")
} else {
    @("v0_simple", "v1_per_label_binary")
}

Write-Host "MedGemma-1.5 schema ablation"
Write-Host "  variants: $($variants -join ', ')"
Write-Host "  expected: core 2 schemas ~6,024 calls / 12-14 h (smoke: 32.7 s per image across the four cells); full 5 schemas ~26-30 h"
Write-Host "  log: $log"

$argv = @(
    "scripts\21_prompt_ablation.py",
    "--model", "medgemma_15_4b_it",
    "--variant"
) + $variants + @(
    "--condition", "original", "blank",
    "--manifest", "data/manifests/nih_eval_1732.csv",
    "--images-root", "data/nih/images",
    "--output-root", "outputs/ablation_mg15",
    "--split", "internal_test,source_val",
    "--limit", "$Limit",
    "--batch-size", "8",
    "--max-new-tokens", "2048",
    "--resume"
)

$errLog = "$log.err"
$proc = Start-Process -FilePath $py -ArgumentList $argv -RedirectStandardOutput $log `
    -RedirectStandardError $errLog -NoNewWindow -PassThru -Wait
$code = $proc.ExitCode
Get-Content $log -Tail 40
if (Test-Path $errLog) { Get-Content $errLog -Tail 20 }
if ($code -ne 0) { throw "MedGemma-1.5 ablation failed with exit $code; see $log / $errLog" }
Write-Host "done: outputs/ablation_mg15/medgemma_15_4b_it"
