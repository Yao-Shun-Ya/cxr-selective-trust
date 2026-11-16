param([int]$Limit = 1506)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$logDir = Join-Path $root "outputs\round2_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("paraphrase_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

$py = Join-Path $root ".venv-vl\Scripts\python.exe"
Write-Host "Prompt paraphrase (MedGemma-4B, same JSON contract, two wordings)"
Write-Host "  expected: 2 paraphrases x 2 conditions x 1,506 = 6,024 calls / 3.5-4 h (smoke: 1.5-2.5 s/call at bs16)"
Write-Host "  log: $log"

$argv = @(
    "scripts\21_prompt_ablation.py",
    "--model", "medgemma_4b_it",
    "--variant", "p1_paraphrase_short", "p2_paraphrase_second_reader",
    "--condition", "original", "blank",
    "--manifest", "data/manifests/nih_eval_1732.csv",
    "--images-root", "data/nih/images",
    "--output-root", "outputs/paraphrase_mg4",
    "--split", "internal_test,source_val",
    "--limit", "$Limit",
    "--batch-size", "16",
    "--max-new-tokens", "256",
    "--resume"
)

$errLog = "$log.err"
$proc = Start-Process -FilePath $py -ArgumentList $argv -RedirectStandardOutput $log `
    -RedirectStandardError $errLog -NoNewWindow -PassThru -Wait
$code = $proc.ExitCode
Get-Content $log -Tail 40
if (Test-Path $errLog) { Get-Content $errLog -Tail 20 }
if ($code -ne 0) { throw "paraphrase run failed with exit $code; see $log / $errLog" }
Write-Host "done: outputs/paraphrase_mg4/medgemma_4b_it"
