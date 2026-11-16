param([switch]$Try30B)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$logDir = Join-Path $root "outputs\round2_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("reasoning_probe_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

$py = Join-Path $root ".venv-modern\Scripts\python.exe"
$model = if ($Try30B) { "qwen3_vl_30b_a3b_awq4bit" } else { "qwen3_vl_8b_thinking" }
$out = if ($Try30B) { "outputs/reasoning_probe_qwen3_30b_round2" } else { "outputs/reasoning_probe_qwen3_thinking_round2" }

Write-Host "Reasoning-channel probe: $model"
Write-Host "  pass if the reply closes a reasoning block (</think>) OR a non-empty preamble before JSON discriminates noise from resolution"
Write-Host "  log: $log"

$argv = @(
    "scripts\95_probe_reasoning_marker.py",
    "--model", $model,
    "--limit", "4",
    # Match the grid's generation budget closely enough that the gate measures
    # the channel the grid will see: at 1,024 tokens the probe's replies are
    # still inside the reasoning block, so the close tag never appears and the
    # gate would under-report the channel it is supposed to detect.
    "--max-new-tokens", "2048",
    "--out", $out
)
if ($Try30B) { $argv += @("--max-memory", "30GiB") }

$errLog = "$log.err"
$proc = Start-Process -FilePath $py -ArgumentList $argv -RedirectStandardOutput $log `
    -RedirectStandardError $errLog -NoNewWindow -PassThru -Wait
$code = $proc.ExitCode
Get-Content $log -Tail 40
if (Test-Path $errLog) { Get-Content $errLog -Tail 20 }
if ($code -eq 0) {
    Write-Host "PROBE PASS: a readable reasoning channel exists. You may launch the 7-cell grid."
} elseif ($code -eq 2) {
    Write-Host "PROBE FAIL: no </think> close tag and no discriminative preamble (noise vs resolution). Do NOT launch the grid."
} else {
    Write-Host "PROBE ERROR: loader/runtime failure (exit $code); see $log"
}
exit $code
