# One-line status for the frontier-API probe.
#   powershell -ExecutionPolicy Bypass -File scripts\api_probe\status.ps1
# The rate is measured between two invocations (state cached in status_state.json),
# so a truncated or resumed file does not produce a misleading average.

$base = "D:\Paper\program\outputs\api_probe\deepseek_v4_1_flash"
$target = 9400
$state = Join-Path $base "status_state.json"

$alive = (Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
          Where-Object { $_.CommandLine -match '03_run_api_probe' }).Count -gt 0

$rows = 0; $errors = 0
if (Test-Path "$base\predictions.jsonl") {
    $lines = Get-Content "$base\predictions.jsonl"
    $rows = $lines.Count
    $errors = ($lines | Select-String '"error": "[^"]').Count
}

$now = Get-Date
$rate = $null
if (Test-Path $state) {
    $prev = Get-Content $state -Raw | ConvertFrom-Json
    $mins = ($now - [datetime]$prev.t).TotalMinutes
    if ($mins -gt 0.2 -and $rows -gt $prev.n) {
        $rate = ($rows - $prev.n) / $mins
    }
}
@{ n = $rows; t = $now.ToString("o") } | ConvertTo-Json | Set-Content $state

$eta = "n/a"
if ($rate -and $rate -gt 0) {
    $eta = "{0:N1} h" -f ((($target - $rows) / $rate) / 60)
}
$r = if ($rate) { "{0:N1}/min" -f $rate } else { "warming" }

$last = ""
if (Test-Path "$base\run_console.log") {
    $last = (Get-Content "$base\run_console.log" -Tail 1) -join ""
}

"{0}  rows={1}/{2}  errors={3}  {4}  eta={5}  worker={6}" -f `
    (Get-Date -Format 'HH:mm:ss'), $rows, $target, $errors, $r, $eta,
    $(if ($alive) { "RUNNING" } else { "STOPPED" })
"last: $last"
