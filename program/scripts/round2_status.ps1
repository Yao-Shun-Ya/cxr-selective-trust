<#
  round2_status.ps1 -- one-screen progress report for the round-2 GPU runs.

  Read-only. Safe to run while a stage is working: it only reads the durable
  JSONL files, the stage logs, and nvidia-smi.

  Usage
  -----
      powershell -NoProfile -ExecutionPolicy Bypass -File program\scripts\round2_status.ps1
#>
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot

function Count-Rows {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return 0 }
    return (Get-Content -LiteralPath $Path | Measure-Object -Line).Lines
}

function Last-Progress {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return "" }
    $text = Get-Content -LiteralPath $Path -Raw -ErrorAction SilentlyContinue
    if (-not $text) { return "" }
    $parts = $text -split "[`r`n]+" | Where-Object { $_ -match "\d+/\d+" }
    if (-not $parts) { return "" }
    $line = ($parts | Select-Object -Last 1).Trim()
    if ($line.Length -gt 150) { $line = $line.Substring($line.Length - 150) }
    return $line
}

Write-Host ("=" * 78)
Write-Host ("Round-2 status   {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-Host ("=" * 78)

$gpu = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader 2>$null
Write-Host ("GPU: {0}" -f ($gpu -join "  |  "))
$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "python.exe" }
Write-Host ("python processes: {0}" -f $procs.Count)

# ------------------------------------------------------------------ ablation
$abl = Join-Path $root "outputs\ablation_mg15\medgemma_15_4b_it"
Write-Host ""
Write-Host "[schema ablation] 10 cells x 1,506"
if (Test-Path $abl) {
    Get-ChildItem $abl -Directory | Sort-Object Name | ForEach-Object {
        $v = $_.Name
        $parts = @()
        foreach ($c in @("original", "blank")) {
            $n = Count-Rows (Join-Path $_.FullName "$c\image_reliance_predictions.jsonl")
            $parts += ("{0} {1,4}/1506" -f $c, $n)
        }
        Write-Host ("  {0,-24} {1}" -f $v, ($parts -join "   "))
    }
}
else { Write-Host "  (not started)" }

# ---------------------------------------------------------------- paraphrase
$par = Join-Path $root "outputs\paraphrase_mg4\medgemma_4b_it"
Write-Host ""
Write-Host "[prompt paraphrase] 4 cells x 1,506"
if (Test-Path $par) {
    Get-ChildItem $par -Directory | Sort-Object Name | ForEach-Object {
        $parts = @()
        foreach ($c in @("original", "blank")) {
            $n = Count-Rows (Join-Path $_.FullName "$c\image_reliance_predictions.jsonl")
            $parts += ("{0} {1,4}/1506" -f $c, $n)
        }
        Write-Host ("  {0,-30} {1}" -f $_.Name, ($parts -join "   "))
    }
}

# ------------------------------------------------------------------ second model
$grid = Join-Path $root "outputs\matrix\nih_eval\qwen3_vl_8b_thinking\shift\shift_predictions.jsonl"
Write-Host ""
Write-Host "[second model grid] 7 cells x 1,506 = 10,542"
$n = Count-Rows $grid
if ($n -gt 0) {
    Write-Host ("  total {0}/10542  ({1:N1} %)" -f $n, (100.0 * $n / 10542))
}
else { Write-Host "  (not started)" }

# ------------------------------------------------------------------ newest log
Write-Host ""
Write-Host "[newest stage log]"
$logs = Get-ChildItem (Join-Path $root "outputs\round2_logs") -Filter "*.log" |
    Sort-Object LastWriteTime | Select-Object -Last 1
if ($logs) {
    Write-Host ("  {0}  (updated {1})" -f $logs.Name, $logs.LastWriteTime.ToString("HH:mm:ss"))
    $p = Last-Progress $logs.FullName
    if ($p) { Write-Host ("  {0}" -f $p) }
    else { Get-Content $logs.FullName -Tail 3 | ForEach-Object { Write-Host ("  {0}" -f $_) } }
}
Write-Host ""
