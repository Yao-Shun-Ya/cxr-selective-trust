<#
  bench_bs_sweep.ps1 -- pick the largest safe batch size for the second-model grid.

  Context
  -------
  The grid ran at batch 4 / 2,560 tokens, ~27.5 s per call. Batch 8 previously
  reached 32,121 of 32,760 MiB and stopped completing batches (allocator
  thrash, not compute), so "bigger is better" is only true up to the point where
  at least ~3 GiB stays free.

  What this measures
  ------------------
  For each candidate batch size it runs the same small slice of the real grid
  (same manifest, same split, same seed, same 2,560-token budget), polls
  nvidia-smi every 2 s while the process is alive, and reports:

    * peak VRAM used and the headroom left;
    * seconds per call (the runner's own per-image timing);
    * parse rate and reasoning-close rate, so a size that silently truncates
      replies is rejected even if it is fast;
    * the projected hours for the full 10,542-call grid.

  Usage
  -----
      powershell -NoProfile -ExecutionPolicy Bypass -File program\scripts\bench_bs_sweep.ps1
#>
param(
    [int[]]$BatchSizes = @(4, 5, 6, 7),
    [int]$Images = 2,
    [int]$MinHeadroomMiB = 3072
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"

$py = Join-Path $root ".venv-modern\Scripts\python.exe"
$sweepRoot = Join-Path $root "outputs\bs_sweep"
$logDir = Join-Path $root "outputs\round2_logs"
New-Item -ItemType Directory -Force -Path $sweepRoot, $logDir | Out-Null

# A comma-separated list can arrive as ONE value (4,5,6,7 -> 4567) depending on
# how the caller quotes it, and a single batch of 14 cells then sits at the
# VRAM ceiling and thrashes instead of measuring anything. Refuse to run that.
$bad = $BatchSizes | Where-Object { $_ -lt 1 -or $_ -gt 16 }
if ($bad) {
    throw ("invalid batch sizes: {0}. Pass at most one value, or use the default." -f ($bad -join ", "))
}

$results = @()

foreach ($bs in $BatchSizes) {
    $outDir = Join-Path $sweepRoot ("bs{0}" -f $bs)
    if (Test-Path $outDir) { Remove-Item -Recurse -Force $outDir }
    $log = Join-Path $logDir ("bs_sweep_bs{0}.log" -f $bs)

    Write-Host ("=" * 78)
    Write-Host ("[bs={0}] starting: {1} images x 7 cells = {2} calls" -f $bs, $Images, ($Images * 7))
    $argv = @(
        "scripts\08_shift_benchmark.py",
        "--model", "qwen3_vl_8b_thinking",
        "--manifest", "data/manifests/nih_eval_1732.csv",
        "--images-root", "data/nih/images",
        "--output-dir", $outDir,
        "--split", "internal_test,source_val",
        "--limit", "$Images",
        "--severities", "severe",
        "--batch-size", "$bs",
        "--max-new-tokens", "2560",
        "--seed", "0"
    )
    $t0 = Get-Date
    $proc = Start-Process -FilePath $py -ArgumentList $argv `
        -RedirectStandardOutput $log -RedirectStandardError "$log.err" `
        -NoNewWindow -PassThru

    $peak = 0
    while (-not $proc.HasExited) {
        $line = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ($line) {
            $used = [int]$line.Trim()
            if ($used -gt $peak) { $peak = $used }
        }
        Start-Sleep -Seconds 2
    }
    $wall = ((Get-Date) - $t0).TotalSeconds
    $code = $proc.ExitCode

    $pred = Join-Path $outDir "shift_predictions.jsonl"
    $rows = @()
    if (Test-Path $pred) {
        $rows = Get-Content -LiteralPath $pred | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json }
    }
    $n = $rows.Count
    $perCell = if ($n) { ($rows | Measure-Object -Property elapsed_sec -Average).Average } else { $null }
    $parse = if ($n) { ($rows | Where-Object { $_.parsed.parse_success }).Count / $n } else { $null }
    $close = if ($n) { ($rows | Where-Object { $_.raw_output -match "</think>" }).Count / $n } else { $null }

    $total = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) | Select-Object -First 1
    $headroom = [int]$total.Trim() - $peak
    $proj = if ($perCell) { 10542 * $perCell / 3600 } else { $null }

    $results += [pscustomobject]@{
        BatchSize        = $bs
        Calls            = $n
        Exit             = $code
        WallSec          = [math]::Round($wall, 0)
        SecPerCall       = [math]::Round($perCell, 2)
        PeakMiB          = $peak
        HeadroomMiB      = $headroom
        ParseRate        = [math]::Round($parse, 3)
        CloseRate        = [math]::Round($close, 3)
        ProjectedHours   = [math]::Round($proj, 1)
        Safe             = ($headroom -ge $MinHeadroomMiB) -and ($code -eq 0) -and ($parse -gt 0.95)
    }
    Write-Host ("[bs={0}] {1:N1} s/call  peak {2} MiB  head {3} MiB  parse {4:P0}  close {5:P0}" -f `
        $bs, $perCell, $peak, $headroom, $parse, $close)
}

Write-Host ""
Write-Host ("=" * 78)
$results | Format-Table BatchSize, Calls, SecPerCall, PeakMiB, HeadroomMiB, ParseRate, CloseRate, ProjectedHours, Safe -AutoSize
$csv = Join-Path $sweepRoot "bs_sweep_results.csv"
$results | Export-Csv -Path $csv -NoTypeInformation -Encoding UTF8
Write-Host ("wrote {0}" -f $csv)

$best = $results | Where-Object { $_.Safe } | Sort-Object SecPerCall | Select-Object -First 1
if ($best) {
    Write-Host ""
    Write-Host ("BEST SAFE: bs={0}  {1} s/call  peak {2} MiB  head {3} MiB  projected {4} h" -f `
        $best.BatchSize, $best.SecPerCall, $best.PeakMiB, $best.HeadroomMiB, $best.ProjectedHours)
}
else {
    Write-Host ""
    Write-Host "No candidate met all conditions (headroom, exit code, parse rate)."
}
