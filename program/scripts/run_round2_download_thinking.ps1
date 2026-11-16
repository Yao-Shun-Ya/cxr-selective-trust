<#!
  92_download_thinking_modelscope.ps1 (now run_round2_download_thinking.ps1) --
  fetch Qwen3-VL-8B-Thinking from ModelScope.

  Why ModelScope instead of Hugging Face
  --------------------------------------
  huggingface.co is unreachable from this machine, and hf-mirror.com throttles
  large files to ~0.25 MB/s (measured 2026-11-02: 10 MB range request took
  41.9 s).  ModelScope serves the same official Qwen checkpoint at ~7 MB/s on
  the same link (10 MB in 1.4 s), i.e. ~40 minutes instead of ~18 hours.

  The shard files are byte-identical to the Hugging Face repo (same sha256 in
  `model.safetensors.index.json`), so the model is interchangeable once on
  disk.

  Behaviour
  ---------
  * four safetensors shards download in parallel, each with `curl -C -`
    (HTTP range resume), so an interrupted run restarts where it stopped;
  * small config/tokenizer files download first and sequentially;
  * every file is size-checked against the published repo listing; a single
    mismatch fails the script instead of leaving a half model behind.

  Usage
  -----
      powershell -NoProfile -ExecutionPolicy Bypass `
          -File program\scripts\92_download_thinking_modelscope.ps1

  Exit 0 = all files present and size-matched.  Exit 1 = something is short.
#>
param(
    [string]$ModelDir = "$PSScriptRoot\..\models\Qwen3-VL-8B-Thinking",
    [string]$BaseUrl  = "https://modelscope.cn/models/Qwen/Qwen3-VL-8B-Thinking/resolve/master"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Sizes = [ordered]@{
    ".gitattributes"                     = 2449
    "README.md"                          = 7201
    "chat_template.json"                 = 5412
    "config.json"                        = 1474
    "configuration.json"                 = 73
    "generation_config.json"             = 270
    "merges.txt"                         = 1671839
    "model.safetensors.index.json"       = 67759
    "preprocessor_config.json"           = 390
    "tokenizer.json"                     = 7032403
    "tokenizer_config.json"              = 10781
    "video_preprocessor_config.json"     = 385
    "vocab.json"                         = 2776833
    "model-00001-of-00004.safetensors"   = 4902275944
    "model-00002-of-00004.safetensors"   = 4915962496
    "model-00003-of-00004.safetensors"   = 4999831048
    "model-00004-of-00004.safetensors"   = 2716270024
}

$ModelDir = [System.IO.Path]::GetFullPath($ModelDir)
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
Write-Host "[dir] $ModelDir"

$small, $shards = @(), @()
foreach ($name in $Sizes.Keys) {
    if ($name -like "*safetensors" -and $name -like "model-0000*") {
        if ($name -ne "model.safetensors.index.json") { $shards += $name } else { $small += $name }
    } else {
        $small += $name
    }
}

function Get-One {
    param([string]$Name)
    $url = "$BaseUrl/$Name"
    $dst = Join-Path $ModelDir $Name
    # Small files are re-fetched whole: `curl -C -` fails with "server does not
    # seem to support byte ranges" when a zero-length leftover exists.
    $fresh = $true
    for ($try = 1; $try -le 5; $try++) {
        if ($fresh -and (Test-Path $dst)) { Remove-Item -Force $dst }
        $rangeArgs = if ($fresh) { @() } else { @("-C", "-") }
        & curl.exe -sS -L @rangeArgs --retry 3 --retry-delay 2 --connect-timeout 20 `
            -o $dst $url 2>&1 | Out-Null
        if (Test-Path $dst) {
            $got = (Get-Item $dst).Length
            if ($got -eq $Sizes[$Name]) { return $true }
        }
        Write-Host ("[retry {0}] {1} -> {2}/{3} bytes" -f $try, $Name,
                    $(if (Test-Path $dst) { (Get-Item $dst).Length } else { 0 }), $Sizes[$Name])
        $fresh = $false
        Start-Sleep -Seconds 3
    }
    return $false
}

Write-Host "[small] $($small.Count) files"
foreach ($n in $small) { if (-not (Get-One $n)) { Write-Host "FAILED $n"; exit 1 } }

Write-Host "[shards] $($shards.Count) files in parallel"
$jobs = @()
foreach ($n in $shards) {
    $jobs += Start-Job -Name $n -ScriptBlock {
        param($Name, $Url, $Dst, $Size)
        for ($try = 1; $try -le 6; $try++) {
            & curl.exe -sS -L -C - --retry 3 --retry-delay 2 --connect-timeout 20 -o $Dst $Url 2>&1 | Out-Null
            if ((Test-Path $Dst) -and ((Get-Item $Dst).Length -eq $Size)) { return "OK $Name" }
            Start-Sleep -Seconds 3
        }
        return "SHORT $Name $((Get-Item $Dst).Length)/$Size"
    } -ArgumentList $n, "$BaseUrl/$n", (Join-Path $ModelDir $n), $Sizes[$n]
}

$last = 0
while ($jobs | Where-Object { $_.State -eq "Running" }) {
    Start-Sleep -Seconds 60
    $total = 0
    foreach ($n in $shards) {
        $f = Join-Path $ModelDir $n
        if (Test-Path $f) { $total += (Get-Item $f).Length }
    }
    $rate = ([double]($total - $last)) / 60.0 / 1MB
    $last = $total
    $want = ($shards | ForEach-Object { $Sizes[$_] } | Measure-Object -Sum).Sum
    Write-Host ("[shards] {0:N2}/{1:N2} GB  {2:N2} MB/s" -f ($total/1GB), ($want/1GB), $rate)
}

$jobs | Receive-Job
$jobs | Remove-Job

$bad = @()
foreach ($n in $Sizes.Keys) {
    $f = Join-Path $ModelDir $n
    if (-not (Test-Path $f)) { $bad += "missing $n"; continue }
    if ((Get-Item $f).Length -ne $Sizes[$n]) { $bad += ("size {0}: {1}/{2}" -f $n, (Get-Item $f).Length, $Sizes[$n]) }
}
if ($bad.Count) { Write-Host "[FAIL]"; $bad | ForEach-Object { Write-Host "  $_" }; exit 1 }

Write-Host "[OK] all 17 files size-matched"
exit 0
