# run_all_experiments.ps1 -- one-button driver for the NIH-layer experiment matrix.
#
# WHICH DRIVER DO I WANT?
# -----------------------
#   scripts\run_all_layers.ps1       the three-layer 6K evaluation (NIH + PadChest +
#                                    Open-i). This is the one to use for the full run:
#                                    it picks the design that answers each dataset's
#                                    own question. See docs\48_three_layer_protocol.md.
#   scripts\run_all_experiments.ps1  (this file) the NIH layer only: 19-cell synthetic
#                                    shift + reliance + bf16 control + prompt ablation.
#                                    223.5 h for all 7 models, 133.0 h for the default 5.
#                                    Still correct and still supported; use it when the
#                                    external layers are not wanted or are already done.
#
# Design
# ------
# * **Per model, not per experiment.** CheXagent-8B takes 75 s to load and
#   LLaVA-Med 220 s; iterating experiments outer / models inner would reload each
#   model once per experiment. Instead each model is loaded once and all of its
#   stages run back to back.
# * **Resumable at two levels.** Every stage writes its own JSONL with resume
#   support (already built into 06/08/21), and this driver additionally records
#   completed stages in a state file so a re-run skips whole finished stages.
# * **Live progress and honest ETAs.** Timings are measured on this machine
#   (docs/44, docs/46) and multiplied by a conservative batching factor, because
#   in these experiments a batch mixes different image sizes and therefore pays
#   padding overhead that the single-image benchmark did not.
# * **An environment per model.** Qwen3-VL-8B needs transformers 5.18
#   (.venv-modern); everything else runs in .venv-vl (4.51.3, the environment that
#   produced the published 4-bit results).
#
# Usage
# -----
#   powershell -ExecutionPolicy Bypass -File scripts\run_all_experiments.ps1
#   ... -Limit 100            # quick smoke of the whole pipeline
#   ... -Models medgemma_4b_it
#   ... -Stages shift,reliance
#   ... -Reset                # forget recorded completion and rebuild
#   ... -NoAblation           # skip the expensive prompt ablation
#
# Ctrl-C stops safely; re-run the same command to continue.

[CmdletBinding()]
param(
    # ValueFromRemainingArguments makes "-Models a,b" behave like "-Models a b";
    # without it PowerShell binds the whole "a,b" as one string and every stage
    # is silently skipped.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Models,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Stages,
    [int]      $Limit      = 1732,
    [int]      $AblationLimit = 1732,
    [int]      $BatchSize  = 4,
    [switch]   $NoAblation,
    [switch]   $Reset,
    # Print the plan and the time budget, then exit without running anything.
    # Named -DryRun because PowerShell variable names are case-insensitive and
    # $plan already holds the stage table below.
    [switch]   $DryRun,
    # Manifest the stages read. The default is the 6K-strategy evaluation set
    # (docs/41): NIH internal_test + source_val only, 1,732 images. Point this at
    # data\manifests\nih_shared_500.csv for a fast pipeline check, or at
    # nih_shared_5000.csv together with -Splits to go wider.
    [string]   $Manifest = 'data\manifests\nih_eval_1732.csv',
    # Splits kept from the manifest. This exists because --limit takes the first
    # N rows of the file, NOT a sample: nih_shared_5000.csv is ordered by
    # image_id and contains 3,268 internal_train rows, so without a split filter
    # a large -Limit silently evaluates on training data.
    [string]   $Splits   = 'internal_test,source_val',
    [string]   $StateFile
)

$ErrorActionPreference = 'Continue'
$script:Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
# An explicit -BatchSize wins over the per-model measured sweet spot, so a
# deliberate override still works.
$script:BatchSizeExplicit = $PSBoundParameters.ContainsKey('BatchSize')
Set-Location $script:Root

# Normalise comma-separated lists and drop empties. "-Stages shift,reliance"
# arrives as one string "shift,reliance" in some invocations, so split it.
$Models = @($Models | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } |
            Where-Object { $_ })
$Stages = @($Stages | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } |
            Where-Object { $_ })
if ($Stages.Count -eq 0) { $Stages = @('shift', 'reliance', 'precision', 'ablation') }

$KnownStages = @('shift', 'reliance', 'precision', 'ablation')
$badStages = $Stages | Where-Object { $KnownStages -notcontains $_ }
if ($badStages) {
    Write-Host ("ERROR: unknown stage(s): {0}" -f ($badStages -join ', ')) -ForegroundColor Red
    Write-Host ("       known stages: {0}" -f ($KnownStages -join ', ')) -ForegroundColor Red
    exit 2
}

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
# measured single-image throughput (s/img) -- docs/44, docs/46
$Throughput = @{
    medgemma_4b_it          = 0.41
    medgemma_15_4b_it       = 0.41
    qwen25_vl_7b_instruct   = 1.28
    qwen3_vl_8b_instruct    = 2.18
    chexagent_8b            = 7.14
    internvl3_5_8b          = 4.40
    llava_med_v1_5_7b       = 0.90
}
# measured load time (s) -- CheXagent and LLaVA-Med are the slow ones
$LoadSeconds = @{
    chexagent_8b      = 75
    internvl3_5_8b    = 30
    llava_med_v1_5_7b = 220
}
# Qwen3-VL needs transformers >= 4.57, so it runs in the modern environment
$ModernEnvModels = @('qwen3_vl_8b_instruct')

# Token budget. 128 is NOT enough: the structured prompt asks for 8 findings plus
# 8 probabilities, and the reply exceeds 128 tokens, producing truncated JSON that
# the parser rejects with "no_json_object_found". That yields a silent
# parse-failure rate of 100% (found the hard way during orchestration testing).
# docs/38 specifies 256 for this prompt.
$MaxNewTokens = 256

# Each stage script loads the model itself, so every stage pays a load cost.
$StageStartupSeconds = 35
# padding penalty: batching mixed-size images costs ~20% over the benchmark figure
$PaddingFactor = 1.20

# The recommended matrix. NIH internal_test+source_val is 1,732 images x 19
# shift cells = 32,908 shift calls per model, so every model added costs real
# days. LLaVA-Med (26 h) and InternVL (94 h) are the two slowest and are left
# out of the primary matrix; their wrappers still work, so they can be added
# with -Models when the budget allows.
$DefaultModels = @(
    'medgemma_4b_it'
    'medgemma_15_4b_it'
    'qwen25_vl_7b_instruct'
    'qwen3_vl_8b_instruct'
    'chexagent_8b'
)
# Models that get the prompt-ablation stage (5 variants x 2 conditions).
# One model is enough to make the point that the effect is not a prompt artefact.
$DefaultAblationModels = @('medgemma_4b_it')
$AblationModels = $DefaultAblationModels

# Which models may be handed --load-in-4bit for the *primary* (shift/reliance)
# stages. Only the models whose sweet spot was MEASURED in 4-bit are listed;
# everything else runs at bf16, exactly as benchmarked (docs/44, docs/46).
# Mixing the two precisions inside one table is forbidden, and running an
# unmeasured precision would change throughput and possibly accuracy.
#   medgemma_4b_it        : 4-bit 0.41 s/img / 14.2 GB  (bf16 0.40 / 21.0)
#   medgemma_15_4b_it     : 4-bit 0.41 s/img / 14.2 GB
#   qwen25_vl_7b_instruct : 4-bit 1.28 s/img / 13.5 GB  (bf16 1.31 / 18.1)
#   chexagent_8b          : bf16  7.14 s/img / 29.7 GB only (bs>=6 collapses)
#   internvl3_5_8b        : bf16  4.40 s/img / 16.0 GB only (serial by design)
#   llava_med_v1_5_7b     : bf16  0.90 s/img / 18.1 GB only (fresh conversion)
#   qwen3_vl_8b_instruct  : bf16  2.18 s/img / 23.2 GB only (4-bit not measured)
$FourBitModels = @('medgemma_4b_it', 'medgemma_15_4b_it', 'qwen25_vl_7b_instruct')

# Measured sweet spot (docs/44). Passing a larger batch to a model that was
# benchmarked serially only makes it slower: CheXagent-8B collapses from 7.14 to
# 117 s/img at bs=6, and InternVL ignores batching entirely.
$SweetSpotBatch = @{
    medgemma_4b_it          = 64
    medgemma_15_4b_it       = 64
    qwen25_vl_7b_instruct   = 8
    qwen3_vl_8b_instruct    = 8
    chexagent_8b            = 4
    internvl3_5_8b          = 1
    llava_med_v1_5_7b       = 1
}

# Ceiling for the bf16 precision stage, where VRAM is the binding constraint.
# CheXagent-8B measured 29.7 GB at bs=4 and collapses into driver paging at bs=6.
$PrecisionBatchCap = @{
    chexagent_8b            = 2
    qwen3_vl_8b_instruct    = 4
    medgemma_4b_it          = 4
    medgemma_15_4b_it       = 4
    qwen25_vl_7b_instruct   = 4
    internvl3_5_8b          = 1
    llava_med_v1_5_7b       = 1
}

# ETA calibration.
#
# $Throughput is the single-image benchmark from docs/44, measured with
# max_new_tokens=128 on a uniform 1024x1024 crop. A formal run differs in two
# ways, and both are real costs rather than noise:
#
#   1. token budget -- 256 instead of 128. Isolated cleanly on InternVL, whose
#      first 22 cells were costed at bs=1: 4.40 s/img at 128 tokens became
#      6.04 s/img at 256. InternVL's serial path makes this a pure measurement,
#      so its factor is 1.0 and everything else is scaled by the same 1.37.
#   2. batching mixed-size images -- the benchmark used same-size images, the
#      runs do not. This shows up as a per-model batching factor below.
#
# The batching factors are derived from the ONE completed orchestrator run
# (绗竴娆¤窇.md) by converting its wall-clock per stage into seconds per cell and
# dividing out the token-budget factor. They are honest measurements with a
# small error bar, NOT guesses -- except the two marked PROVISIONAL, which
# extrapolate from a sibling model and should be replaced after the -Limit 20
# pre-run reports real numbers.
#
#   table value / token factor  ->  expected s per cell
#   observed wall / cells       ->  actual s per cell
#   factor = actual / expected
#
# Resulting expectation (s per cell) with factor applied:
#   medgemma_* 0.81 | qwen25 2.09 | qwen3 3.56 | chexagent 5.94
#   internvl 6.04  | llava_med 2.05
$TokenBudgetFactor = 1.37
$BatchingFactor = @{
    medgemma_4b_it          = 0.42   # 17m shift+reliance = 7600 cells -> 0.134 s/cell
    medgemma_15_4b_it       = 0.42   # same family, same measured rate
    qwen25_vl_7b_instruct   = 1.19   # 11m / 380 = 1.74 s/cell
    qwen3_vl_8b_instruct    = 1.19   # PROVISIONAL: same size class as qwen25
    chexagent_8b            = 0.61   # 21m / 380 = 3.32 s/cell
    internvl3_5_8b          = 1.00   # measured directly at 256 tokens
    llava_med_v1_5_7b       = 1.35   # 2m / 380 = 0.32 s/cell
}

$ShiftCells = 19        # 6 shifts x 3 severities + clean
$RelianceConditions = 4 # original, blank, blur, shuffle

# Items produced per image, per stage. Used both for the timing estimate below
# and by Record-Timing. It used to be a local variable inside Record-Timing only,
# so Get-StageSeconds indexed an undefined $ItemsPerImage and PowerShell 5.1
# printed "Cannot index into a null array." once per stage while silently
# disabling the measured-rate calibration.
$ItemsPerImage = @{ shift = $ShiftCells; reliance = $RelianceConditions
                    precision = $RelianceConditions; ablation = 10 }

if ($NoAblation) { $Stages = $Stages | Where-Object { $_ -ne 'ablation' } }
if (-not $Models) { $Models = $DefaultModels }

if (-not $StateFile) { $StateFile = Join-Path $script:Root 'outputs\state\run_all_state.json' }
if ($Reset -and (Test-Path $StateFile)) { Remove-Item $StateFile -Force }

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
function Get-Python([string] $model) {
    if ($ModernEnvModels -contains $model) {
        return (Join-Path $script:Root '.venv-modern\Scripts\python.exe')
    }
    return (Join-Path $script:Root '.venv-vl\Scripts\python.exe')
}

function Format-Duration([double] $seconds) {
    if ($seconds -lt 60)   { return ("{0:N0}s" -f $seconds) }
    if ($seconds -lt 3600) { return ("{0:N0}m" -f ($seconds / 60)) }
    return ("{0:N1}h" -f ($seconds / 3600))
}

function Read-State {
    if (Test-Path $StateFile) {
        try { return (Get-Content $StateFile -Raw | ConvertFrom-Json) } catch { }
    }
    return [pscustomobject]@{ completed = @(); started = (Get-Date).ToString('s') }
}

function Record-Timing($state, [string] $model, [string] $stage, [int] $images, [double] $seconds) {
    $k = $ItemsPerImage[$stage]
    if (-not $k -or $images -le 0 -or $seconds -le 0) { return }
    $perItem = $seconds / ($images * $k)
    if (-not $state.PSObject.Properties['timings']) {
        $state | Add-Member -NotePropertyName timings -NotePropertyValue ([pscustomobject]@{}) -Force
    }
    $entry = [pscustomobject]@{ per_item_sec = [math]::Round($perItem, 4); images = $images }
    if ($state.timings.PSObject.Properties["$model/$stage"]) {
        $state.timings."$model/$stage" = $entry
    } else {
        $state.timings | Add-Member -NotePropertyName "$model/$stage" -NotePropertyValue $entry -Force
    }
}

function Save-State($state) {
    $dir = Split-Path -Parent $StateFile
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $state | ConvertTo-Json -Depth 5 | Set-Content -Path $StateFile -Encoding UTF8
}

function Get-PredictionFile([string] $model, [string] $stage) {
    $base = Join-Path $script:Root ("outputs\matrix\{0}\{1}" -f $model, $stage)
    foreach ($f in @('shift_predictions.jsonl', 'image_reliance_predictions.jsonl')) {
        $p = Join-Path $base $f
        if (Test-Path $p) { return $p }
    }
    # The ablation stage writes one jsonl per (variant, condition) cell, so the
    # file is nested; any one of them is representative enough to sample.
    if (Test-Path $base) {
        $nested = Get-ChildItem -Path $base -Recurse -Filter 'image_reliance_predictions.jsonl' `
                                -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($nested) { return $nested.FullName }
    }
    return $null
}

function Test-StageOutput {
    param([string] $model, [string] $stage, [int] $sample = 200)
    # A stage script can exit 0 while every row it wrote is an error string. The
    # InternVL image-type bug did exactly that: 80/80 reliance rows carried
    # "'Image' object has no attribute 'read'" and the wrapper still returned 0.
    # Sample the file rather than scanning it, and report the dominant error so a
    # broken stage is visible instead of silently "complete".
    $p = Get-PredictionFile $model $stage
    if (-not $p) { return [pscustomobject]@{ ok = $false; note = 'no output file'; n = 0; bad = 0 } }
    $n = 0; $bad = 0; $errs = @{}
    $i = 0
    foreach ($line in (Get-Content $p -TotalCount $sample)) {
        if (-not $line.Trim()) { continue }
        $n++
        if ($line -match '"error"\s*:\s*"([^"]+)"') {
            $bad++
            $msg = $Matches[1]
            if ($errs.ContainsKey($msg)) { $errs[$msg]++ } else { $errs[$msg] = 1 }
        }
    }
    if ($n -eq 0) { return [pscustomobject]@{ ok = $false; note = 'output file empty'; n = 0; bad = 0 } }
    $note = ''
    if ($errs.Count -gt 0) {
        $top = $errs.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 1
        $note = "{0}x {1}" -f $top.Value, $top.Key
    }
    # Tolerate isolated flakes, reject systematic breakage: a real bug hits
    # essentially every row, so a couple of bad rows in a 200-row sample must not
    # cost a whole (multi-hour) stage a re-run.
    $tol = [Math]::Max(2, [int]($n * 0.01))
    return [pscustomobject]@{ ok = ($bad -le $tol); note = $note; n = $n; bad = $bad }
}

function Test-StageDone($state, [string] $key) {
    return ($state.completed -contains $key)
}

function Mark-Done($state, [string] $key) {
    if ($state.completed -notcontains $key) {
        $state.completed = @($state.completed) + $key
        Save-State $state
    }
}

# --------------------------------------------------------------------------
# stage definitions -- each returns the argument list for one invocation
# --------------------------------------------------------------------------
function Get-BatchFor([string] $model, [string] $stage) {
    $bs = $BatchSize
    if (-not $Script:BatchSizeExplicit) {
        if ($SweetSpotBatch.ContainsKey($model)) { $bs = $SweetSpotBatch[$model] }
        # The precision stage runs the SAME model at bf16, which needs more VRAM
        # than 4-bit. CheXagent-8B bf16 already uses 29.7 GB at bs=4; padding
        # mixed-size images pushes that past the 32 GB driver ceiling and WDDM
        # starts paging to host RAM -- the run that did this took 1.6 h for 80
        # images instead of 4 min. So the bf16 stage gets a per-model ceiling.
        if ($stage -eq 'ablation') {
            # Prompt ablation runs 10 cells per image across 5 variants; a large
            # batch on a bf16 8B model (CheXagent-8B bf16 = 29.7 GB at bs=4) walks
            # straight past the driver ceiling, so this stage stays modest.
            $cap = if ($PrecisionBatchCap.ContainsKey($model)) { $PrecisionBatchCap[$model] } else { 4 }
            if ($bs -gt $cap) { $bs = $cap }
        }
        if ($stage -eq 'precision') {
            $cap = if ($PrecisionBatchCap.ContainsKey($model)) { $PrecisionBatchCap[$model] } else { 4 }
            if ($bs -gt $cap) { $bs = $cap }
        }
    }
    return $bs
}

function Get-QuantArgs([string] $model, [string] $stage) {
    if ($stage -eq 'precision') { return @() }   # precision = bf16 by definition
    if ($FourBitModels -contains $model) { return @('--load-in-4bit') }
    return @()
}

function Get-StageCommand([string] $model, [string] $stage) {
    $outRoot = "outputs\matrix\$model"
    $bs = Get-BatchFor $model $stage
    $quant = Get-QuantArgs $model $stage
    switch ($stage) {
        'shift' {
            return @(
                'scripts\08_shift_benchmark.py',
                '--model', $model,
                '--manifest', $Manifest,
                '--split', $Splits,
                '--images-root', 'data\nih\images',
                '--output-dir', "$outRoot\shift",
                '--limit', "$Limit",
                '--batch-size', "$bs",
                '--max-new-tokens', "$MaxNewTokens"
            ) + $quant
        }
        'reliance' {
            return @(
                'scripts\06_image_reliance.py',
                '--model', $model,
                '--manifest', $Manifest,
                '--split', $Splits,
                '--images-root', 'data\nih\images',
                '--output-dir', "$outRoot\reliance",
                '--limit', "$Limit",
                '--batch-size', "$bs",
                '--max-new-tokens', "$MaxNewTokens"
            ) + $quant
        }
        'precision' {
            # bf16 counterpart of the 4-bit primary configuration
            return @(
                'scripts\06_image_reliance.py',
                '--model', $model,
                '--manifest', $Manifest,
                '--split', $Splits,
                '--images-root', 'data\nih\images',
                '--output-dir', "$outRoot\reliance_bf16",
                '--limit', "$Limit",
                '--batch-size', "$bs",
                '--max-new-tokens', "$MaxNewTokens"
            )
        }
        'ablation' {
            return @(
                'scripts\21_prompt_ablation.py',
                '--model', $model,
                '--variant', 'all',
                '--condition', 'original', 'blank',
                '--manifest', $Manifest,
                '--split', $Splits,
                '--images-root', 'data\nih\images',
                '--output-root', "$outRoot\ablation",
                '--limit', "$AblationLimit",
                '--max-new-tokens', '256',
                '--resume'
            ) + $quant
        }
        default { return $null }
    }
}

function Get-BatchSpeedup([string] $model) {
    # Measured throughput at the model's sweet spot / measured single-image rate
    # (docs/44). 1.0 means batching buys nothing at all.
    $m = @{
        medgemma_4b_it        = 1.0    # 4-bit sweet spot IS bs=64: 0.41 s/img already batched
        medgemma_15_4b_it     = 1.0
        qwen25_vl_7b_instruct = 1.0    # 1.28 s/img at bs=8; serial 2.43 s/img
        qwen3_vl_8b_instruct  = 1.0    # 2.18 s/img at bs=8
        chexagent_8b          = 1.0    # 7.14 s/img at bs=4
        internvl3_5_8b        = 1.0    # serial by design; batching is refused
        llava_med_v1_5_7b     = 1.0    # serial by design
    }
    if ($m.ContainsKey($model)) { return $m[$model] }
    return 1.0
}

function Get-StageSeconds([string] $model, [string] $stage) {
    # Base rate: measured single-image throughput (docs/44) x token-budget x
    # batching factor. See the calibration block above.
    $sp = $Throughput[$model]
    if (-not $sp) { $sp = 2.0 }
    $cal = 1.0
    if ($BatchingFactor.ContainsKey($model)) { $cal = $BatchingFactor[$model] }
    $sp = $sp * $TokenBudgetFactor * $cal

    $batchSpeedup = Get-BatchSpeedup $model
    $body = 0
    switch ($stage) {
        'shift'     { $body = $Limit * $ShiftCells * $sp / $batchSpeedup * $PaddingFactor }
        'reliance'  { $body = $Limit * $RelianceConditions * $sp / $batchSpeedup * $PaddingFactor }
        'precision' { $body = $Limit * $RelianceConditions * $sp * $PaddingFactor }
        'ablation'  { $body = $AblationLimit * 2 * 5 * $sp * $PaddingFactor }
        default     { $body = 0 }
    }
    # Every stage script loads the model itself: add that cost every time.
    return ($body + $StageStartupSeconds)
}

# --------------------------------------------------------------------------
# plan + budget
# --------------------------------------------------------------------------
$plan = @()
foreach ($m in $Models) {
    foreach ($s in $Stages) {
        if ($s -eq 'ablation' -and ($AblationModels -notcontains $m)) { continue }
        $plan += [pscustomobject]@{
            Model   = $m
            Stage   = $s
            Seconds = (Get-StageSeconds $m $s)
        }
    }
}

Write-Host ""
Write-Host ("=" * 78)
Write-Host " STAGE-2 EXPERIMENT RUN"
Write-Host ("=" * 78)
Write-Host ("  models      : {0}" -f ($Models -join ', '))
Write-Host ("  stages      : {0}" -f ($Stages -join ', '))
Write-Host ("  images/stage: {0}   (ablation {1})" -f $Limit, $AblationLimit)
Write-Host ("  batch size  : per model, from the measured sweet spot (docs/44);")
Write-Host ("                override with -BatchSize N if needed")
Write-Host ("  precision   : 4-bit for {0}" -f ($FourBitModels -join ', '))
Write-Host ("                bf16  for the rest (as benchmarked)")
Write-Host ("  state file  : {0}" -f $StateFile)
Write-Host ""
Write-Host "  PLANNED WORK (estimates; batching on mixed-size images is slower than"
Write-Host "  the single-image benchmark, so a 1.2x padding factor is applied):"
Write-Host ""
$totalSeconds = 0
foreach ($m in ($plan | Group-Object Model)) {
    $sum = ($m.Group | Measure-Object -Property Seconds -Sum).Sum
    $loadS = 0; if ($LoadSeconds.ContainsKey($m.Name)) { $loadS = $LoadSeconds[$m.Name] }
    $totalSeconds += $sum + $loadS
    $stageList = ($m.Group | ForEach-Object { $_.Stage }) -join ','
    $prec = if ($FourBitModels -contains $m.Name) { '4bit' } else { 'bf16' }
    Write-Host ("    {0,-24} {1,8}   [{2}]  {3} bs={4}" -f `
        $m.Name, (Format-Duration ($sum + $loadS)), $stageList, $prec, (Get-BatchFor $m.Name 'shift'))
}
Write-Host ""
Write-Host ("    {0,-24} {1,8}" -f 'TOTAL', (Format-Duration $totalSeconds))
Write-Host "  Ctrl-C is safe: re-run the same command to continue where it stopped."
Write-Host ""

# --------------------------------------------------------------------------
# data manifest -- what is actually being measured, cell by cell
#
# The plan above answers "how long"; this answers "how much data", which is the
# question that has to hold up in the paper's methods section.
# --------------------------------------------------------------------------
$cellsPerImage = @{ shift = $ShiftCells; reliance = $RelianceConditions
                    precision = $RelianceConditions; ablation = 10 }
$totalCalls = 0
Write-Host ("=" * 78)
Write-Host " DATA MANIFEST"
Write-Host ("=" * 78)
Write-Host ("  {0,-24} {1,-10} {2,10} {3,10} {4,10}" -f 'model', 'stage', 'images', 'cells/img', 'calls')
foreach ($m in ($plan | Group-Object Model)) {
    $subTotal = 0
    foreach ($p in ($m.Group | Sort-Object Stage)) {
        $lim = if ($p.Stage -eq 'ablation') { $AblationLimit } else { $Limit }
        $cells = $cellsPerImage[$p.Stage]
        $calls = $lim * $cells
        $subTotal += $calls
        $tag = if ($p.Stage -eq 'ablation') { '  (subset)' } else { '' }
        Write-Host ("  {0,-24} {1,-10} {2,10} {3,10} {4,10}{5}" -f `
            $m.Name, $p.Stage, $lim, $cells, $calls, $tag)
    }
    Write-Host ("  {0,-24} {1,-10} {2,10} {3,10} {4,10}" -f '', '', '', 'subtotal', $subTotal)
    $totalCalls += $subTotal
}
$ablationCalls = ($plan | Where-Object { $_.Stage -eq 'ablation' } |
                  ForEach-Object { $AblationLimit * 10 } | Measure-Object -Sum).Sum
if (-not $ablationCalls) { $ablationCalls = 0 }
Write-Host ""
Write-Host ("  model calls, all stages        : {0:N0}" -f $totalCalls)
Write-Host ("    primary (shift+reliance+prec): {0:N0}" -f ($totalCalls - $ablationCalls))
Write-Host ("    ablation (3 models only)     : {0:N0}" -f $ablationCalls)
Write-Host ""
Write-Host "  Every call returns one structured JSON verdict: 8 findings"
Write-Host "  (atelectasis, cardiomegaly, consolidation, edema, pleural_effusion,"
Write-Host "   pneumonia, pneumothorax, no_finding), each with a status and a"
Write-Host "   probability, plus a free-text rationale and an image-reliance claim."
Write-Host ""
Write-Host "  Cells per image, by stage:"
Write-Host ("    shift     : clean + 6 shifts x 3 severities = {0}" -f $ShiftCells)
Write-Host "                blur 0.5/1.0/2.0 | noise sigma 0.02/0.05/0.10"
Write-Host "                jpeg q 70/40/20 | brightness 0.8/1.2/1.5"
Write-Host "                contrast 0.7/1.3/1.6 | resolution 0.5/0.25/0.125"
Write-Host ("    reliance  : {0} (original, blank grey, blur r=8, shuffled study)" -f $RelianceConditions)
Write-Host ("    precision : {0}, identical to reliance but bf16 instead of 4-bit" -f $RelianceConditions)
Write-Host "    ablation  : 5 prompt variants x 2 conditions = 10"
Write-Host "                variants : v0_simple, v1_per_label_binary, v2_few_shot,"
Write-Host "                           v3_report_first, v4_normality_pressure"
Write-Host "                conditions: original, blank"
Write-Host ""
Write-Host "  Output layout: outputs\matrix\<model>\{shift,reliance,reliance_bf16,ablation}\"
Write-Host "    shift     -> shift_predictions.jsonl              (1 row per image x cell)"
Write-Host "    reliance  -> image_reliance_predictions.jsonl + per_image.csv + summary.json"
Write-Host "    precision -> same files as reliance, under reliance_bf16\"
Write-Host "    ablation  -> <variant>\<condition>\ per-cell jsonl + progress.json"
Write-Host ""
Write-Host "  Analysis (CPU only, run after the GPU stages):"
Write-Host "    scripts\18_selective_metrics_v2.py    3-level selective metrics + cluster bootstrap"
Write-Host "    scripts\19_accept_clarify_abstain.py  cost-sensitive accept / clarify / abstain"
Write-Host "    scripts\09_shift_analysis.py          degradation curves per shift x severity"
Write-Host ("=" * 78)
Write-Host ""

if ($DryRun) {
    Write-Host "  (-DryRun given: nothing was run.)"
    exit 0
}

$state = Read-State
$done = 0
foreach ($p in $plan) { if (Test-StageDone $state ($p.Model + '/' + $p.Stage)) { $done++ } }
if ($done -gt 0) { Write-Host ("  ({0} stage(s) already recorded complete and will be skipped)`n" -f $done) }

# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------
$runStart = Get-Date
$index = 0
$succeeded = 0
$failed = @()

foreach ($m in ($plan | Group-Object Model)) {
    $model = $m.Name
    $python = Get-Python $model
    $envName = if ($ModernEnvModels -contains $model) { '.venv-modern' } else { '.venv-vl' }

    Write-Host ""
    Write-Host ("#" * 78)
    Write-Host ("# MODEL: {0}    env: {1}" -f $model, $envName)
    Write-Host ("#" * 78)

    foreach ($p in $m.Group) {
        $index++
        $key = "$model/$($p.Stage)"
        if (Test-StageDone $state $key) {
            Write-Host ("  [skip] {0} (already complete)" -f $p.Stage)
            continue
        }
        $cmd = Get-StageCommand $model $p.Stage
        if (-not $cmd) { Write-Host ("  [skip] unknown stage {0}" -f $p.Stage); continue }

        $remaining = 0
        for ($k = $index - 1; $k -lt $plan.Count; $k++) {
            if (-not (Test-StageDone $state ($plan[$k].Model + '/' + $plan[$k].Stage))) {
                $remaining += $plan[$k].Seconds
            }
        }
        $elapsed = (Get-Date) - $runStart
        Write-Host ""
        Write-Host ("=" * 78)
        Write-Host ("[{0}/{1}] {2} :: {3}" -f $index, $plan.Count, $model, $p.Stage)
        Write-Host ("        est {0}   |   elapsed {1}   |   remaining ~{2}" -f `
            (Format-Duration $p.Seconds), (Format-Duration $elapsed.TotalSeconds), (Format-Duration $remaining))
        Write-Host ("        {0} {1}" -f (Split-Path $python -Leaf), ($cmd -join ' '))
        Write-Host ("=" * 78)

        $t0 = Get-Date
        & $python @cmd
        $code = $LASTEXITCODE
        $dt = ((Get-Date) - $t0).TotalSeconds

        if ($code -eq 0) {
            $chk = Test-StageOutput $model $p.Stage
            if (-not $chk.ok) {
                # exit 0 but the rows are garbage: do NOT record it as complete,
                # so the next run retries this stage instead of skipping it.
                $failed += $key
                Write-Host ("  [BAD ] {0} in {1} -- exit 0 but {2} of {3} sampled rows errored" -f `
                    $p.Stage, (Format-Duration $dt), $chk.bad, $chk.n) -ForegroundColor Red
                if ($chk.note) { Write-Host ("         {0}" -f $chk.note) -ForegroundColor Red }
                $poison = Get-PredictionFile $model $p.Stage
                if ($poison) {
                    # the scripts resume by skipping rows already in the file, so
                    # a poisoned file must go or the retry would skip the bad rows
                    Remove-Item $poison -Force
                    Write-Host ("         deleted {0}; a re-run redoes this stage from scratch" -f (Split-Path $poison -Leaf))
                }
                Write-Host "         stage left INCOMPLETE so a re-run retries it"
            }
            else {
                Mark-Done $state $key
                Record-Timing $state $model $p.Stage $Limit $dt
                Save-State $state
                $succeeded++
                Write-Host ("  [done] {0} in {1} (est {2})" -f $p.Stage, (Format-Duration $dt), (Format-Duration $p.Seconds))
            }
        }
        else {
            $failed += $key
            Write-Host ("  [FAIL] {0} exit={1} after {2} -- continuing; re-run to retry this stage" -f `
                $p.Stage, $code, (Format-Duration $dt))
            Write-Host ("         underlying scripts resume, so a retry does not redo finished work")
        }
    }
}

# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
$total = (Get-Date) - $runStart
Write-Host ""
Write-Host ("=" * 78)
Write-Host " RUN SUMMARY"
Write-Host ("=" * 78)
Write-Host ("  wall clock : {0}" -f (Format-Duration $total.TotalSeconds))
Write-Host ("  succeeded  : {0}" -f $succeeded)
if ($failed.Count -gt 0) {
    Write-Host ("  failed     : {0}" -f ($failed -join ', '))
    Write-Host "  -> re-run the same command to retry only those"
} else {
    Write-Host "  failed     : none"
}
Write-Host ""
Write-Host "  Analysis (CPU only, no GPU needed):"
Write-Host "    .\.venv-vl\Scripts\python.exe scripts\18_selective_metrics_v2.py ``"
Write-Host "        --predictions outputs\overnight\medgemma_4b_it_1000\predictions.jsonl ``"
Write-Host "        --manifest data\manifests\nih_pilot_1000.csv ``"
Write-Host "        --output-dir outputs\selective_v2\medgemma_4b_it_1000 --n-boot 1000"
Write-Host "    .\.venv-vl\Scripts\python.exe scripts\19_accept_clarify_abstain.py ``"
Write-Host "        --predictions outputs\overnight\medgemma_4b_it_1000\predictions.jsonl ``"
Write-Host "        --reliance outputs\matrix\medgemma_4b_it\reliance ``"
Write-Host "        --manifest data\manifests\nih_pilot_1000.csv ``"
Write-Host "        --output-dir outputs\trust_v1\medgemma_4b_it --n-boot 1000"
Write-Host ("=" * 78)
