<#
.SYNOPSIS
    One-button driver for the three-layer 6K evaluation, resumable at every level.

.DESCRIPTION
    Each dataset answers exactly one question, so each layer runs the design that
    answers it -- not the same 19-cell synthetic grid three times (which would cost
    621 h and answer the artificially-degraded question twice):

      NIH eval 1506   in-domain:  19-cell synthetic shift + 4-cell reliance + bf16 control
      PadChest 1500   cross-hospital: native shift + blank control            (2 cells)
      Open-i 947      report generation: prose report + blank control          (2 cells)

    Run order is slowest-model-first within each layer, because the slow models are
    the ones most likely to be interrupted and finishing them early means every
    later interruption happens on cheap work.

    Every stage is resumable twice over: a state file skips whole completed stages,
    and each runner's own durable byte-offset sidecar resumes inside a stage. A
    stage that exits non-zero, or that exits 0 with error rows in its output, is
    left INCOMPLETE so a re-run retries it instead of skipping it.

.PARAMETER Models
    Which models to run. Default is the five that fit the budget. The delivered
    seven-model study runs as two passes: the six contract-compliant models as one
    matrix (add internvl3_5_8b) and llava_med_v1_5_7b separately -- its prose
    stages are contract-exempt, kept as evidence and excluded from metrics.
    See docs/48 section 4.6 for the exact commands.

.PARAMETER Layers
    Subset of nih,padchest,openi. Default: all three.

.PARAMETER Limit
    Images per layer. The default uses each layer's full evaluation set.

.PARAMETER DryRun
    Print the plan, the data manifest and the time budget, then exit. Runs nothing.
#>
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Models,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Layers,
    [int]      $Limit        = 0,        # 0 = use each layer's full set
    [int]      $BatchSize    = 0,        # 0 = use the measured sweet spot
    [switch]   $NoAblation,
    [switch]   $Reset,
    [switch]   $DryRun,
    # Where predictions are written. Separate this for a pre-run smoke test:
    #   -OutputRoot outputs\pretest -StateFile outputs\state\pretest_state.json
    # Both must be separated together. The predictions root decides where files
    # land; the state file decides which stages get skipped. Isolating only one of
    # them still lets the two runs collide.
    [string]   $OutputRoot   = 'outputs\matrix',
    [string]   $StateFile,
    # Label coverage below which a shift cell is treated as a coverage failure.
    # The census runs at the end of every run (CPU only); see the COVERAGE block
    # in the run summary. Lower it only with a reason written down.
    [double]   $MinCoverage  = 0.95
)

$ErrorActionPreference = 'Continue'
$script:Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$script:BatchSizeExplicit = $PSBoundParameters.ContainsKey('BatchSize') -and $BatchSize -gt 0
# Resolve a relative root against the program directory so the same string works
# no matter where the caller is standing.
if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
    $script:OutputRoot = Join-Path $script:Root $OutputRoot
} else {
    $script:OutputRoot = $OutputRoot
}
Set-Location $script:Root

# Native-resolution prefill fragments the caching allocator: after an 11.40 MP
# PadChest cell torch held 28.1 GiB reserved with only 2.5 GiB free, and the next
# 16,384-token cell spilled past physical VRAM. Windows does NOT raise OOM for
# that -- WDDM pages VRAM into host RAM, the cell costs ~290 s instead of ~40 s
# at 100% utilisation and 122 W, and the run looks healthy while its rate rots.
# Measured with scripts/_diag_vram_cliff.py (2026-10-21); see docs/55 section 10.
# expandable_segments removes the fragmentation that starts the cascade; the
# wrappers additionally return the cache after each call.
if (-not $env:PYTORCH_CUDA_ALLOC_CONF) {
    $env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
}
Write-Host ("  PYTORCH_CUDA_ALLOC_CONF = {0}" -f $env:PYTORCH_CUDA_ALLOC_CONF)

$Models = @($Models | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$Layers = @($Layers | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })

# --------------------------------------------------------------------------
# measured throughput, in seconds per model CALL (one image + one prompt)
# --------------------------------------------------------------------------
# TWO tables, because one number per model was wrong in BOTH directions. The
# 7-model x 100-image pre-run (docs/50 §2.5) compared the estimate against what
# each stage actually took:
#
#   medgemma_4b ablation   est 5m   took 2.3h   27.6x   (stage was serial)
#   medgemma_15 precision  est 3m   took 19m     6.3x   (bf16, not 4-bit)
#   qwen3 reliance         est 30m  took 10m     0.33x  (over-estimated)
#
# The cause was structural: $SecondsPerCall was ONE constant per model,
# calibrated on 4-bit shift, and then applied unchanged to bf16 ``precision``
# and to the then-serial ``ablation``. A stage that runs 27x slower than
# advertised is how a "1.4 h" stage silently becomes a 39.8 h one.
#
#   $StageSecondsPerCall : "model/stage" -> the median of the per-cell
#          ``elapsed_sec`` the runners themselves record, measured on the
#          100-image pre-run (4-bit where the model is in $FourBitModels, bf16
#          for the precision control and for the non-4-bit models).
#   $SecondsPerCall      : per-model fallback for anything not measured yet.
#
# Lookup order: "model/stage" -> "model/<stage alias>" -> "model".
$StageSecondsPerCall = @{
    # ---- NIH shift: 19-cell synthetic degradation grid
    'medgemma_4b_it/shift'           = 0.613
    # MEAN 2.344 over 608 real shift cells at the 1024-token budget this model
    # needs, condition-grouped batches, bs=32 (median 1.128; 601/608 parseable).
    # The mean is what the plan needs, and here it is 2x the median because the
    # cells that open a reasoning channel are the slow ones by construction.
    # NOTE: the budget is now 2048 (see $MaxNewTokensOverride below), so the cells
    # that used to truncate at 1024 run longer -- this rate is a lower bound until
    # the retry re-measures it.
    'medgemma_15_4b_it/shift'        = 2.344
    'qwen25_vl_7b_instruct/shift'    = 2.944
    # Post-contract native-resolution rerun, 2026-10-20..21: 19,318 cells took
    # 15.8 h = 2.944 s/cell. The old 1.590 was measured when this model still
    # read a 0.59 MP version of every NIH image; the live progress bar during
    # the rerun read 2.93 s/img, so the plan had been understating this stage
    # (and the qwen25 subtotal) by roughly half.
    'qwen3_vl_8b_instruct/shift'     = 1.455
    'internvl3_5_8b/shift'           = 5.722
    # chexagent crashed while offloaded (84.5 s/cell and rising). Post-fix, the
    # fp16 checkpoint at batch 4 measured a MEAN of 2.92 s/cell over 76 real shift
    # cells -- but that run was BEAM SEARCH (the shipped generation_config says
    # num_beams=5), which is also what caused the 13% parse failures the guard
    # rejected. chexagent.py now forces greedy (docs/50 section 2.11); the greedy
    # diagnostic measures 0.477 s/cell over 96 cells at bs=4, 96/96 parseable
    # (outputs/state/chexagent_parse_budget_greedy.json). 0.60 keeps a ~25% margin
    # over that subset for the full 19-cell shift grid; the retry's own elapsed_sec
    # replaces this once it lands.
    #
    # Means, not medians: the plan multiplies this by a call count, so the number
    # it needs is total wall clock / cells. Every rate in this table is a mean for
    # that reason -- they coincide while the distribution is tight (p10-p90 within
    # a few percent for every 4-bit stage) and diverge exactly where a heavy tail
    # exists, which is the case for the thinking stages below.
    'chexagent_8b/shift'             = 0.600
    # ---- NIH reliance: 4-condition image-dependence stage
    'medgemma_4b_it/reliance'        = 0.613
    # MEAN 2.479 over 128 cells, 128/128 parseable (the blank condition opens the
    # reasoning channel on every cell: 100/100 in the pre-run)
    'medgemma_15_4b_it/reliance'     = 2.479
    'qwen25_vl_7b_instruct/reliance' = 2.911
    # Native-resolution rerun: 6,024 cells in 4.9 h (progress bar 2.91 s/img).
    'qwen3_vl_8b_instruct/reliance'  = 1.477
    'internvl3_5_8b/reliance'        = 5.724
    # ---- NIH precision: the bf16 control, same 4 conditions, batch capped at 4
    'medgemma_4b_it/precision'       = 1.869
    # 13.3 s/cell at the 2048-token budget. This is the one expensive consequence
    # of the thinking channel: at bf16 EVERY reply opens it (100% on the pre-run),
    # so the budget must clear the trace and the stage costs ~20x more than at 256.
    # It also means this model's 4-bit and bf16 rows differ in OUTPUT MODE, not
    # only in precision -- a caveat the paper has to state (docs/50 section 2.8).
    # MEAN 13.305 over 64 cells at bs=4, 64/64 parseable, at_cap=0
    # (outputs/state/medgemma15_precision_parse_budget_2048.json).
    'medgemma_15_4b_it/precision'    = 13.305
    'qwen25_vl_7b_instruct/precision' = 3.356
    # bf16 control at native resolution: 3.36 s/img on the live bar (6024 cells).
    # ---- NIH ablation: 5 prompt variants x original/blank, batched 2026-10-05.
    #      The serial stage measured 5.67-9.93 s/cell depending on how much text
    #      the variant asks for (8.28 s/cell on the pre-run as a whole); at batch
    #      32 every variant lands at 0.77-0.97 and the mean is 0.829
    #      (scripts/60_measure_ablation_batch.py, outputs/state/ablation_batch_measure.json).
    'medgemma_4b_it/ablation'        = 0.829
}
# Per-LAYER measurements, empty on purpose: a rate measured on NIH is not a
# measurement of PadChest or Open-i, and the plan says so instead of pretending.
# The Open-i report stage is the one expected to land here first (prose report,
# 192 new tokens, a different prompt) -- measure it before trusting its estimate.
$LayerStageSecondsPerCall = @{
    # Rates measured ON the layer, keyed model/layer/stage. Needed because a rate
    # measured on NIH is not a rate for a native-resolution layer: PadChest images
    # are 6.7x the pixels of the uniform 1024x1024 NIH set and Open-i 4.5x, and
    # since 2026-10-18 nothing is downscaled any more, so per-call cost is not
    # comparable. Measured in isolated smoke mode (4 images, $NonNihBatch).
    #
    # padchest, batch 1, native 19.27 MP ceiling -> 8 cells in 122 s. Before the
    # preprocessing fix this stage "looked" like 1.70 s/cell only because qwen25
    # was reading a 0.59 MP version of the image and answering in a few tokens.
    'qwen25_vl_7b_instruct/padchest/shift' = 20.5
    # 24-image isolated sample on 2026-10-21 (48 cells, 48/48 parsed): 16.3 s for
    # images <= 11 MP, 52.9 s above it, reweighted to the layer's own pixel mix
    # (171/1500 = 11.4% above 11 MP) -> 20.5 s/cell -> 17.1 h for 3,000 cells.
    # The old 15.24 came from a 4-image smoke test that happened to contain no
    # large image. The remaining >11 MP cost is a WDDM paging tail, not a bug:
    # see docs/55 section 10. Do NOT "fix" it by lowering max_pixels.
    # padchest, batch 1 -> 8 cells in 96 s. This is the model that pinned the
    # 32 GB card at batch 8 before the fix (0 rows written, VRAM full).
    'qwen3_vl_8b_instruct/padchest/shift' = 13.9
    # 12-image isolated sample on 2026-10-21 (24 cells, 24/24 parsed, zero paging
    # samples): 12.6 s <= 11 MP, 24.3 s above, reweighted to 171/1500 -> 13.9
    # s/cell -> 11.6 h. The old 12.12 came from a 4-image smoke test.
    # openi, batch 1 -> 8 cells. Faster than padchest per cell because the report
    # is capped at 192 tokens and the images are smaller (4.75 MP vs 7.00 MP).
    # qwen25's number is derived from the driver's wall clock (1m over 8 cells,
    # so +/-0.5 s); qwen3's is the runner's own mean.
    'qwen25_vl_7b_instruct/openi/shift' = 7.5
    'qwen3_vl_8b_instruct/openi/shift'  = 3.40
}
# A stage whose rate has no measurement of its own borrows one. Open-i's 2-cell
# prose report still has no measurement and therefore borrows the NIH shift rate
# -- the plan prints it as "proxy-nih" rather than quietly calling it measured,
# because an Open-i report is prose and cannot be cheaper than an NIH cell.
$StageAlias = @{ native_shift = 'shift'; report = 'shift' }
$SecondsPerCall = @{
    medgemma_4b_it        = 0.61
    medgemma_15_4b_it     = 0.85
    qwen25_vl_7b_instruct = 1.59
    qwen3_vl_8b_instruct  = 1.46
    # greedy, see the shift entry above: 0.60 s/cell, not the 2.92 the beam-search
    # run cost.
    chexagent_8b          = 0.60
    internvl3_5_8b        = 5.72
    # llava_med answers in prose, never in JSON (0/16 parseable after the
    # rope_theta fix -- see docs/50 section 2.7), so its rate is the honest cost
    # of a stage the [BAD] guard will refuse to mark done.
    llava_med_v1_5_7b     = 1.08
}
$LoadSeconds = @{
    chexagent_8b      = 75
    internvl3_5_8b    = 30
    llava_med_v1_5_7b = 220
    qwen3_vl_8b_instruct = 45
}
$SweetSpotBatch = @{
    medgemma_4b_it        = 64
    medgemma_15_4b_it     = 64
    qwen25_vl_7b_instruct = 8
    qwen3_vl_8b_instruct  = 8
    chexagent_8b          = 4
    internvl3_5_8b        = 1
    llava_med_v1_5_7b     = 1
}
# MEASURED batch sizes for the ablation stage only; see Get-AblationBatch.
$AblationBatch = @{
    medgemma_4b_it = 32
}
# Batch size for the two NATIVE-resolution layers (PadChest, Open-i), where the
# model's image-token count scales with the source image. Only the
# Qwen-processor models qualify: every other wrapper resizes to a fixed size
# (CheXagent, LLaVA-Med, MedGemma's Gemma3 processor) or to a capped tile grid
# (InternVL: 448x448 tiles, max 12 patches), so its token count is constant and
# its NIH sweet spot stays valid there.
#
# Without this override the Qwen family ran the native layers at its NIH sweet
# spot of 8, which is sized for 1.05 MP images and is ~6-15x too large here.
# Starting values below are deliberate but are confirmed by
# scripts/76_layer_smoke.py before the full run; the numbers it prints (peak
# VRAM, s/img, parse rate) are what justify them.
$NonNihBatch = @{
    qwen25_vl_7b_instruct = 1
    qwen3_vl_8b_instruct  = 1
}
# Per-model generation budget, where a model needs more than the shared default.
#
# MedGemma-1.5-4B-IT opens a thinking channel (`<unused94>` id 100 ... `<unused95>`
# id 101) before it answers, and 256 tokens run out inside it. Measured at bf16
# over 8 images (scripts/61_measure_medgemma_think_budget.py, output in
# outputs/state/medgemma_think_budget.json):
#
#   budget   parse ok   s/cell
#      256      0/8       2.94
#      512      0/8       5.54
#      768      6/8       8.22
#     1024      8/8      12.50
#
# Both ways of dealing with that are measured; what the study runs, and why, is
# recorded in docs/50 section 2.8. An entry here is a token budget, nothing else:
# no model gets a different prompt.
$MaxNewTokensOverride = @{
    # 1024 clears the 8-image probe above but NOT the precision control: at 1024 the
    # bf16 precision stage lost 3/64 replies to truncation (at_cap=17) and the guard
    # rejected the whole stage. 1536 and 2048 both reach 64/64 with at_cap=0 and the
    # SAME max reply length, 1495 tokens
    # (outputs/state/medgemma15_precision_parse_budget{,_2048}.json) -- trace length
    # is a property of the model, not of the cap. 2048 is the one with margin
    # (1495/2048 = 73%), so the study uses it. See docs/50 section 2.11.
    medgemma_15_4b_it = 2048
}
$MaxNewTokensDefault = 256
# Throughput at the sweet spot divided by throughput one image at a time.
#
# This is 1.0 for every model ON PURPOSE: the rates in $SecondsPerCall were
# measured AT each model's sweet spot, so the batching benefit is already inside
# them. The first version of this estimate divided by 1.5 anyway and reported
# 125.4 h for a configuration that really costs ~188 h -- the ETA was 1.5x
# optimistic, which is exactly the failure mode that made an earlier run say
# "est 7m" and take 1.6 h. Kept as an explicit table so a future serial-only
# model, whose rate was NOT measured batched, can set it below 1.0.
$BatchSpeedup = @{
    medgemma_4b_it        = 1.0
    medgemma_15_4b_it     = 1.0
    qwen25_vl_7b_instruct = 1.0
    qwen3_vl_8b_instruct  = 1.0
    chexagent_8b          = 1.0
    internvl3_5_8b        = 1.0
    llava_med_v1_5_7b     = 1.0
}
# Padding allowance on top of the measured per-call rate.
#
# What is already inside the rate: the runs that produced
# $StageSecondsPerCall/$SecondsPerCall were BATCHED, so batching is accounted for
# exactly once (that is what $BatchSpeedup = 1.0 means below).
#
# What is not: image decode, the per-batch JSONL append + fsync, and the fact
# that a batch runs until its LONGEST member finishes, so a mixed pile of
# radiographs costs more than the median cell suggests. 1.20 is the allowance for
# those three. The pre-run's own stage clock came in 1.6-2.0x above the median
# cell rate, so if anything this is still generous to the schedule.
$PaddingFactor = 1.20
# Only these models have a MEASURED 4-bit number; the rest stay bf16 as benchmarked.
$FourBitModels = @('medgemma_4b_it', 'medgemma_15_4b_it', 'qwen25_vl_7b_instruct')
# The five-model matrix: LLaVA-Med and InternVL are the two slowest and are opt-in.
$DefaultModels = @(
    'medgemma_4b_it', 'medgemma_15_4b_it', 'qwen25_vl_7b_instruct',
    'qwen3_vl_8b_instruct', 'chexagent_8b'
)
$AblationModels = @('medgemma_4b_it')
# Models whose structured reply is a KNOWN contract failure rather than an
# unexpected one. llava_med_v1_5_7b answers in prose and never in the JSON
# contract -- 0 of 1900 shift cells parse even after the rope_theta fix
# (docs/50 S2.7, docs/51) -- and that prose IS the evidence for the
# prose-without-the-contract failure mode the paper reports. The [BAD] guard
# therefore keeps its file and marks the stage done, but tags the stage
# contract-exempt so no downstream metric silently averages prose into a number.
# Every other model stays under the strict guard; this is a named exemption for a
# documented, intrinsic behaviour, not a loosening of the guard.
$KnownContractFailures = @{
    'llava_med_v1_5_7b' = 'prose, no JSON contract (docs/50 S2.7, docs/51)'
}
# A second, independent reason a stage is not structured: the LAYER is free text
# by design. The open-i layer asks every model to WRITE a report, so a row whose
# `parse_success` is false is the intended deliverable, not a broken cell -- the
# report-layer metric reads the raw text through a labeller (42_geographic_shift),
# not the structured parse. This differs from the model exemption above in one
# respect: an EMPTY report is still a useless row and still gates the stage, so the
# open-i branch relaxes only the parse check, not the emptiness check.
$ProseByDesignLayers = @{
    'openi' = 'report layer is free text by design (no JSON contract)'
}
$StageStartupSeconds = 35

if (-not $Models) { $Models = $DefaultModels }
if (-not $Layers) { $Layers = @('nih', 'padchest', 'openi') }
if ($NoAblation) { $AblationModels = @() }

$KnownLayers = @('nih', 'padchest', 'openi')
$bad = $Layers | Where-Object { $KnownLayers -notcontains $_ }
if ($bad) {
    Write-Host ("ERROR: unknown layer(s): {0}" -f ($bad -join ', ')) -ForegroundColor Red
    Write-Host ("       known layers: {0}" -f ($KnownLayers -join ', ')) -ForegroundColor Red
    exit 2
}
$badM = $Models | Where-Object { -not $SecondsPerCall.ContainsKey($_) }
if ($badM) {
    Write-Host ("ERROR: unknown model(s): {0}" -f ($badM -join ', ')) -ForegroundColor Red
    exit 2
}

# --------------------------------------------------------------------------
# layer definitions: cells per image, evaluation-set size, input paths
# --------------------------------------------------------------------------
# The size that matters is the number of images a layer EVALUATES, which is not
# the manifest's row count. `nih_eval_1732.csv` carries 226 `internal_train`
# "donor" rows that exist only for the no_finding repair (docs/48 6.1.2); every
# runner drops them with --split, so the NIH evaluation set is 1,054 internal_test
# + 452 source_val = 1,506. Counting the manifest here, instead of trusting the
# filename, is what keeps the coverage guard's denominator equal to the number of
# cells the runner really writes: the first full run used the raw 1,732, marked a
# complete and correct NIH stage [BAD] at 87% coverage, and deleted it.
function Get-EvalImageCount([string] $manifest, [string] $splits) {
    $path = Join-Path $script:Root $manifest
    if (-not (Test-Path $path)) { return 0 }
    $want = @{}
    foreach ($s in ($splits -split ',')) { $want[$s.Trim()] = $true }
    $n = 0
    foreach ($row in Import-Csv -LiteralPath $path) {
        if ($want.ContainsKey($row.split)) { $n++ }
    }
    return $n
}

function Get-LayerDef() {
    $nihManifest = 'data\manifests\nih_eval_1732.csv'
    $pcManifest  = 'data\manifests\padchest_shift.csv'
    $oiManifest  = 'data\manifests\openi_rrg.csv'
    return @{
        nih = @{
            Images        = Get-EvalImageCount $nihManifest 'internal_test,source_val'
            Manifest      = $nihManifest
            Splits        = 'internal_test,source_val'
            ImagesRoot    = 'data\nih\images'
            Dir           = 'nih_eval'
            Question      = 'in-domain: how much synthetic degradation before the signal breaks'
            Stages        = @{ shift = 19; reliance = 4; precision = 4; ablation = 10 }
        }
        padchest = @{
            Images        = Get-EvalImageCount $pcManifest 'internal_test'
            Manifest      = $pcManifest
            Splits        = 'internal_test'
            ImagesRoot    = 'data\padchest'
            Dir           = 'padchest'
            Question      = 'cross-hospital: does the framework transfer to a different country'
            Stages        = @{ shift = 2 }
        }
        openi = @{
            Images        = Get-EvalImageCount $oiManifest 'external_report'
            Manifest      = $oiManifest
            Splits        = 'external_report'
            ImagesRoot    = 'data\open-i'
            Dir           = 'openi'
            Question      = 'free text: omission and hallucination under report generation'
            Stages        = @{ shift = 2 }
        }
    }
}
$LayerDef = Get-LayerDef

function Get-BatchFor([string] $model, [string] $layer) {
    if ($script:BatchSizeExplicit) { return $BatchSize }
    # Non-NIH layers carry NATIVE images: PadChest averages 7.00 MP (max 15.37)
    # and Open-i 4.75 MP (max 5.92), against NIH's uniform 1.05 MP. Nothing is
    # downscaled any more (base.IMAGE_MAX_PIXELS = 19.27 MP), so a processor
    # whose token count scales with the source image sees ~6-15x more image
    # tokens per picture, and batch 8 pinned the 32 GB card before the first row
    # was written (2026-10-18: padchest/qwen3 stalled at 0/3000, VRAM full).
    # Batch size -- not resolution -- is the VRAM lever on these layers.
    if ($layer -ne 'nih' -and $script:NonNihBatch.ContainsKey($model)) {
        return $script:NonNihBatch[$model]
    }
    $bs = $SweetSpotBatch[$model]
    # PadChest/Open-i are two-cell runs; the bf16 VRAM ceiling is the binding
    # constraint there, so the slowest models stay small.
    if ($layer -ne 'nih' -and $bs -gt 8) { $bs = 8 }
    return $bs
}

# The ablation stage cannot use the shift sweet spot: it asks for far more text
# (v3 wants a narrative, v4 a per-finding justification), and a batch costs what
# its longest member costs. Measured on medgemma_4b_it, 32 images per config:
#
#   variant                  bs=1    bs=8   bs=16   bs=32
#   v0_simple               8.857   1.349   0.856   0.769
#   v1_per_label_binary     5.807   1.358   0.942   0.792
#   v2_few_shot             5.674   1.388   0.985   0.810
#   v3_report_first         9.926   1.651   1.192   0.968
#   v4_normality_pressure   5.673   1.363   0.962   0.807
#
# 32 is the fastest configuration for every variant and still has headroom
# (10.1 GiB peak against 25.8 GiB free). Models that have never been measured
# here get the shift sweet spot capped at 8 -- deliberately conservative, because
# the failure mode of guessing high is an OOM in the middle of a multi-day run.
function Get-AblationBatch([string] $model, [string] $layer) {
    if ($script:BatchSizeExplicit) { return $BatchSize }
    if ($script:AblationBatch.ContainsKey($model)) { return $script:AblationBatch[$model] }
    $bs = Get-BatchFor $model $layer
    if ($bs -gt 8) { $bs = 8 }
    return $bs
}

# Token budget for this model, defaulting to the shared one. Kept as a function so
# the plan can print what each stage will actually use.
function Get-MaxNewTokens([string] $model, [int] $default) {
    if ($script:MaxNewTokensOverride.ContainsKey($model)) {
        return [int]$script:MaxNewTokensOverride[$model]
    }
    return $default
}

# Seconds per model call for this exact (model, layer, stage), plus where the
# number came from. See the table comment for why one number per model is not
# enough -- and why a rate measured on NIH must never be printed as if it were
# measured on PadChest or Open-i.
function Get-RateFor([string] $model, [string] $layer, [string] $stage) {
    $layerKey = "$model/$layer/$stage"
    if ($script:LayerStageSecondsPerCall.ContainsKey($layerKey)) {
        return @{ Seconds = [double]$script:LayerStageSecondsPerCall[$layerKey]
                  Source  = 'measured' }
    }
    $key = "$model/$stage"
    if ($script:StageSecondsPerCall.ContainsKey($key)) {
        $src = if ($layer -eq 'nih') { 'measured' } else { 'proxy-nih' }
        return @{ Seconds = [double]$script:StageSecondsPerCall[$key]; Source = $src }
    }
    $alias = $script:StageAlias[$stage]
    if ($alias -and $script:StageSecondsPerCall.ContainsKey("$model/$alias")) {
        return @{ Seconds = [double]$script:StageSecondsPerCall["$model/$alias"]
                  Source  = 'proxy-stage' }
    }
    return @{ Seconds = [double]$script:SecondsPerCall[$model]; Source = 'default' }
}

function Get-QuantArgs([string] $model) {
    if ($FourBitModels -contains $model) { return @('--load-in-4bit') }
    return @()
}

function Get-LayerPython([string] $model) {
    if ($model -eq 'qwen3_vl_8b_instruct') {
        return (Join-Path $script:Root '.venv-modern\Scripts\python.exe')
    }
    return (Join-Path $script:Root '.venv-vl\Scripts\python.exe')
}

function Get-LayerCommand([string] $model, [string] $layer, [string] $stage, [int] $limit) {
    $def = $LayerDef[$layer]
    $outRoot = Join-Path $script:OutputRoot "$($def.Dir)\$model"
    $bs = Get-BatchFor $model $layer
    $quant = Get-QuantArgs $model

    if ($layer -eq 'nih') {
        switch ($stage) {
            'shift' {
                return @('scripts\08_shift_benchmark.py', '--model', $model,
                    '--manifest', $def.Manifest, '--split', $def.Splits,
                    '--images-root', $def.ImagesRoot, '--output-dir', "$outRoot\shift",
                    '--limit', "$limit", '--batch-size', "$bs",
                    '--max-new-tokens', "$(Get-MaxNewTokens $model 256)") + $quant
            }
            'reliance' {
                return @('scripts\06_image_reliance.py', '--model', $model,
                    '--manifest', $def.Manifest, '--split', $def.Splits,
                    '--images-root', $def.ImagesRoot, '--output-dir', "$outRoot\reliance",
                    '--limit', "$limit", '--batch-size', "$bs",
                    '--max-new-tokens', "$(Get-MaxNewTokens $model 256)") + $quant
            }
            'precision' {
                # bf16 counterpart; 4-bit models only, since a bf16 8B model at
                # batch 4 already sits at 29.7 GB on a 32 GB card.
                $pc = if ($bs -gt 4) { 4 } else { $bs }
                return @('scripts\06_image_reliance.py', '--model', $model,
                    '--manifest', $def.Manifest, '--split', $def.Splits,
                    '--images-root', $def.ImagesRoot, '--output-dir', "$outRoot\reliance_bf16",
                    '--limit', "$limit", '--batch-size', "$pc",
                    '--max-new-tokens', "$(Get-MaxNewTokens $model 256)")
            }
            'ablation' {
                # --batch-size was missing here until 2026-10-05, which is why the
                # stage ran at 8.28 s/cell while every other stage was batched;
                # over the full run that gap is ~38 hours (docs/50 §2.3).
                $ab = Get-AblationBatch $model $layer
                return @('scripts\21_prompt_ablation.py', '--model', $model,
                    '--variant', 'all', '--condition', 'original', 'blank',
                    '--manifest', $def.Manifest, '--split', $def.Splits,
                    '--images-root', $def.ImagesRoot, '--output-root', "$outRoot\ablation",
                    '--flat-output',
                    '--limit', "$limit", '--batch-size', "$ab",
                    '--max-new-tokens', "$(Get-MaxNewTokens $model 256)", '--resume') + $quant
            }
        }
        return $null
    }

    if ($layer -eq 'padchest') {
        # Write into a `shift` subdir, exactly like NIH. Get-PredictionFile resolves
        # the `shift` stage to `$base\shift`; before this the runner wrote straight
        # into `$base`, so the guard found nothing and marked a good stage [BAD].
        return @('scripts\43_run_padchest_shift.py', '--model', $model,
            '--manifest', $def.Manifest, '--split', $def.Splits,
            '--images-root', $def.ImagesRoot, '--output-dir', "$outRoot\shift",
            '--conditions', 'original', 'blank', '--limit', "$limit",
            '--batch-size', "$bs", '--max-new-tokens', "$(Get-MaxNewTokens $model 256)") + $quant
    }

    if ($layer -eq 'openi') {
        # Same `shift` subdir as NIH/PadChest -- see the padchest branch above.
        return @('scripts\44_run_openi_report.py', '--model', $model,
            '--manifest', $def.Manifest, '--split', $def.Splits,
            '--images-root', $def.ImagesRoot, '--output-dir', "$outRoot\shift",
            '--conditions', 'original', 'blank', '--limit', "$limit",
            '--batch-size', "$bs", '--max-new-tokens', "$(Get-MaxNewTokens $model 192)") + $quant
    }
    return $null
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
    return [pscustomobject]@{ completed = @(); contract_exempt = @(); started = (Get-Date).ToString('s'); config_fingerprint = $null }
}

function Get-ConfigFingerprint {
    # Hash the inputs that change what a stage means: the three manifests and
    # the prompt/schema modules. The stage key itself stays layer/model/stage;
    # this fingerprint is a run-level guard, not a replacement for it.
    $root = Split-Path -Parent $PSScriptRoot
    $files = @(
        (Join-Path $root 'data\manifests\nih_eval_1732.csv'),
        (Join-Path $root 'data\manifests\padchest_shift.csv'),
        (Join-Path $root 'data\manifests\openi_rrg.csv'),
        (Join-Path $root 'src\cxr_trust\inference\prompts.py'),
        (Join-Path $root 'src\cxr_trust\inference\prompts_ablation.py'),
        (Join-Path $root 'src\cxr_trust\inference\prompts_report.py'),
        (Join-Path $root 'src\cxr_trust\inference\schema.py')
    )
    $parts = foreach ($f in $files) {
        if (Test-Path $f) {
            $h = (Get-FileHash -Algorithm SHA256 -Path $f).Hash.ToLower()
            "{0}={1}" -f (Split-Path -Leaf $f), $h
        }
    }
    $text = ($parts -join "`n")
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($text)
    return (($sha.ComputeHash($bytes) | ForEach-Object { $_.ToString('x2') }) -join '')
}

function Save-State($state) {
    $dir = Split-Path -Parent $StateFile
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $state | ConvertTo-Json -Depth 5 | Set-Content -Path $StateFile -Encoding UTF8
}

function Test-StageDone($state, [string] $key) { return ($state.completed -contains $key) }

function Mark-Done($state, [string] $key) {
    if ($state.completed -notcontains $key) {
        $state.completed = @($state.completed) + $key
        Save-State $state
    }
}

function Mark-ContractExempt($state, [string] $key) {
    # Recorded separately from `completed` so a downstream metric can exclude a
    # stage whose rows are prose by design, instead of averaging them in.
    if (-not $state.PSObject.Properties['contract_exempt']) {
        $state | Add-Member -NotePropertyName contract_exempt -NotePropertyValue @() -Force
    }
    if ($state.contract_exempt -notcontains $key) {
        $state.contract_exempt = @($state.contract_exempt) + $key
        Save-State $state
    }
}

function Get-PredictionFile([string] $model, [string] $layer, [string] $stage) {
    # Stage-aware on purpose. The previous version searched the model directory and
    # returned the FIRST *_predictions.jsonl it found, so for reliance / precision /
    # ablation it could return the shift stage's file -- which is written first and
    # usually clean. The [BAD] guard would then sample clean shift rows while the
    # stage it was checking had produced nothing but errors, and mark a poisoned
    # stage complete. This is the second line of defence against silent failure, so
    # a hole in it is worse than not having it.
    $def = $LayerDef[$layer]
    $base = Join-Path $script:OutputRoot ("{0}\{1}" -f $def.Dir, $model)

    # exact location first
    $byStage = @{
        'shift'     = @("$base\shift")
        'reliance'  = @("$base\reliance")
        'precision' = @("$base\reliance_bf16")
        'ablation'  = @("$base\ablation")
    }
    $dirs = $byStage[$stage]
    if (-not $dirs) { $dirs = @($base) }
    foreach ($d in $dirs) {
        if (-not (Test-Path $d)) { continue }
        $hit = Get-ChildItem -Path $d -Recurse -File -Filter '*_predictions.jsonl' `
                             -ErrorAction SilentlyContinue |
               Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

function Test-StageOutput {
    param([string] $model, [string] $layer, [string] $stage, [int] $sample = 200,
          [int] $Expected = 0, [switch] $ContractExempt, [switch] $ProseStage)
    # A stage "succeeds" only if its rows are USABLE, not merely free of exceptions.
    #
    # The first version of this guard tested a single pattern,
    # `"error"\s*:\s*"([^"]+)"`, which requires non-empty text between the quotes.
    # A row with `"error": ""` therefore passed -- and so did every other way a
    # reply can be worthless. The pre-run proved how much that misses: three
    # model x stage combinations were marked [done] while producing no usable data
    # at all --
    #   llava_med_v1_5_7b shift     0 of 1900 rows parse (raw_output is "")
    #   llava_med_v1_5_7b reliance  0 of  400
    #   medgemma_15_4b_it precision 0 of  400 (raw is a `<unused94>thought` trace)
    # Worse, a broken model runs FASTER than a working one, so those stages printed
    # `[done] ... 8m (est 1.1h)` and read as over-performance.
    #
    # A row now counts as usable only when all three hold: no error, parse_success
    # true, and raw_output non-empty. Any of them failing marks the stage [BAD],
    # leaves it out of the state file, and deletes the poisoned file.
    #
    # `-Expected` adds the fourth failure mode, found while testing this against the
    # pre-run artifacts: a stage that wrote only 8 of 1900 rows and exited 0 would
    # pass on usable-row RATIO alone, because 8/8 were usable. Coverage is checked
    # separately from quality.
    $p = Get-PredictionFile $model $layer $stage
    if (-not $p) { return [pscustomobject]@{ ok = $false; note = 'no output file'; n = 0; bad = 0; file = ''
                                             err = 0; unparsed = 0; empty = 0; rows = 0; cover = $null
                                             exempt = $ContractExempt; files = @() } }
    $n = 0; $usable = 0; $nErr = 0; $nUnparsed = 0; $nEmpty = 0
    $errs = @{}
    foreach ($line in (Get-Content $p -TotalCount $sample)) {
        if (-not $line.Trim()) { continue }
        $n++
        $isErr = $line -match '"error"\s*:\s*"([^"]+)"'
        if ($isErr) {
            $nErr++
            $msg = $Matches[1]
            if ($errs.ContainsKey($msg)) { $errs[$msg]++ } else { $errs[$msg] = 1 }
        }
        $isUnparsed = $line -match '"parse_success"\s*:\s*false'
        if ($isUnparsed) { $nUnparsed++ }
        $isEmpty = $line -match '"raw_output"\s*:\s*""'
        if ($isEmpty) { $nEmpty++ }
        # A contract-exempt model is EXPECTED to emit prose, or nothing at all --
        # both are the phenomenon under study, not a broken stage, so only an
        # exception disqualifies its rows. A prose-by-design LAYER (open-i report)
        # is expected to emit prose but NOT to emit nothing, so it relaxes the parse
        # check alone and still fails on an empty reply. Non-exempt structured stages
        # keep the full strict rule. The unparsed/empty counts are reported either way.
        $rowBad = $isErr `
            -or (-not $ContractExempt -and $isEmpty) `
            -or (-not ($ContractExempt -or $ProseStage) -and $isUnparsed)
        if (-not $rowBad) { $usable++ }
    }
    if ($n -eq 0) {
        return [pscustomobject]@{ ok = $false; note = 'output file empty'; n = 0; bad = 0
                                 file = ''; err = 0; unparsed = 0; empty = 0; rows = 0; cover = $null
                                 exempt = $ContractExempt; files = @($p) }
    }
    # total row count vs the stage's planned cell count
    $rows = 0
    if ($Expected -gt 0) {
        $rows = (Get-Content $p -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    }
    $cover = $null
    if ($Expected -gt 0) { $cover = [Math]::Round(100.0 * $rows / $Expected, 1) }

    $bad = $n - $usable
    $parts = @()
    if ($nErr) { $parts += "$nErr exception(s)" }
    if ($nUnparsed) {
        $parts += $(if ($ContractExempt -or $ProseStage) { "$nUnparsed unparsed (prose by design: not gated)" }
                     else { "$nUnparsed unparsed" })
    }
    if ($nEmpty) { $parts += "$nEmpty empty reply" }
    if ($Expected -gt 0 -and $rows -lt [int]($Expected * 0.95)) {
        $parts += "only $rows of $Expected rows written ($cover% coverage)"
    }
    $note = $parts -join ', '
    if ($errs.Count -gt 0) {
        $top = $errs.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 1
        $note = ("{0}x {1}" -f $top.Value, $top.Key) + $(if ($parts.Count -gt 1) {
            '; ' + (($parts | Where-Object { $_ -notmatch '^\d+x ' }) -join ', ') } else { '' })
    }
    # Tolerate isolated flakes, reject systematic breakage. 1% is the same
    # tolerance the exception-only guard used.
    $tol = [Math]::Max(2, [int]($n * 0.01))
    $ok = ($bad -le $tol)
    if ($Expected -gt 0 -and $rows -lt [int]($Expected * 0.95)) { $ok = $false }
    return [pscustomobject]@{ ok = $ok; note = $note; n = $n; bad = $bad
                              file = $p; err = $nErr; unparsed = $nUnparsed
                              empty = $nEmpty; rows = $rows; cover = $cover; exempt = $ContractExempt
                              files = @($p) }
}

function Test-AblationOutput {
    param([string] $model, [string] $layer, [int] $Expected = 0, [int] $sample = 200)
    # The ablation stage writes ONE FILE PER CELL (5 prompt variants x 2 image
    # conditions = 10 files), each holding `limit` rows, under
    # `$base\ablation\<variant>\<condition>\image_reliance_predictions.jsonl`.
    # `--flat-output` only strips the model-name level, so the per-cell subdirs
    # remain. The single-file guard (Test-StageOutput) therefore saw one cell's
    # 1,506 rows, compared them to the whole stage's 15,060 planned cells, read
    # 10% coverage, and DELETED a complete stage -- the same denominator/path
    # mismatch that hid behind the earlier 1,732-vs-1,506 NIH bug.
    #
    # This guard sums every cell file for coverage, and samples each file for
    # quality. The `v3_report_first` variant asks the model for a REPORT, so its
    # rows are prose by design and their parse_success=false is a documented
    # property of that prompt, not a broken cell (docs/50 S7.6 records the
    # +0.230 parse-fail). That one variant relaxes the parse check exactly like
    # the open-i report layer; the other nine keep the strict rule, so a real
    # parse collapse in any of them is still caught. Errors and empty replies
    # gate every cell.
    $def = $LayerDef[$layer]
    $abDir = Join-Path $script:OutputRoot ("{0}\{1}\ablation" -f $def.Dir, $model)
    if (-not (Test-Path $abDir)) {
        return [pscustomobject]@{ ok = $false; note = 'no ablation output dir'; n = 0; bad = 0
                                  file = ''; files = @(); err = 0; unparsed = 0; empty = 0
                                  rows = 0; cover = $null; exempt = $false }
    }
    $files = @(Get-ChildItem -Path $abDir -Recurse -File -Filter '*_predictions.jsonl' `
                             -ErrorAction SilentlyContinue | Sort-Object FullName)
    if ($files.Count -eq 0) {
        return [pscustomobject]@{ ok = $false; note = 'no cell output files'; n = 0; bad = 0
                                  file = ''; files = @(); err = 0; unparsed = 0; empty = 0
                                  rows = 0; cover = $null; exempt = $false }
    }
    $perFile = [Math]::Max(20, [int]($sample / [Math]::Max(1, $files.Count)))
    $n = 0; $usable = 0; $nErr = 0; $nUnparsed = 0; $nEmpty = 0; $rows = 0
    $errs = @{}
    foreach ($f in $files) {
        $relax = ($f.FullName -match 'report_first')
        $rows += (Get-Content $f.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
        foreach ($line in (Get-Content $f.FullName -TotalCount $perFile)) {
            if (-not $line.Trim()) { continue }
            $n++
            $isErr = $line -match '"error"\s*:\s*"([^"]+)"'
            if ($isErr) {
                $nErr++
                $msg = $Matches[1]
                if ($errs.ContainsKey($msg)) { $errs[$msg]++ } else { $errs[$msg] = 1 }
            }
            $isUnparsed = $line -match '"parse_success"\s*:\s*false'
            if ($isUnparsed) { $nUnparsed++ }
            $isEmpty = $line -match '"raw_output"\s*:\s*""'
            if ($isEmpty) { $nEmpty++ }
            $rowBad = $isErr -or (-not $relax -and $isUnparsed) -or $isEmpty
            if (-not $rowBad) { $usable++ }
        }
    }
    if ($n -eq 0) {
        return [pscustomobject]@{ ok = $false; note = 'ablation files empty'; n = 0; bad = 0
                                  file = $files[0].FullName; files = $files; err = 0; unparsed = 0
                                  empty = 0; rows = $rows; cover = $null; exempt = $false }
    }
    $cover = $null
    if ($Expected -gt 0) { $cover = [Math]::Round(100.0 * $rows / $Expected, 1) }
    $bad = $n - $usable
    $parts = @()
    if ($nErr) { $parts += "$nErr exception(s)" }
    if ($nUnparsed) { $parts += "$nUnparsed unparsed (report-first variant is prose by design)" }
    if ($nEmpty) { $parts += "$nEmpty empty reply" }
    if ($Expected -gt 0 -and $rows -lt [int]($Expected * 0.95)) {
        $parts += ("only {0} of {1} rows written across {2} cell(s) ({3}% coverage)" -f `
                   $rows, $Expected, $files.Count, $cover)
    }
    $note = $parts -join ', '
    if ($errs.Count -gt 0) {
        $top = $errs.GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 1
        $note = ("{0}x {1}" -f $top.Value, $top.Key) + $(if ($parts.Count -gt 1) {
            '; ' + (($parts | Where-Object { $_ -notmatch '^\d+x ' }) -join ', ') } else { '' })
    }
    $tol = [Math]::Max(2, [int]($n * 0.01))
    $ok = ($bad -le $tol)
    if ($Expected -gt 0 -and $rows -lt [int]($Expected * 0.95)) { $ok = $false }
    return [pscustomobject]@{ ok = $ok; note = $note; n = $n; bad = $bad
                              file = $files[0].FullName; files = $files; err = $nErr
                              unparsed = $nUnparsed; empty = $nEmpty; rows = $rows
                              cover = $cover; exempt = $false }
}

function Get-StageExempt([string] $model, [string] $layer) {
    # Resolve the two independent exemption reasons into the flags the guard takes.
    # `exempt` -> keep the file, tag it contract-exempt, exclude from structured
    #             metrics; `prose` -> this stage's deliverable is free text, so
    #             parse_success=false is expected; `reason` -> what the summary prints.
    $modelExempt = $KnownContractFailures.ContainsKey($model)
    $proseLayer = $ProseByDesignLayers.ContainsKey($layer)
    $reason = $null
    if ($modelExempt) { $reason = $KnownContractFailures[$model] }
    elseif ($proseLayer) { $reason = $ProseByDesignLayers[$layer] }
    return [pscustomobject]@{ exempt = ($modelExempt -or $proseLayer)
                              prose  = $proseLayer
                              reason = $reason }
}

if (-not $StateFile) { $StateFile = Join-Path $script:Root 'outputs\state\run_all_layers_state.json' }
if ($Reset -and (Test-Path $StateFile)) { Remove-Item $StateFile -Force }

# --------------------------------------------------------------------------
# isolation guard
#
# A pre-run smoke test and the real full run must not share either the
# predictions root or the state file:
#   * the state key is layer/model/stage and does NOT encode the image limit, so a
#     -Limit 100 run would mark every stage complete and the full run would then
#     skip them -- capping the study at 100 images;
#   * if the roots are the same, the smoke rows sit inside the real run's
#     directories, where the durable writer treats them as done work.
# Mixing the two is silent and irreversible, so it is refused here rather than
# documented and hoped for.
# --------------------------------------------------------------------------
$defaultRoot = Join-Path $script:Root 'outputs\matrix'
$defaultState = Join-Path $script:Root 'outputs\state\run_all_layers_state.json'
$rootIsDefault = ($script:OutputRoot.TrimEnd('\') -eq $defaultRoot.TrimEnd('\'))
$stateIsDefault = ($StateFile.TrimEnd('\') -eq $defaultState.TrimEnd('\'))

$isPartial = ($Limit -gt 0)

# The guard is written as ONE condition over the whole truth table on purpose.
# The first version enumerated the illegitimate cases branch by branch and missed
# one: `-Limit N -StateFile <isolated>` (production root, isolated state) passed
# the guard and wrote its N images into outputs\matrix\, where the durable writer
# then treated them as finished work for the full run. Enumerating "the bad ones"
# is what let a quadrant go missing; stating the invariant cannot.
#
#   -Limit   root      state     verdict
#   -------  --------  --------  ---------------------------------------------
#   no       default   default   ALLOW  (the full run: what this script is for)
#   no       isolated  default   ALLOW  (full set into another folder; harmless)
#   no       any       isolated  ALLOW  (full set, separate bookkeeping)
#   yes      isolated  isolated  ALLOW  (a pre-run test, fully contained)
#   yes      default   any       REFUSE (partial results land in the production root)
#   yes      any       default   REFUSE (stage keys ignore -Limit, so the full run
#                                        would skip everything this run completed)
if ($isPartial -and ($rootIsDefault -or $stateIsDefault)) {
    Write-Host ""
    Write-Host "REFUSING TO START: a -Limit run must not share anything with the full run." -ForegroundColor Red
    Write-Host ""
    if ($rootIsDefault -and $stateIsDefault) {
        Write-Host "  Both are shared: -OutputRoot is the default and -StateFile is the default." -ForegroundColor Red
        Write-Host "    * the $Limit images would be written into outputs\matrix\, the production root;"
        Write-Host "    * every stage would be marked complete, so the full run would skip them"
        Write-Host "      and the study would be silently capped at $Limit images."
    }
    elseif ($rootIsDefault) {
        Write-Host "  -OutputRoot is shared: only -StateFile was changed." -ForegroundColor Red
        Write-Host "    * the $Limit images would be written into outputs\matrix\, the production root,"
        Write-Host "      where the durable writer counts them as finished work for the full run."
    }
    else {
        Write-Host "  -StateFile is shared: only -OutputRoot was changed." -ForegroundColor Red
        Write-Host "    * stage keys do not encode the image limit, so the full run would skip"
        Write-Host "      every stage this test completed."
    }
    Write-Host ""
    Write-Host "  Separate BOTH of them, or run the whole set:" -ForegroundColor Yellow
    Write-Host "    pre-run test : -Limit N -OutputRoot outputs\pretest ``"
    Write-Host "                            -StateFile outputs\state\pretest_state.json"
    Write-Host "    full run     : omit -Limit entirely"
    exit 3
}

# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
# Slowest model first inside every layer.
$OrderedModels = $Models | Sort-Object { -$SecondsPerCall[$_] }

$plan = @()
foreach ($layer in $Layers) {
    $def = $LayerDef[$layer]
    $lim = if ($Limit -gt 0) { $Limit } else { $def.Images }
    foreach ($stage in $def.Stages.Keys) {
        # ablation runs on a subset of models, and precision is the bf16 control for
        # the 4-bit primary configuration -- running it on a model that already runs
        # bf16 would compare bf16 with itself. Both filters live here; the model-level
        # half of each is applied in the inner loop below.
        if ($stage -eq 'ablation' -and $AblationModels.Count -eq 0) { continue }
        foreach ($m in $OrderedModels) {
            if ($stage -eq 'ablation' -and ($AblationModels -notcontains $m)) { continue }
            if ($stage -eq 'precision' -and ($FourBitModels -notcontains $m)) { continue }
            $rate = Get-RateFor $m $layer $stage
            $secPerCall = $rate.Seconds
            $rateSource = $rate.Source
            $calls = $lim * $def.Stages[$stage]
            $loadS = 0; if ($LoadSeconds.ContainsKey($m)) { $loadS = $LoadSeconds[$m] }
            $speedup = 1.0
            if ($BatchSpeedup.ContainsKey($m)) { $speedup = $BatchSpeedup[$m] }
            $plan += [pscustomobject]@{
                Layer   = $layer
                Model   = $m
                Stage   = $stage
                Limit   = $lim
                Cells   = $def.Stages[$stage]
                Calls   = $calls
                Seconds = ($calls * $secPerCall / $speedup * $PaddingFactor) `
                          + $StageStartupSeconds + $loadS
                Rate    = $secPerCall
                Source  = $rateSource
            }
        }
    }
}

$totalSeconds = 0
foreach ($p in $plan) { $totalSeconds += $p.Seconds }
$totalCalls = 0
foreach ($p in $plan) { $totalCalls += $p.Calls }

Write-Host ""
Write-Host ("=" * 82)
Write-Host " THREE-LAYER 6K EVALUATION -- PLAN"
Write-Host ("=" * 82)
Write-Host ("  models      : {0}" -f ($OrderedModels -join ', '))
Write-Host ("  layers      : {0}" -f ($Layers -join ', '))
Write-Host ("  stages      : {0}" -f (($plan | ForEach-Object { $_.Stage } | Select-Object -Unique) -join ', '))
Write-Host ("  state file  : {0}" -f $StateFile)
Write-Host ("  output root : {0}{1}" -f $script:OutputRoot,
    $(if ($rootIsDefault) { '' } else { '   (ISOLATED -- not the production matrix)' }))
if ($isPartial) {
    Write-Host ("  MODE        : PRE-RUN TEST, {0} images per layer -- isolated from the full run" -f $Limit) -ForegroundColor Yellow
} else {
    Write-Host "  MODE        : FULL RUN"
}
Write-Host ""
Write-Host "  Each layer runs the design that answers ITS question (not one grid three times):"
foreach ($layer in $Layers) {
    $def = $LayerDef[$layer]
    Write-Host ("    {0,-9} {1,5} images   {2}" -f $layer, $def.Images, $def.Question)
}
Write-Host ""
Write-Host ("  {0,-10} {1,-24} {2,-10} {3,7} {4,10} {5,10} {6,8} {7,-9}" -f `
    'layer', 'model', 'stage', 'cells/img', 'calls', 'est', 's/call', 'rate src')
foreach ($g in ($plan | Group-Object Layer)) {
    foreach ($p in ($g.Group | Sort-Object { $_.Seconds } -Descending)) {
        Write-Host ("  {0,-10} {1,-24} {2,-10} {3,7} {4,10} {5,10} {6,8:N3} {7,-9}" -f `
            $p.Layer, $p.Model, $p.Stage, $p.Cells, $p.Calls,
            (Format-Duration $p.Seconds), $p.Rate, $p.Source)
    }
}
Write-Host ""
Write-Host "  rate src: measured  = this (model, layer, stage) was measured on the"
Write-Host "                        100-image pre-run"
Write-Host "            proxy-nih = borrowed from the NIH layer of the same model"
Write-Host "                        (PadChest/Open-i are not measured yet)"
Write-Host "            proxy-stage / default = borrowed from another stage / the"
Write-Host "                        per-model fallback"
Write-Host "  s/call is the rate BEFORE the 1.20 padding allowance."
Write-Host ""
Write-Host ("  TOTAL: {0:N0} model calls, {1} ({2:N1} days of GPU time)" -f `
    $totalCalls, (Format-Duration $totalSeconds), ($totalSeconds / 86400))
Write-Host ""
Write-Host "  Ctrl-C is safe at any point: every stage fsyncs each batch and records a"
Write-Host "  durable byte offset, so a re-run continues instead of redoing work."
Write-Host ("=" * 82)
Write-Host ""

if ($DryRun) {
    Write-Host ("  config fingerprint: {0}" -f (Get-ConfigFingerprint))
    Write-Host "  (-DryRun given: nothing was run.)"
    exit 0
}

# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------
$state = Read-State
$configFingerprint = Get-ConfigFingerprint
if ($state.PSObject.Properties['config_fingerprint'] -and
    $state.config_fingerprint -and $state.config_fingerprint -ne $configFingerprint) {
    Write-Host ""
    Write-Host "  [refuse] configuration fingerprint changed since this state file was written."
    Write-Host "           Manifests or prompt/schema modules differ. Re-run with -Reset, or"
    Write-Host "           use a separate -StateFile for the new configuration."
    Write-Host ""
    exit 3
}
if (-not $state.PSObject.Properties['config_fingerprint']) {
    $state | Add-Member -NotePropertyName config_fingerprint -NotePropertyValue $configFingerprint -Force
    Save-State $state
}
$done = 0
foreach ($p in $plan) {
    if (Test-StageDone $state ("{0}/{1}/{2}" -f $p.Layer, $p.Model, $p.Stage)) { $done++ }
}
if ($done -gt 0) {
    Write-Host ("  ({0} stage(s) already complete and will be skipped)" -f $done)
    Write-Host ""
}

$runStart = Get-Date
$index = 0
$succeeded = 0
$failed = @()

foreach ($layer in $Layers) {
    $layerPlan = $plan | Where-Object { $_.Layer -eq $layer }
    if (-not $layerPlan) { continue }
    $def = $LayerDef[$layer]
    Write-Host ""
    Write-Host ("#" * 82)
    Write-Host ("# LAYER: {0}   ({1} images)   {2}" -f $layer, $def.Images, $def.Question)
    Write-Host ("#" * 82)

    foreach ($p in $layerPlan) {
        $index++
        $key = "{0}/{1}/{2}" -f $p.Layer, $p.Model, $p.Stage
        if (Test-StageDone $state $key) {
            Write-Host ("  [skip] {0} (already complete)" -f $key)
            continue
        }
        $cmd = Get-LayerCommand $p.Model $p.Layer $p.Stage $p.Limit
        if (-not $cmd) { Write-Host ("  [skip] no command for {0}" -f $key); continue }
        $python = Get-LayerPython $p.Model

        $elapsed = (Get-Date) - $runStart
        Write-Host ""
        Write-Host ("=" * 82)
        Write-Host ("[{0}/{1}] {2} :: {3} :: {4}" -f $index, $plan.Count, $p.Layer, $p.Model, $p.Stage)
        Write-Host ("        est {0}  |  elapsed {1}" -f (Format-Duration $p.Seconds), (Format-Duration $elapsed.TotalSeconds))
        Write-Host ("        {0} {1}" -f (Split-Path $python -Leaf), ($cmd -join ' '))
        Write-Host ("=" * 82)

        $t0 = Get-Date
        & $python @cmd
        $code = $LASTEXITCODE
        $dt = ((Get-Date) - $t0).TotalSeconds

        if ($code -eq 0) {
            # pass the planned cell count so the guard can also detect a stage that
            # wrote only a fraction of its work and still exited 0
            $ex = Get-StageExempt $p.Model $p.Layer
            $exempt = $ex.exempt
            if ($p.Stage -eq 'ablation') {
                # ablation writes one file per cell; the single-file guard cannot
                # see the whole stage (see Test-AblationOutput).
                $chk = Test-AblationOutput $p.Model $p.Layer -Expected $p.Calls
            }
            else {
                $chk = Test-StageOutput $p.Model $p.Layer $p.Stage -Expected $p.Calls `
                    -ContractExempt:$exempt -ProseStage:$ex.prose
            }
            if (-not $chk.ok) {
                $failed += $key
                Write-Host ("  [BAD ] {0} in {1} -- exit 0 but {2} of {3} sampled rows errored" -f `
                    $key, (Format-Duration $dt), $chk.bad, $chk.n) -ForegroundColor Red
                if ($chk.note) { Write-Host ("         {0}" -f $chk.note) -ForegroundColor Red }
                Write-Host ("         inspected: {0}" -f $chk.file)
                # QUARANTINE, do not delete. Deleting kept the "a re-run redraws
                # from scratch" property but destroyed the only evidence of WHY
                # the stage was bad -- which cost a whole diagnosis cycle on
                # 2026-10-18: padchest/qwen25 was marked [BAD] at 21.5% parse and
                # the raw replies that would have explained it (prose? truncated
                # JSON? empty?) were gone before anyone could read them.
                # Renaming to `*.bad.<stamp>` keeps the resume property (the
                # runner keys off the real filename and will not find the renamed
                # file), keeps the bytes for inspection, and does not disturb the
                # state file.
                $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
                foreach ($poison in @($chk.files)) {
                    if ($poison -and (Test-Path $poison)) {
                        $dest = "$poison.bad.$stamp"
                        Move-Item -LiteralPath $poison -Destination $dest -Force
                        Write-Host ("         quarantined -> {0}" -f (Split-Path $dest -Leaf)) -ForegroundColor Yellow
                    }
                }
                Write-Host ("         {0} file(s) quarantined as *.bad; a re-run redoes this stage from scratch" -f (@($chk.files).Count))
            }
            else {
                Mark-Done $state $key
                $succeeded++
                if ($exempt) {
                    Mark-ContractExempt $state $key
                    Write-Host ("  [done*] {0} in {1} (est {2}) -- contract-exempt: {3}" -f `
                        $key, (Format-Duration $dt), (Format-Duration $p.Seconds),
                        $ex.reason) -ForegroundColor Yellow
                }
                else {
                    Write-Host ("  [done] {0} in {1} (est {2})" -f $key, (Format-Duration $dt), (Format-Duration $p.Seconds))
                }
            }
        }
        else {
            $failed += $key
            Write-Host ("  [FAIL] {0} exit={1} after {2}" -f $key, $code, (Format-Duration $dt)) -ForegroundColor Red
            Write-Host "         re-run the same command to continue: the stage resumes itself"
        }
    }
}

Write-Host ""
Write-Host ("=" * 82)
Write-Host " RUN SUMMARY"
Write-Host ("=" * 82)
Write-Host ("  wall clock : {0}" -f (Format-Duration ((Get-Date) - $runStart).TotalSeconds))
Write-Host ("  succeeded  : {0}" -f $succeeded)
if ($failed.Count -eq 0) {
    Write-Host "  failed     : none"
} else {
    Write-Host ("  failed     : {0}" -f ($failed -join ', ')) -ForegroundColor Red
    Write-Host "  -> re-run the same command to retry only those"
}
$exemptKeys = @($state.contract_exempt | Where-Object { $_ })
if ($exemptKeys.Count -gt 0) {
    Write-Host "  contract-exempt (prose by design; keep the file, exclude from metrics):" -ForegroundColor Yellow
    foreach ($k in $exemptKeys) { Write-Host ("    - {0}" -f $k) -ForegroundColor Yellow }
}
Write-Host ""

# --------------------------------------------------------------------------
# coverage accounting (CPU only) -- item 6 of docs/50 section 3
# --------------------------------------------------------------------------
# 09_shift_analysis.py implements the census, but a gate nobody calls is not a
# gate. This driver used to only PRINT the analysis commands, so a full run
# produced no census at all, and a cell that loses two thirds of its labels --
# medgemma_15 at resolution/severe -- went out with nothing attached to say so.
# The census therefore runs here, over every NIH shift stage the run just
# produced, and its verdict lands in this summary instead of in a command the
# reader has to remember to type.
$coverageLow = @()
$coverageChecked = 0
if ($Layers -contains 'nih') {
    Write-Host ("=" * 82)
    Write-Host " COVERAGE (CPU only, no GPU needed)"
    Write-Host ("=" * 82)
    foreach ($model in $Models) {
        if ($KnownContractFailures.ContainsKey($model)) {
            Write-Host ("  [--] {0}: contract-exempt (prose by design); no label census" -f $model) -ForegroundColor Yellow
            continue
        }
        $pred = Join-Path $script:OutputRoot ("nih_eval\{0}\shift\shift_predictions.jsonl" -f $model)
        if (-not (Test-Path $pred)) {
            Write-Host ("  [--] {0}: no shift output to census" -f $model)
            continue
        }
        $anaOut = Join-Path $script:OutputRoot ("nih_eval\{0}\shift\analysis" -f $model)
        $python = Get-LayerPython $model
        Write-Host ""
        Write-Host ("  {0}:" -f $model)
        & $python scripts\09_shift_analysis.py --predictions $pred --output-dir $anaOut `
            --min-coverage $MinCoverage --fail-on-low-coverage
        $code = $LASTEXITCODE
        $coverageChecked++
        if ($code -eq 3) {
            $coverageLow += $model
        }
        elseif ($code -ne 0) {
            Write-Host ("  [BAD ] {0}: coverage analysis exited {1}" -f $model, $code) -ForegroundColor Red
        }
    }
    Write-Host ""
    Write-Host ("=" * 82)
    Write-Host " RUN SUMMARY (coverage)"
    Write-Host ("=" * 82)
    if ($coverageChecked -eq 0) {
        Write-Host "  coverage   : no shift output was produced to census"
    }
    elseif ($coverageLow.Count -eq 0) {
        Write-Host ("  coverage   : {0} model(s) censused, every cell >= {1:P0} label coverage" -f `
            $coverageChecked, $MinCoverage)
    }
    else {
        Write-Host ("  coverage   : {0} of {1} model(s) have cells below {2:P0} -- {3}" -f `
            $coverageLow.Count, $coverageChecked, $MinCoverage, ($coverageLow -join ', ')) `
            -ForegroundColor Yellow
    }
    Write-Host ""
}

Write-Host "  ANALYSIS (CPU only, no GPU needed):"
Write-Host "    shift coverage census: already run above, see each model's shift\analysis\"
Write-Host "                           (coverage_report.json + shift_coverage_by_condition.csv)"
Write-Host "    geographic shift + report layer:"
Write-Host "      .\.venv-vl\Scripts\python.exe scripts\42_geographic_shift_analysis.py ``"
# Paths come from $script:OutputRoot, not from a literal. They used to be hardcoded
# to outputs\matrix, so a run made with `-OutputRoot outputs\pretest` printed
# analysis commands pointing at a directory that did not exist -- the isolation was
# applied to the run and not to the summary. An analysis command that silently
# targets an empty directory is the same class of failure as a silent empty stage.
Write-Host ("          --source-dir {0}\nih_eval\medgemma_4b_it ``" -f $script:OutputRoot)
Write-Host "          --source-manifest data\manifests\nih_eval_1732.csv ``"
Write-Host ("          --target-dir {0}\padchest\medgemma_4b_it ``" -f $script:OutputRoot)
Write-Host "          --target-manifest data\manifests\padchest_shift.csv ``"
Write-Host ("          --report-dir {0}\openi\medgemma_4b_it ``" -f $script:OutputRoot)
Write-Host "          --output-dir outputs\geographic\medgemma_4b_it --n-boot 1000"
Write-Host "    selective metrics (NIH): scripts\18_selective_metrics_v2.py"
Write-Host "    accept/clarify/abstain  : scripts\19_accept_clarify_abstain.py"
Write-Host ("=" * 82)

# The coverage assertion is the LAST thing the run does, on purpose: by the time
# it can fail, every stage is finished and marked done in the state file, so a
# re-run skips straight past them. Exit 3 here means "the inference is complete;
# the census says some cells are computed over a selected subset". Report the
# coverage next to the metric -- do not drop the cell and do not lower the bar.
if ($coverageLow.Count -gt 0) {
    Write-Host ""
    Write-Host "exit 3: coverage assertion failed -- NOT an inference failure." -ForegroundColor Yellow
    Write-Host ("        {0} model(s) have shift cells below {1:P0} label coverage: {2}" -f `
        $coverageLow.Count, $MinCoverage, ($coverageLow -join ', ')) -ForegroundColor Yellow
    Write-Host "        All finished stages are marked done; a re-run skips them." -ForegroundColor Yellow
    exit 3
}
