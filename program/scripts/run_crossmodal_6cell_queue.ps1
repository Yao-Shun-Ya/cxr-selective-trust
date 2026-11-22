# Cross-modal 6-cell queue: 3 open-weight models x 3 modalities x 1,788 images.
#
# Cell set (identical to the API arm's cross-modal panel, scripts/api_probe):
#   schema v0: clean, contrast:severe, noise:severe, blank_gray, blank_phase
#   schema v1: blank_gray
# 5 + 1 = 6 cells per image; 1,788 x 6 = 10,728 calls per model, 32,184 total.
#
# CheXagent is deliberately absent: the increment is the three open-weight
# checkpoints that match the CXR arm's main matrix.
#
# Images enter at stored resolution -- this script never applies the API arm's
# 1,024-px client downscale, so the local and remote arms can be compared on
# that axis (docs/58 section 15).
#
# Every stage resumes from its own JSONL; a failed free-VRAM check SKIPs the
# stage and a re-run picks it up without touching finished stages.
#
#   powershell -ExecutionPolicy Bypass -File scripts/run_crossmodal_6cell_queue.ps1

$ErrorActionPreference = "Continue"
$pyModern = "d:\Paper\program\.venv-modern\Scripts\python.exe"
$pyVL = "d:\Paper\program\.venv-vl\Scripts\python.exe"
$log = "d:\Paper\program\outputs\crossmodal_6cell_queue.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

$cellsV0 = @("clean", "contrast:severe", "noise:severe", "blank_gray", "blank_phase")
$cellsV1 = @("blank_gray")

# Same measured peaks and margins as run_crossmodal_queue.ps1.
$plan = @(
    @{ model = "medgemma_15_4b_it";    need = 11000; bs = 8; py = $pyModern; budget = 14 },
    @{ model = "internvl3_5_8b";       need = 18000; bs = 4; py = $pyVL;     budget = 26 },
    @{ model = "qwen3_vl_8b_instruct"; need = 18000; bs = 4; py = $pyModern; budget = 26 }
)
$order = @(
    @("mri", "v0", $cellsV0), @("oct", "v0", $cellsV0), @("derm", "v0", $cellsV0),
    @("mri", "v1", $cellsV1), @("oct", "v1", $cellsV1), @("derm", "v1", $cellsV1)
)

foreach ($p in $plan) {
    foreach ($stage in $order) {
        $modality, $schema, $cells = $stage
        $free = [int](nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1)
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        if ($free -lt $p.need) {
            Add-Content $log "[$stamp] SKIP $modality/$schema/$($p.model) free=${free}MiB need=$($p.need)MiB"
            continue
        }
        Add-Content $log "[$stamp] START $modality/$schema/$($p.model) cells=$($cells -join ',') free=${free}MiB"
        & $p.py "d:\Paper\program\scripts\121_crossmodal_benchmark.py" `
            --model $p.model --modality $modality --schema $schema `
            --cells $cells --batch-size $p.bs --max-new-tokens 1024 `
            --vram-budget-gb $p.budget *>> $log
        $code = $LASTEXITCODE
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content $log "[$stamp] EXIT $code $modality/$schema/$($p.model)"
        if ($code -ne 0) {
            Add-Content $log "[$stamp] STOPPED on non-zero exit; re-run to resume"
            break
        }
    }
}
Add-Content $log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] QUEUE_FINISHED"
