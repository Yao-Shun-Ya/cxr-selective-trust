# Cross-modal inference queue (protocol: docs/58 section 13).
#
# Order: the 4B model first, because it is the only stage that can run safely
# alongside another job on the card; then the three 8B checkpoints. Within a
# model the order is MRI -> OCT -> derm, so the two grayscale cells that carry the
# corpus contrast land before the RGB cell.
#
# Every stage resumes from its own JSONL. A stage whose free-VRAM check fails is
# SKIPPED, not failed: re-running this script later picks it up from zero without
# touching the stages that already finished.
#
#   powershell -ExecutionPolicy Bypass -File scripts/run_crossmodal_queue.ps1

# Continue, not Stop: these checkpoints write deprecation warnings to stderr on
# every load, and with `Stop` PowerShell turns the first warning line into a
# terminating NativeCommandError before the model has even been asked a question.
# Stage success is decided by $LASTEXITCODE, which the runner sets explicitly.
$ErrorActionPreference = "Continue"
$pyModern = "d:\Paper\program\.venv-modern\Scripts\python.exe"
# CheXagent's remote code uses a removed transformers API; it only loads in its
# own fork, which is why the CXR arm also runs it in a separate environment.
$pyChex = "d:\Paper\program\.venv-chexagent\Scripts\python.exe"
# InternVL's remote code calls model.all_tied_weights_keys, which newer
# transformers removed; measured AttributeError in .venv-modern, loads in
# .venv-vl (transformers 4.51.3), which is what produced the CXR InternVL rows.
$pyVL = "d:\Paper\program\.venv-vl\Scripts\python.exe"
$log = "d:\Paper\program\outputs\crossmodal_queue.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

# ``budget`` is passed to the loader as max_memory {0: <budget>, cpu: 0}. That is
# the difference between "fails loudly when it does not fit" and "silently pages
# VRAM into host RAM at 100% utilisation and 20x the per-cell time" -- the failure
# mode documented in docs/43 and docs/50.
$plan = @(
    # ``need`` values are the measured peaks from the pre-flight validation
    # (MedGemma 9.4 GiB, InternVL 16.1, Qwen3 16.9, CheXagent 16.7) plus margin.
    @{ model = "medgemma_15_4b_it";    need = 11000; bs = 8; py = $pyModern; budget = 14 },
    @{ model = "internvl3_5_8b";       need = 18000; bs = 4; py = $pyVL;     budget = 26 },
    @{ model = "qwen3_vl_8b_instruct"; need = 18000; bs = 4; py = $pyModern; budget = 26 },
    @{ model = "chexagent_8b";         need = 18000; bs = 1; py = $pyChex;   budget = 26 }
)
$order = @(
    @("mri", "v0"), @("oct", "v0"), @("derm", "v0"),
    @("mri", "v1"), @("oct", "v1"), @("derm", "v1")
)

foreach ($p in $plan) {
    foreach ($pair in $order) {
        $modality, $schema = $pair
        $free = [int](nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1)
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        if ($free -lt $p.need) {
            Add-Content $log "[$stamp] SKIP $modality/$schema/$($p.model) free=${free}MiB need=$($p.need)MiB"
            continue
        }
        Add-Content $log "[$stamp] START $modality/$schema/$($p.model) free=${free}MiB"
        & $p.py "d:\Paper\program\scripts\121_crossmodal_benchmark.py" `
            --model $p.model --modality $modality --schema $schema `
            --batch-size $p.bs --max-new-tokens 1024 --vram-budget-gb $p.budget *>> $log
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
