# Controlled input-geometry sweep (docs/58 section 15.6).
#
# 2 models x 3 long edges x 4 cells x 400 NIH evaluation studies = 9,600 calls.
# The same 400 studies are used at every edge and for both models (seed fixed),
# so the only moving part is the client-side long edge. The runner records the
# actual sent width/height per row; the internal processor ceiling is unchanged
# and remains part of the measurement.
#
#   powershell -ExecutionPolicy Bypass -File scripts/run_geometry_sweep.ps1

$ErrorActionPreference = "Continue"
$pyModern = "d:\Paper\program\.venv-modern\Scripts\python.exe"
$pyVL = "d:\Paper\program\.venv-vl\Scripts\python.exe"
$log = "d:\Paper\program\outputs\geometry_sweep.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

$plan = @(
    @{ model = "qwen3_vl_8b_instruct"; need = 18000; bs = 4; py = $pyModern; budget = 26 },
    @{ model = "internvl3_5_8b";       need = 18000; bs = 1; py = $pyVL;     budget = 26 }
)
$edges = @(512, 1024, 2048)

foreach ($p in $plan) {
    foreach ($e in $edges) {
        $free = [int](nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | Select-Object -First 1)
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        if ($free -lt $p.need) {
            Add-Content $log "[$stamp] SKIP edge=$e/$($p.model) free=${free}MiB need=$($p.need)MiB"
            continue
        }
        Add-Content $log "[$stamp] START edge=$e/$($p.model) free=${free}MiB"
        & $p.py "d:\Paper\program\scripts\08_shift_benchmark.py" `
            --model $p.model `
            --manifest "data\manifests\nih_eval_1506.csv" `
            --images-root "data\nih\images" `
            --output-dir "outputs\geometry\$($p.model)\edge$e" `
            --split internal_test,source_val --seed 20261118 --limit 400 `
            --shifts contrast noise resolution --severities severe `
            --max-edge $e --max-new-tokens 1024 --batch-size $p.bs `
            --vram-budget-gb $p.budget *>> $log
        $code = $LASTEXITCODE
        $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Add-Content $log "[$stamp] EXIT $code edge=$e/$($p.model)"
        if ($code -ne 0) {
            Add-Content $log "[$stamp] STOPPED on non-zero exit; re-run to resume"
            break
        }
    }
}
Add-Content $log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] GEOMETRY_QUEUE_FINISHED"
