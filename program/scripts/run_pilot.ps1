# Run from D:\Paper\program after env/create_envs.ps1
$ErrorActionPreference = "Stop"
$VLM = ".\.venv-vl\Scripts\python.exe"
$CHEX = ".\.venv-chexagent\Scripts\python.exe"

Write-Host "1/7 Check environment"
& $VLM scripts\00_check_env.py

Write-Host "2/7 Build manifests"
& $VLM scripts\01_build_manifests.py

Write-Host "3/7 Extract first 300 NIH pilot images"
& $VLM scripts\02_extract_nih_images.py --manifest data\manifests\nih_pilot_1000.csv --limit 300

Write-Host "4/7 Run Qwen2.5-VL on 100 images (4-bit)"
& $VLM scripts\03_run_model.py `
  --model qwen25_vl_7b_instruct `
  --manifest data\manifests\nih_pilot_1000.csv `
  --images-root data\nih\images `
  --output-dir outputs\pilot\qwen_nih `
  --limit 100 `
  --load-in-4bit `
  --max-new-tokens 256

Write-Host "5/7 Run MedGemma on 100 images (4-bit)"
& $VLM scripts\03_run_model.py `
  --model medgemma_4b_it `
  --manifest data\manifests\nih_pilot_1000.csv `
  --images-root data\nih\images `
  --output-dir outputs\pilot\medgemma_nih `
  --limit 100 `
  --load-in-4bit `
  --max-new-tokens 256

Write-Host "6/7 Run CheXagent on 5 images (CPU offload; slow on 8GB GPU)"
& $CHEX scripts\03_run_model.py `
  --model chexagent_8b `
  --manifest data\manifests\nih_pilot_1000.csv `
  --images-root data\nih\images `
  --output-dir outputs\pilot\chexagent_nih `
  --limit 5 `
  --max-new-tokens 256

Write-Host "7/7 Evaluate"
& $VLM scripts\04_evaluate.py --predictions outputs\pilot\qwen_nih\predictions.jsonl --output-dir outputs\pilot\qwen_nih\metrics
& $VLM scripts\04_evaluate.py --predictions outputs\pilot\medgemma_nih\predictions.jsonl --output-dir outputs\pilot\medgemma_nih\metrics
& $VLM scripts\04_evaluate.py --predictions outputs\pilot\chexagent_nih\predictions.jsonl --output-dir outputs\pilot\chexagent_nih\metrics
