#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

VLM="./.venv-vl/Scripts/python.exe"
CHEX="./.venv-chexagent/Scripts/python.exe"
[ -x "$VLM" ] || VLM="./.venv-vl/bin/python"
[ -x "$CHEX" ] || CHEX="./.venv-chexagent/bin/python"

"$VLM" scripts/00_check_env.py
"$VLM" scripts/01_build_manifests.py
"$VLM" scripts/02_extract_nih_images.py --manifest data/manifests/nih_pilot_1000.csv --limit 300
"$VLM" scripts/03_run_model.py --model qwen25_vl_7b_instruct --manifest data/manifests/nih_pilot_1000.csv --images-root data/nih/images --output-dir outputs/pilot/qwen_nih --limit 100 --load-in-4bit --max-new-tokens 256
"$VLM" scripts/03_run_model.py --model medgemma_4b_it --manifest data/manifests/nih_pilot_1000.csv --images-root data/nih/images --output-dir outputs/pilot/medgemma_nih --limit 100 --load-in-4bit --max-new-tokens 256
"$CHEX" scripts/03_run_model.py --model chexagent_8b --manifest data/manifests/nih_pilot_1000.csv --images-root data/nih/images --output-dir outputs/pilot/chexagent_nih --limit 5 --max-new-tokens 256
"$VLM" scripts/04_evaluate.py --predictions outputs/pilot/qwen_nih/predictions.jsonl --output-dir outputs/pilot/qwen_nih/metrics
"$VLM" scripts/04_evaluate.py --predictions outputs/pilot/medgemma_nih/predictions.jsonl --output-dir outputs/pilot/medgemma_nih/metrics
"$VLM" scripts/04_evaluate.py --predictions outputs/pilot/chexagent_nih/predictions.jsonl --output-dir outputs/pilot/chexagent_nih/metrics
