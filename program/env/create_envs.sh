#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m venv .venv-vl
./.venv-vl/Scripts/python.exe -m pip install --upgrade pip 2>/dev/null || ./.venv-vl/bin/python -m pip install --upgrade pip
./.venv-vl/Scripts/pip.exe install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || ./.venv-vl/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
./.venv-vl/Scripts/pip.exe install -r env/requirements-vl.txt 2>/dev/null || ./.venv-vl/bin/pip install -r env/requirements-vl.txt

python -m venv .venv-chexagent
./.venv-chexagent/Scripts/python.exe -m pip install --upgrade pip 2>/dev/null || ./.venv-chexagent/bin/python -m pip install --upgrade pip
./.venv-chexagent/Scripts/pip.exe install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || ./.venv-chexagent/bin/pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
./.venv-chexagent/Scripts/pip.exe install -r env/requirements-chexagent.txt 2>/dev/null || ./.venv-chexagent/bin/pip install -r env/requirements-chexagent.txt

echo "Done."
