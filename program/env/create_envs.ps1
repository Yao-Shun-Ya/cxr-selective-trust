# Run from D:\Paper\program in PowerShell.
# Creates three virtual environments: .venv-vl, .venv-chexagent, .venv-modern.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "Creating .venv-vl ..."
py -3.10 -m venv .venv-vl
.\.venv-vl\Scripts\python.exe -m pip install --upgrade pip
.\.venv-vl\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv-vl\Scripts\python.exe -m pip install -r env\requirements-vl.txt

Write-Host "Creating .venv-chexagent ..."
py -3.10 -m venv .venv-chexagent
.\.venv-chexagent\Scripts\python.exe -m pip install --upgrade pip
.\.venv-chexagent\Scripts\python.exe -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
.\.venv-chexagent\Scripts\python.exe -m pip install -r env\requirements-chexagent.txt

Write-Host "Creating .venv-modern ..."
py -3.10 -m venv .venv-modern
.\.venv-modern\Scripts\python.exe -m pip install --upgrade pip
.\.venv-modern\Scripts\python.exe -m pip install torch==2.14.1 torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv-modern\Scripts\python.exe -m pip install -r env\requirements-vl.txt
.\.venv-modern\Scripts\python.exe -m pip install transformers==5.18.0

Write-Host "Done."
Write-Host "VL env:        .\.venv-vl\Scripts\python.exe"
Write-Host "CheXagent env: .\.venv-chexagent\Scripts\python.exe"
Write-Host "Modern env:    .\.venv-modern\Scripts\python.exe"
