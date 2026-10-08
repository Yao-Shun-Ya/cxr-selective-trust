#!/usr/bin/env python
"""Check environment, data paths, and model paths."""

from __future__ import annotations

import importlib.util
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "MISSING"


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = {
        "python": platform.python_version(),
        "packages": {
            "torch": pkg_version("torch"),
            "transformers": pkg_version("transformers"),
            "accelerate": pkg_version("accelerate"),
            "bitsandbytes": pkg_version("bitsandbytes"),
            "qwen-vl-utils": pkg_version("qwen-vl-utils"),
            "pandas": pkg_version("pandas"),
            "pyarrow": pkg_version("pyarrow"),
            "Pillow": pkg_version("Pillow"),
            "scikit-learn": pkg_version("scikit-learn"),
        },
        "modules": {m: has_module(m) for m in ["torch", "transformers", "accelerate", "bitsandbytes", "qwen_vl_utils"]},
        "data": {
            "nih_metadata": (root / "素材/ChestX-ray14/data/Data_Entry_2017_v2020.csv").exists(),
            "nih_archives": count_files(root / "素材/ChestX-ray14/data/images"),
            "rrg_findings_parquet": (root / "素材/CheXpert-plus-RRG/findings section/valid-00000-of-00001.parquet").exists(),
            "rrg_impression_parquet": (root / "素材/CheXpert-plus-RRG/impression section/valid-00000-of-00001.parquet").exists(),
        },
        "models": {
            "chexagent": (root / "models/chexagent8b/CheXagent-8b-main/config.json").exists(),
            "medgemma": (root / "models/medgemma/MedGemma-4B-IT/config.json").exists(),
            "qwen": (root / "models/qwen/Qwen2.5-VL-7B-Instruct/config.json").exists(),
        },
        "outputs": {
            "manifests": count_files(root / "data/manifests"),
            "rrg_images": count_files(root / "data/chexpert_rrg"),
        },
    }
    if has_module("torch"):
        try:
            import torch
            report["cuda"] = {
                "available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            }
        except Exception as exc:
            report["cuda"] = {"error": str(exc)}
    out = root / "outputs/env_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
