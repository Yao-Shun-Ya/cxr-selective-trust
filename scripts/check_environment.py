#!/usr/bin/env python
"""Check whether data/model prerequisites are present for real experiments."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
from pathlib import Path


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dir_status(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {"exists": False, "n_files": 0}
    return {"exists": True, "n_files": sum(1 for _ in p.rglob("*") if _.is_file())}


def main() -> None:
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "modules": {
            m: has_module(m)
            for m in ["numpy", "pandas", "sklearn", "PIL", "pydantic", "yaml", "torch", "transformers"]
        },
        "data_dirs": {
            d: dir_status(d)
            for d in ["data/chexpert", "data/padchest", "data/nih", "data/mimic"]
        },
        "manifests": {
            str(p): p.exists()
            for p in Path("data").glob("**/manifests/*.csv")
        },
        "env_vars": {
            "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
            "AZURE_OPENAI_API_KEY": bool(os.environ.get("AZURE_OPENAI_API_KEY")),
            "HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
        },
    }
    out = Path("artifacts/smoke/environment_check.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
