#!/usr/bin/env python
"""105 - Hash the analysis artifacts that back the paper's numbers.

The prediction JSONL files are large and already covered by the coverage
census (script 90); this manifest covers the derived CSV/JSON tables that the
manuscript cites, so a reader can check that a downloaded table is the one the
paper used.

Output: outputs/MANIFEST.json

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\105_outputs_manifest.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MAX_BYTES = 50 * 1024 * 1024
SUFFIXES = {".csv", ".json"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    files = []
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in SUFFIXES:
            continue
        if p.name == "MANIFEST.json":
            continue
        if p.stat().st_size > MAX_BYTES:
            continue
        files.append({
            "path": p.relative_to(ROOT).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        })
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": "program/outputs",
        "note": "Derived CSV/JSON analysis artifacts; prediction JSONL files are "
                "covered by outputs/analysis_sensitivity/coverage_census.csv.",
        "n_files": len(files),
        "files": files,
    }
    dst = OUT / "MANIFEST.json"
    dst.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {dst} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
