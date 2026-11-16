#!/usr/bin/env python
"""91 - Manifest lock and split-pure NIH evaluation manifest.

CPU-only, read-only with respect to raw data.  Two reproducibility gaps are
closed without deleting the 226 `internal_train` donor rows (which are used to
repair `no_finding` and documented in docs/49):

1. write a split-pure `nih_eval_1506.csv` containing only the rows the paper
   scores (`internal_test` + `source_val`);
2. write `MANIFEST_SHA256.json` for every manifest the paper uses, so a reader
   can verify that the file has not changed between the artifact and the claim.

The donor rows stay in `nih_eval_1732.csv`; nothing in the evaluation path reads
them because every runner passes `--split internal_test,source_val`.

Outputs:
  data/manifests/nih_eval_1506.csv
  data/manifests/MANIFEST_SHA256.json

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\91_manifest_lock.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / "data" / "manifests"
EVAL_SPLITS = {"internal_test", "source_val"}
FILES = ["nih_eval_1732.csv", "nih_eval_1506.csv", "padchest_shift.csv",
         "openi_rrg.csv"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    src = MAN / "nih_eval_1732.csv"
    with src.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [r for r in reader if r["split"] in EVAL_SPLITS]
        fields = reader.fieldnames
    out = MAN / "nih_eval_1506.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out.name}: {len(rows)} rows")

    lock = {}
    for name in FILES:
        p = MAN / name
        if p.exists():
            with p.open(encoding="utf-8-sig") as fh:
                n = sum(1 for _ in csv.DictReader(fh))
            lock[name] = {"rows": n, "bytes": p.stat().st_size, "sha256": sha256(p)}
    (MAN / "MANIFEST_SHA256.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
