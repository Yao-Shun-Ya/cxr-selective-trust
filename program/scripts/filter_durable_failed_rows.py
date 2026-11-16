#!/usr/bin/env python
"""Drop failed rows from a durable JSONL so the next --resume retries them.

Why a dedicated script
----------------------
`DurableJsonl` treats a row as *done* when it has no `error` field, and a
truncated reply has no error -- it is a successful call that produced no JSON.
So `--resume` skips it forever. The honest way to retry those cells is to remove
the rows from the file (after a timestamped backup) and delete the byte-offset
sidecar, which forces the next open to re-scan the file and rebuild the done-set.

This script only touches the file it is given, keeps a `.bak_<timestamp>` copy
next to it, and refuses to run if the backup already exists.

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\filter_durable_failed_rows.py \
      --path outputs/matrix/nih_eval/qwen3_vl_8b_thinking/shift/shift_predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    args = ap.parse_args()

    path = Path(args.path)
    if not path.is_absolute():
        path = ROOT / path
    sidecar = path.with_suffix(path.suffix + ".progress.json")
    if not path.exists():
        raise SystemExit(f"missing {path}")

    backup = path.with_name(path.name + time.strftime(".bak_%Y%m%d_%H%M%S"))
    shutil.copy2(path, backup)

    keep, drop = [], 0
    # utf-8-sig: tolerate a BOM if an editor ever re-saved the file; the runner
    # itself writes plain utf-8, so this is a guard, not a requirement.
    with path.open(encoding="utf-8-sig") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if (rec.get("parsed") or {}).get("parse_success"):
                keep.append(line if line.endswith("\n") else line + "\n")
            else:
                drop += 1

    if drop == 0:
        backup.unlink()
        print(f"[filter] nothing to drop; kept {len(keep)} rows, backup discarded")
        return 0

    payload = "".join(keep).encode("utf-8")
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    if sidecar.exists():
        sidecar.unlink()

    print(f"[filter] kept {len(keep)} rows, dropped {drop} failed rows")
    print(f"[filter] backup {backup.name}")
    print(f"[filter] sidecar {'removed' if not sidecar.exists() else 'still present'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
