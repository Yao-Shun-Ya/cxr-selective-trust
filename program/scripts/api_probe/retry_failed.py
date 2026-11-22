"""Drop failed rows so the next run retries them.

A durable row is a *finished* row, including one whose ``error`` field is set.
That is the right default (it stops a poison cell from looping forever), but a
quota wall or a gateway outage produces thousands of such rows at once. This
helper removes them and the byte-offset sidecar, so the runner rescans the file
and retries exactly those cells.

    python retry_failed.py outputs/api_probe/<slug>/predictions.jsonl

Prints what it removed and leaves a timestamped backup next to the file.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keep = [r for r in rows if not r.get("error")]
    removed = len(rows) - len(keep)

    backup = path.with_suffix(
        path.suffix + f".bak-{datetime.now():%Y%m%d-%H%M%S}")
    path.replace(backup)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                            for r in keep), encoding="utf-8")

    sidecar = Path(str(path) + ".progress.json")
    if sidecar.exists():
        sidecar.unlink()

    print(f"rows={len(rows)} kept={len(keep)} removed={removed}")
    print(f"backup={backup.name}")
    print(f"sidecar removed: {sidecar.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
