"""Denominator census over frozen prediction files.

The harness paper's strongest idea is that a reported denominator should be
recomputable from the artifacts. This script is that recomputation: for every
prediction file it reports how many cells exist, whether each image carries the
same number of cells (a uniform denominator), how many are duplicate keys, and
how many replies failed the output contract.

A file whose cells-per-image varies is not necessarily wrong -- some stages are
deliberately partial -- but the variation has to be *declared*. Undeclared
raggedness is what makes a published denominator unreconstructible.

    python scripts/124_artifact_census.py --root outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis_harness"


def census_file(path: Path) -> dict:
    keys, per_image, errors, parsed = set(), defaultdict(set), Counter(), 0
    rows = dup = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            errors["torn_or_invalid_json"] += 1
            continue
        rows += 1
        key = (str(obj.get("image_id")), str(obj.get("condition")),
               str(obj.get("severity")))
        if key in keys:
            dup += 1
        keys.add(key)
        per_image[str(obj.get("image_id"))].add(key)
        p = obj.get("parsed") or {}
        if p.get("parse_success"):
            parsed += 1
        else:
            errors[str(p.get("parse_error") or "unknown")] += 1

    cells = [len(v) for v in per_image.values()]
    return {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "rows": rows,
        "unique_keys": len(keys),
        "duplicates": dup,
        "images": len(per_image),
        "cells_per_image_min": min(cells) if cells else 0,
        "cells_per_image_max": max(cells) if cells else 0,
        "uniform_denominator": bool(cells) and min(cells) == max(cells),
        "parse_rate": parsed / rows if rows else float("nan"),
        "parse_failures": rows - parsed,
        "top_error": (errors.most_common(1)[0][0] if errors else ""),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "outputs"))
    ap.add_argument("--glob", default="*.jsonl")
    ap.add_argument("--out", default=str(OUT / "artifact_census.csv"))
    args = ap.parse_args()

    files = sorted(p for p in Path(args.root).rglob(args.glob)
                   if "progress" not in p.name)
    if not files:
        raise SystemExit(f"no {args.glob} under {args.root}")
    rows = [census_file(p) for p in files]
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    n = len(df)
    uniform = int(df["uniform_denominator"].sum())
    dup = int((df["duplicates"] > 0).sum())
    print(f"{n} files, {int(df['rows'].sum()):,} rows, "
          f"{int(df['bytes'].sum())/2**30:.2f} GiB")
    print(f"uniform denominator: {uniform}/{n}")
    print(f"files with duplicate keys: {dup}/{n}")
    print(f"parse-failure rows: {int(df['parse_failures'].sum()):,} "
          f"(mean parse rate {df['parse_rate'].mean():.3f})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
