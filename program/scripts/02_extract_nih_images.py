#!/usr/bin/env python
"""Extract NIH images from tar.gz archives.

Examples:
  python scripts/02_extract_nih_images.py --all
  python scripts/02_extract_nih_images.py --manifest data/manifests/nih_pilot_1000.csv
  python scripts/02_extract_nih_images.py --manifest data/manifests/nih_pilot_1000.csv --limit 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.data.nih import extract_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", default=str(ROOT / "素材/ChestX-ray14/data/images"))
    parser.add_argument("--output", default=str(ROOT / "data/nih/images"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not args.all and not args.manifest:
        raise SystemExit("Specify --all or --manifest.")
    if args.all and args.manifest:
        raise SystemExit("Use either --all or --manifest, not both.")

    if args.all:
        n = extract_images(args.archives, args.output, image_ids=None, max_images=args.limit)
    else:
        df = pd.read_csv(args.manifest)
        ids = set(df["image_id"].astype(str).head(args.limit).tolist()) if args.limit else set(df["image_id"].astype(str))
        n = extract_images(args.archives, args.output, image_ids=ids, max_images=args.limit)
    print(f"extracted {n} images to {args.output}")


if __name__ == "__main__":
    main()
