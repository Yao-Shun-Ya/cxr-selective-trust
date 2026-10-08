#!/usr/bin/env python
"""Build a dataset manifest for a locked CXR dataset.

Examples:
  python scripts/build_manifest.py --dataset chexpert \
    --labels-csv data/chexpert/raw/CheXpert-v1.0-small/train.csv \
    --images-root data/chexpert/raw/CheXpert-v1.0-small \
    --output data/chexpert/manifests/pilot_1000.csv --pilot-size 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medvlm_trust.data.manifest import (  # noqa: E402
    build_chexpert_manifest,
    build_nih_manifest,
    build_padchest_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["chexpert", "nih", "padchest"])
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pilot-size", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.dataset == "chexpert":
        df = build_chexpert_manifest(
            labels_csv=args.labels_csv,
            images_root=args.images_root,
            output_csv=args.output,
            pilot_size=args.pilot_size,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    elif args.dataset == "nih":
        df = build_nih_manifest(
            labels_csv=args.labels_csv,
            images_root=args.images_root,
            output_csv=args.output,
            pilot_size=args.pilot_size,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    elif args.dataset == "padchest":
        df = build_padchest_manifest(
            labels_csv=args.labels_csv,
            images_root=args.images_root,
            output_csv=args.output,
            pilot_size=args.pilot_size,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        raise SystemExit(f"Unsupported dataset: {args.dataset}")

    print(f"wrote {len(df)} rows to {args.output}")
    print(df["split"].value_counts().to_string())


if __name__ == "__main__":
    main()
