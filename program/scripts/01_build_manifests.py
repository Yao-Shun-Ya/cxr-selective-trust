#!/usr/bin/env python
"""Build NIH and CheXpert-plus-RRG manifests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.data.nih import build_manifest as build_nih_manifest
from src.cxr_trust.data.chexpert_rrg import export_all as export_rrg


def main() -> None:
    nih_root = ROOT / "素材/ChestX-ray14"
    rrg_root = ROOT / "素材/CheXpert-plus-RRG"
    manifests = ROOT / "data/manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    print("Building NIH full manifest...")
    build_nih_manifest(
        nih_root / "data/Data_Entry_2017_v2020.csv",
        nih_root / "data/train_val_list.txt",
        nih_root / "data/test_list.txt",
        manifests / "nih_full.csv",
        pilot_size=None,
        seed=0,
    )
    print("Building NIH pilot manifest (1000 studies)...")
    build_nih_manifest(
        nih_root / "data/Data_Entry_2017_v2020.csv",
        nih_root / "data/train_val_list.txt",
        nih_root / "data/test_list.txt",
        manifests / "nih_pilot_1000.csv",
        pilot_size=1000,
        seed=0,
    )
    print("Exporting CheXpert-plus-RRG images and manifest...")
    rrg = export_rrg(rrg_root, ROOT / "data/chexpert_rrg")
    rrg.to_csv(manifests / "chexpert_rrg_all.csv", index=False)
    print(f"NIH full: {manifests / 'nih_full.csv'}")
    print(f"NIH pilot: {manifests / 'nih_pilot_1000.csv'}")
    print(f"RRG manifest: {manifests / 'chexpert_rrg_all.csv'} rows={len(rrg)}")


if __name__ == "__main__":
    main()
