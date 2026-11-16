#!/usr/bin/env python
"""111 - Run the trained DenseNet-121 baseline over the 19-cell NIH grid and PadChest.

Why: Reversal 3 (degradation raises recall by moving the operating point) has
only been shown for VLMs.  A supervised CNN answers whether the phenomenon
requires a language head or is a property of the decision rule.  It is cheap:
DenseNet-121 forward passes are milliseconds per image.

Outputs
  outputs/densenet_baseline/grid_predictions.csv
  outputs/densenet_baseline/padchest_predictions.csv

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\111_densenet_grid.py
  .\\.venv-vl\\Scripts\\python.exe scripts\\111_densenet_grid.py --skip-padchest
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "densenet_baseline"
NIH_MAN = ROOT / "data" / "manifests" / "nih_eval_1506.csv"
NIH_IMAGES = ROOT / "data" / "nih" / "images"
PC_MAN = ROOT / "data" / "manifests" / "padchest_shift.csv"
PC_IMAGES = ROOT / "data" / "padchest"

LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax", "no_finding"]

SEVERITIES = {
    "blur": {"mild": 0.5, "moderate": 1.0, "severe": 2.0},
    "noise": {"mild": 0.02, "moderate": 0.05, "severe": 0.10},
    "jpeg": {"mild": 70, "moderate": 40, "severe": 20},
    "brightness": {"mild": 0.8, "moderate": 1.2, "severe": 1.5},
    "contrast": {"mild": 0.7, "moderate": 1.3, "severe": 1.6},
    "resolution": {"mild": 0.5, "moderate": 0.25, "severe": 0.125},
}
LEVELS = ["mild", "moderate", "severe"]

EVAL_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def apply_shift(image: Image.Image, shift: str, severity: str) -> Image.Image:
    image = image.convert("RGB")
    if shift == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(SEVERITIES["blur"][severity])))
    if shift == "noise":
        sigma = float(SEVERITIES["noise"][severity])
        arr = np.asarray(image, dtype=np.float32) / 255.0
        rng = np.random.default_rng(0)
        arr = np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if shift == "jpeg":
        quality = int(SEVERITIES["jpeg"][severity])
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if shift == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(SEVERITIES["brightness"][severity]))
    if shift == "contrast":
        return ImageEnhance.Contrast(image).enhance(float(SEVERITIES["contrast"][severity]))
    if shift == "resolution":
        factor = float(SEVERITIES["resolution"][severity])
        w, h = image.size
        small = image.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    raise ValueError(f"unknown shift: {shift}")


class CellDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_root: Path, shift: str | None,
                 severity: str | None):
        self.frame = frame.reset_index(drop=True)
        self.root = image_root
        self.shift = shift
        self.severity = severity

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int):
        r = self.frame.iloc[i]
        img = Image.open(self.root / r["image_relpath"]).convert("RGB")
        if self.shift is not None:
            img = apply_shift(img, self.shift, self.severity)
        return EVAL_TF(img), str(r["image_id"])


def load_model(device: str):
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, len(LABELS))
    state = torch.load(OUT / "best_model.pt", map_location="cpu")
    m.load_state_dict(state)
    m.to(device).eval()
    return m


@torch.inference_mode()
def predict(frame: pd.DataFrame, image_root: Path, model, device: str,
            shift: str | None = None, severity: str | None = None,
            batch_size: int = 64) -> np.ndarray:
    ds = CellDataset(frame, image_root, shift, severity)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    out, ids = [], []
    for x, idb in dl:
        out.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        ids.extend(idb)
    p = np.concatenate(out, axis=0)
    return pd.DataFrame(p, columns=[f"p_{l}" for l in LABELS]).assign(image_id=ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-padchest", action="store_true")
    ap.add_argument("--skip-nih", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device, flush=True)
    model = load_model(device)

    if not args.skip_nih:
        nih = pd.read_csv(NIH_MAN, low_memory=False)
        nih = nih[nih["image_relpath"].notna()].copy()
        frames = []
        cells = [("clean", None, None)] + [
            (f"{s}_{lv}", s, lv) for s in SEVERITIES for lv in LEVELS
        ]
        for name, shift, sev in cells:
            print(f"[{name}] {len(nih)} images", flush=True)
            d = predict(nih, NIH_IMAGES, model, device, shift, sev, args.batch_size)
            d["condition"] = shift or "clean"
            d["severity"] = sev or "none"
            frames.append(d)
        grid = pd.concat(frames, ignore_index=True)
        grid.to_csv(OUT / "grid_predictions.csv", index=False, encoding="utf-8-sig")
        print(f"wrote {OUT / 'grid_predictions.csv'} ({len(grid)} rows)")

    if not args.skip_padchest:
        pc = pd.read_csv(PC_MAN, low_memory=False)
        pc = pc[pc["image_relpath"].notna()].copy()
        print(f"[padchest] {len(pc)} images", flush=True)
        d = predict(pc, PC_IMAGES, model, device, None, None, args.batch_size)
        d.to_csv(OUT / "padchest_predictions.csv", index=False, encoding="utf-8-sig")
        print(f"wrote {OUT / 'padchest_predictions.csv'} ({len(d)} rows)")

    (OUT / "grid_run.json").write_text(json.dumps({
        "cells": 0 if args.skip_nih else 19,
        "nih_images": 0 if args.skip_nih else len(nih),
        "padchest_images": 0 if args.skip_padchest else len(pc),
        "device": device, "batch_size": args.batch_size,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
