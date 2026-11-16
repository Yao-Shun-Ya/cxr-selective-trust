#!/usr/bin/env python
"""Image-side (no-reference) detectability baseline for the severe shifts.

Why this script exists
----------------------
Section 5 argues that Gaussian noise damages predictions without triggering the
model's own reasoning-channel marker.  A reviewer can object that the marker is
a model-side channel only, and ask whether the degradation is detectable at the
image level at all.  This script answers that with a measured, model-free
baseline: for every internal-test image we apply the same severe degradation as
``08_shift_benchmark.py`` and compute three no-reference statistics.  We then
report the AUC for separating clean from degraded images using a single
statistic.  No model is involved.

Statistics (all computed on the grayscale image, no clean reference needed):
* ``lapvar``  -- variance of the 3x3 Laplacian (high-frequency energy);
* ``std``     -- global intensity standard deviation (contrast);
* ``entropy`` -- 256-bin intensity entropy.

Output: ``outputs/analysis_sensitivity/image_side_detectability.csv``.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import laplace

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "nih_eval_1732.csv"
IMAGE_ROOT = ROOT / "data" / "nih" / "images"
OUT = ROOT / "outputs" / "analysis_sensitivity"

SHIFTS = {
    "noise": ("noise", 0.10),
    "blur": ("blur", 3.0),
    "contrast": ("contrast", 1.6),
}


def apply_shift(image: Image.Image, kind: str, param: float) -> Image.Image:
    image = image.convert("RGB")
    if kind == "noise":
        arr = np.asarray(image, dtype=np.float32) / 255.0
        # same generator/seed as 08_shift_benchmark.py
        rng = np.random.default_rng(0)
        arr = np.clip(arr + rng.normal(0, param, arr.shape), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=param))
    if kind == "contrast":
        return ImageEnhance.Contrast(image).enhance(param)
    raise ValueError(kind)


def stats(image: Image.Image) -> dict[str, float]:
    g = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    lap = laplace(g)
    hist = np.histogram(g, bins=256, range=(0.0, 1.0))[0].astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return {
        "lapvar": float(lap.var()),
        "std": float(g.std()),
        "entropy": float(-(p * np.log2(p)).sum()),
    }


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC for pos > neg, ties counted at 0.5."""
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty(len(allv), dtype=np.float64)
    sorted_vals = allv[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def main() -> int:
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    test = [r for r in rows if r["split"] in {"internal_test", "source_val"}]
    OUT.mkdir(parents=True, exist_ok=True)
    recs = []
    for r in test:
        p = IMAGE_ROOT / r["image_relpath"]
        if not p.exists():
            continue
        clean = Image.open(p)
        recs.append((p, stats(clean)))

    out_rows = []
    for name, (kind, param) in SHIFTS.items():
        deg = [stats(apply_shift(Image.open(p), kind, param)) for p, _ in recs]
        for stat in ("lapvar", "std", "entropy"):
            c = np.array([s[stat] for _, s in recs])
            d = np.array([s[stat] for s in deg])
            raw = auc(d, c)
            direction = "higher in degraded" if raw >= 0.5 else "lower in degraded"
            out_rows.append({
                "condition": name,
                "statistic": stat,
                "n_images": int(len(c)),
                "auc_detectable": round(max(raw, 1.0 - raw), 4),
                "direction": direction,
                "median_clean": round(float(np.median(c)), 6),
                "median_severe": round(float(np.median(d)), 6),
            })
        print(f"{name}: n={len(recs)} -> "
              f"lapvar AUC={out_rows[-3]['auc_detectable']}, "
              f"std AUC={out_rows[-2]['auc_detectable']}, "
              f"entropy AUC={out_rows[-1]['auc_detectable']}")

    with open(OUT / "image_side_detectability.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {OUT / 'image_side_detectability.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
