#!/usr/bin/env python
"""Measure the ablation stage's real seconds-per-cell at several batch sizes.

Why this measurement exists
---------------------------
``21_prompt_ablation.py`` was serial until 2026-10-05: it called
``model.generate`` once per image and measured **8.28 s/cell** against 0.63
s/cell for the batched shift stage on the same model. Batching was added, but the
batch size could not be copied from the shift stage, for a reason that is
specific to this stage:

* the five prompt variants ask for very different amounts of text --
  ``v3_report_first`` wants a region-by-region narrative *before* the labels,
  ``v4_normality_pressure`` wants a per-finding justification, ``v0_simple``
  wants the JSON and nothing else -- and

* a batch's cost is set by its **longest** member, because the decoder runs until
  every sequence finishes. So the shift stage's sweet spot (batch 64 on
  medgemma_4b_it) is not evidence for this one: what matters here is the spread
  of completion lengths inside a batch, and that is a property of the prompt.

This script therefore produces the number the driver needs, for the model that
actually runs the stage, per variant and per batch size, and records peak VRAM so
the largest safe batch is visible rather than guessed.

What it does NOT do
-------------------
It does not write predictions. Its only output is
``outputs/state/ablation_batch_measure.json`` plus a table on stdout, so it can
never contaminate ``outputs/matrix`` or the stage state file.

An OOM here is a *result*, not a failure: the config is recorded as ``oom`` and
the sweep continues, because the point is to find where it stops fitting.

Usage::

    .venv-vl\\Scripts\\python.exe scripts\\60_measure_ablation_batch.py
    .venv-vl\\Scripts\\python.exe scripts\\60_measure_ablation_batch.py \\
        --batch-sizes 1 8 16 32 --probe-bs 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.inference.prompts_ablation import (  # noqa: E402
    PROMPT_VARIANTS,
    build_prompt,
)
from src.cxr_trust.models.registry import MODEL_NAMES, get_model  # noqa: E402


def vram_gib() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info()
    return free / 2**30, total / 2**30


def run_batches(model, images, prompt: str, batch_size: int,
                max_new_tokens: int) -> dict:
    """Time ``len(images)`` cells at one batch size; return per-cell seconds.

    Cells are re-timed from a clean peak-memory counter, so the VRAM number
    belongs to this configuration and not to the largest one run before it.
    """
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    sizes: list[int] = []
    start = time.time()
    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        raws = model.generate_batch(chunk, [prompt] * len(chunk),
                                    max_new_tokens=max_new_tokens)
        sizes.extend(len(r) for r in raws)
    elapsed = time.time() - start
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return {
        "cells": len(images),
        "batches": (len(images) + batch_size - 1) // batch_size,
        "elapsed_sec": round(elapsed, 2),
        "sec_per_cell": round(elapsed / len(images), 3),
        "chars_mean": int(sum(sizes) / len(sizes)),
        "chars_max": max(sizes),
        "peak_vram_gib": round(peak, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="medgemma_4b_it", choices=MODEL_NAMES)
    ap.add_argument("--manifest", default="data/manifests/nih_eval_1732.csv")
    ap.add_argument("--split", default="internal_test,source_val")
    ap.add_argument("--images-root", default="data/nih/images")
    ap.add_argument("--variants", nargs="+", default=list(PROMPT_VARIANTS))
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 16])
    ap.add_argument("--probe-bs", type=int, default=0,
                    help="extra single-batch VRAM probe at this size, on the "
                         "longest variant only (0 = skip)")
    ap.add_argument("--images", type=int, default=32,
                    help="images per batched configuration")
    ap.add_argument("--serial-images", type=int, default=2,
                    help="images for the batch-size-1 baseline, which is slow")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--load-in-4bit", action="store_true", default=True)
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/state/ablation_batch_measure.json")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = pd.read_csv(manifest_path)
    if args.split and "split" in manifest.columns:
        wanted = [s.strip() for s in args.split.split(",") if s.strip()]
        manifest = manifest[manifest["split"].isin(wanted)].copy()
    # Same shuffle-then-limit convention as the runner: the first N rows of the
    # file are ordered by image id and are not a sample.
    manifest = manifest.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    need = max(args.images, args.probe_bs, args.serial_images)
    manifest = manifest.head(need).copy()
    images_root = Path(args.images_root)
    if not images_root.is_absolute():
        images_root = ROOT / images_root

    def load_images(n: int) -> list[Image.Image]:
        return [Image.open(images_root / row["image_relpath"]).convert("RGB")
                for _, row in manifest.head(n).iterrows()]

    print(f"model={args.model} images={len(manifest)} "
          f"4bit={args.load_in_4bit} max_new_tokens={args.max_new_tokens}", flush=True)
    model = get_model(args.model, root=ROOT, device_map="auto",
                      load_in_4bit=args.load_in_4bit)
    t0 = time.time()
    model.load()
    print(f"loaded in {time.time() - t0:.0f}s; vram free {vram_gib()[0]:.1f} GiB", flush=True)

    results: list[dict] = []
    for variant in args.variants:
        system, user = build_prompt(variant)
        prompt = system + chr(10) * 2 + user
        for bs in args.batch_sizes:
            n = args.serial_images if bs == 1 else args.images
            images = load_images(n)
            print(f"  {variant:24s} bs={bs:<3d} n={n} ...", end="", flush=True)
            try:
                row = run_batches(model, images, prompt, bs, args.max_new_tokens)
                row.update({"variant": variant, "batch_size": bs, "status": "ok"})
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                row = {"variant": variant, "batch_size": bs, "status": "oom",
                       "error": str(exc)[:200]}
            except RuntimeError as exc:
                torch.cuda.empty_cache()
                row = {"variant": variant, "batch_size": bs, "status": "error",
                       "error": f"{type(exc).__name__}: {exc}"[:200]}
            row["vram_free_gib"] = round(vram_gib()[0], 2)
            results.append(row)
            if row["status"] == "ok":
                print(f" {row['sec_per_cell']:.3f} s/cell "
                      f"(max {row['chars_max']} chars, peak {row['peak_vram_gib']:.1f} GiB)",
                      flush=True)
            else:
                print(f" {row['status'].upper()}", flush=True)

    if args.probe_bs:
        # One full batch of the longest variant: the VRAM ceiling question, asked
        # once, cheaply.
        variant = "v3_report_first" if "v3_report_first" in args.variants else args.variants[0]
        system, user = build_prompt(variant)
        images = load_images(args.probe_bs)
        print(f"  probe {variant} bs={args.probe_bs} n={len(images)} ...", end="", flush=True)
        try:
            row = run_batches(model, images, system + chr(10) * 2 + user,
                              args.probe_bs, args.max_new_tokens)
            row.update({"variant": variant, "batch_size": args.probe_bs,
                        "status": "ok", "probe": True,
                        "vram_free_gib": round(vram_gib()[0], 2)})
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            torch.cuda.empty_cache()
            row = {"variant": variant, "batch_size": args.probe_bs, "status": "oom",
                   "probe": True, "error": f"{type(exc).__name__}: {exc}"[:200],
                   "vram_free_gib": round(vram_gib()[0], 2)}
        results.append(row)
        print(f" {row['status']}", flush=True)

    ok = [r for r in results if r["status"] == "ok"]
    print(f"\n{'variant':24s} {'bs':>4s} {'s/cell':>8s} {'max chars':>10s} {'peak GiB':>9s}")
    for r in results:
        if r["status"] == "ok":
            print(f"{r['variant']:24s} {r['batch_size']:4d} {r['sec_per_cell']:8.3f} "
                  f"{r['chars_max']:10d} {r['peak_vram_gib']:9.2f}")
        else:
            print(f"{r['variant']:24s} {r['batch_size']:4d} {r['status'].upper():>8s}")

    best = {}
    for variant in args.variants:
        cands = [r for r in ok if r["variant"] == variant and r["vram_free_gib"] > 2.0]
        if cands:
            b = min(cands, key=lambda r: r["sec_per_cell"])
            best[variant] = {"batch_size": b["batch_size"],
                             "sec_per_cell": b["sec_per_cell"],
                             "peak_vram_gib": b["peak_vram_gib"]}
    if best:
        print("\nfastest fitting batch size per variant:")
        for k, v in best.items():
            print(f"  {k:24s} bs={v['batch_size']:<3d} {v['sec_per_cell']:.3f} s/cell")

    payload = {
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "load_in_4bit": bool(args.load_in_4bit),
        "max_new_tokens": args.max_new_tokens,
        "manifest": str(manifest_path),
        "split": args.split,
        "images_per_config": args.images,
        "rows": results,
        "fastest_per_variant": best,
    }
    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
