#!/usr/bin/env python
"""Batch + grouped inference benchmark.

Two throughput questions, one script
------------------------------------
1. **Batch size** -- how much does a batched forward pass beat one-at-a-time?
2. **Grouping** -- the shift benchmark currently re-decodes the *same* base image
   19 times (6 shifts x 3 severities + clean). Grouping a base image's variants
   into one call should remove most of that duplicated I/O and preprocessing.

Run it before scheduling anything, on the actual target GPU. Numbers from an
8GB laptop card do not transfer.

Usage
-----
::

    # single model, both modes
    python scripts/22_benchmark_batch.py --model chexagent_2_3b --n 24

    # compare a 4-bit and a bf16 configuration of the same model
    python scripts/22_benchmark_batch.py --model qwen3_vl_8b_instruct --n 24 --precision 4bit

    # everything in the registry, silently skipping the ones that fail to load
    python scripts/22_benchmark_batch.py --all --n 16

Output
------
Prints a table and writes ``outputs/benchmark/batch_benchmark.json`` with, per
(model, precision): load seconds, serial s/image, batched s/image at each batch
size, the speedup, observed VRAM, and the generated batch size actually used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.models.registry import (  # noqa: E402
    MODEL_NAMES,
    MODEL_OPTIONS,
    MODEL_SPECS,
    resolve_path,
)

SMOKE_PROMPT = ("Look at this chest radiograph and reply with one short JSON "
                'object: {"no_finding": "positive|negative"}.')


def vram_used_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def reset_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_model(name: str, precision: str):
    from src.cxr_trust.models.registry import get_model
    spec = MODEL_SPECS[name]
    opts = dict(MODEL_OPTIONS.get(name, {}))
    load_kwargs = {
        "load_in_4bit": precision == "4bit",
    }
    # do not inherit the 8GB-era device map / memory cap
    t0 = time.time()
    model = get_model(name, root=ROOT, device_map="auto", max_memory=None, **opts,
                      **load_kwargs)
    model.load()
    return model, time.time() - t0


def benchmark_one(name: str, precision: str, n: int, batch_sizes: list[int],
                  image_path: Path, verbose: bool = True) -> dict:
    result = {"model": name, "precision": precision, "n": n}
    reset_vram()
    try:
        model, load_s = load_model(name, precision)
    except Exception as exc:
        result["error"] = f"load_failed: {type(exc).__name__}: {exc}"
        if verbose:
            print(f"  load FAILED: {result['error'][:120]}")
        return result
    result["load_sec"] = round(load_s, 2)

    images = [image_path] * n
    prompts = [SMOKE_PROMPT] * n

    # --- warm-up (first call includes kernel autotuning / cudnn benchmarking) ---
    try:
        model.generate(image_path, SMOKE_PROMPT, max_new_tokens=32)
    except Exception as exc:
        result["error"] = f"generate_failed: {type(exc).__name__}: {exc}"
        if verbose:
            print(f"  generate FAILED: {result['error'][:120]}")
        return result

    # --- serial baseline ---
    try:
        t0 = time.time()
        outs = [model.generate(img, p, max_new_tokens=64) for img, p in zip(images, prompts)]
        serial = (time.time() - t0) / n
        result["serial_s_per_image"] = round(serial, 3)
        result["serial_sample"] = (outs[0] or "")[:80]
    except Exception as exc:
        result["error"] = f"serial_failed: {type(exc).__name__}: {exc}"
        if verbose:
            print(f"  serial FAILED: {result['error'][:120]}")
        return result

    # --- batched ---
    result["batch"] = {}
    if not hasattr(model, "generate_batch"):
        result["supports_batching"] = False
        if verbose:
            print(f"  serial {serial:.2f} s/img  (no batch API on this wrapper)")
        return result

    result["supports_batching"] = True
    for bs in batch_sizes:
        if bs > n:
            continue
        try:
            t0 = time.time()
            outs = model.generate_batch(images[:bs], prompts[:bs], max_new_tokens=64)
            dt = time.time() - t0
        except Exception as exc:
            result["batch"][f"bs{bs}"] = {"error": f"{type(exc).__name__}: {str(exc)[:80]}"}
            continue
        per = dt / bs
        result["batch"][f"bs{bs}"] = {
            "sec_per_image": round(per, 3),
            "speedup_vs_serial": round(serial / per, 2) if per > 0 else None,
            "total_sec": round(dt, 2),
            "sample": (outs[0] or "")[:80] if outs else "",
        }
        if verbose:
            print(f"  batch {bs:>3d}: {per:6.2f} s/img   speedup x{serial/per:5.2f}")

    result["vram_peak_gb"] = round(vram_used_gb(), 2)
    result["batchary_fell_back"] = not getattr(model, "supports_batching", True)

    # free before the next model
    del model
    import gc
    gc.collect()
    reset_vram()
    if verbose:
        print(f"  serial {serial:.2f} s/img  | peak VRAM {result['vram_peak_gb']:.1f} GB")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", default=None,
                    help=f"one or more of: {', '.join(MODEL_NAMES)}")
    ap.add_argument("--all", action="store_true", help="every model in the registry")
    ap.add_argument("--precision", default=None, choices=["4bit", "bf16"],
                    help="override each model's registry default")
    ap.add_argument("--n", type=int, default=24, help="images per measurement")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--manifest", default="data/manifests/nih_pilot_1000.csv")
    ap.add_argument("--images-root", default="data/nih/images")
    ap.add_argument("--output", default="outputs/benchmark/batch_benchmark.json")
    args = ap.parse_args()

    names = MODEL_NAMES if args.all else (args.model or [])
    if not names:
        ap.error("give --model <name> or --all")

    manifest = pd.read_csv(ROOT / args.manifest)
    first = manifest.iloc[0]
    image_path = ROOT / args.images_root / first["image_relpath"]
    if not image_path.exists():
        print(f"[error] image not found: {image_path}")
        return 1

    print(f"image: {image_path}")
    print(f"n={args.n}  batch_sizes={args.batch_sizes}")
    print(f"CUDA: {torch.cuda.is_available()}  "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
    print()

    results = []
    for name in names:
        precision = args.precision or MODEL_SPECS[name].get("precision", "bf16")
        print(f"=== {name}  [{precision}]")
        res = benchmark_one(name, precision, args.n, args.batch_sizes, image_path)
        results.append(res)
        print()

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": str(image_path), "n": args.n, "batch_sizes": args.batch_sizes,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
