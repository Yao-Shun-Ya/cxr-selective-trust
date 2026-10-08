"""One-image smoke test: proves a model loads AND generates before any sweep.

Why this exists
---------------
A 30B model at 4-bit needs ~17 GB of weights; if the load itself does not fit,
the Windows driver pages VRAM into host RAM and the machine thrashes *before*
any per-batch guard can fire. The fix is not a better guard, it is not starting
a sweep until a single image has worked.

Prints progress at every stage so a hang is visible immediately, and enforces a
driver-VRAM ceiling throughout.

Usage::

    python scripts/27_smoke_one.py --model qwen3_vl_30b_a3b_instruct --precision 4bit
    python scripts/27_smoke_one.py --model medgemma_4b_it --precision bf16
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPT = ('Assess this chest X-ray. Reply with one short JSON: '
          '{"no_finding": "positive|negative"}.')


def vram_gb():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetMemoryInfo(h).used / 1024 ** 3
    except Exception:
        return float("nan")


def ram_free_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", default="bf16", choices=["4bit", "bf16"])
    ap.add_argument("--vram-budget-gb", type=float, default=29.0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--manifest", default="data/manifests/nih_shared_500.csv")
    ap.add_argument("--images-root", default="data/nih/images")
    args = ap.parse_args()

    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU {torch.cuda.get_device_name(0)}  {total:.1f} GiB   "
          f"budget {args.vram_budget_gb:.0f} GB   host free {ram_free_gb():.1f} GB",
          flush=True)

    df = pd.read_csv(ROOT / args.manifest).head(1)
    img = ROOT / args.images_root / df.iloc[0]["image_relpath"]
    print(f"image : {img.name}  exists={img.exists()}", flush=True)
    if not img.exists():
        print("ERROR: image not found")
        return 1

    from src.cxr_trust.models.registry import get_model

    print("[1/4] loading (this is where a too-big model thrashes) ...", flush=True)
    t0 = time.time()
    try:
        model = get_model(args.model, root=ROOT, device_map="auto",
                          max_memory={0: f"{int(args.vram_budget_gb)}GiB", "cpu": "0GiB"},
                          load_in_4bit=(args.precision == "4bit"))
        model.load()
    except Exception as exc:
        print(f"LOAD FAILED after {time.time()-t0:.1f}s: {type(exc).__name__}: {str(exc)[:250]}")
        print(f"driver VRAM now {vram_gb():.1f} GB")
        return 1
    print(f"[1/4] loaded in {time.time()-t0:.1f}s   "
          f"driver VRAM {vram_gb():.1f} GB   allocator {torch.cuda.memory_allocated()/1024**3:.1f} GB",
          flush=True)

    print(f"[2/4] generating 1 image (max_new_tokens={args.max_new_tokens}) ...", flush=True)
    t0 = time.time()
    try:
        out = model.generate(img, PROMPT, max_new_tokens=args.max_new_tokens)
    except Exception as exc:
        print(f"GENERATE FAILED after {time.time()-t0:.1f}s: {type(exc).__name__}: {str(exc)[:250]}")
        return 1
    dt = time.time() - t0
    print(f"[2/4] generated in {dt:.1f}s", flush=True)
    print(f"      output: {out[:200]!r}", flush=True)

    print("[3/4] memory after 1 image", flush=True)
    print(f"      driver VRAM  {vram_gb():.1f} GB", flush=True)
    print(f"      allocator    {torch.cuda.max_memory_allocated()/1024**3:.1f} GB", flush=True)
    print(f"      host free    {ram_free_gb():.1f} GB", flush=True)

    ok = vram_gb() <= args.vram_budget_gb
    print(f"[4/4] verdict: {'OK - safe to sweep' if ok else 'OVER BUDGET - do not sweep'}",
          flush=True)
    print(f"      implied budget for batch scaling: "
          f"{args.vram_budget_gb - vram_gb():+.1f} GB of headroom", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
