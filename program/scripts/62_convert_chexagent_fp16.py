#!/usr/bin/env python
"""Pre-convert CheXagent-8b to fp16 on disk (docs/50 section 2.2, option C).

Why this is the only option that works here
-------------------------------------------
The checkpoint is stored in **F32**: 31.15 GB over 7 shards, exactly 8B x 4 bytes.
Three things were measured on 2026-10-06, in this order:

1. ``device_map={"": 0}`` (option A) removes the CPU offload, but the model is
   still fp32, and 31.15 GiB of parameters do not fit in the 30.6 GiB that is
   free on a 32 GiB card: the driver pages VRAM to host RAM during *load*, which
   is the 100%-utilisation / 82 W signature that reset the driver twice during
   the pre-run.
2. ``torch_dtype=torch.float16`` is **silently ignored** for this remote-code
   class. Measured (``scripts/_diag_chexagent_dtype.py``): config.torch_dtype
   reads ``torch.float16`` while the parameters are 31.15 GiB of ``torch.float32``
   -- with and without ``low_cpu_mem_usage=True``. So "just ask for fp16" does
   nothing at all, which is what made the original bug look like a device-map
   bug.
3. What the loader *does* honour is the checkpoint's own dtype: whatever dtype is
   on disk is what lands in the model.

So the fix has to happen before loading. This script rewrites every float tensor
as fp16 (integers, indices and bools are copied unchanged), which halves the file
(31.15 -> ~15.6 GB), makes the load fit with room to spare, makes it about 2x
faster, and -- the point of option C -- makes any future ``device_map="auto"``
estimate correct instead of off by a factor of two.

The original directory is NOT touched.

Usage::

    .venv-vl\\Scripts\\python.exe scripts\\62_convert_chexagent_fp16.py
    .venv-vl\\Scripts\\python.exe scripts\\62_convert_chexagent_fp16.py --check
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SRC_DEFAULT = ROOT / "models" / "CheXagent-8b"
DST_DEFAULT = ROOT / "models" / "CheXagent-8b-fp16"

# Copied verbatim: the remote code, the tokenizer and the processor are what make
# `trust_remote_code=True` work at all, and the config carries the architecture.
COPY_SUFFIXES = {".py", ".json", ".model", ".txt", ".md", ".gitattributes"}
SKIP_NAMES = {"model.safetensors.index.json"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SRC_DEFAULT))
    ap.add_argument("--dst", default=str(DST_DEFAULT))
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--check", action="store_true",
                    help="report what is on disk without writing anything")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"ERROR: {index_path} not found")
        return 1

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    target = getattr(torch, args.dtype)

    by_shard: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(key)

    dt = index.get("metadata", {}).get("total_size", 0) / 2**30
    print(f"source      : {src}")
    print(f"shards      : {len(shards)}   tensors: {len(weight_map)}   "
          f"declared size: {dt:.2f} GiB")
    print(f"destination : {dst}   dtype: {args.dtype}")

    existing = sorted(dst.glob("model-*.safetensors")) if dst.exists() else []
    if existing:
        got = sum(f.stat().st_size for f in existing) / 2**30
        print(f"already present: {len(existing)} converted shard(s), {got:.2f} GiB")
        if got > 10:
            print("nothing to do (delete the destination to reconvert)")
            return 0

    if args.check:
        print("\n--check: nothing written")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    out_map: dict[str, str] = {}
    total_bytes = 0
    for shard in shards:
        with safe_open(str(src / shard), framework="pt") as f:
            converted = {}
            for key in by_shard.get(shard, []):
                tensor = f.get_tensor(key)
                if tensor.is_floating_point():
                    tensor = tensor.to(target)
                converted[key] = tensor.contiguous()
        save_file(converted, str(dst / shard), metadata={"format": "pt"})
        for key in converted:
            out_map[key] = shard
        n_bytes = sum(t.numel() * t.element_size() for t in converted.values())
        total_bytes += n_bytes
        print(f"  {shard}: {len(converted)} tensors -> {n_bytes / 2**30:.2f} GiB",
              flush=True)

    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": out_map},
                   indent=2), encoding="utf-8")

    copied = []
    for p in src.iterdir():
        if not p.is_file() or p.name in SKIP_NAMES:
            continue
        if p.suffix in COPY_SUFFIXES or p.name == ".gitattributes":
            shutil.copy2(p, dst / p.name)
            copied.append(p.name)
    print(f"\ncopied {len(copied)} non-weight file(s): {', '.join(sorted(copied))}")

    # The declared dtype must match the weights, or the whole point of this
    # conversion is lost: `device_map="auto"` estimates from this field. Leaving
    # "float32" here would keep the estimator wrong even though the tensors are
    # fp16 -- the exact failure this script exists to remove.
    cfg_path = dst / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    was = cfg.get("torch_dtype")
    cfg["torch_dtype"] = args.dtype
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"config.json torch_dtype: {was} -> {args.dtype}")
    print(f"converted total: {total_bytes / 2**30:.2f} GiB "
          f"(was {dt:.2f} GiB) -> saved {dt - total_bytes / 2**30:.2f} GiB")
    print(f"\nNEXT: point the registry at {dst.name} and run "
          f"scripts/_diag_chexagent.py; the fp16 model must settle near 15.6 GiB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
