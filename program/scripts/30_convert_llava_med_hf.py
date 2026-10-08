"""Convert microsoft/LLaVA-Med-1.5 weights into HuggingFace Llava layout.

The problem
-----------
``LlavaForConditionalGeneration`` expects these prefixes::

    vision_tower.<...>              (a CLIPVisionModel)
    multi_modal_projector.<...>
    language_model.model.<...>
    language_model.lm_head.weight

but ``microsoft/llava-med-v1.5-mistral-7b`` ships the *original LLaVA* naming::

    model.vision_tower.vision_tower.vision_model.<...>     (391 tensors)
    model.mm_projector.<...>                                (4 tensors)
    model.layers.<...> / model.norm / model.embed_tokens    (288 + others)
    lm_head.weight

Loading it as-is produces "Some weights were not used ... and are newly
initialized: language_model.lm_head.weight, ..." — the vision tower and projector
are silently left uninitialised, which is why generation then fails deep inside
CLIP with *"You have to specify pixel_values"*. This conversion is the actual fix;
config edits alone cannot repair a key mismatch.

Output layout
-------------
Weights are written to a **new directory** so the verified download is never
modified. Non-weight files (tokenizer, configs) are copied from the source; the
config is rewritten by ``scripts/29_fix_llava_med_config.py`` semantics.

Usage
-----
::

    python scripts/30_convert_llava_med_hf.py \
        --src models/llava-med-v1.5-mistral-7b \
        --dst models/llava-med-v1.5-mistral-7b-hf
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VISION_PREFIX = "model.vision_tower.vision_tower."
PROJECTOR_PREFIX = "model.mm_projector."

# The original LLaVA projector is a sequential MLP addressed by integer index
# (0 = linear, 1 = activation, 2 = linear), while HF's LlavaMultiModalProjector
# names its layers ``linear_1`` / ``linear_2``. Without this remap the projector
# loads as "newly initialized" and the model projects noise.
PROJECTOR_INDEX_MAP = {"0": "linear_1", "2": "linear_2"}


def remap(key: str) -> str:
    """Original LLaVA key -> HuggingFace LlavaForConditionalGeneration key."""
    if key.startswith(VISION_PREFIX):
        return "vision_tower." + key[len(VISION_PREFIX):]
    if key.startswith(PROJECTOR_PREFIX):
        rest = key[len(PROJECTOR_PREFIX):]
        head, _, tail = rest.partition(".")
        return f"multi_modal_projector.{PROJECTOR_INDEX_MAP.get(head, head)}.{tail}"
    if key == "lm_head.weight":
        return "language_model.lm_head.weight"
    if key.startswith("model."):
        return "language_model." + key          # model.layers.* -> language_model.model.layers.*
    return key


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(ROOT / "models" / "llava-med-v1.5-mistral-7b"))
    ap.add_argument("--dst", default=str(ROOT / "models" / "llava-med-v1.5-mistral-7b-hf"))
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    ap.add_argument("--max-shard-gb", type=float, default=4.5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not (src / "model.safetensors.index.json").exists():
        print(f"ERROR: {src}/model.safetensors.index.json not found")
        return 1

    index = json.loads((src / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index["weight_map"]
    print(f"source shards : {sorted(set(weight_map.values()))}")
    print(f"source tensors: {len(weight_map)}")

    # show the mapping for a sample of each family
    samples = {}
    for k in weight_map:
        fam = ("vision" if k.startswith(VISION_PREFIX)
               else "projector" if k.startswith(PROJECTOR_PREFIX)
               else "lm_head" if k == "lm_head.weight"
               else "language" if k.startswith("model.")
               else "other")
        samples.setdefault(fam, k)
    print("\nkey remap preview:")
    for fam, k in sorted(samples.items()):
        print(f"  [{fam:9s}] {k}\n            -> {remap(k)}")

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    import torch
    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)

    # ---- 1) load, remap, and convert dtype shard by shard ------------------
    new_state: dict[str, torch.Tensor] = {}
    target = getattr(torch, args.dtype)
    for shard in sorted(set(weight_map.values())):
        part = load_file(str(src / shard))
        for k, v in part.items():
            nk = remap(k)
            if nk in new_state:
                raise RuntimeError(f"duplicate key after remap: {nk}")
            new_state[nk] = v.to(target).contiguous()
        print(f"  loaded {shard}: {len(part)} tensors", flush=True)

    print(f"\nremapped tensors: {len(new_state)}")

    # ---- 2) sanity: every expected family must be present -----------------
    fams = {
        "vision_tower": sum(1 for k in new_state if k.startswith("vision_tower.")),
        "multi_modal_projector": sum(1 for k in new_state if k.startswith("multi_modal_projector.")),
        "language_model": sum(1 for k in new_state if k.startswith("language_model.")),
    }
    print("families:", fams)
    if fams["vision_tower"] == 0 or fams["multi_modal_projector"] == 0:
        print("ERROR: conversion left a family empty; refusing to write")
        return 1

    # ---- 3) save in shards ------------------------------------------------
    by_bytes = sorted(new_state.items(), key=lambda kv: kv[1].numel() * kv[1].element_size(),
                      reverse=True)
    limit = args.max_shard_gb * 1024 ** 3
    shards: list[dict[str, torch.Tensor]] = []
    cur: dict[str, torch.Tensor] = {}
    cur_bytes = 0
    for k, v in by_bytes:
        nb = v.numel() * v.element_size()
        if cur and cur_bytes + nb > limit:
            shards.append(cur); cur = {}; cur_bytes = 0
        cur[k] = v; cur_bytes += nb
    if cur:
        shards.append(cur)

    total = len(shards)
    out_map: dict[str, str] = {}
    for i, shard in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{total:05d}.safetensors"
        save_file(shard, str(dst / name), metadata={"format": "pt"})
        for k in shard:
            out_map[k] = name
        print(f"  wrote {name}: {len(shard)} tensors")

    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum(
            v.numel() * v.element_size() for v in new_state.values())},
            "weight_map": out_map}, indent=2), encoding="utf-8")

    # ---- 4) copy non-weight files ----------------------------------------
    for f in src.iterdir():
        if f.is_file() and f.suffix not in (".safetensors",) and not f.name.endswith(".index.json"):
            if ".orig" in f.name:
                continue
            shutil.copy2(f, dst / f.name)

    print(f"\nwrote {dst}  ({total} shards, {len(new_state)} tensors)")
    print("NEXT: run scripts/29_fix_llava_med_config.py --model-dir <dst>, then "
          "scripts/27_smoke_one.py and CHECK THE OUTPUT is a sensible reading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
