#!/usr/bin/env python
"""Diagnostic: does the explicit pixel hint actually reach qwen_vl_utils?

The audit in outputs/state/input_contract_audit.json reported qwen25's largest
PadChest image as 3584x3584 = 12,845,056 px, which is exactly the LIBRARY default
MAX_PIXELS = 16384*28*28 -- not the shared contract value
base.IMAGE_MAX_PIXELS = 24576*28*28 = 19,267,584. Either the audit script did not
pass the hint, or the hint is ignored. This script settles it by building the same
message twice -- with and without the hint -- against the largest image in a
layer, and printing the resulting image grid.

CPU only: it loads the *processor*, never the weights.

  .\\.venv-vl\\Scripts\\python.exe scripts\\_diag_pixel_hint.py --model qwen25_vl_7b_instruct
  .\\.venv-modern\\Scripts\\python.exe scripts\\_diag_pixel_hint.py --model qwen3_vl_8b_instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cxr_trust.models.base import (  # noqa: E402
    IMAGE_MAX_PIXELS, IMAGE_MIN_PIXELS, image_pixel_hints,
)
from src.cxr_trust.models.registry import resolve_path  # noqa: E402

from PIL import Image  # noqa: E402
from transformers import AutoProcessor  # noqa: E402


def probe(processor, image, hint: bool, label: str) -> dict:
    """Measure BOTH resize stages.

    Stage 1 is ``process_vision_info`` (it honours the per-image max_pixels hint).
    Stage 2 is ``processor.__call__``, which re-runs smart_resize using the
    checkpoint's OWN preprocessor_config.json limits. If stage 2 has a lower
    ceiling than the hint, stage 1's result is undone and the hint is useless --
    which is exactly the discrepancy between the audit table and this probe, so
    only measuring stage 2 settles it.
    """
    from qwen_vl_utils import process_vision_info

    content = {"type": "image", "image": image}
    if hint:
        content.update(image_pixel_hints())
    messages = [[{"role": "user", "content": [content, {"type": "text", "text": "x"}]}]]
    image_inputs, video_inputs = process_vision_info(messages)
    first = image_inputs[0]
    s1_w, s1_h = (first.size if hasattr(first, "size")
                  else (int(first.shape[-1]), int(first.shape[-2])))

    text = processor.apply_chat_template(messages[0], tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    grid = inputs.get("image_grid_thw")
    grid_list = [int(x) for x in grid[0].tolist()] if grid is not None else None
    tokens = None
    if grid_list:
        t, h, w = grid_list
        tokens = (t * h * w) // 4          # merge_size 2 -> /4
    n_pad = int((inputs["attention_mask"] == 0).sum())

    out = {"label": label, "hint": hint, "stage1": [s1_h, s1_w], "grid_thw": grid_list,
           "tokens": tokens, "input_len": int(inputs["input_ids"].shape[1]),
           "pad_tokens": n_pad}
    print(f"  {label:20s} hint={str(hint):5s} stage1={s1_w}x{s1_h} "
          f"grid={grid_list} tokens={tokens} input_len={out['input_len']} "
          f"pad={n_pad}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", default=None,
                    help="default: the LARGEST image found in data/padchest/images")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    if args.image:
        img_path = Path(args.image)
    else:
        cands = sorted((root / "data" / "padchest" / "images").glob("*.jpg"))
        if not cands:
            print("[error] no padchest images found")
            return 2
        img_path = max(cands, key=lambda p: Image.open(p).size[0] * Image.open(p).size[1])

    with Image.open(img_path) as im:
        image = im.convert("RGB")
    print(f"model      : {args.model}")
    print(f"processor  : {root / 'models' / ''}")
    print(f"image      : {img_path.name}  source={image.size[0]}x{image.size[1]} "
          f"= {image.size[0] * image.size[1]:,} px")
    print(f"contract   : IMAGE_MAX_PIXELS={IMAGE_MAX_PIXELS:,} "
          f"IMAGE_MIN_PIXELS={IMAGE_MIN_PIXELS:,}  (library default "
          f"MAX_PIXELS={16384 * 28 * 28:,})")

    proc = AutoProcessor.from_pretrained(str(resolve_path(args.model, root)))
    no_hint = probe(proc, image, False, "without hint")
    with_hint = probe(proc, image, True, "with contract hint")

    print()
    if with_hint["grid_thw"] == no_hint["grid_thw"]:
        print("[WARN] the hint did NOT change the FINAL grid -- stage 2 (the "
              "checkpoint's own preprocessor_config ceiling) overrides the hint.")
        print(f"       stage-1 sizes differed? "
              f"{with_hint['stage1']} vs {no_hint['stage1']} "
              f"-> final grid {with_hint['grid_thw']}")
    else:
        print("[ok  ] the hint IS honoured end to end: the final image grid changed.")
        print(f"       final tokens {no_hint['tokens']} -> {with_hint['tokens']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
