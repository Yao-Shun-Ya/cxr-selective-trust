"""Locate the PadChest memory cliff for Qwen2.5-VL-7B (4-bit).

Why this exists
---------------
The first 17 PadChest originals showed 6.7--23.3 s/cell for images between
2.8 and 9.0 MP, and 269--290 s/cell for the only three images above 11 MP.
No CUDA OOM was raised: on Windows the driver pages VRAM into host RAM
instead of failing, so an oversubscribed call looks like a 20x slowdown.

This script reproduces the production path exactly (same prompt, same
apply_chat_template / process_vision_info / processor call, same greedy
decoding) and splits the wall clock into:

    load      PIL open + convert("RGB")
    pre       chat template + vision info + processor tensors
    gen       model.generate

plus the peak allocated/reserved VRAM and the driver's free/total. Running
the same list twice shows whether cost accumulates across cells or is a
per-image property.

Usage
-----
    python scripts/_diag_vram_cliff.py                 # default image ladder
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/_diag_vram_cliff.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.cxr_trust.inference.prompts import SIMPLE_SYSTEM_PROMPT, build_simple_prompt  # noqa: E402
from src.cxr_trust.models.base import ensure_left_padding, image_pixel_hints  # noqa: E402
from src.cxr_trust.models.registry import get_model  # noqa: E402

# Small -> largest, one image per pixel bucket, all from the PadChest test split.
DEFAULT_IMAGES = [
    "padchest_00720.jpg",   # 2.83 MP
    "padchest_00471.jpg",   # 3.77 MP
    "padchest_00187.jpg",   # 6.99 MP
    "padchest_00674.jpg",   # 9.00 MP
    "padchest_01417.jpg",   # 11.40 MP  <- first slow cell in the run
    "padchest_01157.jpg",   # 12.99 MP  <- 289.9 s in the run
    "padchest_01256.jpg",   # 15.07 MP  <- 273.2 s in the run
]


def one_call(model, processor, prompt: str, path: Path, max_new_tokens: int) -> dict:
    from qwen_vl_utils import process_vision_info

    row: dict = {"image": path.name}

    t0 = time.time()
    with Image.open(path) as im:
        image = im.convert("RGB")
    w, h = image.size
    row["mp"] = round(w * h / 1e6, 2)
    row["load_sec"] = round(time.time() - t0, 2)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    row["free_before_gib"] = round(free_before / 2**30, 2)

    t0 = time.time()
    messages = [[{"role": "user", "content": [
        {"type": "image", "image": image, **image_pixel_hints()},
        {"type": "text", "text": prompt},
    ]}]]
    text = processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    torch.cuda.synchronize()
    row["pre_sec"] = round(time.time() - t0, 2)
    row["input_tokens"] = int(inputs["input_ids"].shape[1])
    row["pixels_sent"] = int(inputs["pixel_values"].shape[0])

    t0 = time.time()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None, repetition_penalty=1.05,
        )
    torch.cuda.synchronize()
    row["gen_sec"] = round(time.time() - t0, 2)
    row["out_tokens"] = int(generated.shape[1] - inputs["input_ids"].shape[1])

    row["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    row["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
    free_after, _ = torch.cuda.mem_get_info()
    row["free_after_gib"] = round(free_after / 2**30, 2)
    row["reserved_total_gib"] = round(torch.cuda.memory_reserved() / 2**30, 2)

    del inputs, generated, image, image_inputs
    torch.cuda.empty_cache()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen25_vl_7b_instruct")
    ap.add_argument("--images-root", default=str(ROOT / "data" / "padchest" / "images"))
    ap.add_argument("--images", nargs="*", default=DEFAULT_IMAGES)
    ap.add_argument("--rounds", type=int, default=2,
                    help="how many times to walk the image list (2 shows accumulation)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    print(f"torch {torch.__version__} | alloc_conf="
          f"{torch.cuda.get_allocator_backend() if hasattr(torch.cuda,'get_allocator_backend') else 'n/a'}",
          flush=True)

    model = get_model(args.model, root=ROOT, load_in_4bit=True)
    t0 = time.time()
    model.load()
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    processor = model.processor
    ensure_left_padding(processor)
    prompt = build_simple_prompt()

    rows = []
    for rnd in range(args.rounds):
        for name in args.images:
            path = Path(args.images_root) / name
            if not path.exists():
                print(f"[skip] missing {path}", flush=True)
                continue
            row = one_call(model.model, processor, prompt, path, args.max_new_tokens)
            row["round"] = rnd
            rows.append(row)
            print(json.dumps(row), flush=True)

    out = ROOT / "outputs" / "state" / "diag_vram_cliff.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "alloc_conf_env": __import__("os").environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
