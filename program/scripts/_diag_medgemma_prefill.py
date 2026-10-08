"""Can MedGemma-1.5 be run WITHOUT its thinking channel, and what does that cost?

Context (scripts/61_measure_medgemma_think_budget.py, 8 images, bf16):

    budget   parse    s/cell
     256     0/8      2.94
     512     0/8      5.54
     768     6/8      8.22
    1024     8/8     12.50

So the model can be made to satisfy the JSON contract by giving it ~4x the token
budget every other model needs -- and it pays for it: 12.5 s/cell against 2.85
s/cell at 256. Over the precision stage alone (6,928 cells) that is 29 h instead
of 6 h.

The token layout is fixed: ``<unused94>`` (id 100) opens the thought channel and
``<unused95>`` (id 101) closes it, the answer follows. The training format renders
a turn that did NOT think as an empty, already-closed block, so priming that block
should start the model at the answer.

This probe measures that claim instead of assuming it:

    A  budget 256, no prefill        (the shipped path -- reproduces the failure)
    B  budget 1024, no prefill       (the expensive working configuration)
    C  prefill "<unused94>thought\\n<unused95>\\n", budget 256,
       cache_implementation="dynamic"
    D  the same prefill with the space the community template uses
       ("<unused94> thought\\n"), budget 256

C/D pass ``cache_implementation="dynamic"`` because appending tokens to the
processor's batch hits a causal-mask size mismatch under this checkpoint's
``"cache_implementation": "hybrid"`` (measured: "The size of tensor a (594) must
match the size of tensor b (599)"), which is a cache-allocation artefact rather
than a model objection.

Usage::

    .venv-vl\\Scripts\\python.exe scripts\\_diag_medgemma_prefill.py
    .venv-vl\\Scripts\\python.exe scripts\\_diag_medgemma_prefill.py --load-in-4bit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.cxr_trust.inference.prompts import (  # noqa: E402
    SIMPLE_SYSTEM_PROMPT,
    build_simple_prompt,
)
from src.cxr_trust.inference.schema import parse_vlm_output  # noqa: E402
from src.cxr_trust.models.registry import get_model  # noqa: E402

PREFILLS = {
    "C. prefill (no space)": "<unused94>thought\n<unused95>\n",
    "D. prefill (with space)": "<unused94> thought\n<unused95>\n",
}


def summarise(tag, raws, elapsed, n):
    ok = sum(1 for r in raws if parse_vlm_output(r).parse_success)
    lens = [len(r) for r in raws]
    print(f"\n--- {tag} ---")
    print(f"  parse {ok}/{n}   chars mean {sum(lens) // n} max {max(lens)}   "
          f"{elapsed / n:.2f} s/cell")
    for r in raws[:2]:
        print(f"    {r[:140]!r}")
        print(f"    ...{r[-120:]!r}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medgemma_15_4b_it")
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--skip-ab", action="store_true",
                    help="only run the prefill configurations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = pd.read_csv(ROOT / "data/manifests/nih_eval_1732.csv")
    manifest = manifest[manifest["split"].isin(["internal_test", "source_val"])]
    manifest = manifest.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    manifest = manifest.head(args.images)
    images = [Image.open(ROOT / "data/nih/images" / r["image_relpath"]).convert("RGB")
              for _, r in manifest.iterrows()]
    prompt = SIMPLE_SYSTEM_PROMPT + chr(10) * 2 + build_simple_prompt()

    print(f"model={args.model} 4bit={args.load_in_4bit} images={len(images)}")
    model = get_model(args.model, root=ROOT, device_map="auto",
                      load_in_4bit=args.load_in_4bit)
    model.load()
    tok = model.processor.tokenizer

    for tag, budget in (("A. shipped path, budget 256", 256),
                        ("B. shipped path, budget 1024", 1024)):
        if args.skip_ab:
            continue
        t0 = time.time()
        raws = model.generate_batch(images, [prompt] * len(images),
                                    max_new_tokens=budget, chunk=args.batch_size)
        summarise(tag, raws, time.time() - t0, len(images))

    for tag, prefill in PREFILLS.items():
        ids = tok(prefill, add_special_tokens=False)["input_ids"]
        inputs = model._prepare_batch(images, [prompt] * len(images))
        # EVERY 2-D input has to grow with the prompt, not just input_ids.
        # Gemma-3's processor also emits `token_type_ids` (it marks which positions
        # are image tokens), and extending only input_ids made the causal mask and
        # that tensor disagree: "The size of tensor a (594) must match the size of
        # tensor b (599)" -- which looks like a cache bug and is not one.
        base_len = inputs["input_ids"].shape[-1]
        device = next(model.model.parameters()).device
        for key in list(inputs):
            value = inputs[key]
            if hasattr(value, "dim") and value.dim() >= 2 and value.shape[-1] == base_len:
                fill = 1 if key == "attention_mask" else 0
                ext = torch.full((value.shape[0], len(ids)), fill,
                                 dtype=value.dtype, device=value.device)
                inputs[key] = torch.cat([value, ext], dim=-1)
                print(f"    extended {key}: {base_len} -> {inputs[key].shape[-1]}")
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]
        t0 = time.time()
        with torch.inference_mode():
            gen = model.model.generate(**inputs, max_new_tokens=256, do_sample=False,
                                       cache_implementation="dynamic")
        elapsed = time.time() - t0
        raws = model.processor.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)
        done = summarise(f"{tag}, budget 256", raws, elapsed, len(images))
        if done == 0:
            print("  (no JSON: this configuration is not usable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
