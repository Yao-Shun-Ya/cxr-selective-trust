"""Why MedGemma-1.5 spends its whole token budget thinking, and what to do.

The pre-run recorded 400/400 parse failures on the ``precision`` (bf16) stage for
this model: every reply begins ``<unused94>thought`` and is 990-1189 characters
long at ``max_new_tokens=256``, i.e. the budget runs out *inside* the reasoning
trace and the JSON contract is never reached. The failure rate is stage-dependent
(precision 100%, reliance 36.5%, shift 10.1%), which is itself a finding: the same
model answers directly on the shorter prompts and thinks on the longer ones.

The token layout is fixed, not guessed:

    <unused94> = token id 100   (opens the thought channel)
    <unused95> = token id 101   (closes it; the answer follows)

So there are exactly two honest ways out, and this probe decides between them:

* **let it think, with a budget that is large enough to finish** (``max_new_tokens``
  >> the observed thought length), or
* **skip the channel** by prefilling an *empty, already-closed* thought block
  right after the generation prompt -- the same shape the training format uses
  for a finished turn, so the model starts at the answer.

Whichever wins has to hold for the *content* too, so each configuration is run
through the real parser and the parsed label count is printed rather than just
"it produced text".

Usage::

    .venv-vl\\Scripts\\python.exe scripts\\_diag_medgemma_think.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.cxr_trust.inference.prompts import (  # noqa: E402
    SIMPLE_SYSTEM_PROMPT,
    build_simple_prompt,
)
from src.cxr_trust.inference.schema import parse_vlm_output  # noqa: E402
from src.cxr_trust.models import registry  # noqa: E402

SYSTEM = "You are a careful chest X-ray interpretation assistant."
OPEN_THOUGHT = "<unused94>"
CLOSE_THOUGHT = "<unused95>"
# The model's own output form is `<unused94>thought\n` (no space); the community
# Ollama template re-renders it with a space, so both are measured.
PREFILLS = {
    "C. prefill empty thought (no space)": f"{OPEN_THOUGHT}thought\n{CLOSE_THOUGHT}\n",
    "D. prefill empty thought (with space)": f"{OPEN_THOUGHT} thought\n{CLOSE_THOUGHT}\n",
}


def make_inputs(processor, images, prompt):
    batch = [[
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "image", "image": img},
                                     {"type": "text", "text": prompt}]},
    ] for img in images]
    return processor.apply_chat_template(
        batch, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", padding=True,
    )


def report(tag, new_ids, tok, elapsed, n_cells):
    text = tok.decode(new_ids, skip_special_tokens=True)
    parsed = parse_vlm_output(text)
    n_findings = len((parsed.to_dict() or {}).get("findings") or {})
    print(f"\n--- {tag} ---")
    print(f"  generated tokens : {len(new_ids)}  ({elapsed:.1f}s for {n_cells} cells, "
          f"{elapsed / n_cells:.2f} s/cell)")
    print(f"  raw decodes to   : {text[:200]!r}")
    print(f"  ...tail          : {text[-160:]!r}")
    print(f"  parse_success    : {parsed.parse_success}  ({parsed.parse_error or ''}) "
          f"findings_parsed={n_findings}")
    return parsed.parse_success


def main() -> int:
    model = registry.get_model("medgemma_15_4b_it", root=ROOT, device_map="auto")
    model.load()
    processor = model.processor
    tok = processor.tokenizer
    prompt = SIMPLE_SYSTEM_PROMPT + chr(10) * 2 + build_simple_prompt()

    nih = ROOT / "data" / "nih" / "images" / "images"
    paths = sorted(nih.glob("*.png"))[:2]
    images = [Image.open(p).convert("RGB") for p in paths]
    print(f"images: {[p.name for p in paths]}")

    for bs in (100, 101):
        ids = tok.convert_ids_to_tokens(bs)
        print(f"  token {bs} = {ids!r}")

    results = {}

    # A / B: the shipped path, at the pre-run budget and at a budget big enough
    # for the thought trace to finish.
    for tag, budget in (("A. shipped path, max_new_tokens=256", 256),
                        ("B. shipped path, max_new_tokens=1024", 1024)):
        start = time.time()
        raws = model.generate_batch(images, [prompt] * len(images),
                                    max_new_tokens=budget)
        elapsed = time.time() - start
        print(f"\n--- {tag} ---")
        for raw in raws:
            parsed = parse_vlm_output(raw)
            print(f"  len={len(raw):5d} parse={parsed.parse_success} {raw[:90]!r}")
            if OPEN_THOUGHT in raw:
                tail = raw.split(CLOSE_THOUGHT, 1)
                print(f"    closes thought: {len(tail) > 1}")
        results[tag] = elapsed

    # C / D: prefill the empty thought block and start at the answer.
    for tag, prefill in PREFILLS.items():
        inputs = make_inputs(processor, images, prompt)
        pre = tok(prefill, add_special_tokens=False)["input_ids"]
        ids = inputs["input_ids"]
        pad = torch.tensor([pre], dtype=ids.dtype, device=ids.device).repeat(ids.shape[0], 1)
        inputs["input_ids"] = torch.cat([ids, pad], dim=-1)
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(pad)], dim=-1)
        inputs = {k: v.to(model.model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]
        start = time.time()
        with torch.inference_mode():
            gen = model.model.generate(**inputs, max_new_tokens=256, do_sample=False)
        elapsed = time.time() - start
        for row in range(gen.shape[0]):
            new_ids = gen[row][prompt_len:].tolist()
            report(f"{tag} [img {row}]", new_ids, tok, elapsed, gen.shape[0])
        results[tag] = elapsed

    print("\nsummary")
    for k, v in results.items():
        print(f"  {v:7.1f}s  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
