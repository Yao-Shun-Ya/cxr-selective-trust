"""Diagnose LLaVA-Med's empty replies at the token level.

The pre-run produced 1,890 empty replies out of 1,900 shift cells on this model
(mean 0.22 s/cell), so it is stopping almost immediately -- not "answering in
prose instead of JSON", which is what the earlier note assumed. The parser is
therefore NOT the suspect and must not be loosened.

The wrapper assembles the LLaVA-1.5 layout by hand:

    prefix("USER: ", add_special_tokens=False) + [<image>] * 576
        + suffix("\\n{prompt} ASSISTANT:", add_special_tokens=False)

The reference LLaVA implementation (``tokenizer_image_token``) keeps the token
that a plain ``tokenizer(chunk)`` call produces, and that call adds Mistral's
**BOS** (``add_bos_token=True``, so ``tokenizer("USER: ")`` is
``[1, 2223, 725, 28747, 28705]``, not ``[2223, ...]``). This probe asks whether
that missing ``<s>`` is why the model emits EOS at step 1, by running the same
image through four assemblies and printing the FIRST GENERATED TOKEN IDS, so an
immediate ``</s>`` is visible rather than inferred from an empty string.

Usage::

    .venv-vl\\Scripts\\python.exe scripts\\_diag_llava.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.cxr_trust.inference.prompts import (  # noqa: E402
    SIMPLE_SYSTEM_PROMPT,
    build_simple_prompt,
)
from src.cxr_trust.models import registry  # noqa: E402

OUT = ROOT / "outputs" / "diagnose_llava"


def decode_ids(tok, ids) -> str:
    return tok.decode(ids, skip_special_tokens=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    model = registry.get_model("llava_med_v1_5_7b", root=ROOT, device_map="auto")
    model.load()
    tok = model.tokenizer
    m = model.model
    prompt = SIMPLE_SYSTEM_PROMPT + chr(10) * 2 + build_simple_prompt()

    image_path = ROOT / "data" / "nih" / "images" / "images" / "00028873_008.png"
    if not image_path.exists():
        candidates = sorted((ROOT / "data" / "nih" / "images" / "images").glob("*.png"))
        image_path = candidates[0]
    print(f"image: {image_path}")
    pil = Image.open(image_path).convert("RGB")

    n = model.num_image_tokens
    prefix_nb = tok("USER: ", add_special_tokens=False)["input_ids"]
    prefix_b = tok("USER: ")["input_ids"]
    suffix_nb = tok(f"\n{prompt} ASSISTANT:", add_special_tokens=False)["input_ids"]
    bos = tok.bos_token_id
    img_id = model.image_token_id
    print(f"tokenizer adds BOS: {tok.add_bos_token} -> tokenizer('USER: ') = {prefix_b}")
    print(f"wrapper uses add_special_tokens=False -> {prefix_nb}")
    print(f"num_image_tokens={n} image_token_id={img_id}")

    def build(with_bos: bool):
        """Exactly the wrapper's sequence, optionally with Mistral's BOS."""
        ids = (([bos] if with_bos else []) + prefix_nb + [img_id] * n + suffix_nb)
        return torch.tensor([ids], dtype=torch.long, device=m.device)

    def run_inputs(input_ids, pixel_values, tag):
        input_ids = input_ids.to(m.device)
        attn = torch.ones_like(input_ids)
        moved = {"input_ids": input_ids, "attention_mask": attn,
                 "pixel_values": pixel_values.to(m.device, dtype=torch.bfloat16)}
        plen = input_ids.shape[-1]
        with torch.inference_mode():
            gen = m.generate(**moved, max_new_tokens=64, do_sample=False,
                             pad_token_id=model.pad_token_id or tok.eos_token_id)
        new = gen[0][plen:].tolist()
        print(f"\n--- {tag} ---")
        print(f"  prompt tokens     : {plen} (image positions: "
              f"{int((input_ids == img_id).sum())})")
        print(f"  first 12 new ids  : {new[:12]}")
        print(f"  decoded           : {decode_ids(tok, new)[:300]!r}")

    px = model.image_processor(images=pil, return_tensors="pt")["pixel_values"]

    run_inputs(build(with_bos=False), px, "A. wrapper as shipped (NO BOS)")
    run_inputs(build(with_bos=True), px, "B. same + BOS <s>")

    # C. the HF-blessed path: let LlavaProcessor tokenise and expand <image>
    try:
        enc = model.processor(text=f"USER: <image>\n{prompt} ASSISTANT:",
                              images=pil, return_tensors="pt")
        run_inputs(enc["input_ids"], enc["pixel_values"], "C. LlavaProcessor end to end")
    except Exception as exc:  # noqa: BLE001
        print(f"\n--- C. LlavaProcessor ---\n  FAILED: {type(exc).__name__}: {exc}")

    # D. BOS + image + question only (no system preamble), in case the preamble is
    #    what the model is echoing back.
    plain = build_simple_prompt()
    suffix_plain = tok(f"\n{plain} ASSISTANT:", add_special_tokens=False)["input_ids"]
    ids_plain = torch.tensor([[bos] + prefix_nb + [img_id] * n + suffix_plain],
                             dtype=torch.long, device=m.device)
    run_inputs(ids_plain, px, "D. BOS + question only (no system preamble)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
