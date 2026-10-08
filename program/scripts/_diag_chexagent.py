"""Verify the CheXagent-8b device-map fix on the card, before any long run.

The pre-run crashed this model twice at the same 16-minute mark with
``CUDA error: unknown error``, after ``accelerate`` had split it between the GPU
and the CPU. Root cause (docs/50 §2.2): the checkpoint is **F32 on disk**
(31.15 GB) while ``chexagent.py`` asks for fp16, and ``device_map="auto"``
estimates from the on-disk metadata, not from the requested dtype.

The fix is ``device_map={"": 0}`` when the model is unquantised, which is one
line and does not touch precision. This script proves it three ways rather than
by "the run did not crash":

1. **Placement** -- ``hf_device_map`` must be ``{'': 0}`` and no parameter may be
   on the ``meta`` device. That is the exact failure the warnings described
   (``Some parameters are on the meta device because they were offloaded to the
   cpu``).
2. **Footprint** -- resident GPU memory must be ~15.6 GiB, i.e. fp16. ~31 GiB
   would mean the fp32 estimate won again.
3. **Speed** -- one real forward pass with the real prompt, timed. The offloaded
   configuration measured 38.69 -> 118.05 s/img; a correctly loaded fp16 model on
   this card is a few seconds.

Usage::

    .venv-vl\\Scripts\\python.exe scripts\\_diag_chexagent.py
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
from src.cxr_trust.models import registry  # noqa: E402


def main() -> int:
    free, total = torch.cuda.mem_get_info()
    print(f"gpu before load: {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB")

    model = registry.get_model("chexagent_8b", root=ROOT, device_map="auto")
    t0 = time.time()
    model.load()
    load_s = time.time() - t0

    resident = torch.cuda.memory_allocated() / 2**30
    hf_map = getattr(model.model, "hf_device_map", None)
    meta = [n for n, p in model.model.named_parameters() if p.device.type == "meta"]
    cpu = [n for n, p in model.model.named_parameters() if p.device.type == "cpu"]

    print(f"\nload time        : {load_s:.0f}s")
    print(f"hf_device_map    : {hf_map}")
    print(f"resident on GPU  : {resident:.2f} GiB  (fp16 expectation ~15.6)")
    print(f"meta params      : {len(meta)}")
    print(f"cpu params       : {len(cpu)}")
    ok_placement = (not meta) and (not cpu)
    ok_footprint = 12.0 < resident < 20.0
    print(f"placement ok     : {ok_placement}")
    print(f"footprint ok     : {ok_footprint}")

    img = sorted((ROOT / "data" / "nih" / "images" / "images").glob("*.png"))[0]
    prompt = SIMPLE_SYSTEM_PROMPT + chr(10) * 2 + build_simple_prompt()
    pil = Image.open(img).convert("RGB")
    print(f"\nforward pass on {img.name} ...")
    t1 = time.time()
    raw = model.generate(pil, prompt, max_new_tokens=256)
    dt = time.time() - t1
    print(f"one image        : {dt:.2f}s  (pre-run offloaded: 38.69 -> 118.05s)")
    print(f"reply            : {raw[:200]!r}")

    verdict = ok_placement and ok_footprint and dt < 15
    print(f"\nVERDICT: {'PASS' if verdict else 'FAIL'} -- "
          f"placement={ok_placement} footprint={ok_footprint} speed={dt < 15}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
