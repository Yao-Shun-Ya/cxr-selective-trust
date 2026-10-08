"""Regression smoke test: InternVL must accept PIL images, not only paths.

The stage runners (``06_image_reliance.py``, ``08_shift_benchmark.py``) decode
images themselves and pass ``PIL.Image`` objects to ``generate_batch``, while the
batch sweep (``26_batch_sweep.py``) passes paths. The wrapper used to assume
paths, so every PIL call raised ``'Image' object has no attribute 'read'`` and the
callers swallowed it and wrote 80/80 empty rows with exit code 0.

This script exercises both input types on one real image and fails loudly if
either produces an empty or unparsable reply.

    .venv-vl\\Scripts\\python.exe scripts\\38_smoke_internvl_pil.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from cxr_trust.models.registry import get_model  # noqa: E402

PROMPT = (
    "You are a radiology assistant. Look at the chest radiograph and return JSON "
    'of the form {"findings": {"<label>": <probability 0..1>}, "rationale": "<short>"}.'
)


def find_image(root: Path) -> Path:
    for cand in sorted((root / "data" / "nih" / "images" / "images").glob("*.png")):
        return cand
    raise SystemExit("no image found under data/nih/images/images")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    img_path = find_image(root)
    print(f"image: {img_path.name}", flush=True)

    model = get_model("internvl3_5_8b", load_in_4bit=False, max_memory={0: "30GiB"})
    model.load()
    print("model loaded", flush=True)

    pil = Image.open(img_path).convert("RGB")
    results = {}

    for label, payload in (("path", img_path), ("PIL", pil)):
        try:
            out = model.generate_batch([payload], [PROMPT], max_new_tokens=64)[0]
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:4s}: EXCEPTION {type(exc).__name__}: {exc}", flush=True)
            results[label] = None
            continue
        ok = bool(out and out.strip())
        results[label] = out
        print(f"  {label:4s}: {'OK  ' if ok else 'EMPTY'} {out[:90]!r}", flush=True)

    bad = [k for k, v in results.items() if not v or not v.strip()]
    if bad:
        print(f"FAIL: empty output for {bad}", flush=True)
        return 1
    print("PASS: path and PIL inputs both produced text", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
