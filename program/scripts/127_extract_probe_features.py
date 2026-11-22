"""Extract frozen vision-tower embeddings for the readout control (probe).

The probe asks a different question from the prompted readout: does the image
representation itself carry the label? We therefore run only the vision tower,
pool its token states, and never touch the language model.

    python scripts/127_extract_probe_features.py --model medgemma_15_4b_it

Output: ``outputs/probe/<model>/features.jsonl`` with one row per image:
``{"image_id": ..., "dim": ..., "embedding_b64": ...}`` (float32, base64).
The file is append-only and resumable by image_id.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.models.registry import get_model  # noqa: E402


def _pool(out) -> torch.Tensor:
    """Pool a vision-tower output to one vector per image."""
    if isinstance(out, (tuple, list)):
        out = out[0]
    if hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state
    if not torch.is_tensor(out):
        raise TypeError(f"unexpected vision output {type(out)!r}")
    if out.dim() == 3:
        out = out.reshape(-1, out.shape[-1])
    return out.float().mean(dim=0)


def _find_vision(root):
    """Locate the vision tower on a conditional-generation model.

    Gemma3 and Qwen3 both nest it one level down (``root.model.vision_tower`` /
    ``root.model.visual``), while InternVL exposes ``vision_model`` directly.
    """
    for obj in (root, getattr(root, "model", None)):
        if obj is None:
            continue
        for name in ("vision_tower", "visual", "vision_model"):
            if hasattr(obj, name):
                return getattr(obj, name)
    return None


def _extract(model, image: Image.Image) -> np.ndarray:
    """Exact per-family path: reuse the wrapper's own image preprocessing."""
    if hasattr(model, "_tile") and hasattr(model.model, "vision_model"):
        tiles, _ = model._tile(image)
        vision = model.model.vision_model
        dev = next(vision.parameters()).device
        dtype = next(vision.parameters()).dtype
        with torch.inference_mode():
            out = vision(pixel_values=tiles.to(device=dev, dtype=dtype))
        return _pool(out).cpu().numpy().astype(np.float32)

    inputs = model._prepare_batch([image], ["Describe this image."])
    vision = _find_vision(model.model)
    if vision is None:
        raise AttributeError(f"no vision tower on {type(model.model).__name__}")
    dev = next(vision.parameters()).device
    dtype = next(vision.parameters()).dtype
    pv = inputs["pixel_values"]
    if isinstance(pv, (list, tuple)):
        pv = torch.stack(list(pv))
    grid = inputs.get("image_grid_thw")
    # Qwen3 exposes the whole image-feature path (patch embed + vision tower +
    # spatial merge) on the inner model; its vision tower's first argument is
    # named ``hidden_states`` and does not accept ``pixel_values=``.
    inner = getattr(model.model, "model", model.model)
    if grid is not None and hasattr(inner, "get_image_features"):
        with torch.inference_mode():
            out = inner.get_image_features(pixel_values=pv.to(dev),
                                           image_grid_thw=grid.to(dev))
        pooled = getattr(out, "pooler_output", out)
        if isinstance(pooled, (tuple, list)):
            pooled = pooled[0]
        return _pool(pooled).cpu().numpy().astype(np.float32)
    kwargs = {}
    if grid is not None:
        kwargs["grid_thw"] = grid.to(dev)
    with torch.inference_mode():
        if kwargs:
            out = vision(pv.to(device=dev, dtype=dtype), **kwargs)
        else:
            out = vision(pixel_values=pv.to(device=dev, dtype=dtype))
    return _pool(out).cpu().numpy().astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["medgemma_15_4b_it", "qwen3_vl_8b_instruct",
                             "internvl3_5_8b"])
    ap.add_argument("--manifest",
                    default=str(ROOT / "data" / "manifests" / "nih_eval_1506.csv"))
    ap.add_argument("--images-root", default=str(ROOT / "data" / "nih" / "images"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-memory-gb", type=float, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else ROOT / "outputs" / "probe" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["image_id"])

    manifest = pd.read_csv(args.manifest)
    if "split" in manifest.columns:
        manifest = manifest[manifest["split"].isin(["internal_test", "source_val"])]
    if args.limit:
        manifest = manifest.head(args.limit)
    todo = manifest[~manifest["image_id"].astype(str).isin(done)]
    print(f"{args.model}: {len(done)} done, {len(todo)} to extract", flush=True)
    if todo.empty:
        return 0

    kwargs = {}
    if args.max_memory_gb:
        kwargs["max_memory"] = {0: f"{args.max_memory_gb:g}GiB", "cpu": "0GiB"}
    model = get_model(args.model, root=ROOT, device_map="auto", **kwargs)
    model.load()
    print(f"loaded {args.model}", flush=True)

    images_root = Path(args.images_root)
    start = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(todo.itertuples(index=False), 1):
            image = Image.open(images_root / row.image_relpath).convert("RGB")
            emb = _extract(model, image)
            fh.write(json.dumps({
                "image_id": str(row.image_id),
                "dim": int(emb.size),
                "embedding_b64": base64.b64encode(emb.tobytes()).decode("ascii"),
            }) + "\n")
            fh.flush()
            if i % 25 == 0:
                rate = i / (time.time() - start)
                print(f"[{i}/{len(todo)}] {rate:.2f} img/s", flush=True)
    print(f"done; {len(todo)} embeddings in {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
