#!/usr/bin/env python
"""Run one open-source VLM on a manifest.

Example:
  python scripts/03_run_model.py \
    --model qwen25_vl_7b_instruct \
    --manifest data/manifests/nih_pilot_1000.csv \
    --images-root data/nih/images \
    --output-dir outputs/pilot/qwen_nih \
    --limit 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.inference.prompts import SIMPLE_SYSTEM_PROMPT, build_simple_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["chexagent_8b", "medgemma_4b_it", "qwen25_vl_7b_instruct"])
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip image_ids already present in raw_outputs.jsonl")
    args = parser.parse_args()

    try:
        import torch
        from src.cxr_trust.inference.runner import run_manifest
        from src.cxr_trust.models.registry import get_model
    except Exception as exc:
        print("Model environment is not ready.")
        print(f"Import error: {exc}")
        print("Please create the recommended environment from program/env/requirements-vl.txt.")
        raise SystemExit(2)

    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text(encoding="utf-8"))
    spec = cfg["primary_models"][args.model]
    model_path = ROOT / spec["path"]
    if not model_path.exists():
        raise SystemExit(f"Model path not found: {model_path}")

    print(f"Loading {args.model} from {model_path}")
    model = get_model(
        args.model,
        model_path=model_path,
        device_map=spec.get("device_map", "auto"),
        load_in_4bit=args.load_in_4bit or spec.get("load_in_4bit", False),
        max_memory=spec.get("max_memory"),
    )
    model.load()
    prompt = SIMPLE_SYSTEM_PROMPT + chr(10) * 2 + build_simple_prompt()
    summary = run_manifest(
        model=model,
        manifest_csv=args.manifest,
        images_root=args.images_root,
        output_dir=args.output_dir,
        prompt=prompt,
        limit=args.limit,
        resume=args.resume,
        max_new_tokens=args.max_new_tokens,
    )
    print(summary)


if __name__ == "__main__":
    main()
