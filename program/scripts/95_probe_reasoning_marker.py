#!/usr/bin/env python
"""95 - Probe whether a second model exposes a readable reasoning marker.

This is a feasibility gate, not an experiment.  It loads one model, runs a
handful of images with the SAME prompt the grid will use, and reports which
markers actually appear in the raw output:

  <think> / </think>   Qwen3-style reasoning channel
  <unused94>           MedGemma-1.5 native marker (reference only)
  "Reasoning:"         a plain-text scratchpad, if the model ignores the tags

If no marker appears, the grid should not be launched for that model: the
two-axis result would have no model-side channel to read.

How Qwen3-VL-Thinking differs from the prompted contract
--------------------------------------------------------
Its chat template PRE-FILLS the opening tag: with ``add_generation_prompt``
the prompt ends in ``<|im_start|>assistant\\n<think>\\n``.  The decoded reply
therefore begins with reasoning text and closes with ``</think>`` -- the
literal string ``<think>`` never appears in the output.  Detecting a
``<think>``/``</think>`` *pair* would report "no channel" for a model that
plainly has one, so the emitted close tag is the signal here, and the probe
records whether the opening tag was pre-filled in the prompt.

The probe deliberately uses the plain grid prompt rather than the prompted
``r0_think_json`` contract: asking the model to write ``<think>`` tags would
produce a tag we asked for, not the native channel we want to measure.

Usage:
  .\\.venv-modern\\Scripts\\python.exe scripts\\95_probe_reasoning_marker.py \
      --model qwen3_vl_8b_thinking --limit 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.inference.prompts import SIMPLE_SYSTEM_PROMPT, build_simple_prompt  # noqa: E402
from src.cxr_trust.inference.prompts_ablation import build_prompt  # noqa: E402
from src.cxr_trust.models.registry import MODEL_NAMES, get_model  # noqa: E402

MARKERS = ["<think>", "</think>", "<unused94>", "Reasoning:", "thought"]


def preamble_before_json(raw: str) -> str:
    """Text before the first JSON object; empty if the reply starts with JSON."""
    i = raw.find("{")
    return raw[:i].strip() if i >= 0 else raw.strip()


def reasoning_text(raw: str) -> str:
    """The emitted trace that ends at the first ``</think>``.

    Qwen's template already wrote the opening tag into the prompt, so the trace
    is simply everything the model produced before it closed the block.
    """
    i = raw.find("</think>")
    if i < 0:
        return ""
    text = raw[:i]
    for opener in ("<think>", "<think>\n"):
        if text.startswith(opener):
            text = text[len(opener):]
            break
    return text.strip()


def _variants(img):
    """(condition, severity, PIL image) probe set: clean, blank, noise, resolution."""
    import numpy as np
    from PIL import Image
    out = [("clean", "none", img), ("blank", "none",
                                    Image.new("RGB", img.size, (128, 128, 128)))]
    arr = np.asarray(img, dtype=np.float32) / 255.0
    rng = np.random.default_rng(0)
    noisy = np.clip(arr + rng.normal(0, 0.10, arr.shape), 0, 1)
    out.append(("noise", "severe", Image.fromarray((noisy * 255).astype(np.uint8))))
    w, h = img.size
    small = img.resize((max(1, int(w * 0.125)), max(1, int(h * 0.125))), Image.BILINEAR)
    out.append(("resolution", "severe", small.resize((w, h), Image.BILINEAR)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--manifest", default="data/manifests/nih_eval_1732.csv")
    ap.add_argument("--images-root", default="data/nih/images")
    ap.add_argument("--split", default="internal_test,source_val")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-memory", default=None,
                    help="optional GPU budget, e.g. 30GiB; forces no CPU offload")
    ap.add_argument("--prompt-variant", default=None,
                    help="optional prompts_ablation variant; default is the plain "
                         "grid prompt, so the probe measures the channel the grid "
                         "will actually see")
    ap.add_argument("--out", default="outputs/reasoning_probe")
    args = ap.parse_args()

    kwargs = {"device_map": "auto", "load_in_4bit": False}
    if args.max_memory:
        kwargs["max_memory"] = {0: args.max_memory, "cpu": "0GiB"}
    model = get_model(args.model, root=ROOT, **kwargs)
    model.load()
    if args.prompt_variant:
        system, user = build_prompt(args.prompt_variant)
        prompt = system + "\n\n" + user
    else:
        prompt = SIMPLE_SYSTEM_PROMPT + "\n\n" + build_simple_prompt()
    print(f"prompt: {args.prompt_variant or 'simple (grid default)'}", flush=True)

    man = pd.read_csv(ROOT / args.manifest, low_memory=False)
    if args.split:
        man = man[man["split"].isin([s.strip() for s in args.split.split(",")])]
    man = man.sample(frac=1.0, random_state=0).head(args.limit)
    images_root = ROOT / args.images_root

    records = []
    for _, row in man.iterrows():
        img = Image.open(images_root / row["image_relpath"]).convert("RGB")
        for condition, severity, variant_img in _variants(img):
            raw = model.generate(variant_img, prompt, max_new_tokens=args.max_new_tokens)
            pre = preamble_before_json(raw)
            trace = reasoning_text(raw)
            records.append({
                "image_id": str(row["image_id"]),
                "condition": condition,
                "severity": severity,
                "raw_output": raw,
                "markers": {m: raw.count(m) for m in MARKERS},
                "has_think_close": "</think>" in raw,
                "has_think_pair": "<think>" in raw and "</think>" in raw,
                "reasoning_chars": len(trace),
                "has_preamble": bool(pre),
                "preamble_chars": len(pre),
                "preamble_preview": pre[:200],
            })
            print(f"\n=== {row['image_id']} | {condition}/{severity} ===")
            print(raw[:900])

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    by_cond = {}
    for cond in ["clean", "blank", "noise", "resolution"]:
        rr = [r for r in records if r["condition"] == cond]
        by_cond[cond] = {
            "n": len(rr),
            "preamble_rate": round(sum(r["has_preamble"] for r in rr) / len(rr), 4) if rr else None,
            "mean_preamble_chars": round(
                sum(r["preamble_chars"] for r in rr) / len(rr), 1) if rr else None,
            "think_close_rate": round(sum(r["has_think_close"] for r in rr) / len(rr), 4) if rr else None,
            "think_pair_rate": round(sum(r["has_think_pair"] for r in rr) / len(rr), 4) if rr else None,
            "mean_reasoning_chars": round(
                sum(r["reasoning_chars"] for r in rr) / len(rr), 1) if rr else None,
        }

    payload = {
        "model": args.model,
        "prompt_variant": args.prompt_variant or "simple",
        "max_new_tokens": args.max_new_tokens,
        "n": len(records),
        "marker_counts": {m: sum(r["markers"][m] for r in records) for m in MARKERS},
        "n_with_think_close": sum(r["has_think_close"] for r in records),
        "n_with_think_pair": sum(r["has_think_pair"] for r in records),
        "n_with_preamble": sum(r["has_preamble"] for r in records),
        "mean_preamble_chars": round(
            sum(r["preamble_chars"] for r in records) / len(records), 1) if records else 0,
        "mean_reasoning_chars": round(
            sum(r["reasoning_chars"] for r in records) / len(records), 1) if records else 0,
        "by_condition": by_cond,
        "records": records,
    }
    (out / f"probe_{args.model}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps({k: payload[k] for k in
                             ["model", "n", "marker_counts", "n_with_think_close",
                              "n_with_think_pair", "n_with_preamble",
                              "mean_reasoning_chars", "by_condition"]},
                            indent=2))
    # A reasoning channel exists if the model closed a reasoning block in at
    # least one reply.  `<think>` alone would fail for Qwen3-VL-Thinking,
    # whose opener is pre-filled by the chat template (see module docstring).
    think_ok = payload["n_with_think_close"] > 0
    noise = by_cond.get("noise", {}).get("preamble_rate") or 0
    res = by_cond.get("resolution", {}).get("preamble_rate") or 0
    preamble_discriminates = (res - noise) >= 0.30
    print(f"\nchannel: think_close={payload['n_with_think_close']}/{payload['n']} "
          f"preamble_delta(res-noise)={res - noise:+.2f}")
    return 0 if (think_ok or preamble_discriminates) else 2


if __name__ == "__main__":
    raise SystemExit(main())
