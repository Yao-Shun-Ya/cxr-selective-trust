#!/usr/bin/env python
"""How many tokens does MedGemma-1.5 need before it answers?

The pre-run recorded 400/400 parse failures on this model's ``precision`` stage
and pinned the mechanism down: every failure is a truncated ``<unused94>thought``
trace (146/146 on reliance, 191/191 on shift, 400/400 on precision), i.e. the
budget ran out *inside* the reasoning channel and the JSON contract was never
reached. So the fix is a token budget that is large enough to finish, and the
question this script answers is **how large, and what it costs**.

It measures the real wrapper path (``MedGemma``/``GenericVLM`` through
``generate_batch``) with the real stage prompt, at several budgets, and reports
per budget:

* the share of replies that reach a parseable JSON object,
* how often the thought channel is opened and whether it is ever closed,
* seconds per cell -- because a budget that costs 3x is a budget worth arguing
  about with the numbers in hand.

An entry whose ``parse_ok`` is below ``--require`` is a budget that still
truncates.

Usage::

    .venv-vl\\Scripts\\python.exe scripts\\61_measure_medgemma_think_budget.py
    .venv-vl\\Scripts\\python.exe scripts\\61_measure_medgemma_think_budget.py \\
        --model medgemma_15_4b_it --budgets 256 512 768 1024 1536
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.inference.prompts import (  # noqa: E402
    SIMPLE_SYSTEM_PROMPT,
    build_simple_prompt,
)
from src.cxr_trust.inference.schema import parse_vlm_output  # noqa: E402
from src.cxr_trust.models.registry import MODEL_NAMES, get_model  # noqa: E402

OPEN_THOUGHT = "<unused94>"
CLOSE_THOUGHT = "<unused95>"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="medgemma_15_4b_it", choices=MODEL_NAMES)
    ap.add_argument("--budgets", nargs="+", type=int, default=[256, 512, 768, 1024])
    ap.add_argument("--manifest", default="data/manifests/nih_eval_1732.csv")
    ap.add_argument("--split", default="internal_test,source_val")
    ap.add_argument("--images-root", default="data/nih/images")
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=0,
                    help="0 = the model's driver sweet spot for this stage")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--require", type=float, default=1.0,
                    help="parse-success share a budget must reach to be usable")
    ap.add_argument("--out", default="outputs/state/medgemma_think_budget.json")
    args = ap.parse_args()

    manifest = pd.read_csv(ROOT / args.manifest)
    if args.split and "split" in manifest.columns:
        wanted = [s.strip() for s in args.split.split(",") if s.strip()]
        manifest = manifest[manifest["split"].isin(wanted)].copy()
    manifest = manifest.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    manifest = manifest.head(args.images)
    images_root = ROOT / args.images_root
    images = [Image.open(images_root / row["image_relpath"]).convert("RGB")
              for _, row in manifest.iterrows()]
    prompt = SIMPLE_SYSTEM_PROMPT + chr(10) * 2 + build_simple_prompt()

    batch_size = args.batch_size or (4 if args.load_in_4bit else 4)
    print(f"model={args.model} 4bit={args.load_in_4bit} images={len(images)} "
          f"batch={batch_size} budgets={args.budgets}", flush=True)

    model = get_model(args.model, root=ROOT, device_map="auto",
                      load_in_4bit=args.load_in_4bit)
    model.load()
    tok = model.processor.tokenizer

    rows = []
    for budget in args.budgets:
        start = time.time()
        raws = model.generate_batch(images, [prompt] * len(images),
                                    max_new_tokens=budget, chunk=batch_size)
        elapsed = time.time() - start
        ok = thought = closed = 0
        chars, toks = [], []
        for raw in raws:
            if parse_vlm_output(raw).parse_success:
                ok += 1
            has_open = OPEN_THOUGHT in raw
            thought += int(has_open)
            closed += int(CLOSE_THOUGHT in raw)
            chars.append(len(raw))
            toks.append(len(tok(raw, add_special_tokens=False)["input_ids"]))
        row = {
            "budget": budget,
            "images": len(images),
            "parse_ok": ok,
            "parse_rate": ok / len(images),
            "thought_opened": thought,
            "thought_closed": closed,
            "chars_mean": int(statistics.mean(chars)),
            "chars_max": max(chars),
            "tokens_mean": int(statistics.mean(toks)),
            "tokens_max": max(toks),
            "elapsed_sec": round(elapsed, 2),
            "sec_per_cell": round(elapsed / len(images), 3),
            "truncated": sum(1 for t in toks if t >= budget),
        }
        rows.append(row)
        print(f"  budget={budget:5d}  parse {ok}/{len(images)}  "
              f"thought {thought}/{len(images)} (closed {closed})  "
              f"tokens mean {row['tokens_mean']:4d} max {row['tokens_max']:4d}  "
              f"{row['sec_per_cell']:.2f} s/cell", flush=True)

    usable = [r for r in rows if r["parse_rate"] >= args.require]
    print(f"\n{'budget':>7s} {'parse':>7s} {'opened':>7s} {'closed':>7s} "
          f"{'tok mean':>9s} {'tok max':>8s} {'s/cell':>7s} {'at cap':>7s}")
    for r in rows:
        print(f"{r['budget']:7d} {r['parse_ok']:3d}/{r['images']:<3d} "
              f"{r['thought_opened']:7d} {r['thought_closed']:7d} "
              f"{r['tokens_mean']:9d} {r['tokens_max']:8d} {r['sec_per_cell']:7.2f} "
              f"{r['truncated']:7d}")

    recommend = None
    if usable:
        # smallest budget that never truncated, with a 1.5x safety margin
        safe = [r for r in usable if r["truncated"] == 0]
        base = min(safe or usable, key=lambda r: r["budget"])
        recommend = int(round(base["tokens_max"] * 1.5 / 64) * 64)
        print(f"\nsmallest never-truncating budget: {base['budget']} "
              f"(max observed {base['tokens_max']} tokens) -> recommend {recommend}")
    else:
        print(f"\nno budget reached the {args.require:.0%} parse requirement")

    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "load_in_4bit": bool(args.load_in_4bit),
        "prompt": "SIMPLE_SYSTEM_PROMPT + build_simple_prompt() (the shift/reliance/precision prompt)",
        "prompt_sha": __import__("hashlib").sha256(prompt.encode()).hexdigest()[:16],
        "images": [str(r) for r in manifest["image_id"].tolist()],
        "rows": rows,
        "recommended_max_new_tokens": recommend,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
