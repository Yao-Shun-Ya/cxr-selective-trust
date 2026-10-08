#!/usr/bin/env python
"""Is CheXagent's parse failure a token-budget problem, or a format problem?

The retry run answered one question and raised another. The driver's quality
guard rejected three stages -- ``chexagent_8b`` shift and reliance, and
``medgemma_15_4b_it`` precision -- because more than 1% of their sampled rows
were unusable. For chexagent the reliance summary says the failures are NOT
uniform across conditions::

    parse_failures_by_condition:  original=11  blank=0  blur=29  shuffle=12
    generation_failures_by_condition: all 0

Two things follow from that table, and they point in opposite directions:

* ``blank`` never fails. So chexagent is not a model that cannot emit the JSON
  contract -- it honours it 100% of the time on a blank control. The failures are
  content-dependent, which is what a length/truncation story looks like.
* The failures are *condition-dependent*, and that is the dangerous part. Dropping
  them does not lose a random 13% of cells, it loses 29% of the ``blur`` cells and
  0% of the ``blank`` cells. A stage that is scored on the survivors would report a
  blur/shift effect computed on a biased subset. So the guard is right to reject
  the file; the fix is to make the cells parse, not to keep them.

This script decides which of the two causes it is, by re-running the SAME cells
(the pretest's own image subset, its own deterministic conditions, greedy decoding)
at several token budgets and reporting, per budget and per condition:

* the share of replies that reach a parseable JSON object,
* how many replies hit the token cap exactly -- a truncated reply is the direct
  signature of "the budget ran out mid-object",
* token length, so the budget can be chosen from data rather than guessed,
* seconds per cell, because a budget that costs 2x is a budget worth arguing about,
* a few failed raw tails, so the failure is visible rather than asserted.

Read the result like this:

* parse rate climbs with the budget and the cap-hits explain the failures
  -> truncation. Add the model to ``$MaxNewTokensOverride`` in run_all_layers.ps1
     and re-run only the rejected stages.
* parse rate is flat in the budget, failures do not hit the cap, and the raw tails
  are prose rather than cut-off JSON
  -> genuine format non-compliance, the same class as llava_med. Then the honest
     move is to run it outside the driver and report the parse rate and its
     condition-dependence as a finding (docs/42).

Usage::

    .venv-vl\\Scripts\\python.exe scripts\\67_measure_chexagent_parse_budget.py
    .venv-vl\\Scripts\\python.exe scripts\\67_measure_chexagent_parse_budget.py \\
        --model chexagent_8b --budgets 256 512 768 1024 --images 32
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.inference.prompts import (  # noqa: E402
    SIMPLE_SYSTEM_PROMPT,
    build_simple_prompt,
)
from src.cxr_trust.inference.schema import parse_vlm_output  # noqa: E402
from src.cxr_trust.models.registry import MODEL_NAMES, get_model  # noqa: E402


def _load_reliance_module():
    """Import scripts/06_image_reliance.py so the conditions are the REAL ones.

    The whole point is to reproduce the pretest cells, so the blank/blur controls
    must come from the same code that built them there. Re-implementing
    ``make_conditions`` here would let the two drift, and a drifted control would
    silently make this measurement answer a different question. The module name
    starts with a digit, so it is loaded by path.
    """
    path = ROOT / "scripts" / "06_image_reliance.py"
    spec = importlib.util.spec_from_file_location("_reliance_06", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="chexagent_8b", choices=MODEL_NAMES)
    ap.add_argument("--budgets", nargs="+", type=int, default=[256, 512, 768, 1024])
    ap.add_argument("--conditions", nargs="+", default=["original", "blank", "blur"])
    ap.add_argument("--manifest", default="data/manifests/nih_eval_1732.csv")
    ap.add_argument("--split", default="internal_test,source_val")
    ap.add_argument("--images-root", default="data/nih/images")
    ap.add_argument("--images", type=int, default=32,
                    help="subset of the pretest's 100 images (same seed, same head)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-beams", type=int, default=None,
                    help="override the wrapper's generation_config.num_beams. "
                         "1 = greedy, which is what the study protocol says every "
                         "model uses -- but chexagent's shipped generation_config "
                         "says num_beams=5, i.e. beam search, and the legacy wrapper "
                         "passes that config straight through. Left at None the "
                         "measurement reproduces the stage exactly as it ran.")
    ap.add_argument("--samples", type=int, default=3,
                    help="failed raw tails to print per budget for eyeballing")
    ap.add_argument("--tail-chars", type=int, default=160)
    ap.add_argument("--out", default="outputs/state/chexagent_parse_budget.json")
    args = ap.parse_args()

    # ---- image subset: reproduce the pretest selection exactly --------------
    # 06/08 select with: split filter -> sample(frac=1, random_state=seed) -> head(N).
    # The pretest used seed=0 and head(100); taking head(32) of the same shuffle is a
    # strict subset of the pretest cells, so a failure here is a failure the guard saw.
    manifest = pd.read_csv(ROOT / args.manifest)
    if args.split and "split" in manifest.columns:
        wanted = [s.strip() for s in args.split.split(",") if s.strip()]
        manifest = manifest[manifest["split"].isin(wanted)].copy()
    manifest = manifest.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    manifest = manifest.head(args.images).copy()

    scratch = ROOT / "outputs" / "state" / "_chexagent_budget_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    r06 = _load_reliance_module()
    records = r06.make_conditions(manifest, ROOT / args.images_root, scratch, seed=args.seed)

    prompt = SIMPLE_SYSTEM_PROMPT + chr(10) * 2 + build_simple_prompt()
    print(f"model={args.model} images={len(records)} conditions={args.conditions} "
          f"budgets={args.budgets} batch={args.batch_size}", flush=True)
    print(f"image_ids: {[r['image_id'] for r in records]}", flush=True)

    model = get_model(args.model, root=ROOT, device_map="auto")
    model.load()
    tok = model.processor.tokenizer

    if args.num_beams is not None:
        gc = getattr(model, "generation_config", None)
        if gc is None:
            print(f"[override] {args.model} carries no generation_config; "
                  f"--num-beams has no effect on this wrapper", flush=True)
        else:
            gc.num_beams = args.num_beams
            gc.do_sample = False
            print(f"[override] generation_config.num_beams -> {gc.num_beams} "
                  f"(do_sample={gc.do_sample})", flush=True)

    def run_condition(condition: str, budget: int):
        imgs = []
        from PIL import Image
        for rec in records:
            p = rec["conditions"].get(condition)
            if p is None:
                continue
            imgs.append(Image.open(p).convert("RGB"))
        raws = model.generate_batch(imgs, [prompt] * len(imgs),
                                    max_new_tokens=budget, chunk=args.batch_size)
        return raws

    rows = []
    failures_seen: dict[int, list[str]] = {}
    for budget in args.budgets:
        failures_seen[budget] = []
        for condition in args.conditions:
            start = time.time()
            raws = run_condition(condition, budget)
            elapsed = time.time() - start

            n = len(raws)
            ok = 0
            at_cap = 0
            toks = []
            for raw in raws:
                parsed = parse_vlm_output(raw)
                if parsed.parse_success:
                    ok += 1
                nt = len(tok(raw, add_special_tokens=False)["input_ids"])
                toks.append(nt)
                if nt >= budget:
                    at_cap += 1
                if not parsed.parse_success and len(failures_seen[budget]) < args.samples:
                    tail = raw.strip().replace(chr(10), " ")[-args.tail_chars:]
                    failures_seen[budget].append(
                        f"[{condition} {nt}tok{' CAP' if nt >= budget else ''}] ...{tail}")

            row = {
                "budget": budget,
                "condition": condition,
                "cells": n,
                "parse_ok": ok,
                "parse_rate": round(ok / n, 4) if n else 0.0,
                "parse_fail": n - ok,
                "at_cap": at_cap,
                "tok_mean": int(statistics.mean(toks)) if toks else 0,
                "tok_max": max(toks) if toks else 0,
                "elapsed_sec": round(elapsed, 2),
                "sec_per_cell": round(elapsed / n, 3) if n else 0.0,
            }
            rows.append(row)
            print(f"  budget={budget:5d} {condition:9s} parse {ok:3d}/{n:<3d} "
                  f"at_cap {at_cap:3d}  tok mean {row['tok_mean']:4d} max {row['tok_max']:4d}"
                  f"  {row['sec_per_cell']:.2f} s/cell", flush=True)

    # ---- per-budget aggregate ----------------------------------------------
    agg = []
    for budget in args.budgets:
        rs = [r for r in rows if r["budget"] == budget]
        cells = sum(r["cells"] for r in rs)
        ok = sum(r["parse_ok"] for r in rs)
        cap = sum(r["at_cap"] for r in rs)
        sec = sum(r["elapsed_sec"] for r in rs)
        agg.append({
            "budget": budget, "cells": cells, "parse_ok": ok,
            "parse_rate": round(ok / cells, 4) if cells else 0.0,
            "parse_fail": cells - ok, "at_cap": cap,
            "sec_per_cell": round(sec / cells, 3) if cells else 0.0,
        })

    print(f"\n{'budget':>7s} {'parse':>9s} {'rate':>7s} {'at_cap':>7s} {'s/cell':>7s}")
    for a in agg:
        print(f"{a['budget']:7d} {a['parse_ok']:4d}/{a['cells']:<4d} "
              f"{a['parse_rate']:7.3f} {a['at_cap']:7d} {a['sec_per_cell']:7.2f}")

    print("\nfailed raw tails (are they cut-off JSON, or prose?):")
    for budget in args.budgets:
        for s in failures_seen[budget]:
            print(f"  {budget}: {s}")

    # ---- verdict -----------------------------------------------------------
    # Truncation looks like: parse rate rising with the budget AND failures that hit
    # the cap. If neither holds, a bigger budget is not the fix.
    first, last = agg[0], agg[-1]
    rising = last["parse_rate"] - first["parse_rate"]
    cap_explains = first["at_cap"] >= first["parse_fail"]
    verdict = "truncation" if (rising > 0.02 and cap_explains) else "unclear"
    if rising <= 0.02:
        verdict = "not_budget_limited"
    print(f"\nparse rate {first['budget']} -> {last['budget']}: "
          f"{first['parse_rate']:.3f} -> {last['parse_rate']:.3f} "
          f"(delta {rising:+.3f}); at-cap at {first['budget']} = {first['at_cap']}, "
          f"failures = {first['parse_fail']}")
    print(f"VERDICT: {verdict}")

    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "prompt": "SIMPLE_SYSTEM_PROMPT + build_simple_prompt() (the shift/reliance prompt)",
        "selection": f"split={args.split} seed={args.seed} head={args.images} "
                     "(subset of the pretest's 100 images)",
        "image_ids": [r["image_id"] for r in records],
        "conditions": args.conditions,
        "budgets": args.budgets,
        "rows": rows,
        "aggregate": agg,
        "failed_raw_tails": failures_seen,
        "verdict": verdict,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {dest}")

    shutil.rmtree(scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())