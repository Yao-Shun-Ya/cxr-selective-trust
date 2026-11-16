"""Reproducible cost table for every scenario that has been quoted by hand.

Why this exists
---------------
Three numbers appeared in planning prose and could not be reproduced from the
driver's own cost model: the "621 h" all-layers synthetic-grid counterfactual, the
"309.9 h" seven-model NIH-only figure, and a "349 h" recomputation that disagreed
with both. Numbers that cannot be reproduced from code cannot go into a methods
section, so this script computes every scenario from ONE cost model and one
measured throughput table, and prints the arithmetic line by line.

The cost model is deliberately the same shape as the orchestrators':

    seconds = calls x seconds_per_call / batch_speedup x padding_factor
              + stage_startup + model_load

where:

* ``calls``             = images x cells_per_image for that stage
* ``seconds_per_call``  measured on this machine at the 256-token budget
                        (docs/44 sweeps, corrected; see the table below)
* ``batch_speedup``     throughput at the model's sweet spot divided by its
                        serial rate. The table's figures are ALREADY batched
                        sweet-spot rates, so this is 1.0 everywhere; it is kept
                        explicit so a future serial-only model can set it.
* ``padding_factor``    1.20, for batching differently sized radiographs
* ``stage_startup``     35 s per stage invocation (interpreter + imports)
* ``model_load``        measured load time for the slow checkpoints

Usage::

    .venv-vl\\Scripts\\python.exe scripts\\47_cost_table.py
    .venv-vl\\Scripts\\python.exe scripts\\47_cost_table.py --markdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Measured seconds per model CALL (one image + one prompt, 256 new tokens).
#
# Split per (model, stage) for the same reason the driver is: the 7-model x
# 100-image pre-run showed that applying ONE per-model constant to 4-bit shift,
# bf16 precision and the then-serial ablation stage was wrong by 27.6x in one
# direction and 0.33x in the other (docs/50 §2.5). The values are the median of
# the per-cell ``elapsed_sec`` the runners recorded during that pre-run.
STAGE_SECONDS_PER_CALL = {
    # NIH shift, 19-cell synthetic degradation grid
    ("medgemma_4b_it", "shift"): 0.613,
    # means: the plan multiplies by a call count, so total-wall-clock/cells is the
    # number it needs (these stages have a heavy tail -- see the driver's comment)
    ("medgemma_15_4b_it", "shift"): 2.344,
    ("qwen25_vl_7b_instruct", "shift"): 1.590,
    ("qwen3_vl_8b_instruct", "shift"): 1.455,
    ("internvl3_5_8b", "shift"): 5.722,
    # GREEDY decoding, batch 4: 0.477 s/cell over 96 cells measured; 0.60 keeps a
    # ~25% margin for the full 19-cell grid (was 2.920, which was BEAM SEARCH --
    # the shipped generation_config's num_beams=5, and the cause of the 13% parse
    # failures; see docs/50 sections 2.2.1/2.11).
    ("chexagent_8b", "shift"): 0.600,
    # NIH reliance, 4 conditions
    ("medgemma_4b_it", "reliance"): 0.613,
    ("medgemma_15_4b_it", "reliance"): 2.479,
    ("qwen25_vl_7b_instruct", "reliance"): 1.585,
    ("qwen3_vl_8b_instruct", "reliance"): 1.477,
    ("internvl3_5_8b", "reliance"): 5.724,
    # NIH precision, the bf16 control (same prompt as reliance, batch capped at 4)
    ("medgemma_4b_it", "precision"): 1.869,
    # 13.3 s/cell: at bf16 this model opens its thought channel on every reply, so
    # the budget has to be 2048 (see docs/50 section 2.11)
    ("medgemma_15_4b_it", "precision"): 13.305,
    ("qwen25_vl_7b_instruct", "precision"): 2.144,
    # NIH ablation, batched 2026-10-05 at bs=32 (was 5.7-9.9 s/cell serial; the
    # pre-run's stage average was 8.28). Mean over the five variants.
    ("medgemma_4b_it", "ablation"): 0.829,
}
# Per-LAYER measurements, empty on purpose: a rate measured on NIH is not a
# measurement of PadChest or Open-i, and the estimate says "proxy-nih" rather
# than pretending otherwise.
LAYER_STAGE_SECONDS_PER_CALL = {}
STAGE_ALIAS = {"native_shift": "shift", "report": "shift"}
SECONDS_PER_CALL = {
    "medgemma_4b_it": 0.61,
    "medgemma_15_4b_it": 0.85,
    "qwen25_vl_7b_instruct": 1.59,
    "qwen3_vl_8b_instruct": 1.46,
    "chexagent_8b": 0.60,
    "internvl3_5_8b": 5.72,
    "llava_med_v1_5_7b": 1.08,
}
MODEL_LOAD_SECONDS = {
    "chexagent_8b": 75,
    "internvl3_5_8b": 30,
    "llava_med_v1_5_7b": 220,
    "qwen3_vl_8b_instruct": 45,
}
BATCH_SPEEDUP = {m: 1.0 for m in SECONDS_PER_CALL}
# Allowance for what the measured rate does not contain: image decode, the
# per-batch append + fsync, and the fact that a batch costs what its LONGEST
# member costs. Batching itself is already inside the rate.
PADDING_FACTOR = 1.20
STAGE_STARTUP_SECONDS = 35


def rate_for(model: str, layer: str, stage: str) -> tuple[float, str]:
    """(seconds per call, where the number came from) for this exact stage."""
    key = (model, layer, stage)
    if key in LAYER_STAGE_SECONDS_PER_CALL:
        return LAYER_STAGE_SECONDS_PER_CALL[key], "measured"
    key = (model, stage)
    if key in STAGE_SECONDS_PER_CALL:
        return (STAGE_SECONDS_PER_CALL[key],
                "measured" if layer == "nih" else "proxy-nih")
    alias = STAGE_ALIAS.get(stage)
    if alias and (model, alias) in STAGE_SECONDS_PER_CALL:
        return STAGE_SECONDS_PER_CALL[(model, alias)], "proxy-stage"
    return SECONDS_PER_CALL[model], "default"


FIVE_MODELS = ["medgemma_4b_it", "medgemma_15_4b_it", "qwen25_vl_7b_instruct",
               "qwen3_vl_8b_instruct", "chexagent_8b"]
SEVEN_MODELS = FIVE_MODELS + ["internvl3_5_8b", "llava_med_v1_5_7b"]

LAYERS = {
    "nih": {"images": 1506, "cells": {"shift": 19, "reliance": 4, "precision": 4,
                                      "ablation": 10}},
    "padchest": {"images": 1500, "cells": {"native_shift": 2}},
    "openi": {"images": 947, "cells": {"report": 2}},
}
LAYER_ALIAS = {"native_shift": "shift", "report": "shift"}


def stage_seconds(model: str, layer: str, stage: str, calls: int,
                  load: bool = True) -> float:
    per_call, _ = rate_for(model, layer, LAYER_ALIAS.get(stage, stage))
    body = (calls * per_call / BATCH_SPEEDUP[model]) * PADDING_FACTOR
    extra = STAGE_STARTUP_SECONDS
    if load:
        extra += MODEL_LOAD_SECONDS.get(model, 0)
    return body + extra


def scenario(models, layers, ablation_models=("medgemma_4b_it",),
             precision_for_four_bit_only=True, label="") -> dict:
    """Total cost of one configuration, with a per-stage breakdown."""
    four_bit = {"medgemma_4b_it", "medgemma_15_4b_it", "qwen25_vl_7b_instruct"}
    rows = []
    for layer in layers:
        spec = LAYERS[layer]
        for stage, cells in spec["cells"].items():
            for m in models:
                if stage == "ablation" and m not in ablation_models:
                    continue
                if stage == "precision" and precision_for_four_bit_only and m not in four_bit:
                    continue
                calls = spec["images"] * cells
                _, src = rate_for(m, layer, LAYER_ALIAS.get(stage, stage))
                rows.append({
                    "layer": layer,
                    "model": m,
                    "stage": LAYER_ALIAS.get(stage, stage),
                    "cells_per_image": cells,
                    "images": spec["images"],
                    "calls": calls,
                    "rate_source": src,
                    "seconds": stage_seconds(m, layer, stage, calls),
                })
    total_calls = sum(r["calls"] for r in rows)
    total_seconds = sum(r["seconds"] for r in rows)
    return {
        "label": label,
        "models": list(models),
        "layers": list(layers),
        "stage_instances": len(rows),
        "calls": total_calls,
        "seconds": total_seconds,
        "hours": total_seconds / 3600,
        "days": total_seconds / 86400,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--out", default="outputs/state/cost_table.json")
    args = ap.parse_args()

    scenarios = [
        scenario(FIVE_MODELS, ["nih", "padchest", "openi"],
                 label="A. per-layer design, 5 models (the delivered default)"),
        scenario(SEVEN_MODELS, ["nih", "padchest", "openi"],
                 label="B. per-layer design, all 7 models"),
        scenario(FIVE_MODELS, ["nih"], label="C. NIH layer only, 5 models"),
        scenario(SEVEN_MODELS, ["nih"], label="D. NIH layer only, all 7 models"),
        # The counterfactual that the per-layer design avoids: apply the 19-cell
        # synthetic grid to every dataset, three times over.
        scenario(FIVE_MODELS, ["nih"], ablation_models=("medgemma_4b_it",),
                 label="_frag_nih5"),
        scenario(SEVEN_MODELS, ["nih"], label="_frag_nih7"),
    ]
    synthetic = []
    for layer, imgs in (("nih", 1506), ("padchest", 1500), ("openi", 947)):
        for models, tag in ((SEVEN_MODELS, "7"), (FIVE_MODELS, "5")):
            calls = imgs * 19 * len(models)
            secs = sum(stage_seconds(m, layer, "shift", imgs * 19) for m in models)
            synthetic.append({"layer": layer, "models": len(models), "tag": tag,
                              "calls": calls, "hours": secs / 3600})
    grid_7 = sum(s["hours"] for s in synthetic if s["tag"] == "7")
    grid_5 = sum(s["hours"] for s in synthetic if s["tag"] == "5")

    out = {
        "model": "seconds = calls x seconds_per_call / batch_speedup x padding "
                 "+ stage_startup + model_load",
        "seconds_per_call": SECONDS_PER_CALL,
        "stage_seconds_per_call": {f"{m}/{s}": v
                                   for (m, s), v in STAGE_SECONDS_PER_CALL.items()},
        "layer_stage_seconds_per_call": {f"{m}/{ly}/{s}": v
                                        for (m, ly, s), v
                                        in LAYER_STAGE_SECONDS_PER_CALL.items()},
        "padding_factor": PADDING_FACTOR,
        "stage_startup_seconds": STAGE_STARTUP_SECONDS,
        "model_load_seconds": MODEL_LOAD_SECONDS,
        "scenarios": [{k: v for k, v in s.items() if k != "rows"} for s in scenarios
                      if not s["label"].startswith("_frag")],
        "counterfactual_19cell_on_all_layers": {
            "all_7_models_hours": round(grid_7, 1),
            "five_models_hours": round(grid_5, 1),
            "per_layer": synthetic,
        },
    }

    print(f"{'scenario':52s} {'stages':>6} {'calls':>10} {'hours':>9} {'days':>6}")
    for s in scenarios:
        if s["label"].startswith("_frag"):
            continue
        print(f"{s['label']:52s} {s['stage_instances']:6d} {s['calls']:10,d} "
              f"{s['hours']:9.1f} {s['days']:6.1f}")
    print(f"\ncounterfactual: 19-cell grid on all 3 layers")
    print(f"  all 7 models : {grid_7:8.1f} h ({grid_7/24:.1f} days)")
    print(f"  five models  : {grid_5:8.1f} h ({grid_5/24:.1f} days)")
    print(f"\nper-layer design (all 5 models) vs that counterfactual: "
          f"{scenarios[0]['hours']:.1f} h vs {grid_5:.1f} h "
          f"-> saves {grid_5 - scenarios[0]['hours']:.1f} h")

    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {dest}")

    if args.markdown:
        print("\n| scenario | stages | calls | hours | days |")
        print("|---|---|---|---|---|")
        for s in scenarios:
            if s["label"].startswith("_frag"):
                continue
            print(f"| {s['label']} | {s['stage_instances']} | {s['calls']:,} | "
                  f"{s['hours']:.1f} | {s['days']:.1f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
