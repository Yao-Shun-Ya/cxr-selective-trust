#!/usr/bin/env python
"""Batch-size sweet-spot sweep across the model matrix.

Purpose
-------
On a fixed memory budget, throughput is **not** monotone in batch size. Past a
point each extra sample costs VRAM without buying speed, and once the WDDM
driver begins oversubscribing dedicated VRAM the throughput actually *drops*.
This tool measures the curve per model and records the chosen operating point,
so the paper's supplementary "engineering insights" rest on measurements rather
than on a guess.

Design choices that matter
--------------------------
* **One model in memory at a time.** Loads happen sequentially and the previous
  model is freed before the next load; anything else makes the VRAM numbers
  meaningless.
* **Two memory readings.** ``torch.cuda.max_memory_allocated`` is what the
  allocator reserved; the NVML/WDDM figure reported by ``nvidia-smi`` is what
  the *driver* counts, and the gap between them is where oversubscription hides.
  Both are recorded.
* **A speed decline is the signal, not the memory number.** A batch size is
  rejected when per-image time rises materially versus the previous step,
  regardless of whether an OOM was raised.
* **System-RAM guard.** Oversubscription pages VRAM into host RAM. If free host
  RAM falls below ``--min-free-ram-gb`` the sweep stops, because that is the
  condition that takes the machine down.
* **Real images, real prompt.** Timings come from actual chest radiographs and
  the project's own prompt, so image-token counts match what the experiments
  will really pay.

Results are appended to a persistent JSON so knowledge accumulates across
sessions instead of being rediscovered.

Usage
-----
::

    # one model, explicit ladder
    python scripts/26_batch_sweep.py --model medgemma_4b_it --precision 4bit \
        --batch-sizes 8 16 32 64 96 --limit-per-batch 24

    # every model, auto ladder derived from a predicted sweet spot
    python scripts/26_batch_sweep.py --all --auto-ladder

    # show what would run, load nothing
    python scripts/26_batch_sweep.py --all --auto-ladder --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# predicted sweet spots -- hypotheses to be confirmed or falsified, not truths
# --------------------------------------------------------------------------- #
# ref_batch / ref_vram_gb come from the 2026-10 measurements recorded in
# docs/43; they are used to derive an auto ladder, never as final answers.
PREDICTIONS = {
    "medgemma_4b_it":            {"params_b": 4.0,  "bf16_gb": 8.0,  "ref_bs": 64, "ref_gb": 22.3, "precisions": ["4bit", "bf16"], "pred_bs": {"4bit": 128, "bf16": 64}},
    "medgemma_15_4b_it":         {"params_b": 4.0,  "bf16_gb": 8.0,  "ref_bs": 64, "ref_gb": 22.1, "precisions": ["4bit", "bf16"], "pred_bs": {"4bit": 128, "bf16": 64}},
    "chexagent_8b":              {"params_b": 8.0,  "bf16_gb": 16.0, "ref_bs": 8,  "ref_gb": 16.6, "precisions": ["bf16"],        "pred_bs": {"bf16": 16}},
    "chexagent_2_3b":            {"params_b": 2.3,  "bf16_gb": 6.0,  "ref_bs": None, "ref_gb": None, "precisions": ["bf16"],      "pred_bs": {"bf16": 32}},
    "qwen25_vl_7b_instruct":     {"params_b": 7.0,  "bf16_gb": 15.5, "ref_bs": 8,  "ref_gb": 15.8, "precisions": ["bf16", "4bit"], "pred_bs": {"bf16": 8, "4bit": 32}},
    "qwen3_vl_8b_instruct":      {"params_b": 8.0,  "bf16_gb": 16.0, "ref_bs": 8,  "ref_gb": 27.2, "precisions": ["bf16", "4bit"], "pred_bs": {"bf16": 8, "4bit": 16}},
    "qwen3_vl_30b_a3b_instruct": {"params_b": 30.0, "bf16_gb": 58.0, "ref_bs": None, "ref_gb": None, "precisions": ["4bit"],      "pred_bs": {"4bit": 2}},
    "internvl3_5_8b":            {"params_b": 8.0,  "bf16_gb": 16.0, "ref_bs": None, "ref_gb": None, "precisions": ["bf16", "4bit"], "pred_bs": {"bf16": 8, "4bit": 16}},
    "llava_med_v1_5_7b":         {"params_b": 7.0,  "bf16_gb": 14.0, "ref_bs": None, "ref_gb": None, "precisions": ["bf16"],      "pred_bs": {"bf16": 16}},
}

# models the registry can instantiate today
SUPPORTED = [
    "medgemma_4b_it", "medgemma_15_4b_it", "qwen25_vl_7b_instruct",
    "qwen3_vl_8b_instruct", "qwen3_vl_30b_a3b_instruct", "internvl3_5_8b",
    "llava_med_v1_5_7b", "chexagent_8b", "chexagent_2_3b",
]

# a real prompt, so image-token counts are representative
BENCH_PROMPT = (
    "Assess these chest X-ray findings: atelectasis, cardiomegaly, consolidation, "
    "edema, pleural_effusion, pneumonia, pneumothorax, no_finding. Return exactly "
    'one JSON object: {"findings": {"<label>": "positive|negative|uncertain"}, '
    '"probability_positive": {"<label>": 0.0}}. No markdown, no text outside JSON.'
)


# --------------------------------------------------------------------------- #
# memory helpers
# --------------------------------------------------------------------------- #
def vram_smi_mb():
    """Driver-side dedicated VRAM via NVML; None when unavailable."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)
        return info.used / 1024 ** 2
    except Exception:
        return None


def free_ram_gb():
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        return float("nan")


def torch_peak_gb():
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def reset_torch_peak():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
# model loading / freeing
# --------------------------------------------------------------------------- #
def load_model(name: str, precision: str, gpu_only: bool = False,
               vram_budget_gib: int = 30):
    """Load a model for timing.

    The memory budget is the important part, and it is a *safety* mechanism, not
    a bookkeeping one:

    * ``BaseVLM._device_map_for`` returns ``{"": 0}`` only when a quantisation
      config is present. Without one it passes ``device_map="auto"``, and ``auto``
      is free to offload layers to CPU (or even disk). An offloaded model still
      runs, so the sweep would happily report a *number* — just not one comparable
      with a fully-resident run, and with its weights hidden from
      ``torch.cuda.max_memory_allocated``.
    * Passing an explicit ``max_memory`` that leaves the CPU no room turns a silent
      offload into a loud failure.

    ``vram_budget_gib`` should be a little below physical size so that the
    allocator raises before the Windows driver starts paging VRAM into host RAM.
    Once paging starts, throughput collapses (measured: 7 → 117 s/img on
    CheXagent-8B) and the machine becomes unresponsive, which is exactly what a
    sweep must avoid rather than measure.
    """
    from src.cxr_trust.models.registry import get_model
    import torch

    kwargs = dict(device_map="auto", max_memory=None, load_in_4bit=(precision == "4bit"))
    if gpu_only:
        kwargs["max_memory"] = {0: f"{vram_budget_gib}GiB", "cpu": "0GiB"}
    t0 = time.time()
    model = get_model(name, root=ROOT, **kwargs)
    model.load()
    return model, time.time() - t0


def free_model(model):
    try:
        del model
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# the sweep itself
# --------------------------------------------------------------------------- #
def sweep(model, name: str, precision: str, batch_sizes, images, prompt,
          max_new_tokens: int, repeats: int, min_free_ram_gb: float,
          slowdown_tolerance: float, verbose: bool = True,
          vram_budget_gib: int = 30, log_file: Path | None = None) -> dict:
    """Measure one (model, precision). Returns a result dict.

    Two independent stop conditions protect the machine rather than the
    measurement:

    1. **Driver-side ceiling.** If ``nvidia-smi`` reports VRAM use above
       ``vram_budget_gib``, the next batch would push the Windows driver into
       paging VRAM into host RAM. Throughput then collapses and the desktop
       becomes unresponsive, so the sweep stops *before* taking that step.
    2. **Host-RAM floor.** Oversubscription consumes host memory; if free RAM
       drops below ``min_free_ram_gb`` the sweep stops.
    """
    import torch

    def say(msg: str):
        if verbose:
            print(msg, flush=True)
        if log_file:
            with log_file.open("a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
                fh.flush()

    res = {"model": name, "precision": precision, "max_new_tokens": max_new_tokens,
           "n_images_available": len(images), "steps": [], "stopped_reason": "",
           "vram_budget_gib": vram_budget_gib}

    say(f"  warm-up ...")
    try:
        model.generate(images[0], prompt, max_new_tokens=min(32, max_new_tokens))
    except Exception as exc:
        res["error"] = f"warmup_failed: {type(exc).__name__}: {exc}"
        say(f"  warm-up FAILED: {res['error'][:140]}")
        return res
    say(f"  warm-up done ({vram_smi_mb() or 0:.0f} MB in use)")

    baseline = None
    prev_per_img = None
    for bs in batch_sizes:
        if bs > len(images):
            res["steps"].append({"batch_size": bs, "skipped": "not enough images"})
            continue

        # --- guard 1: driver-side VRAM ceiling -------------------------
        smi_before = vram_smi_mb()
        if smi_before is not None and smi_before / 1024 > vram_budget_gib:
            res["stopped_reason"] = (
                f"driver VRAM {smi_before/1024:.1f} GB exceeds budget "
                f"{vram_budget_gib} GB; refusing to step into paging")
            say(f"  STOP: {res['stopped_reason']}")
            break
        # --- guard 2: host RAM floor -----------------------------------
        if free_ram_gb() < min_free_ram_gb:
            res["stopped_reason"] = f"host RAM {free_ram_gb():.1f} GB below {min_free_ram_gb} GB"
            say(f"  STOP: {res['stopped_reason']}")
            break

        reset_torch_peak()
        imgs = images[:bs]
        prompts = [prompt] * bs
        per_img = None
        step = {"batch_size": bs}
        say(f"  bs={bs}: running ...")
        try:
            t0 = time.time()
            if hasattr(model, "generate_batch"):
                outs = model.generate_batch(imgs, prompts, max_new_tokens=max_new_tokens)
            else:
                outs = [model.generate(i, p, max_new_tokens=max_new_tokens)
                        for i, p in zip(imgs, prompts)]
            dt = time.time() - t0
            per_img = dt / bs
            step.update({"total_sec": round(dt, 2), "sec_per_img": round(per_img, 3),
                         "n_out": len(outs), "sample": (outs[0] or "")[:60]})
        except RuntimeError as exc:
            msg = str(exc)
            step["error"] = ("oom" if "out of memory" in msg.lower() else "runtime") + f": {msg[:120]}"
            res["steps"].append(step)
            say(f"  bs={bs:<4d} FAILED ({step['error'][:70]})")
            if "out of memory" in msg.lower():
                torch.cuda.empty_cache()
                continue
            continue

        step["torch_peak_gb"] = round(torch_peak_gb(), 2)
        smi = vram_smi_mb()
        step["smi_used_gb"] = round(smi / 1024, 2) if smi else None
        step["free_ram_gb"] = round(free_ram_gb(), 1)

        if baseline is None and per_img:
            baseline = per_img
            step["speedup_vs_first"] = 1.0
        elif per_img and baseline:
            step["speedup_vs_first"] = round(baseline / per_img, 2)

        # decline detection: the real signal, independent of any OOM
        if per_img and prev_per_img:
            if per_img > prev_per_img * (1 + slowdown_tolerance):
                step["verdict"] = "decline"
                step["note"] = "decline point; compare torch_peak vs smi_used"
                res["steps"].append(step)
                say(f"  bs={bs:<4d} {per_img:6.2f}s/img  torch_peak={step['torch_peak_gb']:5.2f}GB "
                    f"smi={step['smi_used_gb']}GB -> DECLINE"
                    f" ({per_img/prev_per_img - 1:+.0%} vs bs-prev); stopping ladder")
                res["stopped_reason"] = f"throughput declined at bs={bs}"
                break
            else:
                step["verdict"] = "improved"
        else:
            step["verdict"] = "baseline"

        res["steps"].append(step)
        prev_per_img = per_img
        say(f"  bs={bs:<4d} {per_img:6.2f}s/img  speedup x{step.get('speedup_vs_first', 1):.2f}  "
            f"torch_peak={step['torch_peak_gb']:5.2f}GB  smi={step['smi_used_gb']}GB  "
            f"host_free={step['free_ram_gb']}GB")

    # ---- pick the operating point -------------------------------------
    ok = [s for s in res["steps"] if s.get("sec_per_img") and s.get("verdict") != "decline"]
    if ok:
        fastest = min(ok, key=lambda s: s["sec_per_img"])
        # prefer the smallest batch within 5% of the fastest: same speed, less VRAM
        near = [s for s in ok if s["sec_per_img"] <= fastest["sec_per_img"] * 1.05]
        chosen = min(near, key=lambda s: s["batch_size"])
        res["recommended"] = {
            "batch_size": chosen["batch_size"],
            "sec_per_img": chosen["sec_per_img"],
            "torch_peak_gb": chosen["torch_peak_gb"],
            "smi_used_gb": chosen["smi_used_gb"],
            "speedup_vs_smallest": chosen.get("speedup_vs_first"),
            "rule": "smallest batch within 5% of best measured throughput",
        }
        if verbose:
            print(f"  -> RECOMMEND bs={chosen['batch_size']} "
                  f"({chosen['sec_per_img']:.2f} s/img, {chosen['torch_peak_gb']:.1f} GB)")
    return res


# --------------------------------------------------------------------------- #
# ladder construction
# --------------------------------------------------------------------------- #
def auto_ladder(name: str, precision: str) -> list[int]:
    """Ladder centred on the prediction: below, at, slightly above, well above.

    Spacing is deliberately **finer between the prediction and twice the
    prediction**, because that is where the fall-off actually happens: measured
    on Qwen2.5-VL-7B (4-bit) the curve is flat to bs=8 and already slower by
    bs=16, so a coarse ladder steps straight over the拐点.
    """
    pred = PREDICTIONS.get(name, {}).get("pred_bs", {}).get(precision)
    if not pred:
        return [4, 8, 16]
    ladder = {max(1, pred // 4), max(1, pred // 2), pred}
    # fine steps from pred to 2*pred
    half = max(1, pred // 2)
    b = pred + half
    while b <= pred * 2:
        ladder.add(b)
        b += half
    ladder.update({int(pred * 2.5), pred * 3})
    return sorted(x for x in ladder if x >= 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--preset", default=None,
                    choices=["modern"],
                    help="'modern' = only the models that need the modern transformers "
                         "venv (run this script with .venv-modern's python)")
    ap.add_argument("--precision", nargs="+", default=None, choices=["4bit", "bf16"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    ap.add_argument("--auto-ladder", action="store_true")
    ap.add_argument("--manifest", default="data/manifests/nih_shared_500.csv")
    ap.add_argument("--images-root", default="data/nih/images")
    ap.add_argument("--limit-per-batch", type=int, default=100,
                    help="images used for each measurement; the largest tested batch "
                         "cannot exceed this (default 100 is enough to time a batch)")
    ap.add_argument("--ladder-max", type=int, default=128,
                    help="never test a batch size above this, regardless of the auto ladder")
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="fixed across models so runs are comparable")
    ap.add_argument("--min-free-ram-gb", type=float, default=10.0)
    ap.add_argument("--gpu-only", action="store_true",
                    help="refuse CPU offload (fails loudly instead of silently running "
                         "a partly-offloaded model, which is not comparable)")
    ap.add_argument("--vram-budget-gb", type=float, default=30.0,
                    help="hard ceiling on driver VRAM use. The sweep refuses to start a "
                         "batch that would exceed it, because past this point the Windows "
                         "driver pages VRAM into host RAM and throughput collapses. "
                         "Physical size is 32 GB; the default leaves ~2 GB of headroom.")
    ap.add_argument("--log", default=None,
                    help="append live progress here (poll with Get-Content -Wait)")
    ap.add_argument("--slowdown-tolerance", type=float, default=0.10,
                    help="relative per-image slowdown that counts as a decline")
    ap.add_argument("--output", default="outputs/benchmark/batch_sweep.json")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, load nothing")
    args = ap.parse_args()

    log_path = None
    if args.log:
        log_path = Path(args.log)
        if not log_path.is_absolute():
            log_path = ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")     # fresh run

    import torch
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM     : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB")
    print(f"budget   : {args.vram_budget_gb:.0f} GB driver-VRAM ceiling (abort before paging)")
    print(f"host RAM : {free_ram_gb():.1f} GB free")
    print(f"torch    : {torch.__version__}")
    if log_path:
        print(f"live log : {log_path}")
        print(f"           watch with:  Get-Content '{log_path}' -Wait -Tail 20")
    print()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    df = pd.read_csv(manifest).head(args.limit_per_batch)
    images_root = Path(args.images_root)
    images = []
    for _, r in df.iterrows():
        p = images_root / r["image_relpath"]
        if p.exists():
            images.append(p)
    if not images:
        print(f"[error] no images found under {images_root} using {manifest}")
        return 1
    print(f"bench images: {len(images)} from {manifest.name}")
    print()

    names = ([n for n in SUPPORTED if n in PREDICTIONS] if args.all
             else (args.model or []))
    if args.preset == "modern":
        # Qwen3-VL needs transformers >= 4.57 and must therefore be launched with
        # .venv-modern's python. The registry is the source of truth for which
        # models that applies to.
        try:
            from src.cxr_trust.models.registry import NEEDS_MODERN_TRANSFORMERS
            names = [n for n in SUPPORTED if n in NEEDS_MODERN_TRANSFORMERS]
        except Exception:
            names = []
        if not names:
            names = ["qwen3_vl_8b_instruct", "qwen3_vl_30b_a3b_instruct"]
    if not names:
        ap.error("give --model <name>, --all, or --preset modern")

    plan = []
    for name in names:
        precisions = args.precision or PREDICTIONS.get(name, {}).get("precisions", ["bf16"])
        for prec in precisions:
            ladder = args.batch_sizes or (auto_ladder(name, prec) if args.auto_ladder
                                          else [4, 8, 16])
            # never ask for more images than we loaded, and respect the explicit cap
            cap = min(len(images), args.ladder_max)
            ladder = [b for b in ladder if b <= cap]
            if not ladder:
                ladder = [min(4, cap)]
            plan.append((name, prec, ladder))

    print("=== PLAN ===")
    for name, prec, ladder in plan:
        pred = PREDICTIONS.get(name, {}).get("pred_bs", {}).get(prec)
        print(f"  {name:28s} {prec:5s} ladder={ladder}   predicted sweet spot={pred}")
    print()

    if args.dry_run:
        print("dry run: nothing loaded")
        return 0

    results = []
    for name, prec, ladder in plan:
        print(f"=== {name} [{prec}] ===")
        try:
            model, load_s = load_model(name, prec, gpu_only=args.gpu_only,
                                       vram_budget_gib=int(args.vram_budget_gb))
        except Exception as exc:
            print(f"  LOAD FAILED: {type(exc).__name__}: {str(exc)[:200]}")
            results.append({"model": name, "precision": prec,
                            "gpu_only": args.gpu_only,
                            "error": f"load_failed: {type(exc).__name__}: {exc}"})
            continue
        print(f"  loaded in {load_s:.1f}s")
        try:
            res = sweep(model, name, prec, ladder, images, BENCH_PROMPT,
                        args.max_new_tokens, 1, args.min_free_ram_gb,
                        args.slowdown_tolerance,
                        vram_budget_gib=args.vram_budget_gb, log_file=log_path)
        finally:
            free_model(model)
        res["load_sec"] = round(load_s, 1)
        res["gpu_only"] = args.gpu_only
        res["predicted_sweet_spot"] = PREDICTIONS.get(name, {}).get("pred_bs", {}).get(prec)
        results.append(res)
        print()

    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "max_new_tokens": args.max_new_tokens,
        "results": results,
    }
    if out.exists():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
            payload["previous_runs"] = prev.get("previous_runs", []) + [{
                "recorded_at": prev.get("recorded_at"),
                "n_results": len(prev.get("results", [])),
            }]
        except Exception:
            pass
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")

    print("\n=== SUMMARY ===")
    for r in results:
        if r.get("error"):
            print(f"  {r['model']:28s} {r['precision']:5s} ERROR {r['error'][:70]}")
        elif r.get("recommended"):
            rec = r["recommended"]
            print(f"  {r['model']:28s} {r['precision']:5s} bs={rec['batch_size']:<4d} "
                  f"{rec['sec_per_img']:.2f} s/img  {rec['torch_peak_gb']:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
