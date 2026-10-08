#!/usr/bin/env python
"""Overnight runner for the formal shift benchmark + analysis."""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VL_PY = ROOT / ".venv-vl/Scripts/python.exe"
if not VL_PY.exists():
    VL_PY = ROOT / ".venv-vl/bin/python"

# Per-model generation budget. The source of truth is $MaxNewTokensOverride in
# scripts/run_all_layers.ps1; it is repeated here because this runner used to
# hardcode 256 for every model, which is how medgemma_15 lost 68 of 100 replies
# in the resolution/severe cell (docs/50 section 2.8). A model that is not in
# this table is REFUSED rather than run at a budget that would silently
# truncate its replies -- the same rule the driver applies to unknown models.
MODEL_MAX_NEW_TOKENS = {
    "medgemma_4b_it": 256,
    "medgemma_15_4b_it": 1024,
    "qwen25_vl_7b_instruct": 256,
    "qwen3_vl_8b_instruct": 256,
    "chexagent_8b": 256,
    "internvl3_5_8b": 256,
    "llava_med_v1_5_7b": 256,
}
# Mirrors $FourBitModels in the driver. Passing --load-in-4bit to a model that
# was not validated under it is a configuration the study never measured.
FOUR_BIT_MODELS = {"medgemma_4b_it", "medgemma_15_4b_it", "qwen25_vl_7b_instruct"}


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def keep_awake():
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def allow_sleep():
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def run(name, cmd, log_path, timeout, retries, sleep_sec):
    for attempt in range(1, retries + 1):
        line = f"[{now()}] START {name} attempt {attempt}/{retries}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.write("CMD " + " ".join(str(x) for x in cmd) + "\n")
            f.flush()
            try:
                p = subprocess.run([str(x) for x in cmd], cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                                   timeout=timeout, text=True, encoding="utf-8", errors="replace")
                if p.returncode == 0:
                    print(f"[{now()}] DONE {name}", flush=True)
                    return True
            except subprocess.TimeoutExpired:
                print(f"[{now()}] TIMEOUT {name}", flush=True)
            except Exception as exc:
                print(f"[{now()}] ERROR {name}: {exc}", flush=True)
        if attempt < retries:
            time.sleep(sleep_sec)
    return False


def run_once(name, cmd, log_path):
    """Run once, no retries, and return the exit code.

    Used for the analysis step, where a non-zero code is a *verdict* -- 3 means
    cells below the coverage threshold -- rather than a transient failure a retry
    could fix. Retrying it would only re-read the same file.
    """
    print(f"[{now()}] START {name} (no retries)", flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{now()}] START {name} (no retries)\n")
        f.write("CMD " + " ".join(str(x) for x in cmd) + "\n")
        f.flush()
        p = subprocess.run([str(x) for x in cmd], cwd=str(ROOT), stdout=f,
                           stderr=subprocess.STDOUT, text=True,
                           encoding="utf-8", errors="replace")
    verdict = "DONE" if p.returncode == 0 else f"EXIT {p.returncode}"
    print(f"[{now()}] {verdict} {name}", flush=True)
    return p.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["medgemma_4b_it"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--shifts", nargs="+", default=["blur", "noise", "jpeg", "brightness", "contrast", "resolution"])
    parser.add_argument("--severities", nargs="+", default=["mild", "moderate", "severe"])
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=int, default=300)
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--output-root", default=str(ROOT / "outputs/shift_benchmark"))
    args = parser.parse_args()

    unknown = [m for m in args.models if m not in MODEL_MAX_NEW_TOKENS]
    if unknown:
        print(f"ERROR: no generation budget known for {unknown}. Add it to "
              f"MODEL_MAX_NEW_TOKENS; do not fall back to the 256 default, which "
              f"truncates replies from models with a reasoning channel.",
              flush=True)
        return 2

    out_root = Path(args.output_root)
    log_dir = ROOT / "outputs/logs"
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    master_log = log_dir / f"shift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    keep_awake()
    below = []
    try:
        manifest = ROOT / "data/manifests/nih_pilot_1000.csv"
        for model in args.models:
            out_dir = out_root / f"{model}_{args.limit}"
            cmd = [
                VL_PY, "scripts/08_shift_benchmark.py",
                "--model", model,
                "--manifest", str(manifest),
                "--images-root", str(ROOT / "data/nih/images"),
                "--output-dir", str(out_dir),
                "--limit", str(args.limit),
                "--shifts", *args.shifts,
                "--severities", *args.severities,
                "--max-new-tokens", str(MODEL_MAX_NEW_TOKENS[model]),
            ]
            if model in FOUR_BIT_MODELS:
                cmd.append("--load-in-4bit")
            # generous timeout: 60s/job x jobs. Raised from 30s when the budget
            # table went in: a 1024-token reasoning reply costs several times a
            # 256-token one, and this runner has no --batch-size, so it pays that
            # cost per image.
            jobs = args.limit * (1 + len(args.shifts) * len(args.severities))
            timeout = int(jobs * 60 + 3600)
            if run(f"shift:{model}", cmd, master_log, timeout, args.max_retries, args.retry_sleep):
                code = run_once(
                    f"analysis:{model}",
                    [VL_PY, "scripts/09_shift_analysis.py",
                     "--predictions", str(out_dir / "shift_predictions.jsonl"),
                     "--output-dir", str(out_dir / "analysis"),
                     "--min-coverage", str(args.min_coverage),
                     "--fail-on-low-coverage"],
                    master_log)
                if code == 3:
                    below.append(model)
                    print(f"[{now()}] COVERAGE {model}: cells below "
                          f"{args.min_coverage:.0%} label coverage -- report the "
                          f"coverage next to the metric, do not drop the cell",
                          flush=True)
        print(f"[{now()}] shift benchmark finished", flush=True)
        if below:
            print(f"[{now()}] exit 3: coverage assertion failed for {below} "
                  f"(the predictions themselves are complete)", flush=True)
            return 3
        return 0
    finally:
        allow_sleep()


if __name__ == "__main__":
    sys.exit(main())
