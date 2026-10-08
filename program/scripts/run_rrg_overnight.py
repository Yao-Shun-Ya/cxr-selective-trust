#!/usr/bin/env python
"""Overnight runner for CheXpert-plus-RRG report/hallucination/image-reliance."""

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
        print(f"[{now()}] START {name} attempt {attempt}/{retries}", flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[{now()}] START {name}\nCMD " + " ".join(str(x) for x in cmd) + "\n")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=264)
    parser.add_argument("--blank-limit", type=int, default=50)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=int, default=300)
    args = parser.parse_args()

    out_dir = ROOT / "outputs/rrg/medgemma"
    log_dir = ROOT / "outputs/logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"rrg_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    keep_awake()
    try:
        run("rrg:medgemma", [
            VL_PY, "scripts/10_run_chexpert_rrg.py",
            "--model", "medgemma_4b_it",
            "--manifest", str(ROOT / "data/manifests/chexpert_rrg_all.csv"),
            "--images-root", str(ROOT / "data/chexpert_rrg"),
            "--output-dir", str(out_dir),
            "--limit", str(args.limit),
            "--blank-limit", str(args.blank_limit),
            "--load-in-4bit",
            "--max-new-tokens", "256",
        ], log_path, timeout=12 * 3600, retries=args.max_retries, sleep_sec=args.retry_sleep)
        run("rrg:analysis", [
            VL_PY, "scripts/11_rrg_report_analysis.py",
            "--predictions", str(out_dir / "rrg_predictions.jsonl"),
            "--output-dir", str(out_dir / "analysis"),
        ], log_path, timeout=3600, retries=2, sleep_sec=60)
        print(f"[{now()}] RRG run finished", flush=True)
    finally:
        allow_sleep()


if __name__ == "__main__":
    main()
