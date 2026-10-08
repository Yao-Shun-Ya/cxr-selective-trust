#!/usr/bin/env python
"""Robust overnight orchestration for the CXR selective-trust pilot.

Features:
- runs extraction, several models, evaluation, and diagnosis in sequence;
- retries each step on failure;
- uses --resume for model inference, so interrupted runs continue;
- writes a master log, per-run progress.log/progress.json, and status.json;
- prevents Windows sleep while running, then restores normal sleep behaviour.
"""

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


def _find_python(name: str) -> Path:
    candidates = [
        ROOT / f".venv-{name}" / "Scripts" / "python.exe",
        ROOT / f".venv-{name}" / "bin" / "python",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find python for env {name}. Expected one of: {candidates}")


VL_PY = None
CHEX_PY = None


def keep_awake() -> None:
    if sys.platform != "win32":
        return
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def allow_sleep() -> None:
    if sys.platform != "win32":
        return
    ES_CONTINUOUS = 0x80000000
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(log_path: Path, message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def update_status(status_path: Path, payload: dict) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_step(name: str, cmd: list[str], log_path: Path, timeout: int, retries: int, retry_sleep: int) -> bool:
    for attempt in range(1, retries + 1):
        log(log_path, f"START {name} (attempt {attempt}/{retries})")
        log(log_path, "CMD " + " ".join(str(x) for x in cmd))
        try:
            with log_path.open("a", encoding="utf-8") as f:
                proc = subprocess.run(
                    [str(x) for x in cmd],
                    cwd=str(ROOT),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
            if proc.returncode == 0:
                log(log_path, f"DONE  {name}")
                return True
            log(log_path, f"FAIL  {name} returncode={proc.returncode}")
        except subprocess.TimeoutExpired:
            log(log_path, f"TIMEOUT {name} after {timeout}s")
        except Exception as exc:
            log(log_path, f"ERROR {name}: {exc}")
        if attempt < retries:
            log(log_path, f"RETRY {name} in {retry_sleep}s ...")
            time.sleep(retry_sleep)
    log(log_path, f"GIVE UP {name}")
    return False


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--models", nargs="+", default=["medgemma_4b_it", "qwen25_vl_7b_instruct"])
    parser.add_argument("--include-chexagent", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=int, default=300)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--output-root", default=str(ROOT / "outputs/overnight"))
    args = parser.parse_args()

    global VL_PY, CHEX_PY
    VL_PY = _find_python("vl")
    try:
        CHEX_PY = _find_python("chexagent")
    except FileNotFoundError:
        CHEX_PY = None

    out_root = Path(args.output_root)
    log_dir = ROOT / "outputs/logs"
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    master_log = log_dir / f"overnight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    status_path = out_root / "status.json"

    status = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "limit": args.limit,
        "steps": {},
        "models": args.models,
        "master_log": str(master_log),
    }
    update_status(status_path, status)
    keep_awake()
    log(master_log, "Overnight run started")
    log(master_log, f"Python VL env: {VL_PY}")
    log(master_log, f"Python CheXagent env: {CHEX_PY}")

    try:
        manifest = ROOT / "data/manifests/nih_pilot_1000.csv"

        # 1) Extraction
        if not args.skip_extract:
            ok = run_step(
                "extract_nih_images",
                [VL_PY, "scripts/02_extract_nih_images.py", "--manifest", str(manifest)],
                master_log,
                timeout=4 * 3600,
                retries=2,
                retry_sleep=args.retry_sleep,
            )
            status["steps"]["extract_nih_images"] = "ok" if ok else "failed"
            update_status(status_path, status)

        # 2) Models
        for model in args.models:
            model_out = out_root / f"{model}_{args.limit}"
            cmd = [
                VL_PY, "scripts/03_run_model.py",
                "--model", model,
                "--manifest", str(manifest),
                "--images-root", str(ROOT / "data/nih/images"),
                "--output-dir", str(model_out),
                "--limit", str(args.limit),
                "--load-in-4bit",
                "--max-new-tokens", "256",
                "--resume",
            ]
            # generous timeouts: expected runtime x ~3 + 1h
            per_image = 20 if "medgemma" in model else 15
            timeout = int(per_image * args.limit * 3 + 3600)
            ok = run_step(f"model:{model}", cmd, master_log, timeout=timeout, retries=args.max_retries, retry_sleep=args.retry_sleep)
            status["steps"][f"model:{model}"] = "ok" if ok else "failed"
            update_status(status_path, status)

            pred = model_out / "predictions.jsonl"
            if line_count(pred) > 0:
                run_step(
                    f"evaluate:{model}",
                    [VL_PY, "scripts/04_evaluate.py", "--predictions", str(pred), "--output-dir", str(model_out / "metrics")],
                    master_log, timeout=1800, retries=2, retry_sleep=60,
                )
                run_step(
                    f"diagnose:{model}",
                    [VL_PY, "scripts/05_diagnose_predictions.py", "--predictions", str(pred), "--output-dir", str(model_out / "diagnosis")],
                    master_log, timeout=1800, retries=2, retry_sleep=60,
                )
            else:
                log(master_log, f"SKIP evaluation for {model}: no predictions")

        # 3) Optional CheXagent, small N only
        if args.include_chexagent and CHEX_PY is not None:
            chex_limit = min(10, args.limit)
            chex_out = out_root / f"chexagent_{chex_limit}"
            cmd = [
                CHEX_PY, "scripts/03_run_model.py",
                "--model", "chexagent_8b",
                "--manifest", str(manifest),
                "--images-root", str(ROOT / "data/nih/images"),
                "--output-dir", str(chex_out),
                "--limit", str(chex_limit),
                "--max-new-tokens", "256",
                "--resume",
            ]
            ok = run_step("model:chexagent", cmd, master_log, timeout=6 * 3600, retries=1, retry_sleep=args.retry_sleep)
            status["steps"]["model:chexagent"] = "ok" if ok else "failed"
            update_status(status_path, status)

        log(master_log, "Overnight run finished")
        status["finished_at"] = datetime.now().isoformat(timespec="seconds")
 
        update_status(status_path, status)
    finally:
        allow_sleep()
        log(master_log, "Sleep behaviour restored")


if __name__ == "__main__":
    main()
