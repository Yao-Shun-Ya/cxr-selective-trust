"""Preflight: check every environment and checkpoint BEFORE committing to a long run.

The first full run lost ~1 hour to two failures that a 20-second check would have
caught:

* ``qwen3_vl_8b_instruct`` ran under the wrong interpreter, so transformers did
  not recognise the ``qwen3_vl`` architecture and all three of its stages failed
  in 3 seconds each.
* ``internvl3_5_8b`` and ``llava_med_v1_5_7b`` loaded fine but every generated row
  was an error string, so their reliance stages crashed in post-processing.

This script answers, per model, without generating anything:

1. does the interpreter that the orchestrator will use actually exist,
2. do its torch / transformers versions match what the wrapper needs,
3. is the checkpoint directory complete (weights + config + tokenizer),
4. is the GPU currently free enough to start.

Run it with the interpreter the orchestrator uses for each model; it re-invokes
itself internally when needed.  CPU only: no weights are loaded, no VRAM used.

    .venv-vl\\Scripts\\python.exe scripts\\39_preflight.py
    .venv-vl\\Scripts\\python.exe scripts\\39_preflight.py --gpu-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_trust.models.registry import MODEL_NAMES, MODEL_PATHS  # noqa: E402

# Which interpreter each model needs, and the minimum transformers version.
# Qwen3-VL needs >= 4.57 (only in .venv-modern); InternVL's remote code is broken
# under transformers 5.x (only in .venv-vl).
VENV = {
    "qwen3_vl_8b_instruct": ".venv-modern",
}
DEFAULT_VENV = ".venv-vl"
MODERN_ONLY = {"qwen3_vl_8b_instruct"}
LEGACY_ONLY = {"internvl3_5_8b", "chexagent_8b", "llava_med_v1_5_7b"}
# Stored on disk and checked here, but deliberately NOT part of a run: its
# per-expert packed layout does not match what transformers 5.x expects, so it
# cannot load (see docs/46 S5.1). Reported as info, never counted as a failure.
EXCLUDED = {"qwen3_vl_30b_a3b_awq4bit": "excluded from the matrix (see docs/46 S5.1)"}

WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.pt")
REQUIRED_FILES = ("config.json",)
TOKENIZER_HINTS = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                   "vocab.json", "spiece.model")

OK = "ok  "
BAD = "FAIL"
WARN = "warn"


def venv_for(model: str) -> str:
    return VENV.get(model, DEFAULT_VENV)


def python_for(model: str) -> Path:
    return ROOT / venv_for(model) / "Scripts" / "python.exe"


def probe_interpreter(model: str) -> dict:
    """Ask an interpreter for its versions. Cheap: no model is loaded."""
    exe = python_for(model)
    out = {"venv": venv_for(model), "exe": str(exe), "exists": exe.exists()}
    if not exe.exists():
        out["error"] = "interpreter missing"
        return out
    code = (
        "import json, sys\n"
        "d = {'python': sys.version.split()[0]}\n"
        "try:\n"
        "    import torch\n"
        "    d['torch'] = torch.__version__\n"
        "    d['cuda'] = bool(torch.cuda.is_available())\n"
        "except Exception as exc:\n"
        "    d['torch'] = 'MISSING'\n"
        "    d['cuda'] = False\n"
        "    d['torch_error'] = str(exc)[:120]\n"
        "try:\n"
        "    import transformers as t\n"
        "    d['transformers'] = t.__version__\n"
        "except Exception:\n"
        "    d['transformers'] = 'MISSING'\n"
        "print('PROBE ' + json.dumps(d))\n"
    )
    try:
        r = subprocess.run([str(exe), "-c", code], capture_output=True, text=True,
                           timeout=180)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    line = ""
    for cand in (r.stdout or "").strip().splitlines():
        if cand.startswith("PROBE "):
            line = cand[len("PROBE "):].strip()
    if not line:
        out["error"] = (r.stderr or "no output").strip().splitlines()[-1:] or ["no output"]
        return out
    out.update(json.loads(line))
    return out


def version_tuple(v: str) -> tuple[int, ...]:
    parts = []
    for chunk in str(v).split("."):
        num = "".join(ch for ch in chunk if ch.isdigit())
        if not num:
            break
        parts.append(int(num))
    return tuple(parts)


def check_versions(model: str, info: dict) -> tuple[str, str]:
    if "error" in info:
        return BAD, info["error"]
    tf = info.get("transformers", "MISSING")
    if tf == "MISSING":
        return BAD, "transformers not importable"
    vt = version_tuple(tf)
    if model in MODERN_ONLY and vt < (4, 57):
        return BAD, f"needs transformers >= 4.57 for qwen3_vl, found {tf}"
    if model in LEGACY_ONLY and vt >= (5,):
        return BAD, f"remote code breaks under transformers 5.x, found {tf}"
    if not info.get("cuda"):
        return WARN, f"torch {info.get('torch')} reports CUDA unavailable"
    return OK, f"python {info.get('python')} torch {info.get('torch')} tf {tf}"


def check_checkpoint(model: str) -> tuple[str, str]:
    rel = MODEL_PATHS.get(model)
    if not rel:
        return BAD, "not in MODEL_PATHS"
    path = ROOT / rel
    if not path.is_dir():
        return BAD, f"missing directory {rel}"
    missing = [f for f in REQUIRED_FILES if not (path / f).exists()]
    weights = [p for g in WEIGHT_GLOBS for p in path.glob(g)]
    index = list(path.glob("*.safetensors.index.json")) + list(path.glob("*.bin.index.json"))
    if not weights and not index:
        missing.append("weights (*.safetensors / *.bin)")
    if not any((path / f).exists() for f in TOKENIZER_HINTS):
        missing.append("tokenizer files")
    if missing:
        return BAD, "missing " + ", ".join(missing)
    n = len(weights)
    size_gb = sum(p.stat().st_size for p in weights) / 1024 ** 3
    return OK, f"{n} weight files, {size_gb:.1f} GB on disk"


def gpu_state() -> tuple[str, str]:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        used, total = [int(x) for x in r.stdout.strip().splitlines()[0].split(",")]
    except Exception as exc:  # noqa: BLE001
        return WARN, f"nvidia-smi unavailable ({type(exc).__name__})"
    free_gb = (total - used) / 1024
    msg = f"{used} / {total} MiB in use; {free_gb:.1f} GB free"
    if free_gb < 20:
        return BAD, msg + " -- close other GPU work before starting"
    return OK, msg


def stray_python() -> tuple[str, str]:
    """Other python processes, excluding this self-check.

    On Windows a venv's `python.exe` is a launcher shim: running it spawns the
    real interpreter as a child, so every invocation shows up as TWO python.exe
    processes (shim + real). `os.getpid()` is the child, so excluding only it
    leaves the shim behind. Exclude our PID and our parent PID, then name the
    survivors so the warning is actionable.
    """
    ps = ("$me = {0}; $pa = {1}; "
          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
          "| Where-Object {{ $_.ProcessId -ne $me -and $_.ProcessId -ne $pa }} "
          "| Select-Object -ExpandProperty CommandLine").format(os.getpid(), os.getppid())
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, text=True, timeout=60)
    except Exception:  # noqa: BLE001
        return WARN, "could not count python processes"
    cmds = [c.strip() for c in (r.stdout or "").splitlines() if c.strip()]
    if not cmds:
        return OK, "no stray python processes"
    shown = "; ".join(c[:80] for c in cmds[:2])
    more = f" (+{len(cmds) - 2} more)" if len(cmds) > 2 else ""
    return WARN, f"{len(cmds)} other python process(es): {shown}{more}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--gpu-only", action="store_true",
                    help="only check the GPU and stray processes")
    args = ap.parse_args()

    models = args.models or sorted(MODEL_NAMES)
    print(f"program root : {ROOT}")
    print()
    print("GPU")
    for label, fn in (("gpu state", gpu_state), ("processes", stray_python)):
        state, msg = fn()
        print(f"  [{state}] {label:<12} {msg}")
    if args.gpu_only:
        return 0

    print()
    print("ENVIRONMENTS (one probe per distinct interpreter)")
    seen: dict[str, dict] = {}
    for m in models:
        v = venv_for(m)
        if v in seen:
            continue
        seen[v] = probe_interpreter(m)
        info = seen[v]
        state = OK if info.get("exists") and "error" not in info else BAD
        print(f"  [{state}] {v:<14} {info.get('exe')}")
        if info.get("error"):
            print(f"          {info['error']}")
        else:
            print(f"          python {info.get('python')}  torch {info.get('torch')}"
                  f"  cuda={info.get('cuda')}  transformers={info.get('transformers')}")

    print()
    print("MODELS")
    failures = 0
    for m in models:
        vstate, vmsg = check_versions(m, seen[venv_for(m)])
        cstate, cmsg = check_checkpoint(m)
        state = BAD if BAD in (vstate, cstate) else (WARN if WARN in (vstate, cstate) else OK)
        if state == BAD and m not in EXCLUDED:
            failures += 1
        if m in EXCLUDED:
            state = "info"
        print(f"  [{state}] {m:<24} {venv_for(m)}")
        if m in EXCLUDED:
            print(f"          note : {EXCLUDED[m]}")
        print(f"          env  : {vmsg}")
        print(f"          ckpt : {cmsg}")

    print()
    if failures:
        print(f"RESULT: {failures} model(s) cannot be run as configured -- fix before starting.")
        return 1
    print("RESULT: all checked models are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
