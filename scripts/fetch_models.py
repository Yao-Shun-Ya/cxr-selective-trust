#!/usr/bin/env python
"""Fetch the model weights pinned by configs/models.lock.yaml.

Weights are deliberately absent from this repository (see the header of
configs/models.lock.yaml for the physical, legal and editorial reasons).
This script is the acquisition path that a reviewer is expected to follow.

Usage
-----
    python scripts/fetch_models.py --list
    python scripts/fetch_models.py --all
    python scripts/fetch_models.py --model medgemma_1_5_4b_it --model qwen3_vl_8b_instruct
    python scripts/fetch_models.py --all --endpoint https://hf-mirror.com
    python scripts/fetch_models.py --all --verify-only

Gated repositories (MedGemma) require you to have accepted the Gemma Terms
of Use and the Healthcare and Life Sciences Use Case Extension on the model
page, and to have run `huggingface-cli login`. This script will not accept
terms on your behalf and will fail loudly rather than silently skipping.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_FILE = REPO_ROOT / "configs" / "models.lock.yaml"


def load_lock(path: Path) -> dict:
    """Parse the lock file.

    A tiny hand-rolled reader is used instead of PyYAML so that weight
    acquisition never depends on the ML environments under program/.env*/.
    Only the two-level mapping shape used by the lock file is supported.
    """
    if not path.is_file():
        raise SystemExit(f"lock file not found: {path}")

    models: dict = {}
    derived: dict = {}
    bucket = models
    current: str | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line == "models:":
            bucket, current = models, None
            continue
        if line == "derived:":
            bucket, current = derived, None
            continue
        if not line.startswith(" "):
            bucket, current = {}, None       # a top-level scalar section, ignore
            continue
        stripped = line.strip()
        if len(line) - len(line.lstrip()) == 2 and stripped.endswith(":"):
            current = stripped[:-1]
            bucket[current] = {}
        elif len(line) - len(line.lstrip()) == 4 and current is not None and ":" in stripped:
            key, _, value = stripped.partition(":")
            value = value.strip()
            # strip trailing inline comments ("value   # note")
            if value and value[0] not in "\"'":
                cut = value.find(" #")
                if cut != -1:
                    value = value[:cut].rstrip()
            if value in ("true", "false"):
                bucket[current][key.strip()] = value == "true"
            else:
                bucket[current][key.strip()] = value

    return {"models": models, "derived": derived}


def human_gib(nbytes: int) -> str:
    return f"{nbytes / 1024 ** 3:.1f} GiB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def verify(lock: dict) -> int:
    """Report which pinned directories are present and how large they are."""
    print(f"{'key':<28} {'present':<8} {'size':>10}  revision")
    print("-" * 84)
    missing = 0
    for key, spec in lock["models"].items():
        target = REPO_ROOT / spec.get("local_dir", "")
        if target.is_dir():
            print(f"{key:<28} {'yes':<8} {human_gib(dir_size(target)):>10}  "
                  f"{spec.get('revision', '?')[:12]}")
        else:
            missing += 1
            print(f"{key:<28} {'NO':<8} {'-':>10}  {spec.get('revision', '?')[:12]}")
    for key, spec in lock["derived"].items():
        target = REPO_ROOT / spec.get("local_dir", "")
        state = "yes" if target.is_dir() else "built?"
        size = human_gib(dir_size(target)) if target.is_dir() else "-"
        print(f"{key:<28} {state:<8} {size:>10}  derived, run produced_by script")
    return missing


def fetch_one(key: str, spec: dict, endpoint: str | None, force: bool) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is required:  pip install huggingface_hub", file=sys.stderr)
        return False

    hf_id = spec["hf_id"]
    revision = spec["revision"]
    target = REPO_ROOT / spec["local_dir"]

    if target.is_dir() and dir_size(target) > 1_000_000_000 and not force:
        print(f"[skip] {key}: already present at {target} "
              f"({human_gib(dir_size(target))}); pass --force to re-check")
        return True

    if spec.get("gated"):
        print(f"[gated] {key}: {hf_id} requires you to have accepted the upstream "
              f"terms and run `huggingface-cli login`.")

    if endpoint:
        import os
        os.environ["HF_ENDPOINT"] = endpoint

    print(f"[get] {key}: {hf_id} @ {revision[:12]} -> {target}")
    try:
        snapshot_download(
            repo_id=hf_id,
            revision=revision,
            local_dir=str(target),
            local_dir_use_symlinks=False,
        )
    except Exception as exc:                                  # noqa: BLE001
        print(f"[FAIL] {key}: {exc}\n"
              f"       If this is a permission error, accept the terms for {hf_id} "
              f"on Hugging Face first. Do not substitute a third-party mirror of "
              f"gated weights.", file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="fetch every downloaded model")
    ap.add_argument("--model", action="append", default=[],
                    help="fetch one model key (repeatable)")
    ap.add_argument("--list", action="store_true", help="verify presence and exit")
    ap.add_argument("--verify-only", action="store_true", help="alias of --list")
    ap.add_argument("--endpoint", default=None,
                    help="HF endpoint override, e.g. https://hf-mirror.com")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if the directory already looks complete")
    args = ap.parse_args()

    lock = load_lock(LOCK_FILE)

    if args.list or args.verify_only or (not args.all and not args.model):
        missing = verify(lock)
        print("\nNothing was downloaded. Use --all or --model <key>.")
        return 0

    keys = list(lock["models"]) if args.all else args.model
    unknown = [k for k in keys if k not in lock["models"]]
    if unknown:
        print(f"unknown model key(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(lock['models'])}", file=sys.stderr)
        return 2

    failures = [k for k in keys if not fetch_one(k, lock["models"][k], args.endpoint, args.force)]

    for key, spec in lock["derived"].items():
        if key in keys or args.all:
            print(f"[derived] {key}: rebuild with {spec.get('produced_by', 'see lock file')}")

    if failures:
        print(f"\nfailed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("\nAll requested models are in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
