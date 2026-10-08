"""Point the Stage-2 experiment scripts at the registry instead of hardcoded pairs.

Why
---
`06_image_reliance.py`, `08_shift_benchmark.py` and `21_prompt_ablation.py` each
hardcode ``choices=["medgemma_4b_it", "qwen25_vl_7b_instruct"]`` and build the
model from ``configs/models.yaml``. That was fine for a two-model pilot, but the
matrix now has seven models (and Qwen3-VL lives in a different virtualenv), so the
scripts must:

* accept every model the registry knows, and
* resolve the path through the registry (single source of truth) rather than
  re-reading a YAML that no longer lists them all.

Transformations applied (idempotent — re-running is safe):

1. ``choices=["medgemma_4b_it", "qwen25_vl_7b_instruct"]`` -> ``choices=MODEL_NAMES``
2. the YAML + ``get_model(...)`` block -> a registry call passing ``root=ROOT``
3. ensure ``MODEL_NAMES`` is imported from the registry

Run:  python scripts/35_wire_scripts_to_registry.py [--check]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "scripts" / "06_image_reliance.py",
    ROOT / "scripts" / "08_shift_benchmark.py",
    ROOT / "scripts" / "21_prompt_ablation.py",
]

HARDCODED = re.compile(
    r'choices=\[\s*"medgemma_4b_it"\s*,\s*"qwen25_vl_7b_instruct"\s*\]')

# the YAML-then-get_model block used by 06/08
YAML_BLOCK = re.compile(
    r'( *)cfg = yaml\.safe_load\(\(ROOT / "configs/models\.yaml"\)\.read_text\(encoding="utf-8"\)\)\n'
    r'.*?model\.load\(\)\n',
    re.S)

REGISTRY_LOAD = '''{ind}model = get_model(
{ind}    args.model,
{ind}    root=ROOT,
{ind}    device_map="auto",
{ind}    load_in_4bit=args.load_in_4bit,
{ind})
{ind}model.load()
'''


def patch(path: Path, check_only: bool) -> tuple[bool, list[str]]:
    src = path.read_text(encoding="utf-8")
    orig = src
    notes: list[str] = []

    # --- 1. model choices -------------------------------------------------
    if HARDCODED.search(src):
        src = HARDCODED.sub("choices=MODEL_NAMES", src)
        notes.append("choices -> MODEL_NAMES")
    elif "choices=MODEL_NAMES" in src:
        notes.append("choices already patched")

    # --- 2. registry-based loading ---------------------------------------
    m = YAML_BLOCK.search(src)
    if m:
        indent = m.group(1)
        src = YAML_BLOCK.sub(REGISTRY_LOAD.format(ind=indent), src, count=1)
        notes.append("yaml+get_model -> registry get_model")
    elif "root=ROOT" in src:
        notes.append("loading already registry-based")
    elif "cfg = yaml.safe_load" in src:
        notes.append("WARNING: yaml block present but not matched; patch by hand")

    # --- 3. import MODEL_NAMES -------------------------------------------
    if "MODEL_NAMES" in src and not re.search(
            r"from src\.cxr_trust\.models\.registry import[^\n]*\bMODEL_NAMES\b", src):
        src = re.sub(
            r"from src\.cxr_trust\.models\.registry import ([^\n]+)",
            lambda mm: ("from src.cxr_trust.models.registry import "
                        + ", ".join(sorted(set(
                            [x.strip() for x in mm.group(1).split(",")] + ["MODEL_NAMES"])))),
            src, count=1)
        if "MODEL_NAMES" not in src.split("\n\n")[0] and "from src.cxr_trust.models.registry import" not in src:
            notes.append("WARNING: no registry import line found")
        else:
            notes.append("imported MODEL_NAMES")

    if src != orig and not check_only:
        path.write_text(src, encoding="utf-8")
    return src != orig, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report without writing")
    args = ap.parse_args()

    for p in TARGETS:
        if not p.exists():
            print(f"MISSING {p.name}")
            continue
        changed, notes = patch(p, args.check)
        flag = "WOULD CHANGE" if (changed and args.check) else ("CHANGED" if changed else "unchanged")
        print(f"{flag:12s} {p.name}")
        for n in notes:
            print(f"             - {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
