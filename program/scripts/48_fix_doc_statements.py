"""One-shot documentation correction: remove statements describing a gap that is fixed.

All six prediction-writing runners now use DurableJsonl, verified by
45_debug_sweep.py (which now treats it as a hard check) and by real crash/resume
runs of each. Several documents still said two of them were flush-only.

This script is kept in the repository rather than run once and deleted, because
the same class of drift will happen again: a doc statement about a code property
is only correct until the code changes.

    .venv-vl\\Scripts\\python.exe scripts\\48_fix_doc_statements.py --dry-run
    .venv-vl\\Scripts\\python.exe scripts\\48_fix_doc_statements.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]   # D:\Paper

D48 = "docs/48_three_layer_protocol.md"

EDITS: dict[str, list[tuple[str, str]]] = {
    D48: [
        ("| `10_run_chexpert_rrg.py` | **no — flush only** | interruption loses the "
         "tail; must not be relied on for a long run |",
         "| `10_run_chexpert_rrg.py` | **yes** | adopted later. It used to open the "
         "file with `\"a\" if done else \"w\"`, so a file holding only failed rows "
         "looked empty and was wiped on the next run |"),
        ("| `21_prompt_ablation.py` | **no — flush only** | same; the NIH `ablation` "
         "stage is the one stage of the three-layer run without the fsync guarantee |",
         "| `21_prompt_ablation.py` | **yes** | adopted later. `raw_outputs.jsonl` "
         "and the parsed file are two durable streams, so the audit trail can never "
         "disagree with the parsed records about how many calls happened |"),
        ("4. **The `ablation` stage is not crash-durable** (`21_prompt_ablation.py` is",
         "4. **RESOLVED — the `ablation` stage is now crash-durable.** "
         "(`21_prompt_ablation.py` was"),
        ("* `10_run_chexpert_rrg.py` and `21_prompt_ablation.py` still use flush-only",
         "* `10_run_chexpert_rrg.py` and `21_prompt_ablation.py` used to use flush-only"),
    ],
    "HANDOVER.md": [
        ("23. **`10_run_chexpert_rrg.py` 与 `21_prompt_ablation.py` 仍是 flush-only**"
         "（未接入 `DurableJsonl`）：",
         "23. ~~`10_run_chexpert_rrg.py` 与 `21_prompt_ablation.py` 仍是 flush-only~~ "
         "**已修复**：两者均已接入 `DurableJsonl`，且 `45_debug_sweep.py` 现在把它作为"
         "硬性检查（六个 runner 全部 durable=yes）："),
        ("[ ] 10_run_chexpert_rrg.py / 21_prompt_ablation.py 接入 DurableJsonl",
         "[x] 10_run_chexpert_rrg.py / 21_prompt_ablation.py 接入 DurableJsonl（已完成）"),
        ("- [ ] 知道 `10_run_chexpert_rrg.py` / `21_prompt_ablation.py` 尚未接入 "
         "`DurableJsonl`；",
         "- [x] 知道全部六个 runner 都已接入 `DurableJsonl`"
         "（`45_debug_sweep.py` 会强制检查）；"),
    ],
    "CONTINUE_HERE.md": [
        ("[ ] 10_run_chexpert_rrg.py 与 21_prompt_ablation.py 仍是 flush-only，"
         "尚未接入 DurableJsonl",
         "[x] 10_run_chexpert_rrg.py 与 21_prompt_ablation.py 已接入 DurableJsonl"
         "（六个 runner 全覆盖）"),
    ],
    "docs/41_6k_curated_benchmark_plan.md": [
        ("**not yet** by `10_run_chexpert_rrg.py` or `21_prompt_ablation.py`.",
         "**also now** by `10_run_chexpert_rrg.py` and `21_prompt_ablation.py`; "
         "all six runners are durable."),
    ],
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = 0
    for name, pairs in EDITS.items():
        p = BASE / name
        if not p.exists():
            print(f"{name}: MISSING")
            continue
        text = p.read_text(encoding="utf-8")
        applied = 0
        for old, new in pairs:
            if old in text:
                applied += text.count(old)
                text = text.replace(old, new)
            else:
                print(f"  ! anchor not found in {name}: {old[:70]!r}")
        total += applied
        if applied and not args.dry_run:
            p.write_text(text, encoding="utf-8")
        print(f"{name}: {applied} statement(s) {'would be ' if args.dry_run else ''}updated")

    print(f"\ntotal: {total} statement(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
