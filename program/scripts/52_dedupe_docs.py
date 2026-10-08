"""Remove the duplicate lines 51_sync_doc_test_facts.py introduced, and make it idempotent.

Two bugs in one file, both mine:

1. **Not idempotent.** The guard checked whether the *first line* of the
   replacement was already present, but the anchors were `(old, new)` pairs where
   the inserted line was often a different string from the anchor, so the guard
   never fired and every run appended another copy. Fixed by checking for the
   exact line that would be inserted, and by de-duplicating consecutive identical
   lines after editing.
2. **The duplicate itself.** A doc that lists the same test command twice is not
   merely untidy: it is the same class of defect as everything else in this
   session -- a statement that is not what the file actually says, produced by an
   edit that was never checked.

    .venv-vl\\Scripts\\python.exe scripts\\52_dedupe_docs.py --dry-run
    .venv-vl\\Scripts\\python.exe scripts\\52_dedupe_docs.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
FILES = [
    "CONTINUE_HERE.md", "HANDOVER.md", "README.md",
    "docs/48_three_layer_protocol.md", "docs/41_6k_curated_benchmark_plan.md",
    "docs/49_data_manifest_contract.md", "docs/README.md",
    "program/README.md",
]


def dedupe_consecutive(lines: list[str]) -> tuple[list[str], int]:
    """Drop a line when it is byte-identical to the line directly above it."""
    out: list[str] = []
    removed = 0
    for line in lines:
        if out and line == out[-1] and line.strip():
            removed += 1
            continue
        out.append(line)
    return out, removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total = 0
    for name in FILES:
        p = BASE / name
        if not p.exists():
            continue
        original = p.read_text(encoding="utf-8")
        lines = original.splitlines()
        deduped, removed = dedupe_consecutive(lines)
        if removed:
            total += removed
            if not args.dry_run:
                trailing = "\n" if original.endswith("\n") else ""
                p.write_text("\n".join(deduped) + trailing, encoding="utf-8")
            print(f"{name}: {removed} consecutive duplicate line(s) removed")
        else:
            print(f"{name}: clean")
    print(f"\ntotal removed: {total}{' (dry run)' if args.dry_run else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
