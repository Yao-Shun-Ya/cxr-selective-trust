"""Sync every document to the current test evidence (idempotent).

The docs quote test counts, and test counts change when a suite is added. Rather
than hand-editing six files each time, the facts live here once:

    41_test_durable.py         37 assertions  (the DurableJsonl unit suite)
    41_test_runner_resume.py   40 assertions  (43/44 through their real CLIs)
    49_test_legacy_durable.py  38 assertions  (06/08/10/21 through their real CLIs)
    -> 115 total

Run after any test change:

    .venv-vl\\Scripts\\python.exe scripts\\51_sync_doc_test_facts.py --dry-run
    .venv-vl\\Scripts\\python.exe scripts\\51_sync_doc_test_facts.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]

LEGACY_LINE = (".venv-vl\\Scripts\\python.exe scripts\\49_test_legacy_durable.py"
               "   # 38 passed, 0 failed")
LEGACY_MD = "* `49_test_legacy_durable.py` — **38 assertions, all passing**: the four " \
            "older runners (06, 08, 10, 21) through their real CLIs, each clean, " \
            "killed mid-run, resumed, and re-resumed to a byte-identical no-op."
LEGACY_BULLET = ("program/scripts/49_test_legacy_durable.py  " \
                 "四个老 runner 的真实崩溃/续跑测试（38 断言，全通过）")
LEGACY_PROSE = ("`41_test_runner_resume.py` (40 assertions, passing, real subprocess "
                "runs with a")

EDITS: dict[str, list[tuple[str, str]]] = {
    "CONTINUE_HERE.md": [
        ("program/scripts/41_test_runner_resume.py  真实子进程崩溃/续跑测试（40 断言，全通过）",
         "program/scripts/41_test_runner_resume.py  真实子进程崩溃/续跑测试（40 断言，全通过）\n"
         + LEGACY_BULLET),
        (".venv-vl\\Scripts\\python.exe scripts\\41_test_runner_resume.py   # 40 passed, 0 failed",
         ".venv-vl\\Scripts\\python.exe scripts\\41_test_runner_resume.py   # 40 passed, 0 failed\n"
         + LEGACY_LINE),
    ],
    "HANDOVER.md": [
        (".venv-vl\\Scripts\\python.exe scripts\\41_test_runner_resume.py    # 40 passed, 0 failed",
         ".venv-vl\\Scripts\\python.exe scripts\\41_test_runner_resume.py    # 40 passed, 0 failed\n"
         + LEGACY_LINE),
        (".venv-vl\\Scripts\\python.exe scripts\\41_test_runner_resume.py\n",
         ".venv-vl\\Scripts\\python.exe scripts\\41_test_runner_resume.py\n"
         ".venv-vl\\Scripts\\python.exe scripts\\49_test_legacy_durable.py\n"),
    ],
    "README.md": [
        (".venv-vl\\Scripts\\python.exe scripts\\41_test_runner_resume.py    # 40 passed, 0 failed",
         ".venv-vl\\Scripts\\python.exe scripts\\41_test_runner_resume.py    # 40 passed, 0 failed\n"
         + LEGACY_LINE),
    ],
    "docs/48_three_layer_protocol.md": [
        (".venv-vl\\Scripts\\python.exe scripts\\41_test_runner_resume.py",
         ".venv-vl\\Scripts\\python.exe scripts\\41_test_runner_resume.py\n"
         ".venv-vl\\Scripts\\python.exe scripts\\49_test_legacy_durable.py"),
        ("* `41_test_runner_resume.py` — **40 assertions, all passing**",
         LEGACY_MD + "\n* `41_test_runner_resume.py` — **40 assertions, all passing**"),
    ],
    "docs/41_6k_curated_benchmark_plan.md": [
        (LEGACY_PROSE,
         "`41_test_runner_resume.py` (40 assertions), `49_test_legacy_durable.py` "
         "(38 assertions) and \n`41_test_durable.py` (37) — 115 in total, all "
         "passing, real subprocess")
    ],
}


def apply(text: str, old: str, new: str) -> tuple[str, int]:
    """Insert ``new`` where ``old`` is, at most once, and never twice.

    The first version guarded on the first line of ``new``, which was usually a
    different string from ``old`` -- so the guard never fired and repeated runs
    appended duplicate lines. The guard is now "is the thing I would insert already
    here?", which is the property that actually matters.
    """
    inserted_lines = [ln for ln in new.splitlines() if ln.strip() and ln not in old]
    if inserted_lines and all(any(ln == existing for existing in text.splitlines())
                              for ln in inserted_lines):
        return text, 0            # already synced
    if old not in text:
        return text, 0
    return text.replace(old, new, 1), 1


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
            text, n = apply(text, old, new)
            applied += n
        total += applied
        if applied and not args.dry_run:
            p.write_text(text, encoding="utf-8")
        print(f"{name}: {applied} update(s){' (dry run)' if args.dry_run else ''}")
    print(f"\ntotal: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
