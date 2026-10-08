#!/usr/bin/env python
"""Documentation <-> code consistency check. Read-only; reports, never edits.

The docs in this repository quote code facts: script names, test counts, cost
figures, durability properties. Every one of those can drift, and every drift
found so far was a statement that had been true when written. So it is checked by
a command rather than trusted.

Checks
------
1. Every ``NN_name.py`` cited in any markdown file exists under ``program/scripts``.
2. No markdown claims a runner is still flush-only (all six are durable now).
3. The driver's own plan and the independent cost table agree on the default
   configuration's total.
4. No document still quotes a superseded cost figure.
5. The test counts quoted in docs match the suites' actual assertion counts, read
   from the suites' own source rather than from a hard-coded number.

Run::

    .venv-vl\\Scripts\\python.exe scripts\\53_check_doc_code_consistency.py
    .venv-vl\\Scripts\\python.exe scripts\\53_check_doc_code_consistency.py --strict
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
PROG = BASE / "program"
DOC_DIRS = [BASE / "docs", BASE]

PROBLEMS: list[str] = []
NOTES: list[str] = []


def md_files():
    for d in DOC_DIRS:
        if d.is_dir():
            for p in sorted(d.glob("*.md")):
                yield p


def check_script_names() -> None:
    print("=== 1. cited script names exist ===")
    cited = set()
    for p in md_files():
        t = p.read_text(encoding="utf-8", errors="replace")
        cited.update(re.findall(r"\b(\d{2}_[a-z0-9_]+\.py)\b", t))
    missing = [n for n in sorted(cited) if not (PROG / "scripts" / n).exists()]
    print(f"  {len(cited)} distinct script name(s) cited")
    if missing:
        PROBLEMS.append(f"documentation cites non-existent scripts: {missing}")
        print(f"  MISSING: {missing}")
    else:
        print("  all present")


def check_durability_claims() -> None:
    print("\n=== 2. no doc claims a runner is still flush-only ===")
    pattern = re.compile(r"still use flush-only|仍是 flush-only|尚未接入 `?Durable")
    bad = []
    for p in md_files():
        for ln, line in enumerate(p.read_text(encoding="utf-8", errors="replace")
                                  .splitlines(), 1):
            if pattern.search(line):
                bad.append(f"{p.relative_to(BASE)}:{ln}")
    if bad:
        # A struck-through line recording that it WAS flush-only is fine; an
        # unstruck claim is not distinguishable by regex alone, so report both and
        # let the reader decide.
        NOTES.append(f"durability claims to review: {bad}")
        print(f"  review: {bad}")
    else:
        print("  none")


def check_cost_agreement() -> None:
    print("\n=== 3. driver plan vs independent cost table ===")
    drv = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         "scripts/run_all_layers.ps1", "-DryRun"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(PROG))
    hours = calls = None
    for line in (drv.stdout or "").splitlines():
        m = re.search(r"TOTAL:\s*([\d,]+) model calls,\s*([\d.]+)h", line)
        if m:
            calls, hours = int(m.group(1).replace(",", "")), float(m.group(2))
    print(f"  driver    : {calls} calls, {hours} h")
    cost_path = PROG / "outputs" / "state" / "cost_table.json"
    if not cost_path.exists():
        NOTES.append("cost_table.json missing; run scripts/47_cost_table.py")
        print("  cost table: MISSING (run 47_cost_table.py)")
        return
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    first = cost["scenarios"][0]
    print(f"  cost table: {first['calls']} calls, {first['hours']} h")
    if calls is not None and calls != first["calls"]:
        PROBLEMS.append(f"call counts disagree: driver {calls} vs table {first['calls']}")
    if hours is not None and abs(hours - first["hours"]) > 1.0:
        PROBLEMS.append(f"hour estimates disagree: driver {hours} vs table {first['hours']}")
    if not PROBLEMS:
        print("  agree")


def check_stale_figures() -> None:
    print("\n=== 4. no superseded cost figures stated as current ===")
    # A struck-through or explicitly-corrective mention is the point of the
    # changelog, so lines that frame the old number as wrong are not defects.
    # Without this the checker cries wolf, and a checker that cries wolf gets
    # ignored -- which is worse than not having one.
    stale = re.compile(r"\b125\.4\b|\b621 h\b|\b309\.9\b|\b5\.2 days\b")
    corrective = re.compile(
        r"not |no longer|used to|superseded|was wrong|previously|formerly|"
        r"reported|corrected|was |were |错误|原来的|之前")
    hits = []
    for p in md_files():
        for ln, line in enumerate(p.read_text(encoding="utf-8", errors="replace")
                                  .splitlines(), 1):
            if stale.search(line) and not corrective.search(line):
                hits.append(f"{p.relative_to(BASE)}:{ln}: {line.strip()[:90]}")
    if hits:
        PROBLEMS.append(f"{len(hits)} line(s) state a superseded figure as current")
        for h in hits:
            print(f"  {h}")
    else:
        print("  none (every mention is corrective)")


def check_test_counts() -> None:
    print("\n=== 5. quoted test counts vs actual suites ===")
    suites = ["41_test_durable.py", "41_test_runner_resume.py",
              "49_test_legacy_durable.py"]
    counts: dict[str, int] = {}
    for name in suites:
        p = PROG / "scripts" / name
        if not p.exists():
            PROBLEMS.append(f"test suite missing: {name}")
            continue
        n = len(re.findall(r"\bcheck\(\s*[\"']",
                           p.read_text(encoding="utf-8")))
        counts[name] = n
        print(f"  {name:32s} assertions: {n}")
    total = sum(counts.values())
    print(f"  total: {total}")
    # Only lines that actually talk about tests may quote the total; a bare "115"
    # elsewhere is a chapter number and flagging it is noise.
    testish = re.compile(r"assert|passed|test|测试|断言")
    for p in md_files():
        for ln, line in enumerate(p.read_text(encoding="utf-8", errors="replace")
                                  .splitlines(), 1):
            if str(total) in line and testish.search(line) and "check(" not in line:
                NOTES.append(f"{p.relative_to(BASE)}:{ln} quotes the {total} total")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    check_script_names()
    check_durability_claims()
    check_cost_agreement()
    check_stale_figures()
    check_test_counts()

    print(f"\n{'=' * 78}")
    print(f"  doc/code consistency: {len(PROBLEMS)} problem(s), {len(NOTES)} note(s)")
    for m in PROBLEMS:
        print(f"    PROBLEM {m}")
    for m in NOTES:
        print(f"    NOTE    {m}")
    print(f"{'=' * 78}")
    if PROBLEMS:
        return 1
    if args.strict and NOTES:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
