#!/usr/bin/env python
"""Audit every key the analysis code READS against the keys the runners WRITE.

Why this exists
---------------
A real bug of this exact shape was found on 2026-10-06:
``59_probe_pretest_findings.py`` read ``probability_pos`` where the schema writes
``probability_positive``. Python does not raise for a missing dict key reached
through ``.get``, so that column silently held ``None`` for every row, the
comparison it fed quietly used the *status* instead of the probability, and no
test, sweep or review noticed for a whole round. There is no exception to catch:
the only way to see it is to compare what is read against what is written.

So this script builds both sides and diffs them:

1. **Written.** The schema's own keys (produced behaviourally, by calling
   ``ParsedPrediction.to_dict()`` rather than by grepping for strings), every key
   in every prediction JSONL that actually exists under ``outputs/``, and the row
   dicts of the six runners (parsed statically, because PadChest and Open-i have
   no artifacts on disk yet).
2. **Read.** Every string literal used as ``x["key"]`` or ``x.get("key")``
   anywhere under ``scripts/`` and ``src/``.

The interesting output is not "keys read that are unknown" -- most of those are
config, argparse or plotting parameters and there are hundreds of them. It is the
**near misses**: a read key that does not exist in the written universe but is
within a small edit distance of one that does. That is what a typo looks like, and
it is the difference between a metric computed on data and a metric computed on
``None``.

Exit code is 1 when a near miss is found, 0 otherwise, so this can be wired into
the sweep as a standing guard.

Usage::

    .venv-vl\\Scripts\\python.exe scripts\\64_audit_row_keys.py
    .venv-vl\\Scripts\\python.exe scripts\\64_audit_row_keys.py --show-unknown
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keys that are close to a written key but are legitimately something else
# (a local dict, a config field, an argument). Every entry here is a decision, so
# keep the list short and say why.
KNOWN_NEAR_MISS_OK = {
    # matplotlib / pandas keyword arguments that shadow schema-ish words
    "probability",            # plotting axis label dicts
    "confidence",             # the schema writes it too, but also a metrics column
    "prob_positive",          # local variable name, never a dict lookup key
    "parse_success_rate",     # a computed summary field, not a row key
    "n_parse_success",        # census output column (written by 09, not by runners)
    "n_parse_fail",           # census output column
    "parse_coverage",         # census output column
    "label_coverage",         # census output column
    "n_label_planned",        # census output column
    "n_label_instances",      # census output column
    "n_gt_dropped",           # census output column
    "n_attempted",            # census output column
    "n_parse_ok",             # census output column
    "n_marker",               # census column added by 59
    "n_marker_unparsed",      # census column added by 59
    "coverage_ok",            # coverage_report.json / shift_summary.json field
    "min_label_coverage",     # shift_summary.json field
    "min_coverage",           # argparse dest
    "raw_outputs",            # a filename (raw_outputs.jsonl), not a row key
    "image_reliance",         # a metrics/summary name, not a row key
    "elapsed_sec",            # written by every runner; listed to be explicit
}

# Files whose string literals are mostly not row keys; still scanned, because a
# typo inside them is exactly what we are hunting, but the near-miss report names
# the file so triage is quick.
SCAN_DIRS = ("scripts", "src")

RUNNER_FILES = (
    "scripts/06_image_reliance.py",
    "scripts/08_shift_benchmark.py",
    "scripts/10_run_chexpert_rrg.py",
    "scripts/21_prompt_ablation.py",
    "scripts/43_run_padchest_shift.py",
    "scripts/44_run_openi_report.py",
)

# Shorter than this and the edit-distance test matches half the universe.
MIN_INTERESTING_LEN = 10
# How much longer/shorter than the read key a written key may be and still count
# as "the same name, truncated or extended".
MAX_AFFIX_SLACK = 8


def collect_keys(obj, prefix="", out=None, depth=0) -> set[str]:
    """Every key and dotted key-path in a nested JSON object."""
    if out is None:
        out = set()
    if depth > 6:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out.add(str(k))
            out.add(path)
            collect_keys(v, path, out, depth + 1)
    elif isinstance(obj, list):
        for item in obj[:2]:
            collect_keys(item, prefix, out, depth + 1)
    return out


def written_from_schema() -> set[str]:
    """The parser's own contract, produced by running it, not by reading it."""
    from src.cxr_trust.inference.schema import FindingPrediction, ParsedPrediction

    sample = ParsedPrediction(
        findings={"cardiomegaly": FindingPrediction("cardiomegaly", "positive", 0.5, 0.5)},
        rationale="r", image_reliance_claim="c", parse_success=True, parse_error="",
    )
    keys = collect_keys(sample.to_dict())
    # the model-side contract, same names, checked so both directions are covered
    keys |= {"findings", "probability_positive", "confidence", "rationale",
             "image_reliance_claim"}
    return keys


def written_from_artifacts(max_files: int = 400) -> tuple[set[str], list[str]]:
    keys: set[str] = set()
    seen: list[str] = []
    files = sorted(ROOT.glob("outputs/**/*_predictions.jsonl"))
    files += sorted(ROOT.glob("outputs/**/raw_outputs.jsonl"))
    for path in files[:max_files]:
        seen.append(str(path.relative_to(ROOT)))
        try:
            with path.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= 40 or not line.strip():
                        continue
                    keys |= collect_keys(json.loads(line))
        except (json.JSONDecodeError, OSError):
            continue
    return keys, seen


def dict_keys_in_source(path: Path) -> set[str]:
    """String keys of every dict literal in one file (the runners' row shapes)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    out.add(k.value)
    return out


def keys_written_in_source(path: Path) -> set[str]:
    """Dict-literal keys PLUS subscript assignments (``row["k"] = ...``).

    The assignment form matters: a metrics frame built with
    ``out["n_findings"] = g.size()`` and a stats dict filled with
    ``blank_stats["pct_identical_to_original"] = ...`` both write keys that look
    like schema keys to an edit-distance test. Counting them as "written here"
    removes exactly those false positives without weakening the check that caught
    the original bug -- the bug was a key that its file *never* wrote.
    """
    out = dict_keys_in_source(path)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store) \
                and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str):
            out.add(node.slice.value)
    return out


def written_from_runners() -> set[str]:
    keys: set[str] = set()
    for rel in RUNNER_FILES:
        p = ROOT / rel
        if p.exists():
            keys |= dict_keys_in_source(p)
    return keys


def reads_in_file(path: Path) -> dict[str, list[int]]:
    """String literals used as x["key"] or x.get("key") in one file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return {}
    found: dict[str, list[int]] = defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) \
                and isinstance(node.slice.value, str):
            found[node.slice.value].append(node.lineno)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "get" and node.args \
                and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            found[node.args[0].value].append(node.lineno)
    return found


def edit_distance(a: str, b: str, cap: int = 4) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def near_miss(key: str, universe: set[str], cap: int = 2) -> list[str]:
    """Written keys this read key is probably a typo of.

    Precision matters more than recall here: a checker that cries wolf gets
    ignored (the same argument `53` makes for its stale-figure rule). Two filters
    do the work:

    * **length** -- a short key (``auc``, ``batch``, ``Path``) can be within one
      edit of half the universe and is never where a silent-``None`` bug hides;
    * **same-file writes** -- if the file writes that key itself as a dict literal,
      it belongs to a local structure (a metrics row, a download record), not to
      the prediction schema.
    """
    if len(key) < MIN_INTERESTING_LEN:
        return []
    hits = []
    for cand in universe:
        if cand == key or len(cand) < MIN_INTERESTING_LEN:
            continue
        # Prefix/substring relationship: this is how the real bug looked
        # (`probability_pos` for `probability_positive`), and an edit-distance test
        # misses it -- a truncated name is 5 edits away, not 2. The length gate
        # above is what keeps this from firing on every short word.
        if key in cand or cand in key:
            if abs(len(cand) - len(key)) <= MAX_AFFIX_SLACK:
                hits.append(cand)
                continue
        if edit_distance(key, cand, cap) <= cap:
            hits.append(cand)
    return sorted(hits)[:3]


def population_census(max_files: int = 400, max_rows: int = 80,
                      min_rows: int = 20) -> dict[str, list[int]]:
    """Per dotted key-path: [rows seen, rows with a non-null value].

    The second half of the same failure class. The typo that started this audit
    made a *reader* see ``None``; a key that is written but never populated makes
    every reader see ``None`` for a different reason, and neither raises. Only a
    census can tell "this field is empty" from "this field is missing".
    """
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, (dict, list)):
                    walk(v, path)
                else:
                    stats[path][0] += 1
                    if v not in (None, "", [], {}):
                        stats[path][1] += 1
        elif isinstance(obj, list):
            for item in obj[:2]:
                walk(item, prefix)

    files = sorted(ROOT.glob("outputs/**/*_predictions.jsonl"))
    files += sorted(ROOT.glob("outputs/**/raw_outputs.jsonl"))
    for path in files[:max_files]:
        try:
            with path.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if i >= max_rows or not line.strip():
                        continue
                    walk(json.loads(line))
        except (json.JSONDecodeError, OSError):
            continue
    return {k: v for k, v in stats.items() if v[0] >= min_rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show-unknown", action="store_true",
                    help="also list read keys unrelated to any written key")
    ap.add_argument("--max-edit-distance", type=int, default=2)
    ap.add_argument("--selftest", action="store_true",
                    help="positive control: assert that the 2026-10-06 bug "
                         "(reading probability_pos) would be flagged")
    args = ap.parse_args()

    schema_keys = written_from_schema()
    artifact_keys, artifact_files = written_from_artifacts()
    runner_keys = written_from_runners()
    written = schema_keys | artifact_keys | runner_keys

    print("=== written universe ===")
    print(f"  schema (behavioural) : {len(schema_keys)} keys")
    print(f"  artifacts            : {len(artifact_keys)} keys "
          f"from {len(artifact_files)} file(s) under outputs/")
    print(f"  runner row dicts     : {len(runner_keys)} keys (static, incl. 43/44)")
    print(f"  union                : {len(written)}")

    read: dict[str, list[tuple[str, int]]] = defaultdict(list)
    local_writes: dict[str, set[str]] = {}
    for d in SCAN_DIRS:
        for path in sorted((ROOT / d).rglob("*.py")):
            if "__pycache__" in str(path) or path.name == Path(__file__).name:
                continue
            rel = str(path.relative_to(ROOT))
            local_writes[rel] = keys_written_in_source(path)
            for key, lines in reads_in_file(path).items():
                for ln in lines:
                    read[key].append((rel, ln))

    # A key that this same file writes as a dict literal belongs to a local
    # structure (a metrics row, an API record), so it is not a schema read.
    def is_local_only(key: str) -> bool:
        sites = read[key]
        return bool(sites) and all(key in local_writes.get(rel, set())
                                   for rel, _ in sites)

    print(f"\n=== read keys ===")
    print(f"  {len(read)} distinct string keys read as x[\"key\"] / x.get(\"key\") "
          f"across {len(SCAN_DIRS)} source trees")

    unknown = {k: v for k, v in read.items() if k not in written}
    suspects: dict[str, list[str]] = {}
    for key in unknown:
        if key in KNOWN_NEAR_MISS_OK or is_local_only(key):
            continue
        hits = near_miss(key, written, args.max_edit_distance)
        if hits:
            suspects[key] = hits

    print(f"\n=== near misses (read key is NOT written, but resembles one that is) ===")
    if not suspects:
        print("  none")
    else:
        for key in sorted(suspects):
            sites = read[key]
            print(f"  READ  {key!r}   ~  {', '.join(repr(h) for h in suspects[key])}")
            for rel, ln in sites[:6]:
                print(f"          {rel}:{ln}")
            if len(sites) > 6:
                print(f"          ... {len(sites) - 6} more site(s)")

    if args.show_unknown:
        print(f"\n=== read keys unrelated to anything written ({len(unknown)}; "
              f"config/argparse/plotting, informational) ===")
        for key in sorted(unknown):
            if key in suspects:
                continue
            rel, ln = read[key][0]
            print(f"  {key:32s} {rel}:{ln}")

    print(f"\n=== populated-ness census (keys written but never given a value) ===")
    census = population_census()
    always_null = {k: v for k, v in census.items() if v[1] == 0}
    partly = {k: v for k, v in census.items()
              if v[1] > 0 and v[1] < 0.5 * v[0] and "." not in k}
    if not always_null:
        print("  no key is empty across every row sampled")
    else:
        print(f"  {len(always_null)} key(s) are null/empty in every row sampled "
              f"-- a reader of these gets nothing, silently:")
        for k in sorted(always_null):
            print(f"    {k:38s} {always_null[k][0]:6d} rows")
    if partly:
        print(f"\n  partially populated top-level keys (informational):")
        for k in sorted(partly):
            seen, ok = partly[k]
            print(f"    {k:38s} {ok}/{seen} rows non-empty "
                  f"({100.0 * ok / seen:.0f}%)")

    print(f"\n=== verdict ===")
    if args.selftest:
        # A checker that prints "none" is only useful if it can print something.
        # This is the exact bug of 2026-10-06: 59_probe_pretest_findings.py read
        # `probability_pos`, a key no runner ever writes, and every value it fed
        # downstream was None.
        hits = near_miss("probability_pos", written, args.max_edit_distance)
        ok = "probability_positive" in hits
        print(f"  positive control: near_miss('probability_pos') = {hits}")
        print(f"  {'PASS' if ok else 'FAIL'}: the scan "
              f"{'does' if ok else 'does NOT'} flag the known bug")
        return 0 if ok else 1
    if suspects:
        print(f"  {len(suspects)} suspicious read key(s) -- each one is a silent None "
              f"waiting to be averaged over")
        print("  fix the read site, or add an explicit entry to KNOWN_NEAR_MISS_OK "
              "with the reason")
        return 1
    print("  no near misses: every row key that analysis code reads is one a runner "
          "writes")
    if artifact_files:
        print(f"  (checked against {len(artifact_files)} artifact file(s))")
    else:
        print("  WARNING: no artifacts under outputs/ -- the check fell back to the "
              "schema and the runners' source only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
