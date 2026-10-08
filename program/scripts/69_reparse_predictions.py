#!/usr/bin/env python
"""Re-parse stored predictions with the current parser.

Why this exists
---------------
``schema.extract_json_object`` gained two fixes (docs/50 section 2.12):

* HTML line-break tags (``<br>``) are normalised to newlines before extraction --
  CheXagent writes a complete, well-formed object with ``<br>`` instead of ``\\n``
  in 16 of its 1,900 shift rows;
* a brace group that fails to parse no longer ends the search -- a thought-trace
  model can put a malformed draft before the final object, and stopping there lost
  one medgemma_15 precision row.

Every prediction row stores BOTH ``raw_output`` and the ``parsed`` derived from it,
so rows written before the fix hold a stale ``parsed``. This walks the prediction
files, recomputes ``parsed`` with the SAME rule the writers use, and reports exactly
which rows change.

The rule mirrors the writers (``08_shift_benchmark.py:271`` and friends)::

    parsed = parse_vlm_output(raw) if not error else parse_vlm_output("")

Guarantees
----------
The change is a strict superset: a row that parsed before must parse to the SAME
findings afterwards, and no row may go parsed -> unparsed. Any violation is a bug
in the parser, not a data event, and this script exits 3 on it.

Safety
------
* dry run by default;
* every file that changes is copied to ``outputs/_reparse_backup/<rel>`` first;
* writes are atomic (tmp + fsync + os.replace) and the ``DurableJsonl`` sidecar
  ``<file>.progress.json`` is rewritten with the new byte size and record count.
  Without that the next driver run would truncate the file back to the old offset
  and silently drop the tail.

Usage::

    .venv-vl\\Scripts\\python.exe scripts\\69_reparse_predictions.py
    .venv-vl\\Scripts\\python.exe scripts\\69_reparse_predictions.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr_trust.inference.schema import parse_vlm_output  # noqa: E402

OUTPUTS = ROOT / "outputs"
BACKUP = OUTPUTS / "_reparse_backup"
EVIDENCE = ROOT / "outputs" / "state" / "reparse_evidence.json"


def _is_backup(path: Path) -> bool:
    for part in path.parts:
        low = part.lower()
        if low == "_reparse_backup" or low.endswith("_backup"):
            return True
    return False


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".progress.json")


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def _reparse_file(path: Path, apply: bool) -> dict:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    stats = {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "rows": len(rows),
        "parsed_before": 0,
        "parsed_after": 0,
        "recovered": 0,        # was unparsed, now parses
        "regression": 0,       # was parsed, now does not  -> must never happen
        "content_change": 0,   # parsed both times, findings differ -> must never happen
        "error_label_change": 0,  # failed both times, different parse_error
        "examples": [],
    }

    for idx, row in enumerate(rows):
        if "parsed" not in row:
            continue
        stored = row.get("parsed") or {}
        was_ok = bool(stored.get("parse_success"))
        stats["parsed_before"] += int(was_ok)

        raw = row.get("raw_output") or ""
        err = row.get("error") or ""
        new = (parse_vlm_output(raw) if not err else parse_vlm_output("")).to_dict()
        is_ok = bool(new.get("parse_success"))
        stats["parsed_after"] += int(is_ok)

        if _canon(stored) != _canon(new):
            if not was_ok and is_ok:
                stats["recovered"] += 1
                kind = "recovered"
            elif was_ok and not is_ok:
                stats["regression"] += 1
                kind = "REGRESSION"
            elif was_ok and is_ok:
                stats["content_change"] += 1
                kind = "CONTENT_CHANGE"
            else:
                stats["error_label_change"] += 1
                kind = "error_label_change"
            if len(stats["examples"]) < 5:
                stats["examples"].append({
                    "row_index": idx,
                    "kind": kind,
                    "image_id": row.get("image_id"),
                    "condition": row.get("condition"),
                    "before_error": stored.get("parse_error", ""),
                    "after_error": new.get("parse_error", ""),
                })
        row["parsed"] = new

    changed = stats["recovered"] + stats["regression"] + stats["content_change"] + stats["error_label_change"]
    if changed and apply:
        rel = path.relative_to(OUTPUTS)
        dst = BACKUP / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        side = _sidecar(path)
        if side.exists():
            shutil.copy2(side, dst.with_suffix(dst.suffix + ".progress.json"))

        payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
        _atomic_write_bytes(path, payload)

        if side.exists():
            body = json.loads(side.read_text(encoding="utf-8"))
            body["bytes"] = path.stat().st_size
            body["records"] = len(rows)
            body["truncated_bytes_on_open"] = 0
            _atomic_write_bytes(side, json.dumps(body, ensure_ascii=False).encode("utf-8"))

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write the recomputed parsed fields (default: dry run)")
    ap.add_argument("--glob", default="**/*.jsonl",
                    help="glob under outputs/ (default: **/*.jsonl)")
    args = ap.parse_args()

    files = sorted(p for p in OUTPUTS.glob(args.glob) if p.is_file() and not _is_backup(p))
    if not files:
        print("no prediction files matched")
        return 1

    print("=" * 96)
    print("REPARSE PREDICTIONS -- recompute `parsed` from `raw_output` with the current parser")
    print("=" * 96)
    print(f"  mode : {'APPLY' if args.apply else 'DRY RUN'}")
    print(f"  root : {OUTPUTS.relative_to(ROOT)}")
    print(f"  glob : {args.glob}   ({len(files)} files)")
    print()

    results = []
    touched = 0
    total_rows = total_before = total_after = 0
    total_recovered = total_regression = total_content = total_errlabel = 0

    for path in files:
        try:
            stats = _reparse_file(path, args.apply)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR  {path.relative_to(ROOT)}: {exc}")
            results.append({"file": str(path.relative_to(ROOT)), "error": str(exc)})
            total_regression += 1
            continue
        results.append(stats)
        total_rows += stats["rows"]
        total_before += stats["parsed_before"]
        total_after += stats["parsed_after"]
        total_recovered += stats["recovered"]
        total_regression += stats["regression"]
        total_content += stats["content_change"]
        total_errlabel += stats["error_label_change"]
        if stats["recovered"] or stats["regression"] or stats["content_change"] or stats["error_label_change"]:
            touched += 1
            flag = "  <-- CHANGED"
            print("%-64s rows=%5d %5d -> %5d parsed  rec=%d reg=%d diff=%d%s"
                  % (stats["file"], stats["rows"], stats["parsed_before"], stats["parsed_after"],
                     stats["recovered"], stats["regression"], stats["content_change"], flag))
            for ex in stats["examples"]:
                print("        %-16s %s cond=%s  %s -> %s"
                      % (ex["kind"], ex["image_id"], ex["condition"],
                         ex["before_error"] or "(parsed)", ex["after_error"] or "(parsed)"))

    print()
    print("-" * 96)
    print("files=%d   files_changed=%d" % (len(files), touched))
    print("rows=%d   parsed %d -> %d" % (total_rows, total_before, total_after))
    print("recovered=%d  regression=%d  content_change=%d  error_label_change=%d"
          % (total_recovered, total_regression, total_content, total_errlabel))

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "mode": "apply" if args.apply else "dry_run",
        "glob": args.glob,
        "files_scanned": len(files),
        "files_changed": touched,
        "rows": total_rows,
        "parsed_before": total_before,
        "parsed_after": total_after,
        "recovered": total_recovered,
        "regression": total_regression,
        "content_change": total_content,
        "error_label_change": total_errlabel,
        "per_file": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"evidence -> {EVIDENCE.relative_to(ROOT)}")

    if total_regression or total_content:
        print()
        print("REPARSE: PROBLEM -- the parser changed rows that already parsed. "
              "The fix must be a strict superset; investigate before applying.")
        return 3
    if not args.apply and touched:
        print()
        print("DRY RUN -- nothing written. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())