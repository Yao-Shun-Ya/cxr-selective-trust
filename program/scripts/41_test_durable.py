#!/usr/bin/env python
"""Crash-safety tests for ``DurableJsonl``.

These are not unit tests of a happy path. Every case here simulates a real way a
>10-day run loses data, and asserts what the runner must do about it:

1. **Torn tail** -- the process was killed mid-write, leaving half a JSON line.
   On reopen the file must be truncated back to the last durable offset, so the
   torn row cannot be counted as a result.
2. **Unflushed tail** -- rows were written but the progress sidecar was not
   updated. They must be treated as suspect and redone, never silently kept.
3. **Failed row** -- a row whose generation raised must be retried by default,
   because a transient CUDA error must not become a permanent hole.
4. **Stale sidecar** -- a sidecar pointing past the end of the file (file
   deleted or replaced) must not crash and must not invent offsets.
5. **No sidecar** -- upgrading in place: an existing JSONL with complete lines
   must be adopted without redoing that work.

Run::

    .venv-vl\\Scripts\\python.exe scripts\\41_test_durable.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.inference.durable import DurableJsonl, read_jsonl  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(f"{name} -- {detail}")
        print(f"  [FAIL] {name}  {detail}")


def key_of(rec: dict) -> tuple:
    return (str(rec["image_id"]), str(rec["condition"]))


def row(iid: str, cond: str, text: str = "ok") -> dict:
    return {"image_id": iid, "condition": cond, "raw_output": text, "error": ""}


def failing_row(iid: str, cond: str, err: str = "CUDA error") -> dict:
    return {"image_id": iid, "condition": cond, "raw_output": "", "error": err}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="durable_test_") as td:
        tmp = Path(td)

        # ---------------------------------------------------------------- case 1
        print("\ncase 1: normal write, then resume skips completed work")
        p1 = tmp / "a.jsonl"
        with DurableJsonl(p1, key=key_of) as out:
            out.write_many([row("i1", "original"), row("i1", "blank")])
            out.write_many([row("i2", "original")])
            check("1a records on disk", len(out.records) == 3, str(len(out.records)))
            check("1b durable offset > 0", out._offset > 0)
        check("1c progress sidecar written", p1.with_suffix(".jsonl.progress.json").exists())
        with DurableJsonl(p1, key=key_of) as out:
            check("1d no extra work needed for i1/original",
                  not out.needs("i1", "original"))
            check("1e i2/blank still needed", out.needs("i2", "blank"))
            check("1f nothing appended on reopen", out._appended == 0)

        # ---------------------------------------------------------------- case 2
        print("\ncase 2: torn tail is truncated away, not kept")
        p2 = tmp / "b.jsonl"
        with DurableJsonl(p2, key=key_of) as out:
            out.write_many([row("t1", "original")])
        durable = p2.stat().st_size
        with p2.open("ab") as fh:                      # simulate a torn write
            fh.write(b'{"image_id": "t2", "condition": "orig')
            fh.flush()
        check("2a file now longer than durable", p2.stat().st_size > durable)
        with DurableJsonl(p2, key=key_of) as out:
            check("2c torn bytes dropped", out._truncated_bytes > 0,
                  str(out._truncated_bytes))
            check("2d file back to durable size", p2.stat().st_size == durable,
                  f"{p2.stat().st_size} vs {durable}")
            check("2e torn row is NOT counted done", out.needs("t2", "original"))
            check("2f intact row survives", not out.needs("t1", "original"))

        # ---------------------------------------------------------------- case 3
        print("\ncase 3: rows written after the sidecar was stale are redone")
        p3 = tmp / "c.jsonl"
        with DurableJsonl(p3, key=key_of) as out:
            out.write_many([row("s1", "original")])
        stale = json.loads(p3.with_suffix(".jsonl.progress.json").read_text())
        with p3.open("ab") as fh:                      # complete row, no fsync/marker
            fh.write((json.dumps(row("s2", "original")) + "\n").encode("utf-8"))
            fh.flush()
        with DurableJsonl(p3, key=key_of) as out:
            check("3a unmarked complete row was truncated",
                  out._truncated_bytes > 0, str(out._truncated_bytes))
            check("3b it will be redone", out.needs("s2", "original"))
            check("3c marked row kept", not out.needs("s1", "original"))
        check("3d sidecar preserved the old durable size",
              stale["bytes"] > 0 and stale["records"] == 1, json.dumps(stale))

        # ---------------------------------------------------------------- case 4
        print("\ncase 4: failed rows are retried by default, kept when asked")
        p4 = tmp / "d.jsonl"
        with DurableJsonl(p4, key=key_of) as out:
            out.write_many([row("f1", "original"), failing_row("f2", "original")])
        with DurableJsonl(p4, key=key_of) as out:
            check("4a failed row retried", out.needs("f2", "original"))
            check("4b good row not retried", not out.needs("f1", "original"))
        with DurableJsonl(p4, key=key_of, retry_failed=False) as out:
            check("4c failed row kept when retry_failed=False",
                  not out.needs("f2", "original"))
        check("4d both rows still on disk", len(read_jsonl(p4)) == 2)

        # ---------------------------------------------------------------- case 5
        print("\ncase 5: missing or absurd sidecar does not corrupt state")
        p5 = tmp / "e.jsonl"
        with DurableJsonl(p5, key=key_of) as out:
            out.write_many([row("n1", "original"), row("n2", "blank")])
        side = p5.with_suffix(".jsonl.progress.json")
        side.unlink()                                   # simulate an upgrade
        with DurableJsonl(p5, key=key_of) as out:
            check("5a adopted existing complete lines", len(out.records) == 2)
            check("5b no truncation without a sidecar", out._truncated_bytes == 0)
            check("5c known work recognised", not out.needs("n1", "original"))
        side.write_text(json.dumps({"bytes": 10 ** 9, "records": 999}))
        with DurableJsonl(p5, key=key_of) as out:
            check("5d absurd offset ignored, file intact", len(out.records) == 2,
                  str(len(out.records)))
        p6 = tmp / "f.jsonl"
        p6.write_text("")                               # empty file, no sidecar
        with DurableJsonl(p6, key=key_of) as out:
            check("5e empty file opens cleanly", out.records == [] and out.needs("x", "y"))

        # ---------------------------------------------------------------- case 6
        print("\ncase 6: exception inside the block still leaves resumable state")
        p7 = tmp / "g.jsonl"
        try:
            with DurableJsonl(p7, key=key_of) as out:
                out.write_many([row("e1", "original")])
                raise RuntimeError("simulated crash mid-stage")
        except RuntimeError:
            pass
        with DurableJsonl(p7, key=key_of) as out:
            check("6a written work survived the exception", len(out.records) == 1)
            check("6b and is not redone", not out.needs("e1", "original"))

        # ---------------------------------------------------------------- case 7
        print("\ncase 7: retrying a failed cell must not keep the failed attempt")
        p8 = tmp / "h.jsonl"

        # run 1: one good cell, one that fails
        with DurableJsonl(p8, key=key_of) as out:
            out.write_many([row("r1", "original"), failing_row("r2", "original")])
        check("7a both attempts recorded initially", len(read_jsonl(p8)) == 2)

        # run 2: r2 fails again and r3 succeeds. The superseded failed attempts for
        # r2 are dropped at open, so the file must not grow by one per retry.
        with DurableJsonl(p8, key=key_of) as out:
            check("7b failed cell still pending", out.needs("r2", "original"))
            out.write_many([failing_row("r2", "original"), row("r3", "original")])
        lines_after_2 = read_jsonl(p8)
        check("7c a retry does not grow the file", len(lines_after_2) == 3,
              f"{len(lines_after_2)} lines: {[r['image_id'] for r in lines_after_2]}")

        # run 3: r2 finally succeeds. There must be exactly one r2 row afterwards,
        # and it must be the successful one -- keeping both would mean any reader
        # that counts rows, or takes the first row per key, reports the failure.
        with DurableJsonl(p8, key=key_of) as out:
            out.write_many([row("r2", "original")])
        on_disk = read_jsonl(p8)
        r2_rows = [r for r in on_disk if r["image_id"] == "r2"]
        check("7d exactly one r2 row survives", len(r2_rows) == 1,
              f"{len(r2_rows)} r2 rows: {[bool(r.get('error')) for r in r2_rows]}")
        check("7e the surviving r2 row is the successful one",
              bool(r2_rows) and not r2_rows[0].get("error"),
              f"error={r2_rows[0].get('error') if r2_rows else 'n/a'!r}")
        check("7f three unique keys, three lines", len(on_disk) == 3
              and len({(r["image_id"], r["condition"]) for r in on_disk}) == 3,
              f"{len(on_disk)} lines")
        check("7g no failed rows remain", not [r for r in on_disk if r.get("error")])
        with DurableJsonl(p8, key=key_of) as out:
            check("7h nothing left to do", all(
                not out.needs(*k) for k in [("r1", "original"), ("r2", "original"),
                                            ("r3", "original")]))
        with DurableJsonl(p8, key=key_of) as out:
            check("7i a further no-op resume collapses nothing", out._deduped == 0,
                  str(out._deduped))

        # ---------------------------------------------------------------- case 8
        print("\ncase 8: dedupe does not fire when every key is unique")
        p9 = tmp / "i.jsonl"
        with DurableJsonl(p9, key=key_of) as out:
            out.write_many([row(f"u{i}", "original") for i in range(5)])
        with DurableJsonl(p9, key=key_of) as out:
            check("8a nothing collapsed", out._deduped == 0, str(out._deduped))
            check("8b all five records intact", len(out.records) == 5)

    print(f"\n{'=' * 70}")
    print(f"  durable jsonl: {len(PASSED)} passed, {len(FAILED)} failed")
    for f in FAILED:
        print(f"    FAIL {f}")
    print(f"{'=' * 70}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
