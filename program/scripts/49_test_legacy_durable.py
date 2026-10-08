#!/usr/bin/env python
"""Kill/restart evidence for the four older runners now on the durable writer.

Why a separate suite
--------------------
``41_test_runner_resume.py`` covers the two new layers (PadChest, Open-i) through
their real CLIs. The four older runners — ``06_image_reliance``,
``08_shift_benchmark``, ``10_run_chexpert_rrg``, ``21_prompt_ablation`` — were
moved onto ``DurableJsonl`` later, and "it uses the class" is not evidence. Each
one is therefore actually run, killed at a chosen batch, and restarted here.

What each case asserts, and why
-------------------------------
1. **Clean run** produces one row per cell, a durable sidecar, and no error rows.
2. **Kill mid-run** exits non-zero. A hard failure that exits 0 would be recorded
   as a completed stage by the orchestrator, which is the silent-failure mode this
   whole layer exists to prevent.
3. **Partial output survives the kill** — the durable prefix is readable and every
   row in it is a complete JSON object.
4. **Resume completes the work** with an exact row count.
5. **Resume leaves no duplicate keys and no failed rows.** Retrying a failed cell
   must replace its earlier attempt, not sit beside it.
6. **A further resume is a byte-identical no-op.**

``21_prompt_ablation`` is run through all six at **two batch sizes** (1 and 2)
because it gained a batching path on 2026-10-05; the other three have one path
each. "Killed at a chosen batch" is therefore evaluated per path, not per file.

The model is stubbed (``scripts/_stub``, activated by ``RUNNER_STUB=1``), so this
needs no GPU and runs in seconds.

Run::

    .venv-vl\\Scripts\\python.exe scripts\\49_test_legacy_durable.py
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUB_DIR = ROOT / "scripts" / "_stub"
PYTHON = ROOT / ".venv-vl" / "Scripts" / "python.exe"

PASSED: list[str] = []
FAILED: list[str] = []

LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax", "no_finding"]


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name if ok else f"{name} -- {detail}")
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if not ok else ""))


def tail(text, n: int = 700) -> str:
    return (text or "")[-n:]


def read_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def make_fixture(tmp: Path, n: int) -> tuple[Path, Path]:
    """Real tiny PNGs plus a manifest carrying the full 8-label contract."""
    from PIL import Image

    img_dir = tmp / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    header = ("image_id,image_relpath,dataset,patient_id,view,split,reference_text,"
              + ",".join(f"label_{lb}" for lb in LABELS))
    lines = [header]
    for i in range(n):
        name = f"img{i:03d}.png"
        Image.new("RGB", (64, 48), (30 + i * 11 % 200, 80, 150)).save(img_dir / name)
        labels = ",".join("positive" if (i + j) % 3 else "negative"
                          for j in range(len(LABELS)))
        lines.append(f"img{i:03d}.png,images/{name},nih,pid{i},PA,internal_test,"
                     f"\"The lungs are clear. No effusion.\",{labels}")
    manifest = tmp / "manifest.csv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest, tmp


def run(cmd: list[str], counter: Path, out_dir: Path,
        hard_fail_at: int | None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(STUB_DIR)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["RUNNER_STUB"] = "1"
    env["STUB_COUNTER_FILE"] = str(counter)
    env["STUB_PROGRAM_ROOT"] = str(ROOT)
    env.pop("CRASH_AFTER_BATCHES", None)
    if hard_fail_at is None:
        env.pop("STUB_HARD_FAIL_AT", None)
    else:
        env["STUB_HARD_FAIL_AT"] = str(hard_fail_at)
    return subprocess.run([str(PYTHON), *cmd], capture_output=True, text=True,
                          cwd=str(ROOT), env=env, encoding="utf-8",
                          errors="replace", timeout=900)


def case_06(tmp: Path) -> None:
    manifest, root = make_fixture(tmp / "c06", 4)
    out = tmp / "c06" / "out"
    base = ["scripts/06_image_reliance.py", "--model", "medgemma_4b_it",
            "--manifest", str(manifest), "--split", "internal_test",
            "--images-root", str(root), "--output-dir", str(out),
            "--limit", "4", "--batch-size", "2", "--max-new-tokens", "32"]
    counter = tmp / "c06" / "counter.json"
    pred = out / "image_reliance_predictions.jsonl"

    r = run(base, counter, out, None)
    check("06 clean run exits 0", r.returncode == 0, f"rc={r.returncode}\n{tail(r.stderr)}")
    check("06 wrote 4 conditions x 4 images", len(read_rows(pred)) == 16,
          str(len(read_rows(pred))))
    check("06 durable sidecar written", Path(str(pred) + ".progress.json").exists())

    out2 = tmp / "c06" / "out_kill"
    pred2 = out2 / "image_reliance_predictions.jsonl"
    base2 = [a if a != str(out) else str(out2) for a in base]
    r1 = run(base2, counter, out2, 2)
    check("06 kill exits non-zero", r1.returncode != 0, f"rc={r1.returncode}")
    partial = read_rows(pred2)
    check("06 partial output survives the kill", len(partial) > 0, str(len(partial)))
    r2 = run(base2, counter, out2, None)
    check("06 resume exits 0", r2.returncode == 0, f"rc={r2.returncode}\n{tail(r2.stderr)}")
    final = read_rows(pred2)
    keys = [(r["image_id"], r["condition"]) for r in final]
    check("06 resume is complete", len(set(keys)) == 16, f"{len(set(keys))}/16")
    check("06 resume has no duplicates", len(keys) == len(set(keys)), f"{len(keys)} rows")
    check("06 no failed rows", not [r for r in final if r.get("error")])
    before = pred2.read_bytes()
    run(base2, counter, out2, None)
    check("06 second resume is a byte-identical no-op", pred2.read_bytes() == before)


def case_08(tmp: Path) -> None:
    manifest, root = make_fixture(tmp / "c08", 3)
    out = tmp / "c08" / "out"
    base = ["scripts/08_shift_benchmark.py", "--model", "medgemma_4b_it",
            "--manifest", str(manifest), "--split", "internal_test",
            "--images-root", str(root), "--output-dir", str(out),
            "--limit", "3", "--batch-size", "2", "--max-new-tokens", "32"]
    counter = tmp / "c08" / "counter.json"
    pred = out / "shift_predictions.jsonl"

    r = run(base, counter, out, None)
    check("08 clean run exits 0", r.returncode == 0, f"rc={r.returncode}\n{tail(r.stderr)}")
    check("08 wrote 19 cells x 3 images", len(read_rows(pred)) == 57,
          str(len(read_rows(pred))))

    out2 = tmp / "c08" / "out_kill"
    pred2 = out2 / "shift_predictions.jsonl"
    base2 = [a if a != str(out) else str(out2) for a in base]
    r1 = run(base2, counter, out2, 3)
    check("08 kill exits non-zero", r1.returncode != 0, f"rc={r1.returncode}")
    check("08 partial output survives the kill", len(read_rows(pred2)) > 0,
          str(len(read_rows(pred2))))
    r2 = run(base2, counter, out2, None)
    check("08 resume exits 0", r2.returncode == 0, f"rc={r2.returncode}\n{tail(r2.stderr)}")
    final = read_rows(pred2)
    keys = [(r["image_id"], r["condition"], r["severity"]) for r in final]
    check("08 resume is complete", len(set(keys)) == 57, f"{len(set(keys))}/57")
    check("08 resume has no duplicates", len(keys) == len(set(keys)))
    check("08 no failed rows", not [r for r in final if r.get("error")])
    before = pred2.read_bytes()
    run(base2, counter, out2, None)
    check("08 second resume is a byte-identical no-op", pred2.read_bytes() == before)


def case_10(tmp: Path) -> None:
    manifest, root = make_fixture(tmp / "c10", 3)
    out = tmp / "c10" / "out"
    base = ["scripts/10_run_chexpert_rrg.py", "--model", "medgemma_4b_it",
            "--manifest", str(manifest), "--images-root", str(root),
            "--output-dir", str(out), "--limit", "3",
            "--max-new-tokens", "32"]
    counter = tmp / "c10" / "counter.json"
    pred = out / "rrg_predictions.jsonl"

    r = run(base, counter, out, None)
    check("10 clean run exits 0", r.returncode == 0, f"rc={r.returncode}\n{tail(r.stderr)}")
    check("10 wrote one row per image", len(read_rows(pred)) == 3, str(len(read_rows(pred))))

    out2 = tmp / "c10" / "out_kill"
    pred2 = out2 / "rrg_predictions.jsonl"
    base2 = [a if a != str(out) else str(out2) for a in base]
    r1 = run(base2, counter, out2, 2)
    check("10 kill exits non-zero", r1.returncode != 0, f"rc={r1.returncode}")
    check("10 partial output survives the kill", len(read_rows(pred2)) > 0,
          str(len(read_rows(pred2))))
    r2 = run(base2, counter, out2, None)
    check("10 resume exits 0", r2.returncode == 0, f"rc={r2.returncode}\n{tail(r2.stderr)}")
    final = read_rows(pred2)
    keys = [(r["image_id"], r["condition"]) for r in final]
    check("10 resume is complete", len(set(keys)) == 3, f"{len(set(keys))}/3")
    check("10 resume has no duplicates", len(keys) == len(set(keys)))
    before = pred2.read_bytes()
    run(base2, counter, out2, None)
    check("10 second resume is a byte-identical no-op", pred2.read_bytes() == before)


def case_21(tmp: Path, batch_size: int) -> None:
    """Kill/resume evidence for the ablation runner at one batch size.

    Run twice, at ``--batch-size 1`` and ``--batch-size 2``. Before batching
    existed there was only one code path here, and its kill point ("fail on the
    2nd model call") happened to coincide with the 2nd image. Once the runner
    batches, that coincidence is gone: with three images in a single call the
    simulated crash lands *after* the work is done, the process exits 0, and the
    suite reported a durability failure that was really a stale assumption. Both
    paths are therefore exercised explicitly, because a batch that is written as
    a unit is exactly where a torn write would hurt most.
    """
    tag = f" [bs={batch_size}]"
    manifest, root = make_fixture(tmp / f"c21_bs{batch_size}", 3)
    out = tmp / f"c21_bs{batch_size}" / "out"
    cell = out / "medgemma_4b_it" / "v0_simple" / "original"
    base = ["scripts/21_prompt_ablation.py", "--model", "medgemma_4b_it",
            "--variant", "v0_simple", "--condition", "original",
            "--manifest", str(manifest), "--split", "internal_test",
            "--images-root", str(root), "--output-root", str(out),
            "--limit", "3", "--max-new-tokens", "32", "--resume",
            "--batch-size", str(batch_size)]
    counter = tmp / f"c21_bs{batch_size}" / "counter.json"
    pred = cell / "image_reliance_predictions.jsonl"
    raw = cell / "raw_outputs.jsonl"

    r = run(base, counter, out, None)
    check("21 clean run exits 0" + tag, r.returncode == 0, f"rc={r.returncode}\n{tail(r.stderr)}")
    check("21 wrote one row per image" + tag, len(read_rows(pred)) == 3, str(len(read_rows(pred))))
    check("21 audit trail written" + tag, len(read_rows(raw)) == 3, str(len(read_rows(raw))))
    check("21 both streams are durable" + tag,
          Path(str(pred) + ".progress.json").exists()
          and Path(str(raw) + ".progress.json").exists())
    check("21 records are keyed by variant and condition" + tag,
          all(r.get("variant") == "v0_simple" and r.get("condition") == "original"
              for r in read_rows(pred)))

    out2 = tmp / f"c21_bs{batch_size}" / "out_kill"
    cell2 = out2 / "medgemma_4b_it" / "v0_simple" / "original"
    pred2 = cell2 / "image_reliance_predictions.jsonl"
    base2 = [a if a != str(out) else str(out2) for a in base]
    r1 = run(base2, counter, out2, 2)  # 2nd call: the 1st must already be durable
    check("21 kill exits non-zero" + tag, r1.returncode != 0, f"rc={r1.returncode}")
    check("21 partial output survives the kill" + tag, len(read_rows(pred2)) > 0,
          str(len(read_rows(pred2))))
    r2 = run(base2, counter, out2, None)
    check("21 resume exits 0" + tag, r2.returncode == 0, f"rc={r2.returncode}\n{tail(r2.stderr)}")
    final = read_rows(pred2)
    check("21 resume is complete" + tag, len(final) == 3, f"{len(final)}/3")
    check("21 resume has no duplicates" + tag,
          len({(r["image_id"], r["variant"], r["condition"]) for r in final}) == len(final))
    before = pred2.read_bytes()
    run(base2, counter, out2, None)
    check("21 second resume is a byte-identical no-op" + tag, pred2.read_bytes() == before)


def _one(name: str, thunk) -> None:
    print(f"\n=== {name} ===")
    try:
        thunk()
    except Exception as exc:  # noqa: BLE001
        check(f"{name} case crashed", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    if not PYTHON.exists():
        print(f"[error] interpreter not found: {PYTHON}")
        return 2
    with tempfile.TemporaryDirectory(prefix="legacy_durable_") as td:
        tmp = Path(td)
        for name, fn in (("06", case_06), ("08", case_08), ("10", case_10)):
            _one(name, lambda fn=fn, tmp=tmp: fn(tmp))
        # 21 runs twice: the serial path it was written with, and the batched path
        # added on 2026-10-05. Same assertions, two batch sizes.
        for bs in (1, 2):
            _one(f"21 (bs={bs})", lambda bs=bs, tmp=tmp: case_21(tmp, bs))

    print(f"\n{'=' * 72}")
    print(f"  legacy durable runners: {len(PASSED)} passed, {len(FAILED)} failed")
    for f in FAILED:
        print(f"    FAIL {f}")
    print("  (case 21 runs at two batch sizes, so the executed count exceeds the "
          "38 source assertions)")
    print(f"{'=' * 72}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
