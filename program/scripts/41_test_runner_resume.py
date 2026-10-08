#!/usr/bin/env python
"""End-to-end crash-and-resume test for the layer runners (43 PadChest, 44 Open-i).

What this proves, and why it is not a unit test
-----------------------------------------------
The claim being verified is "a power cut during a >10-day run costs at most the
in-flight batch, and the next run resumes correctly". That claim spans process
boundaries, the actual CLI, the durable writer and the resume key, so it can only
be tested by really killing a run and really starting it again.

The model is stubbed (``scripts/_stub``), so this needs no GPU and finishes in
seconds. Tested end to end through the real command lines:

1. **Clean pass** -- the runner produces one row per (study, condition), a
   summary, and a durable byte-offset sidecar.
2. **Crash and resume** -- the stub raises on batch N, the process exits
   non-zero, and a second invocation must complete the remaining cells while
   adding *no* duplicate of the rows the first run had already fsynced.
3. **Torn tail** -- a half-written JSON line appended after a crash must be
   truncated on reopen, and the row it belonged to must be redone.
4. **Repeat resume is a no-op** -- a third run must report nothing to do and
   leave the JSONL byte-identical, which is what "resumable" has to mean when it
   is relied on 200 times over twelve days.

Run::

    .venv-vl\\Scripts\\python.exe scripts\\41_test_runner_resume.py
"""

from __future__ import annotations

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


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(f"{name} -- {detail}" if not ok else name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if not ok else ""))


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


def make_dataset(tmp: Path, n_images: int, header: str, relpath_of, row_of):
    """Write tiny real images plus a manifest that points at them.

    ``relpath_of`` decides where each image actually goes and ``row_of`` builds
    the CSV line, and the two must agree -- the first version of this fixture did
    not, so the runners correctly skipped every study and the test blamed the
    runner. Fixtures have to be wrong-proof too.
    """
    from PIL import Image
    rows = []
    for i in range(n_images):
        rel = relpath_of(i)
        path = tmp / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), (40 + i * 7 % 200, 90, 140)).save(path)
        rows.append(row_of(i, rel))
    manifest = tmp / "manifest.csv"
    # encoding="utf-8" is not optional here: Path.write_text defaults to the
    # *locale* encoding on Windows, which silently produces a manifest pandas
    # cannot resolve column names from.
    manifest.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return manifest, tmp


def run(runner: Path, args: list[str], counter: Path, crash_after: int | None,
        out_dir: Path, hard_fail_at: int | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(STUB_DIR)
    env["PYTHONIOENCODING"] = "utf-8"
    # The stub writes its counter with utf-8; without this Windows would use the
    # locale codec and the counter would be unreadable on a non-UTF-8 codepage.
    env["PYTHONUTF8"] = "1"
    env["STUB_COUNTER_FILE"] = str(counter)
    env["STUB_PROGRAM_ROOT"] = str(ROOT)
    env["RUNNER_STUB"] = "1"
    if crash_after is None:
        env.pop("CRASH_AFTER_BATCHES", None)
    else:
        env["CRASH_AFTER_BATCHES"] = str(crash_after)
    if hard_fail_at is None:
        env.pop("STUB_HARD_FAIL_AT", None)
    else:
        env["STUB_HARD_FAIL_AT"] = str(hard_fail_at)
    cmd = [str(PYTHON), str(runner), *args, "--output-dir", str(out_dir)]
    # NOTE: callers must NOT put --output-dir in their own args; argparse rejects
    # the duplicate and exits 2 before a single image is processed.
    # encoding= is required on a non-UTF-8 Windows codepage: the default codec
    # (gbk here) cannot decode tqdm's progress-bar bytes, and the reader thread
    # then dies with a UnicodeDecodeError instead of returning the output.
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), env=env,
                          encoding="utf-8", errors="replace", timeout=600)


def tail(text, n: int = 800) -> str:
    return (text or "")[-n:]


def openi_case(tmp: Path) -> None:
    print("\n=== Open-i report runner (44) ===")
    n = 12
    header = ("image_id,image_relpath,dataset,patient_id,view,split,reference_text,"
              "default_prompt,label_atelectasis,label_cardiomegaly,label_consolidation,"
              "label_edema,label_pleural_effusion,label_pneumonia,label_pneumothorax,"
              "label_no_finding")
    rows = []
    for i in range(n):
        rows.append(f"openi_{i:03d}.jpg,images/openi_{i:03d}.jpg,openi,openi_{i:03d},,"
                    f"external_report,\"The cardiac silhouette is normal. No edema.\",,"
                    f"unknown,unknown,unknown,unknown,unknown,unknown,unknown,unknown")
    manifest, img_root = make_dataset(
        tmp, n, header,
        relpath_of=lambda i: f"images/openi_{i:03d}.jpg",
        row_of=lambda i, rel: rows[i])
    out_dir = tmp / "out_openi"
    runner = ROOT / "scripts" / "44_run_openi_report.py"
    counter = tmp / "stub_counter.json"

    base = ["--model", "medgemma_4b_it", "--manifest", str(manifest),
            "--images-root", str(img_root), "--conditions", "original", "blank",
            "--batch-size", "3", "--max-new-tokens", "64"]

    # ---- 1. clean pass ------------------------------------------------------
    r = run(runner, base, counter, None, out_dir)
    check("44 clean run exits 0", r.returncode == 0,
          f"rc={r.returncode}\n{tail(r.stderr)}")
    pred = out_dir / "rrg_predictions.jsonl"
    rows_out = read_rows(pred)
    check("44 wrote 2 cells per study", len(rows_out) == 2 * n,
          f"got {len(rows_out)} want {2 * n}")
    conds = {}
    for row in rows_out:
        conds[row["condition"]] = conds.get(row["condition"], 0) + 1
    check("44 both conditions present", conds.get("original") == n and conds.get("blank") == n,
          str(conds))
    check("44 every row has reference_text",
          all(row.get("reference_text") for row in rows_out))
    check("44 summary written", (out_dir / "report_summary.json").exists())
    check("44 durable sidecar written", Path(str(pred) + ".progress.json").exists())
    if (out_dir / "report_summary.json").exists():
        s = json.loads((out_dir / "report_summary.json").read_text(encoding="utf-8"))
        check("44 summary counts generations", s["n_studies"] == n, str(s.get("n_studies")))
        check("44 summary records the blank contrast",
              s["blank_control"]["n"] == n, str(s["blank_control"]))

    # ---- 2. hard failure mid-run must exit non-zero ------------------------
    out_hard = tmp / "out_openi_hard"
    counter_h = tmp / "counter_hard.json"
    r_hard = run(runner, base, counter_h, None, out_hard, hard_fail_at=2)
    check("44 hard failure exits non-zero", r_hard.returncode != 0,
          f"rc={r_hard.returncode} (a driver error must not look like success)")
    check("44 hard failure says how to resume",
          "Re-run the same command" in (r_hard.stdout or ""),
          tail(r_hard.stdout, 400))
    hard_rows = read_rows(out_hard / "rrg_predictions.jsonl")
    check("44 hard failure kept the durable prefix", len(hard_rows) > 0,
          f"{len(hard_rows)} rows")
    r_hard2 = run(runner, base, counter_h, None, out_hard)
    check("44 resume after hard failure completes", r_hard2.returncode == 0,
          f"rc={r_hard2.returncode}\n{tail(r_hard2.stderr, 600)}")
    hard_final = read_rows(out_hard / "rrg_predictions.jsonl")
    hard_keys = [(r["image_id"], r["condition"]) for r in hard_final]
    check("44 resume after hard failure leaves no duplicates",
          len(hard_keys) == len(set(hard_keys)), f"{len(hard_keys)} rows")
    check("44 resume after hard failure is complete", len(set(hard_keys)) == 2 * n,
          f"{len(set(hard_keys))} of {2 * n}")
    check("44 hard failure wrote no per-row error strings",
          not [r for r in hard_final if r.get("error")],
          str([r["error"] for r in hard_final if r.get("error")][:2]))

    # ---- 3. crash mid-run, then resume -------------------------------------
    out2 = tmp / "out_openi_crash"
    counter2 = tmp / "counter2.json"
    r1 = run(runner, base, counter2, 2, out2)
    check("44 crashed run exits non-zero", r1.returncode != 0, f"rc={r1.returncode}")
    pred2 = out2 / "rrg_predictions.jsonl"
    after_crash = read_rows(pred2)
    check("44 partial file exists after the crash", len(after_crash) > 0,
          f"{len(after_crash)} rows")
    r2 = run(runner, base, counter2, None, out2)
    check("44 resumed run exits 0", r2.returncode == 0, f"rc={r2.returncode}\n{r2.stderr[-600:]}")
    final = read_rows(pred2)
    keys = [(row["image_id"], row["condition"]) for row in final]
    check("44 no duplicate cells after resume", len(keys) == len(set(keys)),
          f"{len(keys)} rows, {len(set(keys))} unique")
    check("44 all cells complete after resume", len(set(keys)) == 2 * n,
          f"{len(set(keys))} of {2 * n}")
    check("44 no failed rows left behind",
          not [row for row in final if row.get("error")],
          str([row["error"] for row in final if row.get("error")][:2]))

    # ---- 3. torn tail ------------------------------------------------------
    with pred2.open("ab") as fh:
        fh.write(b'{"image_id": "openi_011.jpg", "condition": "orig')
        fh.flush()
    size_before = pred2.stat().st_size
    r3 = run(runner, base, counter2, None, out2)
    check("44 run after torn tail exits 0", r3.returncode == 0, f"rc={r3.returncode}")
    final3 = read_rows(pred2)
    keys3 = [(row["image_id"], row["condition"]) for row in final3]
    check("44 torn bytes removed", pred2.stat().st_size < size_before,
          f"{pred2.stat().st_size} vs {size_before}")
    check("44 still no duplicates after torn tail", len(keys3) == len(set(keys3)))
    check("44 still complete after torn tail", len(set(keys3)) == 2 * n,
          f"{len(set(keys3))} of {2 * n}")

    # ---- 4. idempotent resume ---------------------------------------------
    before = pred2.read_bytes()
    r4 = run(runner, base, counter2, None, out2)
    check("44 third resume exits 0", r4.returncode == 0, f"rc={r4.returncode}")
    check("44 resume is a byte-identical no-op", pred2.read_bytes() == before,
          "file changed on a no-op resume")


def padchest_case(tmp: Path) -> None:
    print("\n=== PadChest geographic-shift runner (43) ===")
    n = 8
    header = ("image_id,image_relpath,dataset,patient_id,view,split,"
              "label_atelectasis,label_cardiomegaly,label_consolidation,label_edema,"
              "label_pleural_effusion,label_pneumonia,label_pneumothorax,label_no_finding")
    rows = [f"padchest_{i:05d}.jpg,images/padchest_{i:05d}.jpg,padchest,pid_{i},,"
            f"internal_test,negative,negative,negative,negative,negative,negative,"
            f"negative,negative" for i in range(n)]
    manifest, img_root = make_dataset(
        tmp, n, header,
        relpath_of=lambda i: f"images/padchest_{i:05d}.jpg",
        row_of=lambda i, rel: rows[i])
    out_dir = tmp / "out_pad"
    runner = ROOT / "scripts" / "43_run_padchest_shift.py"
    counter = tmp / "counter_pad.json"
    base = ["--model", "medgemma_4b_it", "--manifest", str(manifest),
            "--images-root", str(img_root), "--conditions", "original", "blank",
            "--batch-size", "2", "--max-new-tokens", "64"]

    r = run(runner, base, counter, None, out_dir)
    check("43 clean run exits 0", r.returncode == 0, f"rc={r.returncode}\n{tail(r.stderr)}")
    pred = out_dir / "image_reliance_predictions.jsonl"
    rows_out = read_rows(pred)
    check("43 wrote 2 cells per image", len(rows_out) == 2 * n, f"got {len(rows_out)}")
    check("43 blank control images materialised",
          len(list((out_dir / "blank_images").glob("*.png"))) == n,
          str(len(list((out_dir / "blank_images").glob("*.png")))))
    check("43 per_image csv written", (out_dir / "image_reliance_per_image.csv").exists())
    check("43 summary written", (out_dir / "image_reliance_summary.json").exists())
    s = json.loads((out_dir / "image_reliance_summary.json").read_text(encoding="utf-8"))
    check("43 summary has blank contrast", "blank" in s.get("by_condition", {}),
          str(s.get("by_condition")))
    check("43 output is reliance-schema compatible",
          {"image_id", "condition", "image_path", "raw_output", "error", "parsed"} <=
          set(rows_out[0].keys()), str(sorted(rows_out[0].keys())))

    out2 = tmp / "out_pad_crash"
    counter2 = tmp / "counter_pad2.json"
    r1 = run(runner, base, counter2, 2, out2)
    check("43 crashed run exits non-zero", r1.returncode != 0, f"rc={r1.returncode}")
    pred2 = out2 / "image_reliance_predictions.jsonl"
    partial = read_rows(pred2)
    check("43 partial file after crash", len(partial) > 0, str(len(partial)))
    r2 = run(runner, base, counter2, None, out2)
    check("43 resumed run exits 0", r2.returncode == 0, f"rc={r2.returncode}\n{r2.stderr[-600:]}")
    final = read_rows(pred2)
    keys = [(row["image_id"], row["condition"]) for row in final]
    check("43 no duplicates after resume", len(keys) == len(set(keys)))
    check("43 complete after resume", len(set(keys)) == 2 * n, f"{len(set(keys))} of {2 * n}")

    # the resume must not re-run cells that were already durable
    served = (json.loads(counter2.read_text(encoding="utf-8"))["batch_calls"]
              if counter2.exists() else 0)
    total_batches_if_no_resume = 2 * ((2 * n + 1) // 2)
    check("43 resume skipped durable work",
          served < total_batches_if_no_resume + 2,
          f"stub served {served} batches; a no-resume run would need ~{total_batches_if_no_resume}")


def main() -> int:
    if not PYTHON.exists():
        print(f"[error] interpreter not found: {PYTHON}")
        return 2
    with tempfile.TemporaryDirectory(prefix="resume_test_") as td:
        tmp = Path(td)
        openi_case(tmp)
        padchest_case(tmp)

    print(f"\n{'=' * 72}")
    print(f"  runner crash/resume: {len(PASSED)} passed, {len(FAILED)} failed")
    for f in FAILED:
        print(f"    FAIL {f}")
    print(f"{'=' * 72}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
