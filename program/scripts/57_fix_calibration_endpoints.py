"""Recompute the calibration endpoints so both ends use the corrected convention.

Why this is needed
------------------
``05_discussion`` and ``04_results`` quote a temperature-scaling pair for
``no_finding``: an uncalibrated ECE and a post-scaling ECE. The uncalibrated end now
comes from the regenerated ``metrics_by_label.csv`` (corrected bin convention), but
the post-scaling end still lives in ``calibration_by_label.csv``, which was written
BEFORE the convention was corrected. The paper therefore quotes a pair whose two
ends were produced by different conventions -- exactly the "two accounts mixed"
failure this whole pass exists to remove.

The fix is not to edit the number by hand: it is to re-run the calibration script so
both ends are produced by the same convention in the same pass. Then the pair is
internally consistent and reproducible.

Run::

    .venv-vl\\Scripts\\python.exe scripts\\57_fix_calibration_endpoints.py
    .venv-vl\\Scripts\\python.exe scripts\\57_fix_calibration_endpoints.py --write
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROG = Path(__file__).resolve().parents[1]
MODELS = ["medgemma_4b_it_1000", "qwen25_vl_7b_instruct_1000"]
LABELS = ["no_finding", "cardiomegaly", "consolidation", "edema"]


def read_pair(model: str) -> pd.DataFrame:
    p = PROG / "outputs" / "overnight" / model / "calibration_selective" / \
        "calibration_by_label.csv"
    if not p.exists():
        print(f"  {model}: no calibration_by_label.csv")
        return pd.DataFrame()
    return pd.read_csv(p)


def show(tag: str) -> None:
    print(f"\n--- {tag}")
    print(f"  {'model':28s} {'label':16s} {'ece_before':>11} {'ece_after':>10}")
    for m in MODELS:
        d = read_pair(m)
        if d.empty or "label" not in d.columns:
            continue
        for lb in LABELS:
            r = d[d["label"] == lb]
            if not len(r):
                continue
            row = r.iloc[0]
            print(f"  {m:28s} {lb:16s} {row.get('ece_before'):11.4f} "
                  f"{row.get('ece_after'):10.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="actually re-run the calibration script")
    args = ap.parse_args()

    show("BEFORE (ece_after is from the pre-correction convention)")
    if not args.write:
        print("\n(dry run: pass --write to re-run 07_calibration_selective.py)")
        return 0

    for m in MODELS:
        pred = PROG / "outputs" / "overnight" / m / "predictions.jsonl"
        out = PROG / "outputs" / "overnight" / m / "calibration_selective"
        if not pred.exists():
            print(f"skipped {m}: {pred} missing")
            continue
        print(f"\nre-running calibration for {m} ...")
        r = subprocess.run([sys.executable,
                            str(PROG / "scripts" / "07_calibration_selective.py"),
                            "--predictions", str(pred), "--output-dir", str(out)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", cwd=str(PROG), timeout=1800)
        print((r.stdout or "").strip().splitlines()[-1] if r.stdout else "",
              flush=True)
        if r.returncode != 0:
            print(f"  FAILED rc={r.returncode}: {(r.stderr or '')[-300:]}")

    show("AFTER (both ends from the corrected convention)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
