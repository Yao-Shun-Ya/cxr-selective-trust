#!/usr/bin/env python
"""Decide the calibration *sample universe* with data, not preference.

The disagreement
----------------
Two metric paths in this repository score the same predictions on different rows:

* ``04_evaluate.py`` drops a row unless it has both a definite call (``y_pred``)
  and a probability. A model that answers ``uncertain`` has no ``y_pred``, so those
  rows are removed -- MedGemma loses 186 of 1800 test decisions (10.3%).
* ``07_calibration_selective.py`` and ``55_recompute_ece_impact.py`` keep every row
  whose ``probability_positive`` is numeric, regardless of the status the model gave.

Hence 0.232 (paper, Table 1, from ``metrics_by_label.csv``) vs 0.217 (raw
predictions) for MedGemma internal_test. Both are "correct"; they answer different
questions. The paper documents neither choice.

What this script establishes, so the choice can be made on evidence
------------------------------------------------------------------
1. **Do the dropped rows carry real probabilities?** If they do, they are ordinary
   probability statements and the standard ECE estimand (Guo et al., sklearn) covers
   them. If they do not, dropping them is the only option and there is no decision.
2. **In which direction does each universe move the number?** A restriction that
   happens to flatter the model is a much bigger problem than a restriction that
   does not, because it invites the reading "you dropped the hard cases".
3. **Is the hedging itself reported?** If the abstention/uncertain rate is already a
   headline quantity, then the definite-only universe is not hiding anything.

Run::

    .venv-vl\\Scripts\\python.exe scripts\\56_decide_calibration_universe.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.evaluation.metrics import expected_calibration_error  # noqa: E402

LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax", "no_finding"]
SPLIT = "internal_test"


def findings_of(obj: dict) -> dict:
    parsed = obj.get("parsed")
    if isinstance(parsed, dict) and isinstance(parsed.get("findings"), dict):
        return parsed["findings"]
    f = obj.get("findings")
    return f if isinstance(f, dict) else {}


def collect(path: Path) -> dict:
    """Per label: status counts, and (y_true, y_prob) for each universe."""
    per = {lb: {"status": Counter(), "definite_y": [], "definite_p": [],
                "anynum_y": [], "anynum_p": [],
                "unc_y": [], "unc_p": []} for lb in LABELS}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(obj.get("split")) != SPLIT:
            continue
        gt = obj.get("ground_truth") or {}
        findings = findings_of(obj)
        for lb in LABELS:
            truth = gt.get(f"label_{lb}")
            if truth not in ("positive", "negative"):
                continue
            y = 1.0 if truth == "positive" else 0.0
            entry = findings.get(lb) or {}
            status = str(entry.get("status", "not_mentioned"))
            p = entry.get("probability_positive")
            per[lb]["status"][status] += 1
            if isinstance(p, (int, float)):
                per[lb]["anynum_y"].append(y)
                per[lb]["anynum_p"].append(float(p))
                if status == "uncertain":
                    per[lb]["unc_y"].append(y)
                    per[lb]["unc_p"].append(float(p))
            if status in ("positive", "negative") and isinstance(p, (int, float)):
                per[lb]["definite_y"].append(y)
                per[lb]["definite_p"].append(float(p))
    return per


def ece(ys, ps) -> float:
    return expected_calibration_error(ys, ps) if len(ys) >= 20 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medgemma_4b_it_1000")
    args = ap.parse_args()

    path = ROOT / "outputs" / "overnight" / args.model / "predictions.jsonl"
    if not path.exists():
        print(f"missing {path}")
        return 2
    per = collect(path)

    print(f"model: {args.model}   split: {SPLIT}")
    print(f"\n=== 1. do the dropped rows carry real probabilities? ===")
    print(f"  {'label':18s} {'n_total':>7} {'definite':>8} {'uncertain':>9} "
          f"{'unc w/ prob':>11} {'unc w/o':>8}")
    tot_unc = tot_unc_num = tot_unc_nonum = 0
    for lb in LABELS:
        st = per[lb]["status"]
        n_def = st.get("positive", 0) + st.get("negative", 0)
        n_unc = st.get("uncertain", 0)
        n_unc_num = len(per[lb]["unc_p"])
        n_unc_nonum = n_unc - n_unc_num
        tot_unc += n_unc
        tot_unc_num += n_unc_num
        tot_unc_nonum += n_unc_nonum
        print(f"  {lb:18s} {sum(st.values()):7d} {n_def:8d} {n_unc:9d} "
              f"{n_unc_num:11d} {n_unc_nonum:8d}")
    print(f"  {'TOTAL':18s} {'':7s} {'':8s} {tot_unc:9d} {tot_unc_num:11d} "
          f"{tot_unc_nonum:8d}")
    print(f"\n  uncertain rows WITH a numeric probability: {tot_unc_num} of {tot_unc} "
          f"({100.0 * tot_unc_num / max(1, tot_unc):.1f}%)")
    if tot_unc_nonum == 0:
        print("  -> every uncertain row still states a probability, so those rows are")
        print("     ordinary probability statements and REMAIN valid inputs to ECE.")
    else:
        print(f"  -> {tot_unc_nonum} uncertain rows have NO probability; those cannot be")
        print("     scored by ECE under any universe, so the two universes differ only")
        print("     on the {tot_unc_num} rows that do carry one.")

    print(f"\n=== 2. direction: which universe flatters the model? ===")
    print(f"  {'label':18s} {'ECE definite':>13} {'ECE any-num':>12} "
          f"{'ECE unc-only':>13} {'delta':>9}")
    d_def, d_any = [], []
    for lb in LABELS:
        a = ece(per[lb]["definite_y"], per[lb]["definite_p"])
        b = ece(per[lb]["anynum_y"], per[lb]["anynum_p"])
        u = ece(per[lb]["unc_y"], per[lb]["unc_p"])
        if np.isfinite(a) and np.isfinite(b):
            d_def.append(a)
            d_any.append(b)
            print(f"  {lb:18s} {a:13.4f} {b:12.4f} "
                  f"{(f'{u:.4f}' if np.isfinite(u) else '   n/a'):>13} {b - a:+9.4f}")
    if d_def:
        md, ma = float(np.mean(d_def)), float(np.mean(d_any))
        print(f"\n  {'MACRO':18s} {md:13.4f} {ma:12.4f} {'':13s} {ma - md:+9.4f}")
        print(f"  paper Table 1 (definite universe): 0.2324   raw recompute: {ma:.4f}")
        if ma < md:
            print("  -> including uncertain rows LOWERS ECE, i.e. the definite-only")
            print("     universe is the HARSHER one. The paper is not flattered by its")
            print("     choice; if anything it understates calibration.")
        elif ma > md:
            print("  -> including uncertain rows RAISES ECE, i.e. the definite-only")
            print("     universe flatters the model by dropping rows it hedged on.")
        else:
            print("  -> no difference.")

    print(f"\n=== 3. is the hedging itself reported? ===")
    tot_rows = max(len(per[lb]["anynum_y"]) for lb in LABELS) or 1
    print(f"  uncertain share of scored decisions: "
          f"{tot_unc_num + tot_unc_nonum} of "
          f"{sum(sum(per[lb]['status'].values()) for lb in LABELS)} "
          f"({100.0 * (tot_unc + tot_unc_nonum) / max(1, sum(sum(per[lb]['status'].values()) for lb in LABELS)):.1f}%)")
    print("  the paper reports an abstention/uncertain rate separately? "
          "check 04_results.tex -- if yes, the definite-only universe hides nothing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
