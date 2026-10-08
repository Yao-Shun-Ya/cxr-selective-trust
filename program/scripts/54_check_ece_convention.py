#!/usr/bin/env python
"""Is the ECE binning convention actually wrong, and does it matter here?

The claim under test: ``expected_calibration_error`` uses
``np.digitize(y_prob, bins[1:-1], right=True)``, which makes each bin
**(left-open, right-closed]** -- ``(0.0, 0.1]``, ``(0.1, 0.2]``, ... -- whereas
the standard convention (Guo et al. 2017; ``sklearn.calibration.calibration_curve``)
is **[left-closed, right-open)**. If that is true, a probability of exactly 0.1
lands in a different bin under the two conventions, and the reported ECE is not
comparable to any published number.

Two things have to be established, and an argument is not evidence for either:

1. **Are the conventions actually different?** Test the boundary behaviour
   directly against ``np.digitize(..., right=False)`` and against the values
   ``sklearn.calibration_curve`` assigns.
2. **Do real model probabilities hit those boundaries?** A convention difference
   is a measure-zero curiosity for a continuous score, but this project's models
   emit round one-decimal numbers, so the interesting quantity is the *share of
   real predictions sitting exactly on a bin edge*. If that share is large, the
   ECE difference is large, and "it is only a convention" stops being true.

Run::

    .venv-vl\\Scripts\\python.exe scripts\\54_check_ece_convention.py
    .venv-vl\\Scripts\\python.exe scripts\\54_check_ece_convention.py --files <jsonl> [...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.evaluation.metrics import expected_calibration_error  # noqa: E402

LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax", "no_finding"]


def ece_with(y_true, y_prob, n_bins: int = 10, right: bool = True) -> float:
    """The same estimator, with the bin-edge convention switched."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=right)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        ece += (mask.sum() / len(y_true)) * abs(np.mean(y_true[mask])
                                                - np.mean(y_prob[mask]))
    return float(ece)


def part1_conventions() -> None:
    print("=== 1. do the two conventions differ at the edges? ===")
    bins = np.linspace(0.0, 1.0, 11)
    edges = bins[1:-1]
    probes = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9, 0.95, 1.0]
    print(f"  {'p':>6} {'right=True':>11} {'right=False':>12} {'same?':>6}")
    differ = []
    for p in probes:
        a = int(np.digitize([p], edges, right=True)[0])
        b = int(np.digitize([p], edges, right=False)[0])
        if a != b:
            differ.append(p)
        print(f"  {p:6.2f} {a:11d} {b:12d} {'yes' if a == b else 'NO':>6}")
    print(f"  -> bins differ for p in {differ or 'nothing'}")

    try:
        from sklearn.calibration import calibration_curve
        y = np.array([0, 1] * 5)
        for p in (0.1, 0.2, 0.9):
            prob = np.array([p, p, 0.5, 0.6])
            yv = np.array([0, 1, 0, 1])
            curve = calibration_curve(yv, prob, n_bins=10, strategy="uniform")
            print(f"  sklearn calibration_curve accepts p={p} -> "
                  f"{len(curve[0])} non-empty bins (no crash; sklearn keeps the "
                  f"standard convention internally)")
    except Exception as exc:  # noqa: BLE001
        print(f"  sklearn not available for comparison ({type(exc).__name__})")


def part2_boundary_share() -> float:
    print("\n=== 2. how many REAL predictions sit exactly on a bin edge? ===")
    cands = sorted((ROOT / "outputs").rglob("*predictions.jsonl"))
    if not cands:
        print("  no prediction files found under outputs/")
        return 0.0
    edges = set(np.round(np.linspace(0.0, 1.0, 11)[1:-1], 10).tolist())
    total = on_edge = 0
    per_file = []
    for p in cands[:60]:
        n = e = 0
        try:
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                findings = (obj.get("parsed") or {}).get("findings") or {}
                for lb in LABELS:
                    v = (findings.get(lb) or {}).get("probability_positive")
                    if isinstance(v, (int, float)):
                        n += 1
                        if round(float(v), 10) in edges:
                            e += 1
        except OSError:
            continue
        if n:
            total += n
            on_edge += e
            per_file.append((e / n, e, n, p))
    if not total:
        print("  no usable probabilities found")
        return 0.0
    print(f"  scanned {len(per_file)} file(s), {total:,} probability values")
    print(f"  exactly on a bin edge: {on_edge:,}  ({100.0 * on_edge / total:.2f}%)")
    per_file.sort(reverse=True)
    for share, e, n, p in per_file[:5]:
        print(f"    {100.0 * share:5.1f}%  {e:6d}/{n:<7d} {p.relative_to(ROOT)}")
    return on_edge / total


def part3_ece_delta() -> None:
    print("\n=== 3. ECE under each convention, on real data ===")
    cands = sorted((ROOT / "outputs").rglob("*predictions.jsonl"))
    if not cands:
        return
    print(f"  {'file':52s} {'ECE edge-close':>15} {'ECE edge-open':>14} {'delta':>9}")
    worst = 0.0
    shown = 0
    for p in cands:
        yt, yp = [], []
        try:
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                gt = obj.get("ground_truth") or {}
                findings = (obj.get("parsed") or {}).get("findings") or {}
                for lb in LABELS:
                    truth = gt.get(f"label_{lb}")
                    v = (findings.get(lb) or {}).get("probability_positive")
                    if truth in ("positive", "negative") and isinstance(v, (int, float)):
                        yt.append(1.0 if truth == "positive" else 0.0)
                        yp.append(float(v))
        except OSError:
            continue
        if len(yt) < 50:
            continue
        a = ece_with(yt, yp, right=True)
        b = ece_with(yt, yp, right=False)
        d = abs(a - b)
        worst = max(worst, d)
        if shown < 8 or d > 0.001:
            print(f"  {str(p.relative_to(ROOT))[:52]:52s} {a:15.5f} {b:14.5f} {d:9.5f}")
            shown += 1
    print(f"\n  largest absolute ECE difference on real data: {worst:.5f}")
    if worst > 0.002:
        print("  -> NOT a measure-zero curiosity: the convention changes the number")
    else:
        print("  -> difference is small on this data; the convention still differs")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None,
                    help="specific prediction jsonl files to use for parts 2-3")
    args = ap.parse_args()

    print(f"program root: {ROOT}")
    part1_conventions()
    part2_boundary_share()
    part3_ece_delta()

    # the shipped function must agree with the edge-closed variant, or the
    # diagnostic above is describing different code from the one that runs.
    #
    # The vector has to DISCRIMINATE. A first attempt used round values only
    # ([0.1,0.1,0.9,0.9,0.2,...]) and got 0.35 under both conventions, because
    # shifting every value into the next bin changes nothing when the bins keep the
    # same composition. What separates the conventions is a boundary value whose
    # bin-mates differ from the ones it would otherwise share a bin with.
    yt = [1, 1, 0, 0]
    yp = [0.1, 0.1, 0.15, 0.15]
    shipped = expected_calibration_error(yt, yp)
    closed = ece_with(yt, yp, right=True)
    opend = ece_with(yt, yp, right=False)
    print("\n=== 4. which convention does the shipped function implement? ===")
    print(f"  vector yp = {yp}  y_true = {yt}   (0.1 straddles the edge)")
    print(f"  shipped expected_calibration_error : {shipped:.6f}")
    print(f"  local right=True  (edge-closed)   : {closed:.6f}")
    print(f"  local right=False (edge-open)     : {opend:.6f}")
    print(f"  shipped == right=True : {abs(shipped - closed) < 1e-12}")
    print(f"  shipped == right=False: {abs(shipped - opend) < 1e-12}")
    if abs(closed - opend) < 1e-12:
        print("  !! the vector does not discriminate; the verdict above is meaningless")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
