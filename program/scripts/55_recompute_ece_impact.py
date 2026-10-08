#!/usr/bin/env python
"""Recompute every ECE number the paper quotes, under both bin conventions.

Why this is a separate script
-----------------------------
The ECE bin convention was corrected (see ``54_check_ece_convention.py``). Every
ECE figure that was produced before the correction is therefore on the old
convention and must be regenerated -- not estimated, regenerated, because the
paper's numbers are what a reviewer will try to reproduce.

ECE is a function of saved predictions, so this needs no GPU: it recomputes from
``outputs/**/predictions.jsonl`` and prints old vs new side by side, including the
exact macro-over-labels figures that appear in ``papers/main_v2``.

Run::

    .venv-vl\\Scripts\\python.exe scripts\\55_recompute_ece_impact.py
    .venv-vl\\Scripts\\python.exe scripts\\55_recompute_ece_impact.py --json out.json
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


def ece_variant(y_true, y_prob, right: bool, n_bins: int = 10) -> float:
    y = np.asarray(y_true, float)
    p = np.asarray(y_prob, float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(p, bins[1:-1], right=right)
    if len(y) == 0:
        return float("nan")
    val = 0.0
    for b in range(n_bins):
        m = ids == b
        if np.any(m):
            val += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return float(val)


def findings_of(obj: dict) -> dict:
    """Find the per-label findings, whichever shape this file uses.

    Two shapes exist in this repository and a naive reader silently sees neither:

      * current: ``{"parsed": {"findings": {...}}, ...}``  (runners 06/08/43/44)
      * legacy : ``{"findings": {...}, "parse_success": true, ...}``  (03/04 pipeline,
        which is what ``outputs/overnight/*_1000`` uses -- the files behind the
        paper's headline numbers)

    Reading only the nested form returned zero labels for the overnight files, and
    the script then reported nothing at all rather than saying why. That is the
    same silent-empty failure this project has been bitten by repeatedly, so this
    function is explicit and the caller complains when nothing is found.
    """
    parsed = obj.get("parsed")
    if isinstance(parsed, dict):
        f = parsed.get("findings")
        if isinstance(f, dict) and f:
            return f
    f = obj.get("findings")
    if isinstance(f, dict):
        return f
    return {}


def load(path: Path, split: str | None = None,
         universe: str = "any-numeric") -> tuple[dict, dict, int, int]:
    """Return per-label y_true / y_prob, plus row and decision counts.

    ``universe`` selects which rows count:

    ``any-numeric``
        every row with a declared probability, including rows the model hedged on
        (status ``uncertain``). This is the standard ECE estimand -- Guo et al. and
        ``sklearn`` score the model's probability outputs, and 186 of 186 hedged
        MedGemma rows do carry one. PRIMARY.
    ``definite``
        only rows with a definite call. This is what the published Table 1 used,
        implicitly, and it yields a *harsher* number (macro ECE 0.2316 vs 0.2196 on
        MedGemma internal_test). Kept as the reported sensitivity check, and as a
        flag here so the two accounts -- bin convention and sample universe -- can be
        settled separately instead of being silently mixed.

    ``split`` restricts to one split: the paper's main table is per split, so
    aggregating the whole file reproduces a different number than the printed one.
    """
    yt: dict[str, list[float]] = {lb: [] for lb in LABELS}
    yp: dict[str, list[float]] = {lb: [] for lb in LABELS}
    n_rows = n_skipped = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if split is not None and str(obj.get("split")) != split:
            continue
        n_rows += 1
        gt = obj.get("ground_truth") or {}
        findings = findings_of(obj)
        if not findings:
            n_skipped += 1
            continue
        for lb in LABELS:
            truth = gt.get(f"label_{lb}")
            entry = findings.get(lb) or {}
            v = entry.get("probability_positive")
            if truth not in ("positive", "negative") or not isinstance(v, (int, float)):
                continue
            if universe == "definite" and str(entry.get("status")) not in ("positive",
                                                                           "negative"):
                continue
            yt[lb].append(1.0 if truth == "positive" else 0.0)
            yp[lb].append(float(v))
    return yt, yp, n_rows, n_skipped


def macro_both(yt, yp) -> tuple[float, float, dict]:
    """Macro-over-labels ECE under both conventions, plus the per-label detail."""
    old, new, detail = [], [], {}
    for lb in LABELS:
        if len(yt[lb]) < 20:
            continue
        o = ece_variant(yt[lb], yp[lb], right=True)
        nw = expected_calibration_error(yt[lb], yp[lb])
        assert abs(nw - ece_variant(yt[lb], yp[lb], right=False)) < 1e-12, \
            "shipped ECE disagrees with the edge-open variant"
        old.append(o)
        new.append(nw)
        detail[lb] = {"n": len(yt[lb]), "old": o, "new": nw, "delta": nw - o}
    if not old:
        return float("nan"), float("nan"), {}
    return float(np.mean(old)), float(np.mean(new)), detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="also dump the comparison as JSON")
    ap.add_argument("--definite-only", action="store_true",
                    help="score only definite calls, reproducing the published"
                         " Table 1 universe; the default scores every declared"
                         " probability (the standard estimand). Use this for the"
                         " reported sensitivity check, NOT as the primary number.")
    args = ap.parse_args()

    sources = [p for p in sorted((ROOT / "outputs").rglob("predictions.jsonl"))]
    if not sources:
        print("no predictions.jsonl found under outputs/")
        return 2

    report: dict = {}
    print("ECE bin convention: OLD = (a,b] right=True   NEW = [a,b) right=False")
    print("shipped function now returns the NEW value\n")

    # The paper's main table is per (model, split), so both views are reported:
    # the whole file, and each split inside it. Without the split breakdown the
    # recomputed macro does not match the printed one, for a reason that has
    # nothing to do with the convention.
    for path in sources:
        rel = str(path.relative_to(ROOT))
        universe = "definite" if args.definite_only else "any-numeric"
        yt_all, yp_all, n_rows, n_skipped = load(path, universe=universe)
        total = sum(len(v) for v in yt_all.values())
        if total < 100:
            why = (f"{n_skipped}/{n_rows} rows had no readable findings"
                   if n_skipped else "no ground-truth labels")
            print(f"--- {rel}: SKIPPED ({total} label decisions; {why})\n")
            continue

        # which splits does this file actually contain?
        splits = sorted({str(json.loads(l).get("split"))
                         for l in path.read_text(encoding="utf-8",
                                                 errors="replace").splitlines()
                         if l.strip() and '"split"' in l})
        print(f"--- {rel}   ({n_rows} rows, {total} label decisions)")
        print(f"    splits present: {splits}")

        om, nm, detail = macro_both(yt_all, yp_all)
        print(f"\n    ALL ROWS   {'label':18s} {'n':>6} {'OLD':>9} {'NEW':>9} {'delta':>9}")
        for lb, d in detail.items():
            print(f"    {'':10s} {lb:18s} {d['n']:6d} {d['old']:9.4f} {d['new']:9.4f} "
                  f"{d['delta']:+9.4f}")
        print(f"    {'':10s} {'MACRO (mean over labels)':18s} {'':6s} {om:9.4f} {nm:9.4f} "
              f"{nm - om:+9.4f}")
        report[f"{rel}|ALL"] = {"macro_old": om, "macro_new": nm, "delta": nm - om,
                                "by_label": detail}

        for sp in splits:
            if sp in ("None", ""):
                continue
            yt_s, yp_s, n_s, _ = load(path, split=sp, universe=universe)
            if sum(len(v) for v in yt_s.values()) < 100:
                continue
            o_s, n_sm, det_s = macro_both(yt_s, yp_s)
            print(f"    split={sp:16s} rows={n_s:<5d} macro OLD {o_s:.4f} -> "
                  f"NEW {n_sm:.4f}  ({n_sm - o_s:+.4f})")
            report[f"{rel}|{sp}"] = {"macro_old": o_s, "macro_new": n_sm,
                                     "delta": n_sm - o_s, "by_label": det_s}
        print()


    if report:
        # Per-split entries whose labels all fall under the n>=20 threshold score
        # NaN. Those are real results ("too small to score"), but they must not
        # enter the mean/range -- and abs(NaN) is NaN, which silently poisons
        # max() as well.
        finite = {k: v for k, v in report.items() if np.isfinite(v["delta"])}
        n_nan = len(report) - len(finite)
        d = np.array([v["delta"] for v in finite.values()])
        print("=== summary ===")
        print(f"  files recomputed      : {len(report)}"
              + (f"  ({n_nan} below the per-label n>=20 threshold)" if n_nan else ""))
        if len(d):
            print(f"  macro ECE change      : mean {d.mean():+.4f}, "
                  f"range [{d.min():+.4f}, {d.max():+.4f}]")
            biggest = max(finite.items(), key=lambda kv: abs(kv[1]["delta"]))
            print(f"  largest macro change  : {biggest[1]['delta']:+.4f}  ({biggest[0]})")
        print("\n  Every ECE figure above must be regenerated from these files before")
        print("  it is published; the old values are on a non-standard convention.")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
