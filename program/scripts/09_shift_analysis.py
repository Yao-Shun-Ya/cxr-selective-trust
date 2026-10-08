#!/usr/bin/env python
"""Analyse shift_predictions.jsonl produced by 08_shift_benchmark.py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.evaluation.metrics import (  # noqa: E402
    aurc, auroc, brier, coverage_at_risk, expected_calibration_error,
    binary_f1, high_confidence_error_rate, selective_accuracy,
)
from src.cxr_trust.inference.schema import PRIMARY_LABELS  # noqa: E402

# MedGemma-1.5's reasoning-channel opener. When it is present the reply is not
# "unparseable text", it is a *deliberation trace that ran out of budget*: the
# model committed to the format and then left it. Distinguishing that from a
# model that never attempted the format (llava_med's prose refusal) is a
# one-line test, and the distinction is invisible in the raw files.
MARKER = "<unused94>"


def load_with_census(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the predictions ONCE, returning the metric rows **and** a census.

    Why a census is not optional
    ----------------------------
    This function used to drop every row it could not use and return only what
    survived: unparsed rows (no `parsed.parse_success`) and labels whose ground
    truth is neither `positive` nor `negative` both vanished without a trace, and
    `shift_metrics_by_condition.csv` then reported `n_label_instances` over the
    survivors -- a number that reads like a sample size and is really the
    *remainder*. The pre-run makes the consequence concrete: medgemma_15's blank
    condition had **100 of 100** cells produce no parseable reply, and llava_med
    had 0 of 16 on reliance. An analysis that only reports the survivors describes
    those cells as if nothing had been planned for them.

    So the drop is counted here, per (condition, severity), and travels with the
    metrics: n_attempted, n_parse_ok, n_parse_fail, n_gt_dropped, plus the two
    coverage ratios that make the difference between "this metric is computed on
    less data" and "this metric is computed on different data".

    The same pass also classifies *why* a reply failed, because "did not parse"
    lumps two different events together. `n_marker` counts replies that open the
    reasoning channel, and `n_marker_unparsed` those that opened it and then ran
    out of budget. When `n_marker_unparsed` equals `n_parse_fail`, every failure
    in the cell is a channel switch rather than a model that ignored the format.
    """
    rows: list[dict] = []
    attempts: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        condition = obj.get("condition")
        severity = obj.get("severity")
        parsed = obj.get("parsed") or {}
        ok = bool(parsed.get("parse_success"))
        marker = (obj.get("raw_output") or "").startswith(MARKER)
        gt = obj.get("ground_truth", {}) or {}
        findings = parsed.get("findings") or {}
        n_gt = 0
        for label, fp in findings.items():
            gt_val = gt.get(f"label_{label}")
            if gt_val in {"positive", "negative"}:
                n_gt += 1
            rows.append({
                "image_id": obj["image_id"],
                "split": obj.get("split"),
                "condition": condition,
                "severity": severity,
                "label": label,
                "status": fp.get("status"),
                "p": fp.get("probability_positive"),
                "y_true": (1.0 if gt_val == "positive" else 0.0)
                          if gt_val in {"positive", "negative"} else None,
                "parse_ok": ok,
                "gt_usable": gt_val in {"positive", "negative"},
            })
        attempts.append({
            "image_id": obj["image_id"],
            "condition": condition,
            "severity": severity,
            "parse_ok": ok,
            "marker": marker,
            "marker_unparsed": bool(marker and not ok),
            "n_findings": len(findings),
            "n_labels_with_gt": n_gt,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # `y_true` was written for every label so the census can see what was
        # dropped; the metric rows keep only the usable ones.
        df = df[df["gt_usable"]].drop(columns=["gt_usable"]).reset_index(drop=True)
    if not attempts:
        return df, pd.DataFrame()

    att = pd.DataFrame(attempts)
    labels_per_row = len(PRIMARY_LABELS)
    census = att.groupby(["condition", "severity"]).agg(
        n_attempted=("image_id", "size"),
        n_parse_ok=("parse_ok", "sum"),
        n_findings_mean=("n_findings", "mean"),
        n_marker=("marker", "sum"),
        n_marker_unparsed=("marker_unparsed", "sum"),
    ).reset_index()
    census["n_parse_fail"] = census["n_attempted"] - census["n_parse_ok"]
    census["parse_coverage"] = census["n_parse_ok"] / census["n_attempted"]
    # Planned label instances: every attempted cell would have contributed one
    # instance per label of the contract had the reply parsed and the ground
    # truth been binary. This is the denominator the metric table was missing.
    census["n_label_planned"] = census["n_attempted"] * labels_per_row
    if not df.empty:
        survived = df.groupby(["condition", "severity"]).size().rename("n_label_instances")
        census = census.merge(survived, on=["condition", "severity"], how="left")
    if "n_label_instances" not in census:
        census["n_label_instances"] = 0
    census["n_label_instances"] = census["n_label_instances"].fillna(0).astype(int)
    census["label_coverage"] = census["n_label_instances"] / census["n_label_planned"]
    census["n_gt_dropped"] = census["n_label_planned"] - census["n_label_instances"]
    return df, census


def load_rows(path: Path) -> pd.DataFrame:
    """Metric rows only (kept for callers that do not need the census)."""
    df, _ = load_with_census(path)
    return df


def per_image_flips(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[(df["condition"] == "clean") & (df["severity"] == "none")]
    if clean.empty:
        return pd.DataFrame()
    base_status = {(r.image_id, r.label): r.status for r in clean.itertuples()}
    base_prob = {(r.image_id, r.label): r.p for r in clean.itertuples()}
    out = []
    for (image_id, condition, severity), g in df.groupby(["image_id", "condition", "severity"]):
        if condition == "clean":
            continue
        flips, diffs = 0, []
        for r in g.itertuples():
            b = base_status.get((image_id, r.label))
            if b is not None and b != r.status:
                flips += 1
            bp = base_prob.get((image_id, r.label))
            if bp is not None and r.p is not None:
                diffs.append(abs(float(bp) - float(r.p)))
        out.append({
            "image_id": image_id,
            "condition": condition,
            "severity": severity,
            "flip_count": flips,
            "flip_any": int(flips > 0),
            "mean_abs_prob_diff": float(np.mean(diffs)) if diffs else None,
        })
    return pd.DataFrame(out)


def macro_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    y = df["y_true"].to_numpy()
    p = df["p"].fillna(0.0).to_numpy()
    status = df["status"].fillna("uncertain").to_numpy()
    pred = np.where(status == "positive", 1.0, np.where(status == "negative", 0.0, np.nan))
    error = np.where(np.isnan(pred), 1.0, (pred != y).astype(float))
    risk = 1.0 - np.where(status == "positive", p, np.where(status == "negative", 1 - p, 0.5))
    return {
        "n": int(len(df)),
        "auroc": auroc(y, p),
        "ece": expected_calibration_error(y, p),
        "brier": brier(y, p),
        "f1": binary_f1(y, pred),
        "high_conf_error": high_confidence_error_rate(y, p)[0],
        "error_rate": float(np.mean(error)),
        "aurc": aurc(error, risk),
        "coverage_at_5pct_risk": coverage_at_risk(error, risk, 0.05),
        "selective_accuracy_at_80pct_coverage": selective_accuracy(error, risk, 0.80),
    }


def report_coverage(census: pd.DataFrame, min_coverage: float,
                    out_dir: Path) -> pd.DataFrame:
    """Print the per-cell census, write coverage_report.json, return the bad cells.

    Runs BEFORE the metric rows are used, so a file in which nothing parsed --
    llava_med's prose refusals, or medgemma_15's bf16 precision stage at 400 of
    400 -- still gets a coverage report. That is the case where the census
    matters most, and it used to `return` one line earlier with only
    "no parse-success rows", which reads like an empty stage rather than a total
    structural failure.
    """
    if census.empty:
        return pd.DataFrame()
    low = census[census["label_coverage"] < min_coverage]
    print(f"\n{'condition':12s} {'severity':9s} {'attempted':>9s} {'parsed':>7s} "
          f"{'parse cov':>9s} {'label cov':>9s} {'dropped labels':>14s} "
          f"{'marker':>7s} {'mkr&fail':>9s}")
    for r in census.sort_values(["condition", "severity"]).itertuples():
        print(f"{str(r.condition):12s} {str(r.severity):9s} {r.n_attempted:9d} "
              f"{r.n_parse_ok:7d} {r.parse_coverage:9.3f} {r.label_coverage:9.3f} "
              f"{r.n_gt_dropped:14d} {r.n_marker:7d} {r.n_marker_unparsed:9d}")
    print("\nmarker  = replies opening the reasoning channel (`" + MARKER + "`);")
    print("mkr&fail= of those, the ones that then failed to parse. mkr&fail == "
          "parse-fail")
    print("          means every structural failure in that cell is a channel "
          "switch,")
    print("          not a model that ignored the format.")
    if not low.empty:
        print("\n" + "!" * 78)
        print("COVERAGE WARNING: these cells are below "
              f"{min_coverage:.0%} label coverage, so their metrics are")
        print("computed over a SELECTED subset -- report the coverage, not just "
              "the metric:")
        for r in low.sort_values("label_coverage").itertuples():
            print(f"  {r.condition}/{r.severity}: "
                  f"{r.n_label_instances} of {r.n_label_planned} label instances "
                  f"({r.label_coverage:.1%}); {r.n_parse_fail} of {r.n_attempted} "
                  f"replies did not parse")
        print("!" * 78)
    (out_dir / "coverage_report.json").write_text(json.dumps({
        "min_coverage": min_coverage,
        "cells": census.to_dict(orient="records"),
        "cells_below_threshold": low.to_dict(orient="records"),
        "coverage_ok": bool(low.empty),
    }, indent=2), encoding="utf-8")
    return low


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-coverage", type=float, default=0.95,
                        help="label coverage below which a cell is reported as a "
                             "coverage failure (default 0.95)")
    parser.add_argument("--fail-on-low-coverage", action="store_true",
                        help="exit 3 when any cell is below --min-coverage; the "
                             "report and the warning are written either way")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, census = load_with_census(Path(args.predictions))
    if not census.empty:
        census.to_csv(out_dir / "shift_coverage_by_condition.csv", index=False)
    df.to_csv(out_dir / "shift_flat_predictions.csv", index=False)
    low = report_coverage(census, args.min_coverage, out_dir)
    if df.empty:
        summary = {
            "n_rows": 0,
            "error": "no parse-success rows",
            "coverage_ok": bool(low.empty),
            "n_marker_rows": int(census["n_marker"].sum()) if not census.empty else 0,
            "n_marker_unparsed": (int(census["n_marker_unparsed"].sum())
                                  if not census.empty else 0),
            "n_parse_fail": (int(census["n_parse_fail"].sum())
                             if not census.empty else 0),
        }
        (out_dir / "shift_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print("no parse-success rows")
        if args.fail_on_low_coverage and not low.empty:
            print(f"\nexit 3: {len(low)} cell(s) below "
                  f"{args.min_coverage:.0%} coverage")
            raise SystemExit(3)
        return

    # Per-label metrics first, so condition-level metrics can be macro averages.
    label_rows = []
    for (condition, severity, label), g in df.groupby(["condition", "severity", "label"]):
        m = macro_metrics(g)
        m.update({"condition": condition, "severity": severity, "label": label})
        label_rows.append(m)
    label_df = pd.DataFrame(label_rows)
    label_df.to_csv(out_dir / "shift_metrics_by_finding.csv", index=False)

    rows = []
    for (condition, severity), g in df.groupby(["condition", "severity"]):
        sub = label_df[(label_df["condition"] == condition) & (label_df["severity"] == severity)]
        pooled = macro_metrics(g)
        rows.append({
            "condition": condition,
            "severity": severity,
            "n_label_instances": int(len(g)),
            "n_labels": int(len(sub)),
            "macro_auroc": float(np.nanmean(sub["auroc"])) if len(sub) else float("nan"),
            "macro_ece": float(np.nanmean(sub["ece"])) if len(sub) else float("nan"),
            "macro_brier": float(np.nanmean(sub["brier"])) if len(sub) else float("nan"),
            "macro_f1": float(np.nanmean(sub["f1"])) if len(sub) else float("nan"),
            "pooled_error_rate": pooled.get("error_rate"),
            "pooled_aurc": pooled.get("aurc"),
            "pooled_coverage_at_5pct_risk": pooled.get("coverage_at_5pct_risk"),
            "pooled_selective_accuracy_at_80pct_coverage": pooled.get("selective_accuracy_at_80pct_coverage"),
            "pooled_high_conf_error": pooled.get("high_conf_error"),
        })
    metrics = pd.DataFrame(rows)
    # Coverage travels WITH the metrics, not in a separate file nobody opens.
    if not census.empty:
        metrics = metrics.merge(census, on=["condition", "severity"], how="left")
    metrics.to_csv(out_dir / "shift_metrics_by_condition.csv", index=False)

    flips = per_image_flips(df)
    if not flips.empty:
        flips.to_csv(out_dir / "shift_image_flips.csv", index=False)
        flip_summary = flips.groupby(["condition", "severity"]).agg(
            n=("image_id", "size"),
            image_sensitivity=("flip_any", "mean"),
            mean_flip_count=("flip_count", "mean"),
            mean_abs_prob_diff=("mean_abs_prob_diff", "mean"),
        ).reset_index()
        flip_summary.to_csv(out_dir / "shift_image_sensitivity.csv", index=False)

    summary = {
        "n_rows": int(len(df)),
        "conditions": sorted(df["condition"].unique().tolist()) if not df.empty else [],
        "severities": sorted(df["severity"].unique().tolist()) if not df.empty else [],
        "parse_success_rows": int(len(df)),
        "coverage_ok": bool(low.empty) if not census.empty else True,
        "min_label_coverage": (float(census["label_coverage"].min())
                               if not census.empty else None),
        # Why the failures happened, not just how many. `n_marker_unparsed` over
        # `n_parse_fail` is the channel-switch share of structural failure.
        "n_marker_rows": int(census["n_marker"].sum()) if not census.empty else 0,
        "n_marker_unparsed": (int(census["n_marker_unparsed"].sum())
                              if not census.empty else 0),
        "n_parse_fail": (int(census["n_parse_fail"].sum())
                         if not census.empty else 0),
    }
    (out_dir / "shift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}")

    if args.fail_on_low_coverage and not low.empty:
        print(f"\nexit 3: {len(low)} cell(s) below {args.min_coverage:.0%} coverage")
        raise SystemExit(3)


if __name__ == "__main__":
    main()
