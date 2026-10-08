#!/usr/bin/env python
"""Bootstrap CI and paired clean-vs-shift tests on a flat predictions CSV.

Expected columns: image_id, condition, severity, label, y_true, p
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.evaluation.metrics import auroc, brier, expected_calibration_error  # noqa: E402

METRICS = {
    "auroc": auroc,
    "ece": expected_calibration_error,
    "brier": brier,
}


def macro_metric(df: pd.DataFrame, metric_name: str) -> float:
    vals = []
    for _, g in df.groupby("label"):
        y = g["y_true"].to_numpy()
        p = g["p"].to_numpy()
        vals.append(METRICS[metric_name](y, p))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if len(vals) else float("nan")


def bootstrap_metric(df: pd.DataFrame, metric_name: str, n_boot: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    image_ids = df["image_id"].unique()
    vals = []
    for _ in range(n_boot):
        sample_ids = rng.choice(image_ids, size=len(image_ids), replace=True)
        sample = pd.concat([df[df["image_id"] == i] for i in sample_ids], ignore_index=True)
        vals.append(macro_metric(sample, metric_name))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap(clean: pd.DataFrame, shifted: pd.DataFrame, metric_name: str, n_boot: int = 1000, seed: int = 0):
    merged = clean.merge(shifted, on=["image_id", "label"], suffixes=("_clean", "_shift"))
    if merged.empty:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    image_ids = merged["image_id"].unique()
    deltas = []
    for _ in range(n_boot):
        sample_ids = rng.choice(image_ids, size=len(image_ids), replace=True)
        sample = pd.concat([merged[merged["image_id"] == i] for i in sample_ids], ignore_index=True)
        clean_sample = sample.rename(columns={"y_true_clean": "y_true", "p_clean": "p"})
        shift_sample = sample.rename(columns={"y_true_shift": "y_true", "p_shift": "p"})
        m_c = macro_metric(clean_sample, metric_name)
        m_s = macro_metric(shift_sample, metric_name)
        if np.isfinite(m_c) and np.isfinite(m_s):
            deltas.append(m_s - m_c)
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    delta = float(np.mean(deltas))
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    p_value = 2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))
    p_value = float(min(1.0, max(0.0, p_value)))
    return delta, lo, hi, p_value


def bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / (np.arange(m) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    tmp = np.empty_like(adjusted)
    tmp[order] = adjusted
    out[valid] = tmp
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flat-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--clean-condition", default="clean")
    parser.add_argument("--clean-severity", default="none")
    args = parser.parse_args()

    df = pd.read_csv(args.flat_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_rows = []
    for (condition, severity), g in df.groupby(["condition", "severity"]):
        for metric_name in METRICS:
            mean, lo, hi = bootstrap_metric(g.dropna(subset=["y_true", "p"]), metric_name, n_boot=args.n_boot)
            group_rows.append({
                "condition": condition, "severity": severity, "metric": metric_name,
                "mean": mean, "ci_low": lo, "ci_high": hi, "n": len(g),
            })
    pd.DataFrame(group_rows).to_csv(out_dir / "bootstrap_by_condition.csv", index=False)

    clean = df[(df["condition"] == args.clean_condition) & (df["severity"] == args.clean_severity)]
    paired_rows = []
    for (condition, severity), g in df.groupby(["condition", "severity"]):
        if condition == args.clean_condition and severity == args.clean_severity:
            continue
        for metric_name in METRICS:
            delta, lo, hi, p = paired_bootstrap(clean, g, metric_name, n_boot=args.n_boot)
            paired_rows.append({
                "condition": condition, "severity": severity, "metric": metric_name,
                "delta_shift_minus_clean": delta, "ci_low": lo, "ci_high": hi, "p_value": p,
            })
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired["p_value_bh"] = bh_fdr(paired["p_value"].to_numpy())
    paired.to_csv(out_dir / "paired_clean_vs_shift.csv", index=False)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
