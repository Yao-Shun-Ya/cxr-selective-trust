#!/usr/bin/env python
"""Hierarchical selective-prediction metrics (finding -> image -> patient).

Why this script exists
----------------------
``07_calibration_selective.py`` treats every (image, finding) pair as an
independent observation.  That is wrong in two ways:

1. **No clustering.**  Several findings come from one image, and several images
   can come from one patient.  Resampling label-instances i.i.d. therefore
   understates the variance of risk-coverage summaries.
2. **No hierarchy.**  A deployment decision is made per image (or per patient),
   not per label.  Pooled label-instance risk is not the risk a clinician sees.

This script reports selective prediction at three nested levels:

* ``finding`` -- one risk item per (image, finding) pair (the old pooled view,
  kept for comparability, but now with a *patient-cluster* bootstrap);
* ``image``   -- one risk item per image, aggregating its findings;
* ``patient`` -- one risk item per patient, aggregating that patient's images.

Design rules enforced here (see docs/36_q1_hardening_plan.md section 4.6):

* calibration and decision thresholds are fitted on ``--source-val-split`` ONLY;
* ``--test-split`` is used for evaluation only, never for fitting;
* every confidence interval uses a patient-level cluster bootstrap, so all
  rows/images of a resampled patient move together;
* the operating point is chosen from a pre-registered budget
  (``--target-risk`` coverage@risk, or ``--target-coverage`` risk@coverage),
  not by looking at test performance.

Usage
-----
::

    python scripts/18_selective_metrics_v2.py \
        --predictions outputs/overnight/medgemma_4b_it_1000/predictions.jsonl \
        --manifest data/manifests/nih_pilot_1000.csv \
        --output-dir outputs/selective_v2/medgemma_4b_it_1000

Outputs (written to ``--output-dir``)
-------------------------------------
``selective_v2_summary.json``      headline metrics at all three levels
``risk_coverage_bands.csv``        risk-coverage curves + bootstrap bands
``risk_coverage_metrics.csv``      AURC / coverage@risk / error-detection AUROC
``operating_points.csv``           source_val-selected thresholds and their test outcome
``finding_level_metrics.csv``      per-finding risk-coverage diagnostics
``hierarchical_metrics.csv``       level-to-level risk flow
``selective_v2_manifest.json``     provenance / reproducibility record
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.evaluation.metrics import (  # noqa: E402
    aurc,
    coverage_at_risk,
    error_detection_auroc,
    expected_calibration_error,
)

LEVELS = ("finding", "image", "patient")
AGGREGATIONS = ("max", "mean", "product")

# Fixed coverage grid for band reporting (keeps bootstrap cheap and comparable).
COVERAGE_GRID = np.round(np.arange(0.05, 1.0001, 0.05), 4)


# --------------------------------------------------------------------------- #
# calibration helpers (mirrors 07_calibration_selective.py so results are comparable)
# --------------------------------------------------------------------------- #
def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(y_true, p) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 1.0
    z = _logit(p)

    def nll(log_t):
        t = np.exp(log_t)
        q = np.clip(_sigmoid(z / t), 1e-6, 1 - 1e-6)
        return -np.mean(y_true * np.log(q) + (1 - y_true) * np.log(1 - q))

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x)) if res.success else 1.0


def apply_temperature(p, t):
    return _sigmoid(_logit(p) / t)


def _trapezoid(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 renamed trapz
    return float(fn(y, x))


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_predictions(path: Path) -> pd.DataFrame:
    """Flatten predictions.jsonl into one row per (image, finding) with a usable probability."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        gt = obj.get("ground_truth", {}) or {}
        for label, fp in (obj.get("findings") or {}).items():
            gt_val = gt.get(f"label_{label}")
            if gt_val not in {"positive", "negative"}:
                continue
            p = fp.get("probability_positive")
            if p is None:
                continue
            rows.append({
                "image_id": obj.get("image_id"),
                "split": obj.get("split"),
                "label": label,
                "y_true": 1.0 if gt_val == "positive" else 0.0,
                "p": float(p),
                "status": fp.get("status"),
                "parse_success": bool(obj.get("parse_success", True)),
            })
    return pd.DataFrame(rows)


def attach_patient_id(df: pd.DataFrame, manifest_path: Path | None):
    """Attach patient_id (cluster key). Falls back to image_id when unavailable."""
    if manifest_path is None or not Path(manifest_path).exists():
        df["patient_id"] = df["image_id"]
        return df, "image_id (manifest unavailable)"
    man = pd.read_csv(manifest_path, usecols=lambda c: c in {"image_id", "patient_id"})
    if "patient_id" not in man.columns:
        df["patient_id"] = df["image_id"]
        return df, "image_id (no patient_id column)"
    man = man.drop_duplicates(subset=["image_id"])
    merged = df.merge(man[["image_id", "patient_id"]], on="image_id", how="left")
    missing = int(merged["patient_id"].isna().sum())
    if missing:
        print(f"[warn] {missing} label-instances had no manifest patient_id; "
              f"falling back to image_id for those rows")
        merged["patient_id"] = merged["patient_id"].fillna(merged["image_id"])
    merged["patient_id"] = merged["patient_id"].astype(str)
    return merged, "manifest.patient_id"


# --------------------------------------------------------------------------- #
# risk construction
# --------------------------------------------------------------------------- #
def instance_risk(p: np.ndarray, status: np.ndarray):
    """Per (image, finding) risk item.

    ``pred`` is the model's *selected* decision: explicit status when it emitted
    one, else the thresholded probability.  ``conf`` is the calibrated
    probability of that selected decision; ``risk_score = 1 - conf`` so that
    sorting ascending yields "most trustworthy first".
    """
    p = np.asarray(p, dtype=float)
    status = np.asarray([str(s) if s is not None else "" for s in status])
    pred = np.where(
        status == "positive", 1.0,
        np.where(status == "negative", 0.0, (p >= 0.5).astype(float)),
    )
    emitted = np.isin(status, ["positive", "negative"])
    conf = np.where(pred >= 0.5, p, 1.0 - p)
    # a finding the model declined to call is maximally uncertain
    conf = np.where(emitted, conf, 0.5)
    return pred, conf, 1.0 - conf


def aggregate_to_level(inst: pd.DataFrame, level: str, agg: str) -> pd.DataFrame:
    """Collapse (image, finding) risk items into image-level or patient-level risk items."""
    if level == "finding":
        return inst
    key = "image_id" if level == "image" else "patient_id"
    g = inst.groupby(key, sort=False)

    sub = inst[[key, "risk_score", "error"]].copy()
    if agg == "max":
        out = g["risk_score"].max().rename("risk_score").to_frame()
    elif agg == "mean":
        out = g["risk_score"].mean().rename("risk_score").to_frame()
    else:
        # product risk: any single high-risk finding dominates; stable via log-domain complement
        out = g["risk_score"].apply(
            lambda r: 1.0 - float(np.prod(1.0 - np.asarray(r, dtype=float)))
        ).rename("risk_score").to_frame()

    # a case counts as an error when ANY of its findings is wrong (conservative)
    out["error"] = g["error"].max()
    out["n_findings"] = g.size()
    out["mean_child_risk"] = g["risk_score"].mean()
    out = out.reset_index().sort_values(key, kind="stable").reset_index(drop=True)

    # keep the patient cluster key present at every level (image_id or patient_id)
    if level == "image":
        pat = inst.groupby("image_id", sort=False)["patient_id"].first()
        out["patient_id"] = out["image_id"].map(pat).astype(str)
    return out


def risk_coverage_arrays(error: np.ndarray, risk_score: np.ndarray):
    """Coverage/risk arrays; low risk accepted first (sorted ascending)."""
    error = np.asarray(error, dtype=float)
    risk_score = np.asarray(risk_score, dtype=float)
    mask = np.isfinite(error) & np.isfinite(risk_score)
    error, risk_score = error[mask], risk_score[mask]
    if len(error) == 0:
        return np.array([]), np.array([])
    order = np.argsort(risk_score, kind="stable")
    sorted_error = error[order]
    n = len(error)
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(sorted_error) / np.arange(1, n + 1)
    return coverage, risk


def curve_on_grid(error, risk_score, grid=COVERAGE_GRID):
    """Risk-coverage curve evaluated at fixed coverage points (nan when too few items)."""
    coverage, risk = risk_coverage_arrays(error, risk_score)
    if len(coverage) == 0 or coverage[-1] < grid[-1] - 1e-9:
        return np.full(len(grid), np.nan)
    # coverage is strictly increasing -> np.interp is safe
    return np.interp(grid, coverage, risk)


def coverage_at_risk_fast(error, risk_score, max_risk: float) -> float:
    coverage, risk = risk_coverage_arrays(error, risk_score)
    if len(coverage) == 0:
        return float("nan")
    valid = np.where(risk <= max_risk + 1e-12)[0]
    return float(coverage[valid[-1]]) if len(valid) else 0.0


def risk_at_coverage(error, risk_score, target_coverage: float) -> float:
    coverage, risk = risk_coverage_arrays(error, risk_score)
    if len(coverage) == 0:
        return float("nan")
    idx = int(np.searchsorted(coverage, target_coverage, side="left"))
    idx = min(idx, len(risk) - 1)
    return float(risk[idx])


def threshold_for_target_risk(error, risk_score, max_risk: float):
    """Smallest risk score whose cumulative risk still stays under the budget.

    Returns the threshold ``tau`` such that accepting ``risk_score <= tau``
    keeps empirical risk <= ``max_risk``.  Returns None when even one single
    item already exceeds the budget.
    """
    error = np.asarray(error, dtype=float)
    risk_score = np.asarray(risk_score, dtype=float)
    mask = np.isfinite(error) & np.isfinite(risk_score)
    error, risk_score = error[mask], risk_score[mask]
    if len(error) == 0:
        return None
    order = np.argsort(risk_score, kind="stable")
    e_sorted = error[order]
    r_sorted = risk_score[order]
    cum_risk = np.cumsum(e_sorted) / np.arange(1, len(e_sorted) + 1)
    ok = np.where(cum_risk <= max_risk + 1e-12)[0]
    if len(ok) == 0:
        return None
    return float(r_sorted[ok[-1]])


def threshold_for_target_coverage(risk_score, target_coverage: float):
    risk_score = np.asarray(risk_score, dtype=float)
    risk_score = risk_score[np.isfinite(risk_score)]
    if len(risk_score) == 0:
        return None
    return float(np.quantile(risk_score, target_coverage))


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def cluster_indices(cluster_ids: np.ndarray):
    """List of positional index arrays, one per cluster."""
    order = np.argsort(cluster_ids, kind="stable")
    sorted_ids = cluster_ids[order]
    bounds = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1], True])
    return [order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)]


def cluster_bootstrap_indices(clusters, n_boot: int, rng: np.random.Generator):
    n = len(clusters)
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=n)
        yield np.concatenate([clusters[i] for i in pick])


def percentile_ci(values, level: float = 0.95):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    lo = float(np.percentile(v, (1 - level) / 2 * 100))
    hi = float(np.percentile(v, (1 + level) / 2 * 100))
    return float(np.mean(v)), lo, hi


# --------------------------------------------------------------------------- #
# per-level analysis
# --------------------------------------------------------------------------- #
def analyse_level(items: pd.DataFrame, error: np.ndarray, risk: np.ndarray,
                  clusters, n_boot: int, rng, target_risks, target_coverages,
                  cluster_mode: str):
    """Compute point estimates + cluster-bootstrap CIs for one hierarchy level."""
    n = len(error)
    res = {
        "n_items": int(n),
        "n_clusters": int(len(clusters)),
        "prevalence_error": float(np.mean(error)) if n else float("nan"),
        "aurc": aurc(error, risk),
        "coverage_at_05pct_risk": coverage_at_risk_fast(error, risk, 0.05),
        "coverage_at_10pct_risk": coverage_at_risk_fast(error, risk, 0.10),
    }
    for tc in target_coverages:
        res[f"risk_at_{int(round(tc * 100))}pct_coverage"] = risk_at_coverage(error, risk, tc)

    # error-detection AUROC: can the risk score rank wrong cases above right ones?
    correct = (1.0 - error).astype(float)
    res["error_detection_auroc"] = error_detection_auroc(correct, risk)

    point = {k: res[k] for k in res}

    boot_aurc, boot_cov05, boot_cov10, boot_eda = [], [], [], []
    boot_risk_at = {tc: [] for tc in target_coverages}
    boot_grid = []

    if n_boot > 0 and n > 0 and len(clusters) > 1:
        for idx in cluster_bootstrap_indices(clusters, n_boot, rng):
            e, r = error[idx], risk[idx]
            if len(e) == 0 or len(np.unique(e)) < 1:
                continue
            boot_aurc.append(aurc(e, r))
            boot_cov05.append(coverage_at_risk_fast(e, r, 0.05))
            boot_cov10.append(coverage_at_risk_fast(e, r, 0.10))
            boot_eda.append(error_detection_auroc(1.0 - e, r))
            for tc in target_coverages:
                boot_risk_at[tc].append(risk_at_coverage(e, r, tc))
            boot_grid.append(curve_on_grid(e, r))

    ci = {}
    for name, vals in (
        ("aurc", boot_aurc),
        ("coverage_at_05pct_risk", boot_cov05),
        ("coverage_at_10pct_risk", boot_cov10),
        ("error_detection_auroc", boot_eda),
    ):
        mean, lo, hi = percentile_ci(vals)
        ci[name] = {"mean": mean, "lo": lo, "hi": hi, "n_boot_ok": int(len(np.asarray(vals)[np.isfinite(np.asarray(vals, dtype=float))])) if len(vals) else 0}
    for tc in target_coverages:
        mean, lo, hi = percentile_ci(boot_risk_at[tc])
        ci[f"risk_at_{int(round(tc * 100))}pct_coverage"] = {
            "mean": mean, "lo": lo, "hi": hi,
            "n_boot_ok": int(len(np.asarray(boot_risk_at[tc])[np.isfinite(np.asarray(boot_risk_at[tc], dtype=float))])) if boot_risk_at[tc] else 0,
        }

    band = None
    if boot_grid:
        arr = np.vstack(boot_grid)
        band = pd.DataFrame({
            "level": "finding" if cluster_mode == "instance" else None,
            "coverage": COVERAGE_GRID,
            "risk_point": curve_on_grid(error, risk),
            "risk_mean": np.nanmean(arr, axis=0),
            "risk_lo": np.nanpercentile(arr, 2.5, axis=0),
            "risk_hi": np.nanpercentile(arr, 97.5, axis=0),
            "n_boot_ok": np.sum(np.isfinite(arr), axis=0),
        })
    return point, ci, band


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True,
                        help="predictions.jsonl produced by 03_run_model.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default="data/manifests/nih_pilot_1000.csv",
                        help="manifest providing patient_id for patient-level clustering")
    parser.add_argument("--source-val-split", default="source_val",
                        help="split used ONLY for calibration + threshold selection")
    parser.add_argument("--test-split", default="internal_test",
                        help="split used ONLY for evaluation")
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="patient-cluster bootstrap replicates (0 disables)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-risk", type=float, nargs="+", default=[0.05],
                        help="risk budgets for coverage@risk and threshold selection")
    parser.add_argument("--target-coverage", type=float, nargs="+", default=[0.5, 0.8],
                        help="coverage levels for risk@coverage")
    parser.add_argument("--aggregations", nargs="+", default=["max", "mean", "product"],
                        choices=list(AGGREGATIONS),
                        help="how image/patient risk combines its children's risk")
    parser.add_argument("--min-positives", type=int, default=5,
                        help="findings with fewer positives than this in source_val are not temperature-fitted")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = load_predictions(Path(args.predictions))
    if df.empty:
        print("[error] no usable predictions (need ground_truth + probability_positive)")
        return 1
    df, cluster_source = attach_patient_id(df, Path(args.manifest))
    df.to_csv(out_dir / "flat_predictions_v2.csv", index=False)

    src = df[df["split"] == args.source_val_split].copy()
    test = df[df["split"] == args.test_split].copy()
    if src.empty or test.empty:
        print(f"[error] need both splits: {args.source_val_split}={len(src)}, "
              f"{args.test_split}={len(test)}")
        return 1

    # ---- per-finding temperature, fitted on source_val only -----------------
    temps, cal_rows = {}, []
    for label, g in src.groupby("label"):
        y, p = g["y_true"].to_numpy(), g["p"].to_numpy()
        if len(g) < args.min_positives or len(np.unique(y)) < 2:
            t = 1.0
            note = "identity (insufficient source_val positives)"
        else:
            t = fit_temperature(y, p)
            note = "fitted"
        temps[label] = t
        cal_rows.append({"label": label, "temperature": t, "n_source_val": int(len(g)),
                         "source_val_positives": int(y.sum()), "note": note})
    cal_df = pd.DataFrame(cal_rows).sort_values("label")
    cal_df.to_csv(out_dir / "calibration_by_label_v2.csv", index=False)

    for part in (src, test):
        part["p_calibrated"] = [
            apply_temperature(row.p, temps.get(row.label, 1.0))
            for row in part.itertuples()
        ]

    # ---- risk items at the finding level ------------------------------------
    def build_risk(part: pd.DataFrame) -> pd.DataFrame:
        pred, conf, risk = instance_risk(part["p_calibrated"].to_numpy(),
                                         part["status"].to_numpy())
        out = part.copy()
        out["pred"] = pred
        out["conf"] = conf
        out["risk_score"] = risk
        out["error"] = (pred != out["y_true"].to_numpy()).astype(float)
        return out

    src_i = build_risk(src)
    test_i = build_risk(test)
    test_i.to_csv(out_dir / "risk_items_finding.csv", index=False)

    # ---- global calibration summary on test ---------------------------------
    # Use the canonical implementation rather than a local copy. This script used
    # to define its own `ece` with the same `right=True` binning, which meant that
    # correcting the convention in metrics.py would have left this file -- and every
    # number it reports -- silently on the old convention. Two implementations of
    # one metric is how the two drift apart.
    ece = expected_calibration_error

    calibration_summary = {
        "ece_uncalibrated": ece(test_i["y_true"], test_i["p"]),
        "ece_calibrated": ece(test_i["y_true"], test_i["p_calibrated"]),
        "mean_temperature": float(np.mean(list(temps.values()))),
        "n_labels_identity": int(sum(1 for r in cal_rows if r["note"].startswith("identity"))),
    }

    # ---- analyse each hierarchy level ---------------------------------------
    summary = {
        "levels": {},
        "calibration": calibration_summary,
        "config": {
            "source_val_split": args.source_val_split,
            "test_split": args.test_split,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "target_risks": args.target_risk,
            "target_coverages": args.target_coverage,
            "aggregations": args.aggregations,
            "cluster_source": cluster_source,
            "n_source_val": int(len(src_i)),
            "n_test": int(len(test_i)),
        },
    }
    band_frames, metrics_rows, operating_rows = [], [], []

    # finding level (the comparability baseline vs 07_calibration_selective.py)
    clusters_f = cluster_indices(test_i["patient_id"].to_numpy())
    pt, ci, band = analyse_level(
        test_i, test_i["error"].to_numpy(), test_i["risk_score"].to_numpy(),
        clusters_f, args.n_boot, rng, args.target_risk, args.target_coverage, "instance")
    if band is not None:
        band["level"] = "finding"
        band_frames.append(band)
    summary["levels"]["finding"] = {"point": pt, "ci": ci}
    metrics_rows.append({"level": "finding", "aggregation": "-", **{k: pt[k] for k in
                         ("n_items", "n_clusters", "prevalence_error", "aurc",
                          "coverage_at_05pct_risk", "coverage_at_10pct_risk",
                          "error_detection_auroc")},
                         **{f"{k}_lo": ci[k]["lo"] for k in ci},
                         **{f"{k}_hi": ci[k]["hi"] for k in ci}})

    # image / patient levels, for each aggregation rule
    hierarchical_rows = []
    for level in ("image", "patient"):
        for agg in args.aggregations:
            agg_src = aggregate_to_level(src_i, level, agg)
            agg_test = aggregate_to_level(test_i, level, agg)
            clusters = cluster_indices(agg_test["patient_id"].to_numpy())
            pt, ci, band = analyse_level(
                agg_test, agg_test["error"].to_numpy(), agg_test["risk_score"].to_numpy(),
                clusters, args.n_boot, rng, args.target_risk, args.target_coverage, level)
            if band is not None:
                band["level"] = f"{level}:{agg}"
                band_frames.append(band)
            summary["levels"][f"{level}:{agg}"] = {"point": pt, "ci": ci}
            metrics_rows.append({"level": level, "aggregation": agg, **{k: pt[k] for k in
                                 ("n_items", "n_clusters", "prevalence_error", "aurc",
                                  "coverage_at_05pct_risk", "coverage_at_10pct_risk",
                                  "error_detection_auroc")},
                                 **{f"{k}_lo": ci[k]["lo"] for k in ci},
                                 **{f"{k}_hi": ci[k]["hi"] for k in ci}})

            # ---- operating point chosen on source_val, applied to test -------
            for tr in args.target_risk:
                tau = threshold_for_target_risk(
                    agg_src["error"].to_numpy(), agg_src["risk_score"].to_numpy(), tr)
                if tau is None:
                    operating_rows.append({
                        "level": level, "aggregation": agg, "objective": "coverage_at_risk",
                        "budget": tr, "tau": None, "status": "infeasible_on_source_val",
                    })
                    continue
                acc = agg_test["risk_score"].to_numpy() <= tau
                n_acc = int(acc.sum())
                op_risk = float(agg_test["error"].to_numpy()[acc].mean()) if n_acc else float("nan")
                base = float(agg_test["error"].mean())
                operating_rows.append({
                    "level": level, "aggregation": agg, "objective": "coverage_at_risk",
                    "budget": tr, "tau": tau,
                    "source_val_coverage": float((agg_src["risk_score"].to_numpy() <= tau).mean()),
                    "test_coverage": n_acc / len(agg_test),
                    "test_selective_risk": op_risk,
                    "test_baseline_risk": base,
                    "relative_error_reduction": (base - op_risk) / base if base else float("nan"),
                    "test_deferral_rate": 1 - n_acc / len(agg_test),
                    "workload_reduction": n_acc / len(agg_test),
                    "status": "ok",
                })
            for tc in args.target_coverage:
                tau = threshold_for_target_coverage(agg_src["risk_score"].to_numpy(), tc)
                if tau is None:
                    continue
                acc = agg_test["risk_score"].to_numpy() <= tau
                n_acc = int(acc.sum())
                op_risk = float(agg_test["error"].to_numpy()[acc].mean()) if n_acc else float("nan")
                base = float(agg_test["error"].mean())
                operating_rows.append({
                    "level": level, "aggregation": agg, "objective": "risk_at_coverage",
                    "budget": tc, "tau": tau,
                    "source_val_coverage": float((agg_src["risk_score"].to_numpy() <= tau).mean()),
                    "test_coverage": n_acc / len(agg_test),
                    "test_selective_risk": op_risk,
                    "test_baseline_risk": base,
                    "relative_error_reduction": (base - op_risk) / base if base else float("nan"),
                    "test_deferral_rate": 1 - n_acc / len(agg_test),
                    "workload_reduction": n_acc / len(agg_test),
                    "status": "ok",
                })

            hierarchical_rows.append({
                "level": level, "aggregation": agg,
                "n_source_val_items": int(len(agg_src)),
                "n_test_items": int(len(agg_test)),
                "mean_children_per_item": float(agg_test["n_findings"].mean()),
                "baseline_risk": float(agg_test["error"].mean()),
            })

    # ---- per-finding diagnostics -------------------------------------------
    finding_rows = []
    for label, g in test_i.groupby("label"):
        cl = cluster_indices(g["patient_id"].to_numpy())
        pt, ci, _ = analyse_level(g, g["error"].to_numpy(), g["risk_score"].to_numpy(),
                                  cl, min(args.n_boot, 500), rng, args.target_risk,
                                  args.target_coverage, "instance")
        finding_rows.append({
            "label": label, "n": int(len(g)),
            "prevalence": float(g["y_true"].mean()),
            "temperature": temps.get(label, 1.0),
            "ece_uncalibrated": ece(g["y_true"], g["p"]),
            "ece_calibrated": ece(g["y_true"], g["p_calibrated"]),
            "aurc": pt["aurc"], "aurc_lo": ci["aurc"]["lo"], "aurc_hi": ci["aurc"]["hi"],
            "coverage_at_05pct_risk": pt["coverage_at_05pct_risk"],
            "error_detection_auroc": pt["error_detection_auroc"],
            "n_uncertain_status": int((g["status"].astype(str) == "uncertain").sum()),
        })
    finding_df = pd.DataFrame(finding_rows).sort_values("n", ascending=False)
    finding_df.to_csv(out_dir / "finding_level_metrics.csv", index=False)

    # ---- write everything ---------------------------------------------------
    pd.DataFrame(metrics_rows).to_csv(out_dir / "risk_coverage_metrics.csv", index=False)
    if band_frames:
        pd.concat(band_frames, ignore_index=True).to_csv(
            out_dir / "risk_coverage_bands.csv", index=False)
    op_df = pd.DataFrame(operating_rows)
    op_df.to_csv(out_dir / "operating_points.csv", index=False)
    pd.DataFrame(hierarchical_rows).to_csv(out_dir / "hierarchical_metrics.csv", index=False)

    summary["provenance"] = {
        "script": "scripts/18_selective_metrics_v2.py",
        "predictions": str(args.predictions),
        "manifest": str(args.manifest),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": _safe_version("torch"),
        "transformers": _safe_version("transformers"),
        "gpu": _gpu_name(),
    }
    (out_dir / "selective_v2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # console headline
    print(json.dumps({
        "calibration": calibration_summary,
        "levels": {k: {kk: vv for kk, vv in v["point"].items()
                       if kk in {"n_items", "n_clusters", "prevalence_error", "aurc",
                                  "coverage_at_05pct_risk", "error_detection_auroc"}}
                   for k, v in summary["levels"].items()},
    }, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_dir}")
    return 0


def _safe_version(pkg: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return "unknown"


def _gpu_name() -> str:
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
