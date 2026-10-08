#!/usr/bin/env python
"""Evaluate predictions.jsonl from a run.

Example:
  python scripts/04_evaluate.py \
    --predictions outputs/pilot/qwen_nih/predictions.jsonl \
    --output-dir outputs/pilot/qwen_nih/metrics
"""

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

from src.cxr_trust.evaluation.metrics import (
    auroc, auprc, binary_f1, brier, coverage_at_risk,
    error_detection_auroc, expected_calibration_error, high_confidence_error_rate,
    selective_accuracy, aurc,
)


def load_predictions(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            gt = obj.get("ground_truth", {}) or {}
            for label, fp in (obj.get("findings") or {}).items():
                status = fp.get("status")
                prob = fp.get("probability_positive")
                conf = fp.get("confidence")
                y_true = None
                gt_val = gt.get(f"label_{label}")
                if gt_val == "positive":
                    y_true = 1
                elif gt_val == "negative":
                    y_true = 0
                y_pred = None
                if status == "positive":
                    y_pred = 1
                elif status == "negative":
                    y_pred = 0
                # Derive confidence in the selected status if the model did not
                # explicitly provide it. For negative predictions, confidence is
                # 1 - P(present); for positive, it is P(present).
                #
                # Measured 2026-10-06 across 26,462 prediction rows: NO row carries
                # the model's own `findings[*].confidence` (it is null in 100% of
                # them), so every confidence here is derived. And 2.38% of definite
                # findings (3,914 of 164,348) have a status but no probability, in
                # which case the value below stays None and `metrics_for` imputes
                # 0.5 for it. That imputation is the same convention
                # `07_calibration_selective.py` uses (0.5 = a call that is neither
                # confirmed nor refuted) and it is not wrong -- but it was
                # INVISIBLE: `metrics_by_label.csv` had no way to say how much of an
                # AURC rested on imputed values. It does now (n_conf_imputed,
                # conf_source), the same way the calibration universe and the shift
                # coverage census were made explicit rather than implicit.
                conf_imputed = False
                conf_from_model = conf is not None
                if conf is None:
                    if status == "positive":
                        conf = prob
                    elif status == "negative":
                        conf = (1.0 - prob) if prob is not None else None
                    elif status == "uncertain":
                        conf = 0.5
                    if conf is None:
                        conf_imputed = True
                rows.append({
                    "image_id": obj.get("image_id"),
                    "dataset": obj.get("dataset"),
                    "split": obj.get("split"),
                    "label": label,
                    "y_true": y_true,
                    "y_prob": prob,
                    "confidence": conf,
                    "conf_imputed": conf_imputed,
                    "conf_from_model": conf_from_model,
                    "y_pred": y_pred,
                    "status": status,
                })
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, split, label) metrics, with the sample universe made explicit.

    Two metric families need different rows, and one ``dropna`` used to serve both:

    * **probability metrics** (auroc, auprc, ece, brier, high-confidence error) are
      functions of the *declared probability*. A model that answers ``uncertain``
      still emits a probability -- 186 of 186 such rows do in the MedGemma run -- so
      they belong to the estimand. Guo et al. and ``sklearn`` define ECE over the
      model's probability outputs, and a reviewer recomputing with sklearn lands on
      this universe.
    * **classification and selective metrics** (f1, aurc, coverage, error-detection
      AUROC) need a *definite call* to be right or wrong about, plus a confidence.

    The old single ``dropna(subset=["y_true","y_pred","y_prob"])`` silently removed
    every uncertain row from BOTH families. The paper's Table 1 therefore reported
    calibration on 1614 of 1800 test decisions for MedGemma while saying nothing
    about it, and ``07_calibration_selective.py`` reported a different number on the
    full set for the same data. Two universes, one data set, no declaration: the same
    "two implementations" failure this repository has hit with the ECE bin
    convention and with a duplicated ``ece()``.

    Both counts are now emitted, so no consumer has to infer the universe:
    ``n_prob`` for the probability family and ``n_definite`` for the rest.
    """
    out = []
    for (dataset, split, label), g in df.groupby(["dataset", "split", "label"], dropna=False):
        # probability family: a declared probability is enough
        prob = g.dropna(subset=["y_true", "y_prob"]).copy()
        # classification / selective family: a definite call is required
        defi = g.dropna(subset=["y_true", "y_pred"]).copy()
        if len(prob) == 0 and len(defi) == 0:
            continue

        y_true_p = prob["y_true"].astype(float).to_numpy()
        y_prob = prob["y_prob"].astype(float).to_numpy()

        y_true_d = defi["y_true"].astype(float).to_numpy()
        y_pred = defi["y_pred"].astype(float).to_numpy()
        n_conf_imputed = int(defi["conf_imputed"].sum()) if "conf_imputed" in defi else 0
        n_conf_from_model = (int(defi["conf_from_model"].sum())
                             if "conf_from_model" in defi else 0)
        n_defi = len(defi)
        if n_defi and n_conf_from_model == n_defi:
            conf_source = "model"
        elif n_conf_imputed:
            conf_source = "derived+imputed"
        elif n_defi:
            conf_source = "derived"
        else:
            conf_source = "none"
        conf = defi["confidence"].astype(float).fillna(0.5).to_numpy()
        risk = 1.0 - conf
        error = (y_pred != y_true_d).astype(float)
        correct = (y_pred == y_true_d).astype(float)

        hce, hce_n = (high_confidence_error_rate(y_true_p, y_prob)
                      if len(prob) else (float("nan"), 0))
        out.append({
            "dataset": dataset,
            "split": split,
            "label": label,
            # `n` kept as the definite universe for existing consumers; the two
            # explicit counts are what new work should read.
            "n": len(defi),
            "n_prob": len(prob),
            "n_definite": len(defi),
            # How much of the confidence-based family (error_detection_auroc,
            # aurc, coverage/selective accuracy) rests on an imputed 0.5 rather
            # than on a value derived from the model's probability. `conf_source`
            # says where the score comes from at all: no model in the 6K matrix
            # emits its own `confidence` (null in 26,462/26,462 rows measured
            # 2026-10-06), so in practice this is "derived" or "derived+imputed".
            "n_conf_from_model": n_conf_from_model,
            "n_conf_imputed": n_conf_imputed,
            "conf_source": conf_source,
            "auroc": auroc(y_true_p, y_prob) if len(prob) else float("nan"),
            "auprc": auprc(y_true_p, y_prob) if len(prob) else float("nan"),
            "ece": expected_calibration_error(y_true_p, y_prob) if len(prob) else float("nan"),
            "brier": brier(y_true_p, y_prob) if len(prob) else float("nan"),
            "high_conf_error": hce,
            "high_conf_n": hce_n,
            "f1": binary_f1(y_true_d, y_pred) if len(defi) else float("nan"),
            "error_detection_auroc": (error_detection_auroc(correct, risk)
                                      if len(defi) else float("nan")),
            "aurc": aurc(error, risk) if len(defi) else float("nan"),
            "coverage_at_5pct_risk": (coverage_at_risk(error, risk, 0.05)
                                      if len(defi) else float("nan")),
            "coverage_at_10pct_risk": (coverage_at_risk(error, risk, 0.10)
                                       if len(defi) else float("nan")),
            "selective_accuracy_at_80pct_coverage": (selective_accuracy(error, risk, 0.80)
                                                     if len(defi) else float("nan")),
        })
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = load_predictions(Path(args.predictions))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "predictions_flat.csv", index=False)
    metrics = compute_metrics(df)
    metrics.to_csv(out_dir / "metrics_by_label.csv", index=False)
    print(metrics.to_string(index=False))
    print(f"\nwrote {out_dir / 'metrics_by_label.csv'}")

    # State the provenance of the confidence score out loud, every run. On the
    # 6K matrix this prints "derived+imputed" with a small share; if a future
    # prediction set ever reports its own confidence the label becomes "model",
    # and if the imputed share grows, that is visible here rather than inferred
    # from a metric that quietly got worse.
    if len(metrics):
        total_defi = int(metrics["n_definite"].fillna(0).sum())
        total_imp = int(metrics["n_conf_imputed"].fillna(0).sum())
        sources = ", ".join(sorted(set(metrics["conf_source"].dropna().astype(str))))
        share = f"{100.0 * total_imp / total_defi:.2f}%" if total_defi else "n/a"
        print(f"confidence source : {sources}  ({total_imp} of {total_defi} definite "
              f"findings have no probability and are imputed at 0.5 = {share})")


if __name__ == "__main__":
    main()
