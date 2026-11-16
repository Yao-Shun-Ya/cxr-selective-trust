#!/usr/bin/env python
"""87 - Is the label-level degradation hierarchy explained by prevalence?

CPU-only, read-only.  The manuscript reports that severe-degradation recall
retention is concordant across the four readable models on NIH.  A reviewer
can object that the ordering is just label prevalence: common findings have
more positives and therefore a more stable recall estimate.

This script tests that alternative without any new inference:

1. per model, Spearman(retention, prevalence) over the seven pathology labels;
2. partial Spearman between model pairs after rank-residualising prevalence;
3. exact/permutation p-values for the six readable model pairs;
4. a high/low prevalence split, reporting retention and within-stratum
   pairwise concordance.

Inputs (frozen):
  outputs/analysis_nih_mine/joint_D_label_retention.csv
  outputs/analysis_sensitivity/label_prevalence.csv

Outputs:
  outputs/analysis_sensitivity/label_hierarchy_prevalence.csv
  outputs/analysis_sensitivity/label_hierarchy_pairwise_tests.csv
  outputs/analysis_sensitivity/label_hierarchy_strata.csv

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\87_label_hierarchy_prevalence.py
"""
from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RET = ROOT / "outputs" / "analysis_nih_mine" / "joint_D_label_retention.csv"
PREV = ROOT / "outputs" / "analysis_sensitivity" / "label_prevalence.csv"
OUT = ROOT / "outputs" / "analysis_sensitivity"

PATHOLOGY = ["atelectasis", "cardiomegaly", "consolidation", "edema",
             "pleural_effusion", "pneumonia", "pneumothorax"]
MODELS = ["internvl3_5_8b", "qwen3_vl_8b_instruct",
          "medgemma_15_4b_it", "medgemma_4b_it"]


def exact_spearman_p(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided exact permutation p for Spearman with n <= 8."""
    n = len(x)
    if n > 8:
        raise ValueError("exact permutation only for n <= 8")
    rho = float(spearmanr(x, y).statistic)
    count = 0
    total = 0
    for perm in itertools.permutations(range(n)):
        total += 1
        r = float(spearmanr(x, np.asarray(y)[list(perm)]).statistic)
        if abs(r) >= abs(rho) - 1e-12:
            count += 1
    return count / total


def rank_residual(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank().to_numpy()
    design = np.column_stack([np.ones_like(ranks), ranks])
    beta, *_ = np.linalg.lstsq(design, ranks, rcond=None)
    return ranks - design @ beta


def main() -> int:
    ret = pd.read_csv(RET, encoding="utf-8-sig")
    prev = pd.read_csv(PREV, encoding="utf-8-sig").set_index("label")
    ret = ret[ret["label"].isin(PATHOLOGY) & ret["model"].isin(MODELS)].copy()
    ret["retention"] = pd.to_numeric(ret["retention_severe"], errors="coerce")
    ret["recall_clean"] = pd.to_numeric(ret["recall_clean"], errors="coerce")
    prev_vec = prev.loc[PATHOLOGY, "nih_rate"].to_numpy(dtype=float)

    # 1. Retention vs prevalence, per model.
    rows = []
    for model in MODELS:
        g = ret[ret["model"] == model].set_index("label").loc[PATHOLOGY]
        r = g["retention"].to_numpy(dtype=float)
        keep = np.isfinite(r)
        rho_ret = float(spearmanr(prev_vec[keep], r[keep]).statistic)
        rho_rec = float(spearmanr(prev_vec[keep],
                                  g["recall_clean"].to_numpy(dtype=float)[keep]).statistic)
        rows.append({
            "model": model,
            "n_labels": int(keep.sum()),
            "spearman_retention_vs_prevalence": round(rho_ret, 6),
            "spearman_clean_recall_vs_prevalence": round(rho_rec, 6),
            "mean_retention": round(float(np.nanmean(r)), 6),
            "mean_prevalence": round(float(np.mean(prev_vec[keep])), 6),
        })
    tab1 = pd.DataFrame(rows)
    tab1.to_csv(OUT / "label_hierarchy_prevalence.csv", index=False,
                encoding="utf-8-sig")

    # 2-3. Pairwise concordance, raw and prevalence-partialled.
    piv = ret.pivot(index="label", columns="model", values="retention").loc[PATHOLOGY]
    piv = piv[MODELS]
    pair_rows = []
    for a, b in itertools.combinations(MODELS, 2):
        keep = np.isfinite(piv[a]) & np.isfinite(piv[b])
        x = piv.loc[keep, a].to_numpy(dtype=float)
        y = piv.loc[keep, b].to_numpy(dtype=float)
        rho = float(spearmanr(x, y).statistic)
        p_exact = exact_spearman_p(np.arange(len(x)), y) if len(x) <= 8 else np.nan
        # Partial: residualise each model's retention rank on prevalence rank.
        pv = prev.loc[keep.index, "nih_rate"].to_numpy(dtype=float)
        rx = rank_residual(x) if len(set(x)) > 1 else np.zeros_like(x)
        ry = rank_residual(y) if len(set(y)) > 1 else np.zeros_like(y)
        partial = float(spearmanr(rx, ry).statistic) if np.std(rx) and np.std(ry) else np.nan
        pair_rows.append({
            "model_a": a, "model_b": b, "n_labels": int(len(x)),
            "spearman_raw": round(rho, 6),
            "spearman_partial_prevalence": round(partial, 6),
            "p_exact_two_sided": round(float(p_exact), 6),
        })
    tab2 = pd.DataFrame(pair_rows)
    tab2.to_csv(OUT / "label_hierarchy_pairwise_tests.csv", index=False,
                encoding="utf-8-sig")

    # 4. High/low prevalence strata.
    median_prev = float(np.median(prev_vec))
    strata = {l: ("high" if prev.loc[l, "nih_rate"] >= median_prev else "low")
              for l in PATHOLOGY}
    strat_rows = []
    for model in MODELS:
        g = ret[ret["model"] == model].set_index("label").loc[PATHOLOGY]
        for stratum in ("high", "low"):
            labels = [l for l in PATHOLOGY if strata[l] == stratum]
            vals = g.loc[labels, "retention"].to_numpy(dtype=float)
            strat_rows.append({
                "model": model, "stratum": stratum, "n_labels": len(labels),
                "labels": ";".join(labels),
                "mean_retention": round(float(np.nanmean(vals)), 6),
                "min_retention": round(float(np.nanmin(vals)), 6),
                "max_retention": round(float(np.nanmax(vals)), 6),
            })
    tab3 = pd.DataFrame(strat_rows)
    tab3.to_csv(OUT / "label_hierarchy_strata.csv", index=False,
                encoding="utf-8-sig")

    print(tab1.to_string(index=False))
    print()
    print(tab2.to_string(index=False))
    print()
    print(tab3.to_string(index=False))
    print(f"\nmedian NIH prevalence = {median_prev:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
