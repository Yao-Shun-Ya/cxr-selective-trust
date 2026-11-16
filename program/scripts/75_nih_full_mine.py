#!/usr/bin/env python
"""75 — NIH full-run deep mine (CPU-only, READ-ONLY).

Mines the completed NIH-layer stages of the three-layer 6K run WITHOUT
re-running any model:

  shift        6 models x 19 cells   outputs/matrix/nih_eval/<m>/shift
  reliance     3 models (4-bit)      outputs/matrix/nih_eval/<m>/reliance
  precision    3 models (bf16)       outputs/matrix/nih_eval/<m>/reliance_bf16
  ablation     medgemma_4b_it only   outputs/matrix/nih_eval/<m>/ablation

All metric conventions are inherited from the pipeline, not invented here:
  - parse/label census      -> scripts/09_shift_analysis.py (load_with_census)
  - macro metrics           -> src/cxr_trust/evaluation/metrics.py
  - reliance summary fields -> scripts/06_image_reliance.py
  - contract fields         -> the `contract` block written by 21_prompt_ablation.py

Outputs go to outputs/analysis_nih_mine/ (isolated from the live run; the
driver only writes under outputs/matrix and outputs/state). Safe to run while
the GPU run is in progress: this script opens prediction files only for stages
the state file marks complete, and reads them read-only.

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\75_nih_full_mine.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.evaluation.metrics import (  # noqa: E402
    auroc, expected_calibration_error,
)
from src.cxr_trust.inference.schema import PRIMARY_LABELS  # noqa: E402

MATRIX = ROOT / "outputs" / "matrix" / "nih_eval"
OUT = ROOT / "outputs" / "analysis_nih_mine"
STATE = ROOT / "outputs" / "state" / "run_all_layers_state.json"

MODELS = [
    "internvl3_5_8b",
    "qwen25_vl_7b_instruct",
    "qwen3_vl_8b_instruct",
    "medgemma_15_4b_it",
    "medgemma_4b_it",
    "chexagent_8b",
]
PATHOLOGY = [l for l in PRIMARY_LABELS if l != "no_finding"]
MARKER = "<unused94>"  # medgemma_15 reasoning-channel opener (docs/50 §2.12)
SEV_LEVEL = {"none": 0, "clean": 0, "mild": 1, "moderate": 2, "severe": 3}
DEGRADE_TYPES = ["blur", "noise", "jpeg", "resolution", "contrast", "brightness"]
# Models whose default inference precision is 4-bit (docs/48 §4.6 $FourBitModels);
# for them the `reliance` stage is 4-bit and `reliance_bf16` is the bf16 control.
# Every other model's `reliance` stage is its native precision.
FOUR_BIT_MODELS = {"medgemma_4b_it", "medgemma_15_4b_it", "qwen25_vl_7b_instruct"}
N_BOOT = 400
SEED = 0


def _safe(fn, default=float("nan")):
    try:
        v = fn()
        return default if v is None else float(v)
    except Exception:
        return default


def completed_stages() -> set[str]:
    # utf-8-sig: the state file is written by PowerShell and may carry a BOM.
    try:
        return set(json.loads(STATE.read_text(encoding="utf-8-sig")).get("completed", []))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not read state file {STATE}: {e}")
        return set()


def boot_ci(values, n=N_BOOT, seed=SEED):
    """Percentile bootstrap CI of the mean; values: 1-D array-like."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        m = float(v.mean()) if len(v) else float("nan")
        return m, m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def boot_ratio_ci(tp, gt, n=N_BOOT, seed=SEED):
    """CI for a pooled ratio sum(tp)/sum(gt) by resampling the unit (image)."""
    tp = np.asarray(tp, dtype=float)
    gt = np.asarray(gt, dtype=float)
    keep = ~np.isnan(gt)
    tp, gt = tp[keep], gt[keep]
    if len(gt) < 3 or gt.sum() == 0:
        r = float(tp.sum() / gt.sum()) if gt.sum() else float("nan")
        return r, r, r
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gt), size=(n, len(gt)))
    g = gt[idx].sum(axis=1)
    t = tp[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(g > 0, t / np.where(g > 0, g, 1), np.nan)
    ratios = ratios[~np.isnan(ratios)]
    if len(ratios) == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return float(tp.sum() / gt.sum()), float(lo), float(hi)


# ----------------------------------------------------------------------
# shift
# ----------------------------------------------------------------------

def load_shift(model: str):
    """Stream shift_predictions.jsonl into an attempts frame + a label frame.

    raw_output is dropped after the marker check; nothing is kept that the
    metric code does not need, so six 28,614-row files stay small in memory.
    """
    path = MATRIX / model / "shift" / "shift_predictions.jsonl"
    attempts, rows = [], []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            cond, sev = obj.get("condition"), obj.get("severity")
            parsed = obj.get("parsed") or {}
            ok = bool(parsed.get("parse_success"))
            marker = (obj.get("raw_output") or "").startswith(MARKER)
            img = obj["image_id"]
            attempts.append({"image_id": img, "condition": cond, "severity": sev,
                             "parse_ok": ok, "marker": marker,
                             "marker_unparsed": bool(marker and not ok)})
            if not ok:
                continue
            gt = obj.get("ground_truth", {}) or {}
            findings = parsed.get("findings") or {}
            for label in PRIMARY_LABELS:
                fp = findings.get(label) or {}
                gt_val = gt.get(f"label_{label}")
                rows.append({
                    "image_id": img, "condition": cond, "severity": sev,
                    "label": label, "status": fp.get("status"),
                    "p": fp.get("probability_positive"),
                    "y": (1.0 if gt_val == "positive" else 0.0)
                         if gt_val in {"positive", "negative"} else np.nan,
                })
    return pd.DataFrame(attempts), pd.DataFrame(rows)


def clean_key(att: pd.DataFrame):
    """Discover the (condition, severity) that names the untouched image."""
    c = att[att["severity"].astype(str).isin(["none", "clean"])]
    if len(c):
        cond = c["condition"].mode().iloc[0]
        return cond, c["severity"].mode().iloc[0]
    c = att[att["condition"].astype(str).isin(["clean", "original"])]
    if len(c):
        return c["condition"].mode().iloc[0], c["severity"].mode().iloc[0]
    return None, None


def cell_metrics(att: pd.DataFrame, rows: pd.DataFrame, cond, sev):
    a = att[(att["condition"] == cond) & (att["severity"] == sev)]
    g = rows[(rows["condition"] == cond) & (rows["severity"] == sev)]
    n_att = len(a)
    n_ok = int(a["parse_ok"].sum())
    out = {
        "condition": cond, "severity": sev,
        "n_attempted": n_att, "n_parse_ok": n_ok,
        "parse_coverage": n_ok / n_att if n_att else np.nan,
        "n_marker": int(a["marker"].sum()),
        "n_marker_unparsed": int(a["marker_unparsed"].sum()),
    }
    if g.empty:
        return out
    path = g[g["label"].isin(PATHOLOGY)]
    y = path["y"].to_numpy()
    status = path["status"].fillna("not_mentioned").to_numpy()
    tp = int(((status == "positive") & (y == 1)).sum())
    gt_pos = int((y == 1).sum())
    fp = int(((status == "positive") & (y == 0)).sum())
    out["path_recall"] = tp / gt_pos if gt_pos else np.nan
    out["path_precision"] = tp / (tp + fp) if (tp + fp) else np.nan
    out["n_gt_pos"] = gt_pos
    out["n_path_fp"] = fp
    # Predicted-positive rate over pathology slots. This is the confound dial:
    # a condition that raises recall AND raises this together is a threshold
    # shift (everything looks abnormal), not a robustness gain.
    out["pred_pos_rate"] = float((status == "positive").mean())
    # per-label discrimination + calibration, macro-averaged
    aus, eces = [], []
    for label, lg in g.groupby("label"):
        yy = lg["y"].to_numpy()
        pp = lg["p"].fillna(0.0).to_numpy()
        if len(np.unique(yy[~np.isnan(yy)])) > 1:
            aus.append(_safe(lambda: auroc(yy, pp)))
        eces.append(_safe(lambda: expected_calibration_error(yy, pp)))
    out["macro_auroc"] = float(np.nanmean(aus)) if aus else np.nan
    out["macro_ece"] = float(np.nanmean(eces)) if eces else np.nan
    nf = g[g["label"] == "no_finding"]
    out["nofinding_pos_rate"] = float(
        (nf["status"].fillna("") == "positive").mean()) if len(nf) else np.nan
    out["prob_completeness"] = float(g["p"].notna().mean())
    # failure-shape census on parsed slots
    st = g["status"].fillna("not_mentioned").value_counts(normalize=True)
    for k in ("positive", "negative", "uncertain", "not_mentioned"):
        out[f"status_share_{k}"] = float(st.get(k, 0.0))
    return out


def shift_flips(rows: pd.DataFrame, clean_cond, clean_sev):
    """Image-level answer change vs the clean cell (09's per_image_flips,
    recomputed here so it can be joined with the reliance per-image frame)."""
    clean = rows[(rows["condition"] == clean_cond) & (rows["severity"] == clean_sev)]
    base = {(r.image_id, r.label): r.status for r in clean.itertuples()}
    out = []
    for (img, cond, sev), g in rows.groupby(["image_id", "condition", "severity"]):
        if cond == clean_cond and sev == clean_sev:
            continue
        flips = 0
        diffs = []
        for r in g.itertuples():
            b = base.get((img, r.label))
            if b is not None and b != r.status:
                flips += 1
        out.append({"image_id": img, "condition": cond, "severity": sev,
                    "flip_count": flips, "flip_any": int(flips > 0)})
    return pd.DataFrame(out)


def mine_shift(model: str, done: set[str]):
    if f"nih/{model}/shift" not in done:
        return None
    t0 = time.time()
    att, rows = load_shift(model)
    clean_cond, clean_sev = clean_key(att)
    cells = []
    for (cond, sev), _ in att.groupby(["condition", "severity"]):
        cells.append(cell_metrics(att, rows, cond, sev))
    cells = pd.DataFrame(cells)
    cells["sev_level"] = cells["severity"].map(SEV_LEVEL)

    flips = shift_flips(rows, clean_cond, clean_sev)
    if not flips.empty:
        fs = flips.groupby(["condition", "severity"]).agg(
            shift_image_sensitivity=("flip_any", "mean"),
            shift_mean_flip_count=("flip_count", "mean"),
        ).reset_index()
        cells = cells.merge(fs, on=["condition", "severity"], how="left")

    clean = cells[(cells["condition"] == clean_cond) & (cells["severity"] == clean_sev)]
    sev = cells[cells["severity"] == "severe"] if "severe" in set(cells["severity"]) else cells.iloc[0:0]

    def _ci_ratio(sub, col_tp_num=True):
        # pooled recall CI by resampling images: rebuild tp/gt per image
        p = rows[rows["label"].isin(PATHOLOGY)]
        p = p[p["condition"].isin(sub["condition"]) & p["severity"].isin(sub["severity"])] \
            if len(sub) else p.iloc[0:0]
        if p.empty:
            return (np.nan,) * 3
        per = p.groupby("image_id").apply(
            lambda gg: pd.Series({
                "tp": int(((gg["status"] == "positive") & (gg["y"] == 1)).sum()),
                "gt": int((gg["y"] == 1).sum())}),
            include_groups=False)
        return boot_ratio_ci(per["tp"].to_numpy(), per["gt"].to_numpy())

    headline = {
        "model": model,
        "clean_condition": clean_cond,
        "n_images": int(att["image_id"].nunique()),
        "n_cells": int(len(cells)),
    }
    if len(clean):
        c = clean.iloc[0]
        headline.update({
            "clean_path_recall": c.get("path_recall"),
            "clean_macro_auroc": c.get("macro_auroc"),
            "clean_macro_ece": c.get("macro_ece"),
            "clean_nofinding_rate": c.get("nofinding_pos_rate"),
            "clean_parse_coverage": c.get("parse_coverage"),
            "clean_prob_completeness": c.get("prob_completeness"),
            "clean_shift_image_sensitivity": c.get("shift_image_sensitivity"),
        })
        headline["clean_path_recall_ci"] = _ci_ratio(clean)
    if len(sev):
        pooled_tp = (sev["path_recall"] * sev["n_gt_pos"]).sum()
        pooled_gt = sev["n_gt_pos"].sum()
        headline.update({
            "severe_path_recall_pooled": pooled_tp / pooled_gt if pooled_gt else np.nan,
            "severe_macro_ece_mean": sev["macro_ece"].mean(),
            "severe_nofinding_rate_mean": sev["nofinding_pos_rate"].mean(),
            "severe_parse_coverage_min": sev["parse_coverage"].min(),
            "severe_shift_image_sensitivity_mean": sev.get(
                "shift_image_sensitivity", pd.Series(dtype=float)).mean(),
        })
    if headline.get("clean_path_recall") and not math.isnan(
            headline.get("severe_path_recall_pooled", np.nan)):
        headline["recall_retention_severe"] = (
            headline["severe_path_recall_pooled"] / headline["clean_path_recall"])
    headline["nofinding_drift_clean_to_severe"] = (
        headline.get("severe_nofinding_rate_mean", np.nan)
        - headline.get("clean_nofinding_rate", np.nan))
    # illusion check: pooled accuracy on clean vs the always-negative ceiling
    cl = rows[(rows["condition"] == clean_cond) & (rows["severity"] == clean_sev)
              & rows["label"].isin(PATHOLOGY)]
    if len(cl):
        pred = np.where(cl["status"] == "positive", 1.0,
                        np.where(cl["status"] == "negative", 0.0, -1.0))
        headline["clean_pooled_accuracy"] = float(
            (pred == cl["y"].to_numpy()).mean())
        headline["majority_negative_ceiling"] = float((cl["y"] == 0).mean())
    # reasoning-channel engagement (medgemma_15): marker rate by severity
    a_clean = att[(att["condition"] == clean_cond) & (att["severity"] == clean_sev)]
    a_sev = att[att["severity"] == "severe"]
    headline["marker_rate_clean"] = float(a_clean["marker"].mean()) if len(a_clean) else np.nan
    headline["marker_rate_severe"] = float(a_sev["marker"].mean()) if len(a_sev) else np.nan
    headline["marker_unparsed_rate_severe"] = (
        float(a_sev["marker_unparsed"].mean()) if len(a_sev) else np.nan)
    # failure shape (all cells pooled, parsed slots only)
    st = rows["status"].fillna("not_mentioned").value_counts(normalize=True)
    for k in ("positive", "negative", "uncertain", "not_mentioned"):
        headline[f"slot_share_{k}"] = float(st.get(k, 0.0))
    headline["elapsed_sec"] = round(time.time() - t0, 1)

    # per degradation type, severe vs clean recall (dose-response headline)
    type_rows = []
    for dt in DEGRADE_TYPES:
        for s in ("mild", "moderate", "severe"):
            r = cells[(cells["condition"] == dt) & (cells["severity"] == s)]
            if len(r):
                cl_row = cells[(cells["condition"] == clean_cond)
                               & (cells["severity"] == clean_sev)]
                type_rows.append({
                    "model": model, "type": dt, "severity": s,
                    "sev_level": SEV_LEVEL.get(s, np.nan),
                    "path_recall": r.iloc[0].get("path_recall"),
                    "path_precision": r.iloc[0].get("path_precision"),
                    "pred_pos_rate": r.iloc[0].get("pred_pos_rate"),
                    "n_path_fp": r.iloc[0].get("n_path_fp"),
                    "macro_ece": r.iloc[0].get("macro_ece"),
                    "nofinding_pos_rate": r.iloc[0].get("nofinding_pos_rate"),
                    "parse_coverage": r.iloc[0].get("parse_coverage"),
                    # deliberation dose-response (only medgemma_15 opens the
                    # reasoning channel; 0 elsewhere) — see docs/50 §2.12
                    "marker_rate": (r.iloc[0].get("n_marker", 0.0)
                                    / r.iloc[0].get("n_attempted", 1)),
                    "marker_unparsed_rate": (r.iloc[0].get("n_marker_unparsed", 0.0)
                                             / r.iloc[0].get("n_attempted", 1)),
                    "shift_image_sensitivity": r.iloc[0].get("shift_image_sensitivity"),
                    # deltas vs the clean cell make the confound readable in one row
                    "d_recall_vs_clean": (
                        r.iloc[0].get("path_recall")
                        - (cl_row.iloc[0].get("path_recall") if len(cl_row) else np.nan)),
                    "d_precision_vs_clean": (
                        r.iloc[0].get("path_precision")
                        - (cl_row.iloc[0].get("path_precision") if len(cl_row) else np.nan)),
                    "d_pred_pos_rate_vs_clean": (
                        r.iloc[0].get("pred_pos_rate")
                        - (cl_row.iloc[0].get("pred_pos_rate") if len(cl_row) else np.nan)),
                })
    return {
        "cells": cells, "headline": headline, "flips": flips,
        "type_curve": pd.DataFrame(type_rows),
        "rows_by_image": rows,
    }


# ----------------------------------------------------------------------
# reliance / precision
# ----------------------------------------------------------------------

def load_reliance(model: str, stage: str):
    d = MATRIX / model / stage
    sj = d / "image_reliance_summary.json"
    if not sj.exists():
        return None
    summary = json.loads(sj.read_text(encoding="utf-8"))
    per = d / "image_reliance_per_image.csv"
    per_df = pd.read_csv(per) if per.exists() else pd.DataFrame()
    return {"summary": summary, "per_image": per_df, "stage": stage}


def reliance_table(done: set[str]):
    rows = []
    frames = {}
    for model in MODELS:
        for stage in ("reliance", "reliance_bf16"):
            if stage == "reliance":
                prec = "4bit" if model in FOUR_BIT_MODELS else "native"
                key = f"nih/{model}/reliance"
                if key not in done:
                    continue
            else:
                prec = "bf16"
                if f"nih/{model}/precision" not in done:
                    continue
            r = load_reliance(model, stage)
            if r is None:
                continue
            frames[(model, prec)] = r
            s = r["summary"]
            for cond, c in s.get("conditions", {}).items():
                rows.append({
                    "model": model, "precision": prec, "condition": cond,
                    "n_valid_pairs": c.get("n_valid_pairs"),
                    "image_sensitivity": c.get("image_sensitivity"),
                    "mean_flip_count": c.get("mean_flip_count"),
                    "vrs_mean_abs_prob_diff": c.get("mean_abs_prob_diff"),
                    "mean_text_jaccard": c.get("mean_text_jaccard"),
                    "pct_identical_text": c.get("pct_identical_text"),
                    "parse_failures": s.get("parse_failures_by_condition", {}).get(cond),
                })
    return pd.DataFrame(rows), frames


# ----------------------------------------------------------------------
# ablation
# ----------------------------------------------------------------------

ABLATION_VARIANTS = ["v0_simple", "v1_per_label_binary", "v2_few_shot",
                     "v3_report_first", "v4_normality_pressure"]


def wilson_ci(k: int, n: int, z: float = 1.959963985):
    """Wilson score interval — correct at the extremes (0/n, n/n) where a
    normal-approximation interval collapses to a zero-width point."""
    if n <= 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def mine_ablation(done: set[str]):
    model = "medgemma_4b_it"
    if f"nih/{model}/ablation" not in done:
        return None
    # ground truth lives in the manifest, not in ablation rows; rebuild the
    # same split filter the stage used (docs/48: internal_test,source_val)
    man = pd.read_csv(ROOT / "data" / "manifests" / "nih_eval_1732.csv")
    man = man[man["split"].isin(["internal_test", "source_val"])]
    gt_of = {}
    for r in man.itertuples():
        gt_of[str(r.image_id)] = {
            l: getattr(r, f"label_{l}") for l in PRIMARY_LABELS}
    rows = []
    for variant in ABLATION_VARIANTS:
        for cond in ("original", "blank"):
            d = MATRIX / model / "ablation" / variant / cond
            sj = d / "parse_failure_summary.json"
            if not sj.exists():
                continue
            s = json.loads(sj.read_text(encoding="utf-8"))
            rec = {
                "variant": variant, "condition": cond,
                "parse_failure_rate": s.get("parse_failure_rate"),
                "contract_violation_rate": s.get("contract_violation_rate"),
                "mean_labels_defaulted": s.get("mean_labels_defaulted"),
                "mean_probs_missing": s.get("mean_probs_missing"),
            }
            # per-row contract + answer stats from the predictions file
            pj = d / "image_reliance_predictions.jsonl"
            if pj.exists():
                path_tp = path_gt = 0
                path_fp = 0
                nf_pos = 0
                n_parsed = 0
                blind_complete = 0
                n_rows = 0
                status_counts = {k: 0 for k in
                                 ("positive", "negative", "uncertain",
                                  "not_mentioned")}
                n_with_gt = 0
                n_contract_clean_blank = 0
                n_abstain_blank = 0
                with pj.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        n_rows += 1
                        parsed = obj.get("parsed") or {}
                        if not parsed.get("parse_success"):
                            continue
                        n_parsed += 1
                        findings = parsed.get("findings") or {}
                        gt = gt_of.get(str(obj.get("image_id")), {})
                        if gt:
                            n_with_gt += 1
                        nf = findings.get("no_finding") or {}
                        nf_pos += int(nf.get("status") == "positive")
                        for label in PRIMARY_LABELS:
                            fp = findings.get(label) or {}
                            st = fp.get("status") or "not_mentioned"
                            status_counts[st if st in status_counts
                                          else "not_mentioned"] += 1
                            if label == "no_finding" or not gt:
                                continue
                            gt_val = gt.get(label)
                            if gt_val == "positive":
                                path_gt += 1
                                path_tp += int(st == "positive")
                            elif gt_val == "negative":
                                path_fp += int(st == "positive")
                        ct = obj.get("contract") or {}
                        if cond == "blank":
                            if not ct.get("contract_violation") \
                                    and ct.get("n_probs_missing") == 0:
                                blind_complete += 1
                            # The prompt-flip claim needs a denominator that is
                            # "replied at all": a reply that parsed but stated
                            # nothing is not evidence of abstention.
                            if parsed.get("parse_success"):
                                n_contract_clean_blank += 1
                                st_set = {((findings.get(l) or {}).get("status"))
                                          for l in PRIMARY_LABELS}
                                # Abstention is counted on the *abstention slot*
                                # (status_share_uncertain), not on an all-uncertain
                                # criterion: no_finding legitimately answers
                                # "negative" on a blank image, so requiring every
                                # slot to be uncertain would score 0 for a model
                                # that abstains on all six pathology slots.
                                if "uncertain" in st_set and "positive" not in st_set:
                                    n_abstain_blank += 1
                rec["n_rows"] = n_rows
                rec["n_rows_with_gt"] = n_with_gt
                rec["path_recall"] = path_tp / path_gt if path_gt else np.nan
                rec["path_precision"] = (path_tp / (path_tp + path_fp)
                                         if (path_tp + path_fp) else np.nan)
                rec["nofinding_pos_rate"] = nf_pos / n_parsed if n_parsed else np.nan
                tot = sum(status_counts.values()) or 1
                for k, v in status_counts.items():
                    rec[f"status_share_{k}"] = v / tot
                if cond == "blank":
                    rec["blind_full_completion_rate"] = blind_complete / n_parsed \
                        if n_parsed else np.nan
                    r_ = wilson_ci(blind_complete, n_parsed)
                    rec["blind_full_completion_ci"] = r_[1:]
                    rec["all_uncertain_rate"] = (n_abstain_blank
                                                 / n_contract_clean_blank
                                                 if n_contract_clean_blank else np.nan)
                    r_ = wilson_ci(n_abstain_blank, n_contract_clean_blank)
                    rec["all_uncertain_ci"] = r_[1:]
            rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# cross-model syntheses
# ----------------------------------------------------------------------

def per_image_cross(shift_res: dict, rel_frames: dict):
    """Join reliance blank flips with shift severe flips on the same image.

    Only models that have BOTH a reliance per-image frame and a completed
    shift are joined. Precision regimes are reported per source so a 4-bit
    shift vs bf16 reliance join is flagged, not hidden.
    """
    out = []
    for (model, prec), r in rel_frames.items():
        per = r["per_image"]
        if per is None or per.empty or model not in shift_res:
            continue
        # Use the runner's own column names verbatim. Renaming them to
        # `rel_flip_count` invented a key that CONTAINS a real one, which the
        # row-key audit (64_audit_row_keys.py) correctly reports as a near miss:
        # analysis may only read keys a runner writes.
        blank = per[per["condition"] == "blank"][
            ["image_id", "flip_any", "flip_count", "mean_abs_prob_diff"]]
        flips = shift_res[model]["flips"]
        if flips.empty:
            continue
        sev = flips[flips["severity"] == "severe"]
        if sev.empty:
            continue
        sf = sev.groupby("image_id")["flip_count"].mean().rename(
            "shift_severe_mean_flip").reset_index()
        j = blank.merge(sf, on="image_id", how="inner").dropna(
            subset=["flip_count", "shift_severe_mean_flip"])
        if len(j) < 30:
            continue
        # np.corrcoef returns NaN for the WHOLE array if any element is NaN, and
        # the std guard passes happily on a series whose NaN pandas skipped — so
        # the pairwise-complete subset is the only safe input (this is exactly
        # how medgemma_15 lost its correlation: `flip_count` is None → NaN when a
        # condition had no structured reply).
        corr = float(np.corrcoef(j["flip_count"].to_numpy(),
                                 j["shift_severe_mean_flip"].to_numpy())[0, 1]) \
            if len(j) > 2 and j["flip_count"].std() > 0 \
            and j["shift_severe_mean_flip"].std() > 0 else float("nan")
        both = float(((j["flip_any"] == 1)
                      & (j["shift_severe_mean_flip"] > 0)).mean())
        shift_prec = "4bit" if model in FOUR_BIT_MODELS else "native"
        spearman = float(j["flip_count"].corr(
            j["shift_severe_mean_flip"], method="spearman")) \
            if len(j) > 2 else float("nan")
        out.append({
            "model": model, "reliance_precision": prec,
            "shift_precision": shift_prec,
            # A join whose two sides ran at different precisions is NOT a clean
            # within-regime correlation; keep the flag in the row so the reading
            # cannot silently cross regimes (medgemma_15: bf16 reliance vs 4-bit shift).
            "precision_mismatch": int(prec != shift_prec),
            "n_images_joined": len(j),
            "pearson_rel_blank_vs_shift_severe": corr,
            "spearman_rel_blank_vs_shift_severe": spearman,
            "pct_images_flipping_in_both": 100 * both,
            "rel_blank_flip_rate": float(j["flip_any"].mean()),
            "shift_severe_flip_rate": float((j["shift_severe_mean_flip"] > 0).mean()),
        })
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    done = completed_stages()
    ledger = {"generated_by": "scripts/75_nih_full_mine.py",
              "completed_stages_seen": sorted(done), "models": MODELS}

    # ---- shift ---------------------------------------------------------
    shift_res = {}
    headlines = []
    type_curves = []
    all_cells = []
    for m in MODELS:
        r = mine_shift(m, done)
        if r is None:
            print(f"[skip] {m}: shift not complete")
            continue
        shift_res[m] = r
        headlines.append(r["headline"])
        type_curves.append(r["type_curve"])
        c = r["cells"].copy()
        c.insert(0, "model", m)
        all_cells.append(c)
        print(f"[shift] {m}: clean_recall="
              f"{r['headline'].get('clean_path_recall'):.3f} "
              f"severe_recall={r['headline'].get('severe_path_recall_pooled', float('nan')):.3f} "
              f"({r['headline']['elapsed_sec']}s)")
    head_df = pd.DataFrame(headlines)
    head_df.to_csv(OUT / "table_s1_shift_headline.csv", index=False)
    tc = pd.concat(type_curves) if type_curves else pd.DataFrame()
    if not tc.empty:
        tc.to_csv(OUT / "fig2_recall_vs_severity_curves.csv", index=False)
        # Confound check the type curves exist to answer: a "robustness" cell is
        # only real if recall moves WITHOUT the model just calling everything
        # abnormal. `pred_pos_rate` is the primary discriminator, not precision:
        # when TP and FP rise together precision can stay flat while the model
        # has plainly become more trigger-happy.
        conf = tc[tc["severity"] == "severe"].copy()
        conf["verdict"] = np.where(
            (conf["d_recall_vs_clean"] > 0.02)
            & (conf["d_pred_pos_rate_vs_clean"] > 0.02),
            "threshold_shift",
            np.where(conf["d_recall_vs_clean"] < -0.02, "recall_loss", "stable"))
        conf = conf[["model", "type", "path_recall", "path_precision",
                     "pred_pos_rate", "d_recall_vs_clean",
                     "d_precision_vs_clean", "d_pred_pos_rate_vs_clean",
                     "n_path_fp", "verdict"]]
        conf.to_csv(OUT / "fig5_threshold_shift_confound.csv", index=False)
        ledger["threshold_shift_confound"] = conf.to_dict(orient="records")
    if all_cells:
        pd.concat(all_cells).to_csv(OUT / "shift_cell_metrics_all.csv", index=False)
    ledger["shift_headline"] = headlines

    # ---- reliance / precision ------------------------------------------
    rel_df, rel_frames = reliance_table(done)
    rel_df.to_csv(OUT / "fig1_reliance_spectrum.csv", index=False)
    ledger["reliance_spectrum"] = rel_df.to_dict(orient="records")
    print(f"[reliance] {len(rel_df)} model×precision×condition rows")

    # quantization delta (4bit vs bf16 on the same model, reliance design)
    qrows = []
    for model in (MODELS if not rel_df.empty else []):
        a = rel_df[(rel_df["model"] == model) & (rel_df["precision"] == "4bit")]
        b = rel_df[(rel_df["model"] == model) & (rel_df["precision"] == "bf16")]
        if a.empty or b.empty:
            continue
        for cond in ("blank", "blur", "shuffle"):
            ra = a[a["condition"] == cond]
            rb = b[b["condition"] == cond]
            if ra.empty or rb.empty:
                continue
            qrows.append({
                "model": model, "condition": cond,
                "vrs_4bit": ra.iloc[0]["vrs_mean_abs_prob_diff"],
                "vrs_bf16": rb.iloc[0]["vrs_mean_abs_prob_diff"],
                "delta_vrs_bf16_minus_4bit":
                    rb.iloc[0]["vrs_mean_abs_prob_diff"]
                    - ra.iloc[0]["vrs_mean_abs_prob_diff"],
                "sens_4bit": ra.iloc[0]["image_sensitivity"],
                "sens_bf16": rb.iloc[0]["image_sensitivity"],
                "jaccard_4bit": ra.iloc[0]["mean_text_jaccard"],
                "jaccard_bf16": rb.iloc[0]["mean_text_jaccard"],
            })
    qdf = pd.DataFrame(qrows)
    qdf.to_csv(OUT / "table_s6_quantization_delta.csv", index=False)
    ledger["quantization_delta"] = qrows

    # per-image reliance-vs-shift join
    cross = per_image_cross(shift_res, rel_frames)
    cross.to_csv(OUT / "fig4_per_image_cross_join.csv", index=False)
    ledger["per_image_cross"] = cross.to_dict(orient="records")

    # bootstrap CIs on VRS blank (per model×precision with data)
    vrs_ci = []
    for (model, prec), r in rel_frames.items():
        per = r["per_image"]
        if per is None or per.empty:
            continue
        blank = per[per["condition"] == "blank"]["mean_abs_prob_diff"].dropna()
        if blank.empty:
            continue
        m_, lo, hi = boot_ci(blank.to_numpy())
        sens = per[per["condition"] == "blank"]["flip_any"].dropna()
        sm, slo, shi = boot_ci(sens.to_numpy())
        vrs_ci.append({"model": model, "precision": prec,
                       "vrs_blank": m_, "vrs_blank_ci_lo": lo, "vrs_blank_ci_hi": hi,
                       "image_sensitivity_blank": sm,
                       "image_sensitivity_ci_lo": slo, "image_sensitivity_ci_hi": shi})
    pd.DataFrame(vrs_ci).to_csv(OUT / "reliance_bootstrap_ci.csv", index=False)
    ledger["reliance_bootstrap_ci"] = vrs_ci

    # ---- ablation ------------------------------------------------------
    abl = mine_ablation(done)
    if abl is not None:
        abl.to_csv(OUT / "table_s7_ablation.csv", index=False)
        ledger["ablation"] = abl.to_dict(orient="records")
        print(f"[ablation] {len(abl)} cells")

    # ---- cross-model map -----------------------------------------------
    if len(head_df) and len(rel_df):
        blank = rel_df[rel_df["condition"] == "blank"].copy()
        blank["key"] = blank["model"]
        best = blank.sort_values("precision").groupby("key").tail(1)
        cmap = head_df.merge(
            best[["key", "vrs_mean_abs_prob_diff", "image_sensitivity",
                  "mean_text_jaccard", "precision"]].rename(
                columns={"key": "model", "precision": "reliance_precision"}),
            on="model", how="left")
        cmap.to_csv(OUT / "fig3_cross_map.csv", index=False)
        ledger["cross_map"] = cmap.to_dict(orient="records")

    (OUT / "nih_mine_ledger.json").write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
