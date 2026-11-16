#!/usr/bin/env python
"""77 — cross-parameter joint mine (CPU-only, READ-ONLY).

Six joint analyses on top of the already-mined NIH layer. Metric conventions
are NOT re-invented here: script 75 is loaded as a module and its readers /
miners are reused verbatim (``completed_stages``, ``load_shift``, ``load_reliance``,
``mine_shift``, ``mine_ablation``, ``boot_ci``, ``boot_ratio_ci``, ``wilson_ci``,
``FOUR_BIT_MODELS``).

Data boundary (the project red line, inherited from the state file):
  * only stages listed in outputs/state/run_all_layers_state.json ``completed``
    are read. qwen25_vl_7b_instruct's NIH stages were removed from that list and
    its files archived under ``_pre_contract_20261019/`` — filtering by state
    excludes it automatically. We never open those files.
  * llava_med_v1_5_7b is contract-exempt and never enters this script.
  * nothing under outputs/matrix or outputs/state is written. Output goes to
    outputs/analysis_nih_mine/ as joint_*.csv + joint_parameters.json.

Every reported number carries its denominator / coverage (n, n_valid_pairs,
parse coverage). No fitted thresholds are introduced; where a descriptive cut
is used it is tagged "descriptive, not fitted".

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\77_cross_parameter_mine.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# ---- load script 75 as a module so metric conventions match exactly --------
_spec = importlib.util.spec_from_file_location(
    "nih_full_mine", ROOT / "scripts" / "75_nih_full_mine.py")
m75 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m75)

from scipy.stats import spearmanr, kendalltau  # noqa: E402
from scipy.cluster.hierarchy import linkage, fcluster  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402

MATRIX = m75.MATRIX
OUT = m75.OUT
PRIMARY_LABELS = m75.PRIMARY_LABELS
PATHOLOGY = m75.PATHOLOGY
MARKER = m75.MARKER
DEGRADE_TYPES = m75.DEGRADE_TYPES
FOUR_BIT_MODELS = m75.FOUR_BIT_MODELS
ALL_STATUS = ("positive", "negative", "uncertain", "not_mentioned")

# 5 models with a clean NIH shift + reliance; qwen25 is filtered by state.
MODELS5 = ["internvl3_5_8b", "qwen3_vl_8b_instruct", "medgemma_15_4b_it",
           "medgemma_4b_it", "chexagent_8b"]
SPATIAL = ["blur", "jpeg", "resolution"]
PHOTOMETRIC = ["noise", "contrast", "brightness"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def safe_sp(x, y, method="spearman"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), len(x)
    if method == "kendall":
        return float(kendalltau(x, y).statistic), len(x)
    return float(spearmanr(x, y).statistic), len(x)


def simple_spearman(x, y, method="spearman"):
    """Rank correlation without scipy's tie machinery, for n=5 sub-cases."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 3:
        return float("nan"), n
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = math.sqrt((rx * rx).sum() * (ry * ry).sum())
    if den == 0:
        return float("nan"), n
    return float((rx * ry).sum() / den), n


def status_counts_factory():
    return {k: 0 for k in ALL_STATUS}


def read_reliance_jsonl(model: str, stage: str):
    """Per-condition census from a reliance predictions file.

    Returns a dict cond -> counters (rows / parsed / marker / prob-complete /
    status shares / no_finding-positive). Only READ; nothing written.
    """
    path = MATRIX / model / stage / "image_reliance_predictions.jsonl"
    if not path.exists():
        return None
    per = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            cond = obj.get("condition")
            c = per.setdefault(cond, {
                "n_rows": 0, "n_parsed": 0, "n_marker": 0, "n_prob": 0,
                "n_slots": 0, "nf_pos": 0, "status": status_counts_factory()})
            c["n_rows"] += 1
            parsed = obj.get("parsed") or {}
            if not parsed.get("parse_success"):
                continue
            c["n_parsed"] += 1
            if (obj.get("raw_output") or "").startswith(MARKER):
                c["n_marker"] += 1
            findings = parsed.get("findings") or {}
            for label in PRIMARY_LABELS:
                fp = findings.get(label) or {}
                st = fp.get("status") or "not_mentioned"
                if st not in ALL_STATUS:
                    st = "not_mentioned"
                c["status"][st] += 1
                c["n_slots"] += 1
                if fp.get("probability_positive") is not None:
                    c["n_prob"] += 1
            nf = findings.get("no_finding") or {}
            if nf.get("status") == "positive":
                c["nf_pos"] += 1
    out = {}
    for cond, c in per.items():
        slots = c["n_slots"] or 1
        out[cond] = {
            "condition": cond,
            "n_rows": c["n_rows"],
            "n_parsed": c["n_parsed"],
            "parse_coverage": c["n_parsed"] / c["n_rows"] if c["n_rows"] else np.nan,
            "n_slots": c["n_slots"],
            "n_prob_pairs": c["n_prob"],
            "prob_completeness": c["n_prob"] / slots,
            "marker_rate": c["n_marker"] / c["n_parsed"] if c["n_parsed"] else np.nan,
            "n_marker": c["n_marker"],
            "nofinding_pos_rate": c["nf_pos"] / c["n_parsed"] if c["n_parsed"] else np.nan,
            "n_nofinding_pos": c["nf_pos"],
            **{f"status_share_{k}": c["status"][k] / slots for k in ALL_STATUS},
        }
    return out


def recall_by_label(df: pd.DataFrame):
    """(recall, n_gt_pos) per label over a slot frame."""
    out = {}
    for label, gg in df.groupby("label"):
        y = gg["y"].to_numpy(dtype=float)
        ngt = int(np.nansum(y == 1))
        tp = int((((gg["status"] == "positive").to_numpy()) & (y == 1)).sum())
        out[label] = (tp / ngt if ngt else np.nan, ngt)
    return out


# ---------------------------------------------------------------------------
# A. dependency (blank VRS)  x  robustness (severe recall retention)
# ---------------------------------------------------------------------------

def analysis_A(shift_res, rel_frames, use_bf16_for_4bit=False):
    rows = []
    for model in MODELS5:
        if model in FOUR_BIT_MODELS and use_bf16_for_4bit:
            prec = "bf16"
        else:
            prec = "4bit" if model in FOUR_BIT_MODELS else "native"
        if model not in shift_res or (model, prec) not in rel_frames:
            continue
        rel = rel_frames[(model, prec)]
        s = rel["summary"]
        blank = s["conditions"].get("blank", {})
        h = shift_res[model]["headline"]
        rows.append({
            "model": model, "reliance_precision": prec,
            "vrs_blank": blank.get("mean_abs_prob_diff"),
            "n_valid_pairs_blank": blank.get("n_valid_pairs"),
            "clean_path_recall": h.get("clean_path_recall"),
            "severe_path_recall_pooled": h.get("severe_path_recall_pooled"),
            "recall_retention_severe": h.get("recall_retention_severe"),
            "n_images": h.get("n_images"),
            "severe_parse_coverage_min": h.get("severe_parse_coverage_min"),
            "clean_parse_coverage": h.get("clean_parse_coverage"),
        })
    df = pd.DataFrame(rows)
    sp, n_sp = safe_sp(df["vrs_blank"], df["recall_retention_severe"], "spearman")
    kt, _ = safe_sp(df["vrs_blank"], df["recall_retention_severe"], "kendall")
    # direction stability: sign of the correlation when each model is dropped
    loo = []
    for m in df["model"]:
        sub = df[df["model"] != m]
        s, _ = safe_sp(sub["vrs_blank"], sub["recall_retention_severe"], "spearman")
        loo.append({"dropped_model": m, "spearman_n4": s,
                    "sign": (1 if s > 0 else -1 if s < 0 else 0)})
    loo_df = pd.DataFrame(loo)
    n_same_dir = int((loo_df["sign"] == (1 if sp > 0 else -1)).sum())
    verdict = ("model-invariant" if n_same_dir >= 4 else "model-specific")
    meta = {
        "n_models": len(df), "n_pairs": n_sp,
        "spearman_vrs_blank_vs_retention": sp,
        "kendall_vrs_blank_vs_retention": kt,
        "direction": "higher dependency -> lower retention" if sp < 0
                     else "higher dependency -> higher retention",
        "leave_one_out_sign_consistency": f"{n_same_dir}/{len(loo)}",
        "verdict": verdict,
        "note": "descriptive, not fitted; n=5 rank correlation, no regression.",
    }
    return df, meta, loo_df


# ---------------------------------------------------------------------------
# B / E. failure shape  x  precision (4bit vs bf16)
# ---------------------------------------------------------------------------

def precision_profile(model: str, rel4_cache, rel16_cache):
    """Full dependency + failure-shape profile at both precisions."""
    out = []
    for prec, cache in (("4bit", rel4_cache), ("bf16", rel16_cache)):
        if cache is None:
            continue
        summ = cache["summary"]
        cen = cache["census"]
        for cond in ("original", "blank", "blur", "shuffle"):
            c = summ.get("conditions", {}).get(cond)
            cs = (cen or {}).get(cond, {})
            row = {
                "model": model, "precision": prec, "condition": cond,
                "n_valid_pairs": (c or {}).get("n_valid_pairs"),
                "n_structured": (c or {}).get("n_structured"),
                "vrs_mean_abs_prob_diff": (c or {}).get("mean_abs_prob_diff"),
                "image_sensitivity": (c or {}).get("image_sensitivity"),
                "mean_flip_count": (c or {}).get("mean_flip_count"),
                "mean_text_jaccard": (c or {}).get("mean_text_jaccard"),
                "pct_identical_text": (c or {}).get("pct_identical_text"),
                "parse_coverage": cs.get("parse_coverage"),
                "n_prob_pairs": cs.get("n_prob_pairs"),
                "prob_completeness": cs.get("prob_completeness"),
                "marker_rate": cs.get("marker_rate"),
                "n_marker": cs.get("n_marker"),
                "nofinding_pos_rate": cs.get("nofinding_pos_rate"),
                "n_nofinding_pos": cs.get("n_nofinding_pos"),
                **{f"status_share_{k}": cs.get(f"status_share_{k}") for k in ALL_STATUS},
            }
            out.append(row)
    return pd.DataFrame(out)


def analysis_B_E():
    rows = []
    detail = {}
    for model in ("medgemma_15_4b_it", "medgemma_4b_it"):
        c4 = m75.load_reliance(model, "reliance")
        c16 = m75.load_reliance(model, "reliance_bf16")
        for c in (c4, c16):
            if c is not None:
                c["census"] = read_reliance_jsonl(model, c["stage"])
        prof = precision_profile(model, c4, c16)
        rows.append(prof)
        detail[model] = prof
    prof = pd.concat(rows) if rows else pd.DataFrame()

    # placement shift side-by-side (blank condition is the VRS pivot)
    pair = []
    for model in ("medgemma_15_4b_it", "medgemma_4b_it"):
        a = prof[(prof["model"] == model) & (prof["precision"] == "4bit")
                 & (prof["condition"] == "blank")]
        b = prof[(prof["model"] == model) & (prof["precision"] == "bf16")
                 & (prof["condition"] == "blank")]
        if a.empty or b.empty:
            continue
        a, b = a.iloc[0], b.iloc[0]
        pair.append({
            "model": model, "condition": "blank",
            "vrs_4bit": a["vrs_mean_abs_prob_diff"],
            "vrs_bf16": b["vrs_mean_abs_prob_diff"],
            "delta_vrs_bf16_minus_4bit":
                b["vrs_mean_abs_prob_diff"] - a["vrs_mean_abs_prob_diff"],
            "image_sensitivity_4bit": a["image_sensitivity"],
            "image_sensitivity_bf16": b["image_sensitivity"],
            "mean_flip_count_4bit": a["mean_flip_count"],
            "mean_flip_count_bf16": b["mean_flip_count"],
            "mean_text_jaccard_4bit": a["mean_text_jaccard"],
            "mean_text_jaccard_bf16": b["mean_text_jaccard"],
            "pct_identical_text_4bit": a["pct_identical_text"],
            "pct_identical_text_bf16": b["pct_identical_text"],
            "marker_rate_4bit": a["marker_rate"],
            "marker_rate_bf16": b["marker_rate"],
            "prob_completeness_4bit": a["prob_completeness"],
            "prob_completeness_bf16": b["prob_completeness"],
            "nofinding_pos_rate_4bit": a["nofinding_pos_rate"],
            "nofinding_pos_rate_bf16": b["nofinding_pos_rate"],
            "share_positive_4bit": a["status_share_positive"],
            "share_positive_bf16": b["status_share_positive"],
            "share_negative_4bit": a["status_share_negative"],
            "share_negative_bf16": b["status_share_negative"],
            "share_uncertain_4bit": a["status_share_uncertain"],
            "share_uncertain_bf16": b["status_share_uncertain"],
            "share_not_mentioned_4bit": a["status_share_not_mentioned"],
            "share_not_mentioned_bf16": b["status_share_not_mentioned"],
            "n_valid_pairs_4bit": a["n_valid_pairs"],
            "n_valid_pairs_bf16": b["n_valid_pairs"],
            "n_prob_pairs_4bit": a["n_prob_pairs"],
            "n_prob_pairs_bf16": b["n_prob_pairs"],
            "parse_coverage_4bit": a["parse_coverage"],
            "parse_coverage_bf16": b["parse_coverage"],
        })
    return prof, pd.DataFrame(pair)


# ---------------------------------------------------------------------------
# C. degradation-type clustering in a 4-D delta space
# ---------------------------------------------------------------------------

def analysis_C(shift_res):
    rows = []
    for model, r in shift_res.items():
        cells = r["cells"]
        clean = cells[(cells["condition"] == r["headline"]["clean_condition"])
                      & (cells["severity"] == r["headline"].get("clean_severity",
                                                                "none"))]
        if clean.empty:
            clean = cells[cells["severity"].isin(["clean", "none"])]
        if clean.empty:
            continue
        c0 = clean.iloc[0]
        for dt in DEGRADE_TYPES:
            s = cells[(cells["condition"] == dt) & (cells["severity"] == "severe")]
            if s.empty:
                continue
            s = s.iloc[0]
            rows.append({
                "model": model, "type": dt,
                "n_gt_pos_severe": s.get("n_gt_pos"),
                "n_attempted_severe": s.get("n_attempted"),
                "parse_coverage_severe": s.get("parse_coverage"),
                "d_recall_vs_clean":
                    s["path_recall"] - c0["path_recall"],
                "d_pred_pos_rate_vs_clean":
                    s["pred_pos_rate"] - c0["pred_pos_rate"],
                "d_macro_ece_vs_clean":
                    s["macro_ece"] - c0["macro_ece"],
                "d_nofinding_rate_vs_clean":
                    s["nofinding_pos_rate"] - c0["nofinding_pos_rate"],
            })
    per_model = pd.DataFrame(rows)
    dims = ["d_recall_vs_clean", "d_pred_pos_rate_vs_clean",
            "d_macro_ece_vs_clean", "d_nofinding_rate_vs_clean"]

    # cross-model mean vector per type -> 6 x 4 matrix
    mat = per_model.groupby("type")[dims].mean().reindex(DEGRADE_TYPES)
    n_models_per_type = (per_model.groupby("type")["model"].nunique()
                         .reindex(DEGRADE_TYPES))

    # z-score each dim across the 6 types (descriptive scale fix), then cluster
    z = mat.copy()
    for d in dims:
        col = z[d].to_numpy(dtype=float)
        sd = col.std(ddof=0)
        z[d] = (col - col.mean()) / sd if sd > 0 else 0.0
    D = pdist(z.to_numpy(dtype=float), metric="euclidean")
    link = linkage(D, method="average")
    labels_average = fcluster(link, t=2, criterion="maxclust")
    link_single = linkage(D, method="single")
    labels_single = fcluster(link_single, t=2, criterion="maxclust")
    cluster_df = pd.DataFrame({
        "type": z.index,
        "cluster_average_link": labels_average,
        "cluster_single_link": labels_single,
        "inferred_family_spatial": [1 if t in SPATIAL else 0 for t in z.index],
    }).merge(n_models_per_type.rename("n_models").reset_index()
             .rename(columns={"index": "type"}), on="type", how="left")
    for d in dims:
        cluster_df[d] = mat[d].to_numpy()
        cluster_df["z_" + d] = z[d].to_numpy()

    # pairwise correlation matrix between degradation types (4-dim features)
    types = list(z.index)
    corr = np.full((len(types), len(types)), np.nan)
    for i, ti in enumerate(types):
        for j, tj in enumerate(types):
            corr[i, j] = np.corrcoef(z.loc[ti].to_numpy(dtype=float),
                                     z.loc[tj].to_numpy(dtype=float))[0, 1]
    corr_df = pd.DataFrame(corr, index=types, columns=types).reset_index() \
        .rename(columns={"index": "type"})

    # cross-validation with medgemma_15's reasoning-channel marker rate
    marker = {}
    for model, r in shift_res.items():
        tc = r["type_curve"]
        sev = tc[(tc["severity"] == "severe") & (tc["type"].isin(DEGRADE_TYPES))]
        marker[model] = sev.groupby("type")["marker_rate"].mean()
    marker_df = pd.DataFrame(marker)  # type x model
    md = marker_df.mean(axis=1)
    marker_by_type = pd.DataFrame({
        "type": md.index, "mean_marker_rate_severe": md.to_numpy(),
        "family": ["spatial" if t in SPATIAL else "photometric" for t in md.index],
    })
    # marker for the medgemma_15 model specifically (only it opens the channel)
    if "medgemma_15_4b_it" in marker_df.columns:
        marker_by_type["marker_rate_medgemma_15_severe"] = (
            marker_df["medgemma_15_4b_it"].to_numpy())
    fam_marker = marker_by_type.groupby("family")["marker_rate_medgemma_15_severe"] \
        .mean().to_dict() if "marker_rate_medgemma_15_severe" in marker_by_type \
        else {}

    # does the (unsupervised) average-link split actually reproduce the
    # spatial/photometric grouping handed in by the hypothesis?
    avg_cluster = cluster_df.set_index("type")["cluster_average_link"].to_dict()
    fam_pred = {t: ("spatial" if t in SPATIAL else "photometric") for t in types}
    # map cluster id -> majority family
    from collections import Counter
    c_fam = {}
    for cid in set(avg_cluster.values()):
        fams = [fam_pred[t] for t in types if avg_cluster[t] == cid]
        c_fam[cid] = Counter(fams).most_common(1)[0][0]
    hit = sum(1 for t in types if c_fam[avg_cluster[t]] == fam_pred[t])
    meta = {
        "n_models": int(per_model["model"].nunique()),
        "types": types, "dims": dims,
        "cluster_average_link": avg_cluster,
        "cluster_single_link": cluster_df.set_index("type")["cluster_single_link"].to_dict(),
        "cluster_vs_hypothesis_agreement": f"{hit}/{len(types)}",
        "hypothesis": "spatial {blur,jpeg,resolution} vs photometric "
                      "{noise,contrast,brightness}",
        "marker_rate_by_family_medgemma_15": fam_marker,
        "verdict": ("model-invariant (gap reproduced across "
                    f"{int(per_model['model'].nunique())} models)"
                    if bool(fam_marker)
                    and fam_marker.get("spatial", 0) > fam_marker.get("photometric", 0)
                    else "not confirmed"),
        "note": "clustering on z-scored cross-model-mean 4-D delta vectors; "
                "2-cluster cut is descriptive, not fitted.",
    }
    return per_model, cluster_df, corr_df, marker_by_type, meta


# ---------------------------------------------------------------------------
# D. label-level robustness hierarchy
# ---------------------------------------------------------------------------

def analysis_D(shift_res):
    ret_rows = []
    vec = {}   # model -> retention vector over labels
    for model, r in shift_res.items():
        rows = r["rows_by_image"]
        clean = rows[(rows["condition"] == r["headline"]["clean_condition"])
                     & (rows["severity"] == r["headline"].get("clean_severity", "none"))]
        if clean.empty:
            clean = rows[rows["severity"].isin(["clean", "none"])]
        sev = rows[(rows["severity"] == "severe")]
        rc = recall_by_label(clean)
        rs = recall_by_label(sev)
        v = {}
        for label in PRIMARY_LABELS:
            rec_c, ngt_c = rc.get(label, (np.nan, 0))
            rec_s, ngt_s = rs.get(label, (np.nan, 0))
            ret = rec_s / rec_c if rec_c and not math.isnan(rec_c) and rec_c > 0 \
                else np.nan
            v[label] = ret
            ret_rows.append({
                "model": model, "label": label,
                "recall_clean": rec_c, "n_gt_pos_clean": ngt_c,
                "recall_severe_pooled": rec_s, "n_gt_pos_severe": ngt_s,
                "retention_severe": ret,
                "n_images_clean": clean["image_id"].nunique(),
                "n_images_severe": sev["image_id"].nunique(),
            })
        vec[model] = v
    df = pd.DataFrame(ret_rows)

    # cross-model rank correlation of the 8-label retention vector
    models = list(vec.keys())
    pw = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = [vec[models[i]][l] for l in PRIMARY_LABELS]
            b = [vec[models[j]][l] for l in PRIMARY_LABELS]
            s, n = simple_spearman(a, b)
            k, _ = safe_sp(a, b, "kendall")
            pw.append({"model_a": models[i], "model_b": models[j],
                       "spearman": s, "kendall": k, "n_labels": n})
    pw_df = pd.DataFrame(pw)

    # mean retention ranking across models
    mean_ret = df.groupby("label")["retention_severe"].mean().to_dict()
    rank_series = pd.Series(mean_ret).rank(ascending=False)
    mean_df = pd.DataFrame({
        "label": list(mean_ret.keys()),
        "mean_retention_severe": [mean_ret[l] for l in mean_ret],
        "rank_by_mean_retention": [rank_series[l] for l in mean_ret],
        "n_models": [int(df[df["label"] == l]["model"].nunique()) for l in mean_ret],
    })
    # per-label direction consistency: sign of each model's retention vs 1
    cons = []
    for label in PRIMARY_LABELS:
        sub = df[df["label"] == label]["retention_severe"].dropna()
        n_below = int((sub < 1.0).sum())
        cons.append({"label": label, "n_models": len(sub),
                     "n_models_retention_below_1": n_below,
                     "consistency": f"{n_below}/{len(sub)}"})
    cons_df = pd.DataFrame(cons)
    # is the label ordering stable? require pairwise Spearman >= 0.5 for >=7/10 pairs
    n_pos = int((pw_df["spearman"] > 0.5).sum())
    meta = {
        "n_models": len(models), "n_labels": len(PRIMARY_LABELS),
        "pairwise_spearman_min": float(pw_df["spearman"].min()),
        "pairwise_spearman_median": float(pw_df["spearman"].median()),
        "n_pairs_spearman_gt_0.5": f"{n_pos}/{len(pw_df)}",
        "verdict": ("model-invariant: label ordering stable across models"
                    if n_pos >= 7 else
                    "model-specific: label ordering varies across models"),
        "top_labels_by_mean_retention": mean_df.sort_values(
            "rank_by_mean_retention")["label"].tolist(),
        "note": "retention = recall(severe, all 6 types pooled) / recall(clean); "
                "rank correlations on the 8-label vector, descriptive, not fitted.",
    }
    return df, pw_df, mean_df, cons_df, meta


# ---------------------------------------------------------------------------
# F. schema x condition interaction (medgemma_4b_it ablation, 10 cells)
# ---------------------------------------------------------------------------

def analysis_F(done):
    abl = m75.mine_ablation(done)
    if abl is None:
        return None, None
    abl = abl.copy()
    abl["contract_compliance_rate"] = 1.0 - abl["contract_violation_rate"]
    abl["parse_coverage"] = 1.0 - abl["parse_failure_rate"]
    keep = ["variant", "condition", "n_rows", "n_rows_with_gt",
            "parse_failure_rate", "parse_coverage", "contract_violation_rate",
            "contract_compliance_rate", "nofinding_pos_rate", "path_recall",
            "path_precision", "all_uncertain_rate"] + \
        [f"status_share_{k}" for k in ALL_STATUS]
    keep = [k for k in keep if k in abl.columns]
    tab = abl[keep].copy()

    # interaction: is the best schema on `original` also best on `blank`?
    o = tab[tab["condition"] == "original"].set_index("variant")
    b = tab[tab["condition"] == "blank"].set_index("variant")
    best_o = o["path_recall"].idxmax() if o["path_recall"].notna().any() else None
    best_b = b["path_recall"].idxmax() if b["path_recall"].notna().any() else None
    comp_o = o["contract_compliance_rate"].idxmax() \
        if o["contract_compliance_rate"].notna().any() else None
    comp_b = b["contract_compliance_rate"].idxmax() \
        if b["contract_compliance_rate"].notna().any() else None
    # blank separation: v1 vs the other 4
    v1 = b.loc["v1_per_label_binary"] if "v1_per_label_binary" in b.index else None
    others = b.drop(index="v1_per_label_binary") if (
        "v1_per_label_binary" in b.index) else b
    meta = {
        "n_cells": len(tab),
        "best_recall_schema_original": best_o,
        "best_recall_schema_blank": best_b,
        "recall_best_matches": bool(best_o == best_b),
        "most_compliant_schema_original": comp_o,
        "most_compliant_schema_blank": comp_b,
        "compliance_best_matches": bool(comp_o == comp_b),
        "blank_v1_contract_compliance": (None if v1 is None
                                         else float(v1["contract_compliance_rate"])),
        "blank_other4_mean_contract_compliance":
            (None if others.empty
             else float(others["contract_compliance_rate"].mean())),
        "blank_v1_status_share_uncertain":
            (None if v1 is None else float(v1["status_share_uncertain"])),
        "blank_other4_mean_status_share_uncertain":
            (None if others.empty
             else float(others["status_share_uncertain"].mean())),
        "blank_v1_all_uncertain_rate":
            (None if v1 is None else float(v1.get("all_uncertain_rate", np.nan))),
        "interaction_verdict": ("schema effect interacts with condition: best "
                                "schema differs between original and blank"
                                if best_o != best_b or comp_o != comp_b
                                else "no interaction detected"),
        "note": "compliance = 1 - contract_violation_rate (contract block); "
                "denominator n_rows printed per cell. descriptive, not fitted.",
    }
    return tab, meta


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    done = m75.completed_stages()
    ledger = {
        "generated_by": "scripts/77_cross_parameter_mine.py",
        "completed_stages_seen": sorted(done),
        "models5": MODELS5,
        "data_boundary": ("only stages in state.completed are read; qwen25 NIH "
                          "stages removed from state -> excluded; llava_med never "
                          "entered; outputs/matrix and outputs/state are read-only."),
    }

    # ---- run shift mining for the 5 healthy models ----------------------
    shift_res = {}
    for m in MODELS5:
        r = m75.mine_shift(m, done)
        if r is None:
            print(f"[skip] {m}: shift not complete")
            continue
        shift_res[m] = r
    print(f"[shift] mined {len(shift_res)} models: {list(shift_res)}")

    # ---- reliance frames -------------------------------------------------
    rel_df, rel_frames = m75.reliance_table(done)
    # restrict to the 5 models (drops nothing extra once state excludes qwen25)
    rel_df = rel_df[rel_df["model"].isin(MODELS5)]

    # ---- A ---------------------------------------------------------------
    a_df, a_meta, a_loo = analysis_A(shift_res, rel_frames)
    a_df.to_csv(OUT / "joint_A_dependency_robustness.csv", index=False)
    a_loo.to_csv(OUT / "joint_A_leave_one_out.csv", index=False)
    # sensitivity: use the bf16 reliance frame for the 4-bit models (alternate
    # dependency regime) — shows how the n=5 trend depends on the precision used
    a_alt, a_alt_meta, _ = analysis_A(shift_res, rel_frames,
                                      use_bf16_for_4bit=True)
    a_alt.to_csv(OUT / "joint_A_dependency_robustness_bf16alt.csv", index=False)
    ledger["A_dependency_robustness"] = {"table": a_df.to_dict(orient="records"),
                                         "bf16_sensitivity": a_alt_meta, **a_meta}
    print(f"[A] spearman={a_meta['spearman_vrs_blank_vs_retention']:.3f} "
          f"n={a_meta['n_models']} -> {a_meta['verdict']}")

    # per-image evidence: reliance-blank flip vs shift-severe flip (reuse 75)
    cross = m75.per_image_cross(shift_res, rel_frames)
    cross = cross[cross["model"].isin(MODELS5)] if not cross.empty else cross
    cross.to_csv(OUT / "joint_A_per_image_evidence.csv", index=False)
    a_meta["per_image_evidence"] = cross.to_dict(orient="records")

    # ---- B / E -----------------------------------------------------------
    bprof, bpair = analysis_B_E()
    bprof.to_csv(OUT / "joint_B_EF_failure_shape_profiles.csv", index=False)
    bpair.to_csv(OUT / "joint_B_precision_pairwise.csv", index=False)
    # short B table (blank condition side-by-side, all precision deltas)
    b_meta = {"n_models_compared": len(bpair),
              "kinds": ["medgemma_15_4b_it", "medgemma_4b_it"],
              "pairs": bpair.to_dict(orient="records"),
              "note": "VRS/image_sensitivity/flip_count/jaccard from the "
                      "reliance summary; status shares and marker/prob "
                      "completeness from the predictions jsonl census "
                      "(denominators n_valid_pairs / n_prob_pairs printed)."}
    ledger["B_failure_shape_precision"] = b_meta
    ledger["E_precision_reliance_failure_3way"] = b_meta
    print(f"[B/E] {len(bpair)} model pairs profiled")

    # ---- C ---------------------------------------------------------------
    c_per_model, c_clusters, c_corr, c_marker, c_meta = analysis_C(shift_res)
    c_per_model.to_csv(OUT / "joint_C_type_deltas_per_model.csv", index=False)
    c_clusters.to_csv(OUT / "joint_C_type_clusters.csv", index=False)
    c_corr.to_csv(OUT / "joint_C_type_correlation.csv", index=False)
    c_marker.to_csv(OUT / "joint_C_marker_by_type.csv", index=False)
    ledger["C_type_clusters"] = c_meta
    print(f"[C] average-link clusters: {c_meta['cluster_average_link']}")

    # ---- D ---------------------------------------------------------------
    d_df, d_pw, d_mean, d_cons, d_meta = analysis_D(shift_res)
    d_df.to_csv(OUT / "joint_D_label_retention.csv", index=False)
    d_pw.to_csv(OUT / "joint_D_pairwise_spearman.csv", index=False)
    d_mean.to_csv(OUT / "joint_D_label_ranking.csv", index=False)
    d_cons.to_csv(OUT / "joint_D_label_direction.csv", index=False)
    ledger["D_label_hierarchy"] = d_meta
    print(f"[D] {d_meta['verdict']} (pairs>0.5: "
          f"{d_meta['n_pairs_spearman_gt_0.5']})")

    # ---- F ---------------------------------------------------------------
    f_tab, f_meta = analysis_F(done)
    if f_tab is not None:
        f_tab.to_csv(OUT / "joint_F_schema_condition.csv", index=False)
        ledger["F_schema_condition"] = f_meta
        print(f"[F] {f_meta['interaction_verdict']}")

    (OUT / "joint_parameters.json").write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False, default=float),
        encoding="utf-8")
    print(f"\nwrote {OUT / 'joint_parameters.json'}")
    print("csv:", [p.name for p in sorted(OUT.glob('joint_*.csv'))])


if __name__ == "__main__":
    main()
