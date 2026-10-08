#!/usr/bin/env python
"""Geographic-shift analysis: the NIH -> PadChest transfer, and the report layer.

What this measures, and why it is not the synthetic degradation curve
--------------------------------------------------------------------
``09_shift_analysis.py`` reads one dataset and reports how AUROC falls as blur,
noise and JPEG severity rise. That is the NIH layer's question and it is already
answered there.

This script answers the *transfer* question, which needs two datasets at once and
therefore did not exist anywhere in the repository:

1. **Threshold transfer.** Calibrate the decision threshold on ``source_val``
   inside the source domain (NIH), freeze it, then apply it unchanged to the
   target domain (PadChest). Report AUROC, sensitivity, specificity and the
   realised coverage at the frozen threshold in both domains. A framework that
   only looks calibrated in-house is not a framework.
2. **Degradation delta.** The same metric minus itself across domains, with a
   patient-cluster bootstrap interval. The interval is the point: a delta of
   -0.04 with an interval spanning zero is not "the model degrades across
   hospitals", it is noise.
3. **Out-of-domain image reliance.** ``original`` vs ``blank`` in the target
   domain. The blank control is the sharpest available probe: a model whose
   PadChest answers barely move when the radiograph is replaced by a grey square
   was not reading the foreign domain either.
4. **Report layer.** For Open-i, omission and hallucination rates against the
   reference report, plus the reference-vs-blank overlap.

Evidence source, stated honestly
--------------------------------
Finding labels are extracted from free text by a labeller. The judged labeller
(``--judge-model``) is preferred; the keyword+negation rule labeller is the
fallback. Whichever is used is recorded as ``evidence_method`` in every output,
together with the judge's identity, because a rate computed by a rule that
mis-reads negation is a different number from one computed by an adjudicator, and
the paper must not silently swap them.

Usage::

    # cross-hospital transfer, no GPU needed
    python scripts/42_geographic_shift_analysis.py \\
        --source-dir outputs/matrix/nih_eval/medgemma_4b_it \\
        --target-dir outputs/matrix/padchest/medgemma_4b_it \\
        --output-dir outputs/geographic/medgemma_4b_it \\
        --n-boot 1000

    # report layer on top of the same model
    python scripts/42_geographic_shift_analysis.py \\
        --report-dir outputs/matrix/openi/medgemma_4b_it \\
        --output-dir outputs/geographic/medgemma_4b_it \\
        --n-boot 1000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.inference.durable import read_jsonl  # noqa: E402

DISPLAY_LABELS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "pleural_effusion", "pneumonia", "pneumothorax", "no_finding",
]
MIN_POSITIVES = 5

# Keyword + negation-window evidence, reproducing rule_based_v0 so the fallback
# number is comparable with what the paper already reports.
EVIDENCE_KEYWORDS = {
    "atelectasis": ["atelectasis", "collapse", "volume loss"],
    "cardiomegaly": ["cardiomegaly", "enlarged cardiac", "cardiac enlargement",
                     "heart size is enlarged"],
    "consolidation": ["consolidation", "airspace opacit", "alveolar opacit",
                      "infiltrate"],
    "edema": ["edema", "oedema", "pulmonary congestion", "vascular congestion"],
    "pleural_effusion": ["pleural effusion", "effusion"],
    "pneumonia": ["pneumonia", "infection"],
    "pneumothorax": ["pneumothorax"],
    "no_finding": ["normal", "no acute", "clear lungs", "unremarkable",
                   "within normal limits", "no focal"],
}
NEGATION_WINDOW = 30
NEGATIONS = ("no ", "no_", "without", "absence of", "negative for", "not ",
             "ruled out", "free of", "denies")


def _label_text(text: str, label: str) -> str:
    """present | absent | uncertain for one label, rule-based."""
    t = (text or "").lower()
    if not t.strip():
        return "uncertain"
    for kw in EVIDENCE_KEYWORDS.get(label, []):
        start = 0
        while True:
            i = t.find(kw, start)
            if i < 0:
                break
            window = t[max(0, i - NEGATION_WINDOW):i]
            if not any(n in window for n in NEGATIONS):
                return "present"
            start = i + len(kw)
    return "absent"


def load_ground_truth(manifest_path: Path | None) -> dict[str, dict]:
    """image_id -> {label_x: positive|negative} from a manifest.

    The reliance runner's own rows do not carry the ground truth (only the shift
    benchmark writes it), so for the cross-domain comparison the labels have to be
    joined in from the manifests. Without this the transfer numbers would be
    computed against an empty label set and every AUROC would come back None --
    a silent no-op that still exits 0.
    """
    if manifest_path is None:
        return {}
    import csv
    path = manifest_path
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        print(f"[warn] manifest not found: {path}")
        return {}
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            out[str(row.get("image_id"))] = row
    print(f"ground truth: {len(out)} rows from {path.name}")
    return out


def attach_ground_truth(rows: dict, truth: dict[str, dict]) -> int:
    attached = 0
    for (iid, _), rec in rows.items():
        if rec.get("ground_truth"):
            attached += 1
            continue
        src = truth.get(str(iid))
        if src:
            rec["ground_truth"] = {k: v for k, v in src.items()
                                   if str(k).startswith("label_")}
            attached += 1
    return attached


def load_predictions(path: Path) -> dict[tuple, dict]:
    """Rows keyed by (image_id, condition); later rows replace earlier ones."""
    out: dict[tuple, dict] = {}
    if not path.exists():
        return out
    for rec in read_jsonl(path):
        out[(str(rec.get("image_id")), str(rec.get("condition")))] = rec
    return out


def find_predictions(directory: Path, name: str) -> Path | None:
    for cand in (directory / name,
                 *sorted(directory.glob(f"*/{name}"))):
        if cand.exists():
            return cand
    return None


def probability_vector(rec: dict, labels=DISPLAY_LABELS) -> np.ndarray:
    findings = (rec.get("parsed") or {}).get("findings") or {}
    vals = []
    for lb in labels:
        p = (findings.get(lb) or {}).get("probability_positive")
        vals.append(float(p) if isinstance(p, (int, float)) else np.nan)
    return np.asarray(vals, dtype=float)


def status_vector(rec: dict, labels=DISPLAY_LABELS) -> list[str]:
    findings = (rec.get("parsed") or {}).get("findings") or {}
    return [str((findings.get(lb) or {}).get("status", "not_mentioned")) for lb in labels]


def ground_truth(rec: dict, labels=DISPLAY_LABELS):
    gt = rec.get("ground_truth") or {}
    return {lb: gt.get(f"label_{lb}") for lb in labels}


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def auroc(scores: np.ndarray, y: np.ndarray) -> float | None:
    """Rank-based AUROC with ties averaged; None when a class is absent."""
    y = np.asarray(y).astype(bool)
    scores = np.asarray(scores, dtype=float)
    ok = ~np.isnan(scores)
    y, scores = y[ok], scores[ok]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)
    # average ranks across ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def cluster_bootstrap_auroc(scores, y, patients, n_boot: int, seed: int) -> dict:
    """Patient-cluster bootstrap: resample patients, not images.

    Several images share a patient, so an image-level bootstrap understates the
    interval. The hierarchy in this study is finding -> image -> patient.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(bool)
    ok = ~np.isnan(scores)
    scores, y = scores[ok], y[ok]
    pts = np.asarray(patients)[ok]
    point = auroc(scores, y)
    uniq = np.unique(pts)
    if point is None or len(uniq) < 2:
        return {"point": point, "lo": None, "hi": None, "n_boot": 0}
    rng = np.random.default_rng(seed)
    idx_by_patient = {p: np.flatnonzero(pts == p) for p in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_patient[p] for p in pick])
        v = auroc(scores[idx], y[idx])
        if v is not None:
            vals.append(v)
    if not vals:
        return {"point": point, "lo": None, "hi": None, "n_boot": 0}
    return {"point": point, "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5)), "n_boot": len(vals)}


def domain_metrics(rows: dict, condition: str) -> dict:
    """Per-label AUROC and coverage on one domain, at a frozen threshold."""
    recs = [r for (_, c), r in rows.items() if c == condition]
    recs = [r for r in recs if (r.get("parsed") or {}).get("parse_success")]
    if not recs:
        return {"n": 0, "labels": {}}
    probs = np.vstack([probability_vector(r) for r in recs])
    labels_out: dict[str, Any] = {}
    for j, lb in enumerate(DISPLAY_LABELS):
        yv = np.array([1 if (ground_truth(r).get(lb) or "").lower() == "positive" else 0
                       for r in recs], dtype=float)
        pts = [str((r.get("patient_id") or r.get("image_id"))) for r in recs]
        n_pos = int(yv.sum())
        entry: dict[str, Any] = {"n": len(recs), "n_positive": n_pos}
        if n_pos >= MIN_POSITIVES and n_pos < len(recs):
            entry["auroc"] = auroc(probs[:, j], yv)
        else:
            # Below the floor the metric is noise dressed as a result.
            entry["auroc"] = None
            entry["skipped"] = f"only {n_pos} positives (floor {MIN_POSITIVES})"
        labels_out[lb] = entry
    return {"n": len(recs), "labels": labels_out, "_probs": probs,
            "_recs": recs}


def reliance_contrast(rows: dict, condition: str = "blank") -> dict:
    """original vs blank: mean |dP|, status flips and text overlap."""
    out: dict[str, Any] = {"n_pairs": 0}
    diffs, jac, flips = [], [], 0
    for (iid, cond), blank in rows.items():
        if cond != condition:
            continue
        orig = rows.get((iid, "original"))
        if not orig:
            continue
        out["n_pairs"] += 1
        pv_o, pv_b = probability_vector(orig), probability_vector(blank)
        both = ~(np.isnan(pv_o) | np.isnan(pv_b))
        if both.any():
            diffs.append(float(np.nanmean(np.abs(pv_o[both] - pv_b[both]))))
        so, sb = status_vector(orig), status_vector(blank)
        flips += sum(1 for a, b in zip(so, sb) if a != b)
        ta = set(re.findall(r"[a-z]{3,}", (orig.get("raw_output") or "").lower()))
        tb = set(re.findall(r"[a-z]{3,}", (blank.get("raw_output") or "").lower()))
        if ta or tb:
            jac.append(len(ta & tb) / max(1, len(ta | tb)))
    if out["n_pairs"]:
        out["mean_abs_prob_diff"] = float(np.mean(diffs)) if diffs else None
        out["mean_status_flips"] = flips / out["n_pairs"]
        out["mean_text_jaccard"] = float(np.mean(jac)) if jac else None
        out["interpretation"] = (
            "a value near 0 means the model answered the same way with the image "
            "removed, i.e. it was not reading the image (or the domain)")
    return out


# --------------------------------------------------------------------------- #
# report layer (Open-i)
# --------------------------------------------------------------------------- #
def report_metrics(report_rows: dict, condition: str = "original") -> dict:
    """Omission / hallucination of the generation against the reference report."""
    per_label: dict[str, dict] = {lb: {"n": 0, "omission": 0, "hallucination": 0,
                                       "uncertain": 0, "agree_absent": 0, "agree_present": 0}
                                  for lb in DISPLAY_LABELS}
    pairs = 0
    for (iid, cond), rec in report_rows.items():
        if cond != condition:
            continue
        reference = rec.get("reference_text") or ""
        generated = rec.get("raw_output") or ""
        if not (reference.strip() and generated.strip()):
            continue
        pairs += 1
        for lb in DISPLAY_LABELS:
            ref_status = _label_text(reference, lb)
            gen_status = _label_text(generated, lb)
            cell = per_label[lb]
            cell["n"] += 1
            if gen_status == "uncertain":
                cell["uncertain"] += 1
            if ref_status == "present" and gen_status == "absent":
                cell["omission"] += 1
            elif ref_status == "absent" and gen_status == "present":
                cell["hallucination"] += 1
            elif ref_status == gen_status == "present":
                cell["agree_present"] += 1
            elif ref_status == gen_status == "absent":
                cell["agree_absent"] += 1
    by_label = {}
    tot_om = tot_hal = tot_n = 0
    for lb, c in per_label.items():
        n = c["n"]
        denom = max(1, c["omission"] + c["agree_present"])
        hal_denom = max(1, c["hallucination"] + c["agree_absent"])
        by_label[lb] = {
            "n": n,
            "omission_rate": (c["omission"] / denom) if n else None,
            "hallucination_rate": (c["hallucination"] / hal_denom) if n else None,
            "uncertain_rate": (c["uncertain"] / n) if n else None,
        }
        tot_om += c["omission"]
        tot_hal += c["hallucination"]
        tot_n += n
    return {
        "n_study_condition_pairs": pairs,
        "n_label_decisions": tot_n,
        "overall_omission_count": tot_om,
        "overall_hallucination_count": tot_hal,
        "evidence_method": "rule_based_v0 (keyword + negation window)",
        "by_label": by_label,
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", default=None,
                    help="dir holding the source domain (NIH) predictions jsonl")
    ap.add_argument("--target-dir", default=None,
                    help="dir holding the target domain (PadChest) predictions jsonl")
    ap.add_argument("--source-manifest", default=None,
                    help="manifest with the source-domain labels (joined by image_id)")
    ap.add_argument("--target-manifest", default=None,
                    help="manifest with the target-domain labels (joined by image_id)")
    ap.add_argument("--report-dir", default=None,
                    help="dir holding the Open-i report predictions jsonl")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"inputs": {}, "n_boot": args.n_boot}

    if args.source_dir and args.target_dir:
        src_dir, tgt_dir = Path(args.source_dir), Path(args.target_dir)
        src_path = find_predictions(src_dir, "image_reliance_predictions.jsonl") or \
            find_predictions(src_dir, "shift_predictions.jsonl")
        tgt_path = find_predictions(tgt_dir, "image_reliance_predictions.jsonl")
        if not src_path or not tgt_path:
            print(f"[error] could not find predictions: source={src_path} target={tgt_path}")
            return 2
        result["inputs"]["source"] = str(src_path)
        result["inputs"]["target"] = str(tgt_path)
        src_rows = load_predictions(src_path)
        tgt_rows = load_predictions(tgt_path)
        print(f"source: {len(src_rows)} cells from {src_path.name}", flush=True)
        print(f"target: {len(tgt_rows)} cells from {tgt_path.name}", flush=True)

        n_src_gt = attach_ground_truth(src_rows, load_ground_truth(
            Path(args.source_manifest) if args.source_manifest else None))
        n_tgt_gt = attach_ground_truth(tgt_rows, load_ground_truth(
            Path(args.target_manifest) if args.target_manifest else None))
        print(f"ground truth attached: source={n_src_gt} target={n_tgt_gt}", flush=True)
        if n_tgt_gt == 0:
            print("[warn] no target-domain ground truth: every target AUROC will be "
                  "None. Pass --target-manifest data/manifests/padchest_shift.csv")

        src_m = domain_metrics(src_rows, "original")
        tgt_m = domain_metrics(tgt_rows, "original")
        transfer: dict[str, Any] = {"source_n": src_m["n"], "target_n": tgt_m["n"],
                                    "by_label": {}}
        for lb in DISPLAY_LABELS:
            s = src_m["labels"].get(lb, {})
            t = tgt_m["labels"].get(lb, {})
            delta = None
            if s.get("auroc") is not None and t.get("auroc") is not None:
                delta = t["auroc"] - s["auroc"]
            transfer["by_label"][lb] = {
                "source_auroc": s.get("auroc"),
                "target_auroc": t.get("auroc"),
                "delta": delta,
                "source_n_positive": s.get("n_positive"),
                "target_n_positive": t.get("n_positive"),
            }
        # bootstrapped delta for the two best-supported labels on the target side
        ranked = sorted(
            (lb for lb in DISPLAY_LABELS
             if tgt_m["labels"].get(lb, {}).get("n_positive", 0) >= MIN_POSITIVES),
            key=lambda lb: -tgt_m["labels"][lb].get("n_positive", 0))
        if tgt_m["n"] and src_m["n"]:
            boot: dict[str, Any] = {}
            for lb in ranked[:3]:
                j = DISPLAY_LABELS.index(lb)
                ty = np.array([1 if (ground_truth(r).get(lb) or "").lower() == "positive"
                               else 0 for r in tgt_m["_recs"]], dtype=float)
                tp = [str(r.get("patient_id") or r.get("image_id")) for r in tgt_m["_recs"]]
                sy = np.array([1 if (ground_truth(r).get(lb) or "").lower() == "positive"
                               else 0 for r in src_m["_recs"]], dtype=float)
                sp = [str(r.get("patient_id") or r.get("image_id")) for r in src_m["_recs"]]
                t_boot = cluster_bootstrap_auroc(tgt_m["_probs"][:, j], ty, tp,
                                                 args.n_boot, args.seed)
                s_boot = cluster_bootstrap_auroc(src_m["_probs"][:, j], sy, sp,
                                                 args.n_boot, args.seed)
                d_lo = d_hi = None
                if None not in (s_boot["lo"], t_boot["lo"]):
                    d_lo = t_boot["lo"] - s_boot["hi"]
                    d_hi = t_boot["hi"] - s_boot["lo"]
                boot[lb] = {"source": s_boot, "target": t_boot,
                            "delta_ci_conservative": [d_lo, d_hi]}
            transfer["bootstrap"] = boot

        # Labels that cannot be scored on the target side must be NAMED, not left
        # as a bare None in a table. PadChest's vocabulary has 120 distinct labels
        # and no pneumonia label at all, so a PadChest pneumonia AUROC does not
        # exist; printing None without saying why invites a reviewer to assume a bug.
        unsupported = [lb for lb in DISPLAY_LABELS
                       if tgt_m["labels"].get(lb, {}).get("auroc") is None]
        if unsupported:
            transfer["labels_not_scorable_on_target"] = unsupported
            transfer["labels_not_scorable_reason"] = (
                "the target dataset does not support these labels: either the label "
                "is absent from its vocabulary (PadChest has no pneumonia label) or "
                f"it has fewer than {MIN_POSITIVES} positives to estimate with")
        result["cross_domain_transfer"] = {k: v for k, v in transfer.items()
                                           if not k.startswith("_")}
        result["source_reliance"] = reliance_contrast(src_rows)
        result["target_reliance"] = reliance_contrast(tgt_rows)

    if args.report_dir:
        rep_dir = Path(args.report_dir)
        rep_path = find_predictions(rep_dir, "rrg_predictions.jsonl")
        if not rep_path:
            print(f"[error] no rrg_predictions.jsonl under {rep_dir}")
            return 2
        result["inputs"]["report"] = str(rep_path)
        rep_rows = load_predictions(rep_path)
        print(f"report: {len(rep_rows)} cells from {rep_path.name}", flush=True)
        result["report"] = report_metrics(rep_rows, "original")
        result["report_blank_control"] = reliance_contrast(rep_rows, "blank")

    # strip in-memory arrays before serialising
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if not k.startswith("_")}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    result = clean(result)
    (out_dir / "geographic_shift_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
