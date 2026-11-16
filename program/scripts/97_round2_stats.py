#!/usr/bin/env python
"""97 - Round-2 statistics: patient-cluster bootstrap, BH-FDR, contract coverage.

CPU-only, read-only.  ``96_analyze_round2.py`` reports point estimates; this
script adds the inference a reviewer asks for:

* per-label recall change **original -> blank** (schema ablation, paraphrase)
  and **clean -> severe** (second reasoning model), with a patient-cluster
  bootstrap interval, a two-sided bootstrap p-value, and Benjamini-Hochberg
  q-values across the seven pathology labels;
* contract coverage per cell (parse rate, contract-violation rate, mean labels
  defaulted, ``no_finding`` positive rate) -- the ablation can destroy
  information without losing accuracy, and only these columns show it;
* reasoning-channel statistics for the second model (share of replies that
  close a ``<think>`` block, mean/median trace characters) per condition.

Inputs (whichever exist):
  outputs/ablation_mg15/medgemma_15_4b_it/<variant>/<condition>/image_reliance_predictions.jsonl
  outputs/paraphrase_mg4/medgemma_4b_it/<variant>/<condition>/image_reliance_predictions.jsonl
  outputs/matrix/nih_eval/medgemma_4b_it/ablation/<variant>/<condition>/image_reliance_predictions.jsonl
  outputs/matrix/nih_eval/qwen3_vl_8b_thinking/shift/shift_predictions.jsonl
  data/manifests/nih_eval_1732.csv

Outputs:
  outputs/analysis_round2/round2_label_ci.csv
  outputs/analysis_round2/round2_contract.csv
  outputs/analysis_round2/round2_marker.csv

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\97_round2_stats.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis_round2"
# Split-pure evaluation manifest: 1,506 rows (1,054 internal_test + 452 source_val).
# The older 1,732-row file also carries 226 internal_train rows, which would leak
# training images into an evaluation table.
MAN = ROOT / "data" / "manifests" / "nih_eval_1506.csv"

PATHOLOGY = ["atelectasis", "cardiomegaly", "consolidation", "edema",
             "pleural_effusion", "pneumonia", "pneumothorax"]
N_BOOT = 1000
SEED = 0

ABLATION_ROOTS = [
    ("schema_ablation", ROOT / "outputs" / "ablation_mg15" / "medgemma_15_4b_it"),
    ("paraphrase", ROOT / "outputs" / "paraphrase_mg4" / "medgemma_4b_it"),
    ("paraphrase_baseline",
     ROOT / "outputs" / "matrix" / "nih_eval" / "medgemma_4b_it" / "ablation"),
]
THINK = (ROOT / "outputs" / "matrix" / "nih_eval" / "qwen3_vl_8b_thinking"
         / "shift" / "shift_predictions.jsonl")


def load_truth() -> tuple[dict[str, dict[str, bool]], dict[str, str]]:
    truth: dict[str, dict[str, bool]] = {}
    patients: dict[str, str] = {}
    with open(MAN, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            iid = r["image_id"]
            patients[iid] = str(r["patient_id"])
            truth[iid] = {l: str(r.get(f"label_{l}") or "").lower() == "positive"
                          for l in PATHOLOGY}
    return truth, patients


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values, monotone from the largest p downwards."""
    order = np.argsort(p)
    ranked = p[order]
    n = len(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        val = min(prev, ranked[i] * n / (i + 1))
        q[i] = val
        prev = val
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def is_pos(rec: dict, label: str) -> bool:
    f = (rec.get("parsed") or {}).get("findings") or {}
    return str((f.get(label) or {}).get("status") or "") == "positive"


def cell_stats(records: list[dict], truth: dict[str, dict[str, bool]]) -> dict:
    n = len(records)
    ok = sum(1 for r in records if (r.get("parsed") or {}).get("parse_success"))
    viol = sum(1 for r in records if (r.get("contract") or {}).get("contract_violation"))
    defaulted = sum(int((r.get("contract") or {}).get("n_labels_defaulted") or 0)
                    for r in records)
    probs_missing = sum(int((r.get("contract") or {}).get("n_probs_missing") or 0)
                        for r in records)
    nof = [is_pos(r, "no_finding") for r in records
           if (r.get("parsed") or {}).get("parse_success")]
    return {
        "n": n,
        "parse_rate": round(ok / n, 4) if n else None,
        "contract_violation_rate": round(viol / n, 4) if n else None,
        "mean_labels_defaulted": round(defaulted / ok, 3) if ok else None,
        "mean_probs_missing": round(probs_missing / ok, 3) if ok else None,
        "no_finding_positive_rate": round(sum(nof) / len(nof), 4) if nof else None,
    }


def counts_by_image(records: list[dict],
                    truth: dict[str, dict[str, bool]]) -> dict[str, dict[str, list[int]]]:
    """image_id -> label -> [true positives, ground-truth positives]."""
    out: dict[str, dict[str, list[int]]] = {}
    for r in records:
        parsed = r.get("parsed") or {}
        if not parsed.get("parse_success"):
            continue
        iid = str(r.get("image_id"))
        t = truth.get(iid)
        if t is None:
            continue
        acc = out.setdefault(iid, {l: [0, 0] for l in PATHOLOGY})
        for label in PATHOLOGY:
            acc[label][0] += int(t[label] and is_pos(r, label))
            acc[label][1] += int(t[label])
    return out


def paired_bootstrap(a: dict, b: dict, patients: dict[str, str]) -> list[dict]:
    """Per-label recall(b) - recall(a) with patient-cluster bootstrap."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return []
    pts = np.array([patients.get(i, i) for i in ids])
    uniq = sorted(set(pts.tolist()))
    by_patient = {q: np.where(pts == q)[0] for q in uniq}
    rng = np.random.default_rng(SEED)
    rows = []
    for label in PATHOLOGY:
        a_tp = np.array([a[i][label][0] for i in ids], dtype=float)
        a_gt = np.array([a[i][label][1] for i in ids], dtype=float)
        b_tp = np.array([b[i][label][0] for i in ids], dtype=float)
        b_gt = np.array([b[i][label][1] for i in ids], dtype=float)
        if a_gt.sum() == 0 or b_gt.sum() == 0:
            continue
        ra, rb = a_tp.sum() / a_gt.sum(), b_tp.sum() / b_gt.sum()
        boots = np.empty(N_BOOT, dtype=float)
        for k in range(N_BOOT):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([by_patient[q] for q in pick])
            ag, bg = a_gt[idx].sum(), b_gt[idx].sum()
            boots[k] = np.nan if (ag == 0 or bg == 0) else b_tp[idx].sum() / bg - a_tp[idx].sum() / ag
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        p = 2.0 * min(np.mean(boots <= 0), np.mean(boots >= 0))
        rows.append({
            "label": label,
            "n_images": len(ids), "n_patients": len(uniq),
            "recall_a": round(float(ra), 6), "recall_b": round(float(rb), 6),
            "delta": round(float(rb - ra), 6),
            "ci_lo": round(float(lo), 6), "ci_hi": round(float(hi), 6),
            "p_boot": round(float(min(1.0, max(0.0, p))), 6),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    truth, patients = load_truth()
    contract_rows: list[dict] = []
    ci_rows: list[dict] = []

    # ------------------------------------------------------------------ #
    # schema ablation + paraphrase: original vs blank
    # ------------------------------------------------------------------ #
    for experiment, root in ABLATION_ROOTS:
        if not root.exists():
            print(f"[skip] {experiment}: {root} absent")
            continue
        for variant in sorted({p.name for p in root.glob("*") if p.is_dir()}):
            cells = {}
            for condition in ("original", "blank"):
                path = root / variant / condition / "image_reliance_predictions.jsonl"
                recs = read_jsonl(path)
                if not recs:
                    continue
                cells[condition] = recs
                st = cell_stats(recs, truth)
                contract_rows.append({"experiment": experiment, "variant": variant,
                                      "condition": condition, **st})
            if len(cells) != 2:
                continue
            a = counts_by_image(cells["original"], truth)
            b = counts_by_image(cells["blank"], truth)
            for row in paired_bootstrap(a, b, patients):
                ci_rows.append({"experiment": experiment, "variant": variant,
                                "contrast": "original->blank", **row})
            print(f"[ablation] {experiment}/{variant}: "
                  f"{len(cells['original'])} original, {len(cells['blank'])} blank")

    # ------------------------------------------------------------------ #
    # second reasoning model: clean vs pooled severe
    # ------------------------------------------------------------------ #
    if THINK.exists():
        recs = read_jsonl(THINK)
        clean = [r for r in recs if r.get("condition") == "clean"]
        severe = [r for r in recs if r.get("severity") == "severe"]
        marker_rows = []
        # The grid holds only strictly valid replies, so the informative number
        # is not "did it close" (always yes here) but how many studies the model
        # answered at all: 1,506 for clean and 6 x 1,506 for the severe cells.
        for name, bucket, denom in (("clean", clean, 1506), ("severe", severe, 9036)):
            lens = [len((r.get("raw_output") or "")) for r in bucket]
            marker_rows.append({
                "model": "qwen3_vl_8b_thinking", "condition": name,
                "n_answered": len(bucket), "n_studies": denom,
                "answered_rate": round(len(bucket) / denom, 4) if denom else None,
                "mean_chars": round(float(np.mean(lens)), 1) if lens else None,
                "median_chars": round(float(np.median(lens)), 1) if lens else None,
            })
        a = counts_by_image(clean, truth)
        b = counts_by_image(severe, truth)
        for row in paired_bootstrap(a, b, patients):
            ci_rows.append({"experiment": "second_model",
                            "variant": "qwen3_vl_8b_thinking",
                            "contrast": "clean->severe", **row})
        pd.DataFrame(marker_rows).to_csv(out_dir / "round2_marker.csv", index=False,
                                         encoding="utf-8-sig")
        print(f"[second model] {len(clean)} clean, {len(severe)} severe records")
    else:
        print(f"[skip] second model: {THINK} absent")

    ci = pd.DataFrame(ci_rows)
    if len(ci):
        ci["q_bh"] = np.nan
        for (exp, var), g in ci.groupby(["experiment", "variant"]):
            ci.loc[g.index, "q_bh"] = np.round(
                bh(g["p_boot"].to_numpy(dtype=float)), 6)
        ci.to_csv(out_dir / "round2_label_ci.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(contract_rows).to_csv(out_dir / "round2_contract.csv", index=False,
                                       encoding="utf-8-sig")

    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
