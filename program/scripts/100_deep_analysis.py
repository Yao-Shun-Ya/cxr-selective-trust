#!/usr/bin/env python
"""100 - Deep cross-experiment analysis for round 2 (CPU only).

Three experiments finished cleanly (schema ablation, paraphrase, supervised
baseline) and one (the reasoning-model grid) is being repaired for truncated
replies.  This script turns them into the comparison tables the paper needs:

A. schema ablation     - recall / contract / no_finding / parse per variant x
                         condition, and the original->blank gap per label;
B. paraphrase          - per-image agreement with the MedGemma-4B v0 baseline
                         (Cohen's kappa), so a wording artefact is visible;
C. baseline vs VLMs    - DenseNet-121 per-label AUROC next to every VLM's clean
                         macro AUROC, i.e. is the task learnable at all;
D. reasoning grid      - termination rate, trace length, recall and per-label
                         AUROC per cell, plus the trace-length/correctness
                         association.

Outputs (outputs/analysis_round2/):
  deep_schema.csv
  deep_paraphrase_agreement.csv
  deep_baseline_vs_vlm.csv
  deep_thinking_cells.csv

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\100_deep_analysis.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis_round2"
# Split-pure evaluation manifest: 1,506 rows (1,054 internal_test + 452 source_val).
# The older 1,732-row file also carries 226 internal_train rows, which would leak
# training images into an evaluation table.
MAN = ROOT / "data" / "manifests" / "nih_eval_1506.csv"
ABL = ROOT / "outputs" / "ablation_mg15" / "medgemma_15_4b_it"
PAR = ROOT / "outputs" / "paraphrase_mg4" / "medgemma_4b_it"
BASE4 = ROOT / "outputs" / "matrix" / "nih_eval" / "medgemma_4b_it" / "ablation"
GRID = (ROOT / "outputs" / "matrix" / "nih_eval" / "qwen3_vl_8b_thinking"
        / "shift" / "shift_predictions.jsonl")
CELL_METRICS = ROOT / "outputs" / "analysis_nih_mine" / "shift_cell_metrics_all.csv"
DENSE = ROOT / "outputs" / "densenet_baseline" / "metrics.json"

PATHOLOGY = ["atelectasis", "cardiomegaly", "consolidation", "edema",
             "pleural_effusion", "pneumonia", "pneumothorax"]
LABELS8 = PATHOLOGY + ["no_finding"]


def load_truth() -> dict[str, dict]:
    d = pd.read_csv(MAN, low_memory=False)
    out = {}
    for _, r in d.iterrows():
        out[str(r["image_id"])] = {
            "y": {l: str(r.get(f"label_{l}") or "").lower() == "positive" for l in LABELS8},
            "patient": str(r.get("patient_id", "")),
        }
    return out


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


def status_pos(rec: dict, label: str) -> bool:
    f = (rec.get("parsed") or {}).get("findings") or {}
    return str((f.get(label) or {}).get("status") or "") == "positive"


def prob(rec: dict, label: str) -> float | None:
    f = (rec.get("parsed") or {}).get("findings") or {}
    p = (f.get(label) or {}).get("probability_positive")
    if isinstance(p, bool) or not isinstance(p, (int, float)):
        return None
    return float(p) if 0.0 <= float(p) <= 1.0 else None


def cell_rows(records: list[dict], truth: dict) -> dict:
    """Pooled per-label recall and rates for one cell."""
    ok = [r for r in records if (r.get("parsed") or {}).get("parse_success")]
    out = {
        "n": len(records), "n_parsed": len(ok),
        "parse_rate": round(len(ok) / len(records), 4) if records else None,
        "contract_violation_rate": round(
            sum(1 for r in records if (r.get("contract") or {}).get("contract_violation"))
            / len(records), 4) if records else None,
        "no_finding_pos_rate": round(
            sum(status_pos(r, "no_finding") for r in ok) / len(ok), 4) if ok else None,
        "pred_pos_rate": round(
            sum(status_pos(r, l) for r in ok for l in PATHOLOGY)
            / (len(ok) * len(PATHOLOGY)), 4) if ok else None,
    }
    rec = {}
    for l in PATHOLOGY:
        tp = sum(1 for r in ok if truth.get(str(r.get("image_id")), {}).get("y", {}).get(l)
                 and status_pos(r, l))
        gt = sum(1 for r in ok if truth.get(str(r.get("image_id")), {}).get("y", {}).get(l))
        rec[l] = round(tp / gt, 4) if gt else None
    out["path_recall"] = rec
    out["path_recall_pooled"] = round(
        sum(rec[l] * sum(1 for r in ok if truth.get(str(r.get("image_id")), {}).get("y", {}).get(l))
            for l in PATHOLOGY) /
        max(1, sum(sum(1 for r in ok if truth.get(str(r.get("image_id")), {}).get("y", {}).get(l))
                   for l in PATHOLOGY)), 4)
    return out


# --------------------------------------------------------------------------- #
# A. schema ablation
# --------------------------------------------------------------------------- #
def schema_tables(truth: dict) -> pd.DataFrame:
    rows = []
    for variant_dir in sorted(p for p in ABL.iterdir() if p.is_dir()):
        for cond_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
            path = cond_dir / "image_reliance_predictions.jsonl"
            if not path.exists():
                continue
            recs = read_jsonl(path)
            if not recs:
                continue
            m = cell_rows(recs, truth)
            for l in PATHOLOGY:
                rows.append({"variant": variant_dir.name, "condition": cond_dir.name,
                             "label": l, "recall": m["path_recall"][l],
                             "n": m["n"], "n_parsed": m["n_parsed"],
                             "parse_rate": m["parse_rate"],
                             "contract_violation_rate": m["contract_violation_rate"],
                             "no_finding_pos_rate": m["no_finding_pos_rate"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# B. paraphrase agreement vs the MedGemma-4B v0 baseline
# --------------------------------------------------------------------------- #
def paraphrase_agreement() -> pd.DataFrame:
    rows = []
    for cond in ("original", "blank"):
        base = read_jsonl(BASE4 / "v0_simple" / cond / "image_reliance_predictions.jsonl")
        base_status = {str(r.get("image_id")): r for r in base
                       if (r.get("parsed") or {}).get("parse_success")}
        for variant in ("p1_paraphrase_short", "p2_paraphrase_second_reader"):
            recs = read_jsonl(PAR / variant / cond / "image_reliance_predictions.jsonl")
            for label in LABELS8:
                pairs = []
                for r in recs:
                    iid = str(r.get("image_id"))
                    if not (r.get("parsed") or {}).get("parse_success"):
                        continue
                    b = base_status.get(iid)
                    if b is None:
                        continue
                    pairs.append((status_pos(b, label), status_pos(r, label)))
                if len(pairs) < 10:
                    continue
                a = [p[0] for p in pairs]
                b = [p[1] for p in pairs]
                kappa = cohen_kappa_score(a, b) if len(set(a)) > 1 or len(set(b)) > 1 else 0.0
                rows.append({
                    "condition": cond, "variant": variant, "label": label,
                    "n_pairs": len(pairs),
                    "agreement": round(float(np.mean([x == y for x, y in pairs])), 4),
                    "kappa": round(float(kappa), 4),
                    "base_pos_rate": round(float(np.mean(a)), 4),
                    "variant_pos_rate": round(float(np.mean(b)), 4),
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# C. supervised baseline vs VLMs
# --------------------------------------------------------------------------- #
def baseline_vs_vlm() -> pd.DataFrame:
    rows = []
    if DENSE.exists():
        m = json.loads(DENSE.read_text(encoding="utf-8"))["eval_1506"]
        for l in LABELS8:
            rows.append({"model": "densenet121_supervised", "label": l,
                         "auroc": round(float(m[l]), 4), "source": "eval_1506"})
        rows.append({"model": "densenet121_supervised", "label": "macro_pathology",
                     "auroc": round(float(m["macro_pathology_auroc"]), 4),
                     "source": "eval_1506"})
    if CELL_METRICS.exists():
        d = pd.read_csv(CELL_METRICS)
        clean = d[d["condition"] == "clean"]
        for _, r in clean.iterrows():
            rows.append({"model": r["model"], "label": "macro_pathology",
                         "auroc": round(float(r["macro_auroc"]), 4),
                         "source": "clean_grid"})
            rows.append({"model": r["model"], "label": "parse_coverage",
                         "auroc": round(float(r["parse_coverage"]), 4),
                         "source": "clean_grid"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# D. reasoning grid
# --------------------------------------------------------------------------- #
def thinking_tables(truth: dict) -> tuple[pd.DataFrame, dict]:
    recs = read_jsonl(GRID)
    rows = []
    cells = defaultdict(list)
    for r in recs:
        cells[f"{r.get('condition')}/{r.get('severity')}"].append(r)
    # Coverage is the cell's valid-answer count over the fixed 1,506 studies.
    # The grid file only holds strictly valid replies (closing tag + JSON after
    # it), so `n` is exactly the number of studies the model answered.
    for cell, rr in sorted(cells.items()):
        ok = [r for r in rr if (r.get("parsed") or {}).get("parse_success")]
        chars = [len(r.get("raw_output") or "") for r in rr]
        m = {
            "cell": cell, "n": len(rr),
            "coverage": round(len(rr) / 1506, 4),
            "mean_chars": round(float(np.mean(chars)), 1),
            "median_chars": round(float(np.median(chars)), 1),
            "no_finding_pos_rate": round(sum(status_pos(r, "no_finding") for r in ok) / len(ok), 4) if ok else None,
        }
        for l in PATHOLOGY:
            tp = sum(1 for r in ok if truth.get(str(r.get("image_id")), {}).get("y", {}).get(l) and status_pos(r, l))
            gt = sum(1 for r in ok if truth.get(str(r.get("image_id")), {}).get("y", {}).get(l))
            m[f"recall_{l}"] = round(tp / gt, 4) if gt else None
            ys, ps = [], []
            for r in ok:
                y = truth.get(str(r.get("image_id")), {}).get("y", {}).get(l)
                p = prob(r, l)
                if y is not None and p is not None:
                    ys.append(int(y))
                    ps.append(p)
            m[f"auroc_{l}"] = (round(float(roc_auc_score(ys, ps)), 4)
                              if len(set(ys)) == 2 else None)
        rows.append(m)

    # Paired damage: every severe cell is compared with clean on the SAME studies,
    # i.e. those answered in both. Coverage differs by cell (clean 92 %, resolution
    # 81 %), so an unpaired recall difference would mix damage with a changing
    # study set; the paired version holds the studies fixed.
    by_cell_img = {c: {str(r.get("image_id")): r for r in rr} for c, rr in cells.items()}
    clean = by_cell_img.get("clean/none", {})
    paired = []
    for cell, imgs in sorted(by_cell_img.items()):
        if cell == "clean/none":
            continue
        common = sorted(set(imgs) & set(clean))
        rec_c, rec_s = [], []
        for l in PATHOLOGY:
            gt = sum(1 for i in common if truth.get(i, {}).get("y", {}).get(l))
            if not gt:
                continue
            tpc = sum(1 for i in common if truth[i]["y"].get(l) and status_pos(clean[i], l))
            tps = sum(1 for i in common if truth[i]["y"].get(l) and status_pos(imgs[i], l))
            rec_c.append(tpc / gt)
            rec_s.append(tps / gt)
        paired.append({
            "cell": cell, "n_paired": len(common),
            "recall_clean_paired": round(float(np.mean(rec_c)), 4) if rec_c else None,
            "recall_cell_paired": round(float(np.mean(rec_s)), 4) if rec_s else None,
            "delta_paired": round(float(np.mean(rec_s) - np.mean(rec_c)), 4) if rec_c else None,
        })
    if paired:
        pd.DataFrame(paired).to_csv(OUT / "deep_thinking_paired.csv", index=False,
                                    encoding="utf-8-sig")
        print("\nD2. paired clean->severe recall on commonly answered studies")
        print(pd.DataFrame(paired).to_string(index=False))

    # trace length vs correctness, per image (parsed rows only)
    assoc = {}
    per_image = defaultdict(lambda: {"chars": [], "correct": 0, "gt": 0})
    for r in recs:
        if not (r.get("parsed") or {}).get("parse_success"):
            continue
        iid = str(r.get("image_id"))
        t = truth.get(iid)
        if t is None:
            continue
        acc = per_image[iid]
        acc["chars"].append(len(r.get("raw_output") or ""))
        for l in PATHOLOGY:
            if t["y"].get(l):
                acc["gt"] += 1
                acc["correct"] += int(status_pos(r, l))
    xs, ys = [], []
    for iid, a in per_image.items():
        if a["gt"] == 0 or not a["chars"]:
            continue
        xs.append(float(np.mean(a["chars"])))
        ys.append(a["correct"] / a["gt"])
    from scipy.stats import spearmanr
    if xs:
        rho, p = spearmanr(xs, ys)
        assoc = {"n_images": len(xs), "spearman_rho": round(float(rho), 4),
                 "p": float(p), "mean_chars": round(float(np.mean(xs)), 1),
                 "mean_recall": round(float(np.mean(ys)), 4)}
    return pd.DataFrame(rows), assoc


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    truth = load_truth()

    schema = schema_tables(truth)
    schema.to_csv(OUT / "deep_schema.csv", index=False, encoding="utf-8-sig")
    piv = schema.pivot_table(index=["variant"], columns=["condition"],
                             values="recall", aggfunc="mean").round(3)
    print("A. schema ablation - mean per-label recall (original | blank)")
    print(piv.to_string())

    para = paraphrase_agreement()
    para.to_csv(OUT / "deep_paraphrase_agreement.csv", index=False, encoding="utf-8-sig")
    print("\nB. paraphrase vs v0 (MedGemma-4B) - mean kappa / agreement")
    print(para.groupby(["condition", "variant"])[["agreement", "kappa"]].mean().round(3).to_string())

    base = baseline_vs_vlm()
    base.to_csv(OUT / "deep_baseline_vs_vlm.csv", index=False, encoding="utf-8-sig")
    print("\nC. supervised baseline vs VLMs (macro pathology AUROC, clean)")
    print(base[base.label.isin(["macro_pathology", "parse_coverage"])]
          .pivot_table(index="model", columns="label", values="auroc", aggfunc="max").to_string())

    think, assoc = thinking_tables(truth)
    think.to_csv(OUT / "deep_thinking_cells.csv", index=False, encoding="utf-8-sig")
    print("\nD. reasoning grid - coverage / chars per cell")
    print(think[["cell", "n", "coverage", "mean_chars", "median_chars"]].to_string(index=False))
    print("\n   trace length vs per-image recall:", json.dumps(assoc))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
