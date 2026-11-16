#!/usr/bin/env python
"""96 - Analyze the four round-2 GPU experiments (CPU only).

Reads the outputs of:
  * outputs/ablation_mg15/medgemma_15_4b_it/...
  * outputs/paraphrase_mg4/medgemma_4b_it/...
  * outputs/densenet_baseline/metrics.json
  * outputs/matrix/nih_eval/qwen3_vl_8b_instruct/shift_think/...

and writes a single summary JSON plus a flat CSV.  It never calls a model.

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\96_analyze_round2.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis_round2"
LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax"]
DEGRADATIONS = ["blur", "noise", "jpeg", "brightness", "contrast", "resolution"]


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def is_pos(r: dict, label: str) -> bool:
    f = (r.get("parsed") or {}).get("findings") or {}
    return str((f.get(label) or {}).get("status") or "") == "positive"


def gt_pos(r: dict, label: str) -> bool:
    return str((r.get("ground_truth") or {}).get(f"label_{label}") or "") == "positive"


def summarize_reliance(path: Path) -> dict:
    rs = rows(path)
    if not rs:
        return {}
    n = len(rs)
    ok = sum(1 for r in rs if (r.get("parsed") or {}).get("parse_success"))
    viol = sum(1 for r in rs if (r.get("contract") or {}).get("contract_violation"))
    nof = [is_pos(r, "no_finding") for r in rs if (r.get("parsed") or {}).get("parse_success")]
    rec = {}
    for label in LABELS:
        pairs = [(gt_pos(r, label), is_pos(r, label)) for r in rs
                 if (r.get("parsed") or {}).get("parse_success")]
        den = sum(1 for y, _ in pairs if y)
        rec[label] = round(sum(1 for y, p in pairs if y and p) / den, 4) if den else None
    return {
        "n": n, "parse_rate": round(ok / n, 4),
        "contract_violation_rate": round(viol / n, 4),
        "no_finding_positive_rate": round(sum(nof) / len(nof), 4) if nof else None,
        "recall": rec,
    }


def marker_preamble(rec: dict) -> bool:
    raw = rec.get("raw_output") or ""
    i = raw.find("{")
    return bool(raw[:i].strip()) if i >= 0 else bool(raw.strip())


def marker_unused94(rec: dict) -> bool:
    return (rec.get("raw_output") or "").startswith("<unused94>")


def marker_think_pair(rec: dict) -> bool:
    """True when the reply closed a reasoning block.

    Qwen3-VL-Thinking's chat template pre-fills `<think>` into the prompt, so the
    model's own text holds the close tag and not the opener; requiring the pair
    would report a channel rate of 0 for a model that always reasons.  The close
    tag is therefore the marker, and `reasoning_chars` below measures how much
    trace it emitted.
    """
    raw = rec.get("raw_output") or ""
    return "</think>" in raw


def summarize_shift(path: Path, marker_fn) -> dict:
    rs = rows(path)
    if not rs:
        return {}
    clean = [r for r in rs if r.get("condition") == "clean"]
    severe = [r for r in rs if r.get("severity") == "severe"]
    if not clean or not severe:
        return {}
    out = {"n": len(rs), "n_clean": len(clean), "n_severe": len(severe),
           "marker_rate_clean": round(sum(marker_fn(r) for r in clean) / len(clean), 4),
           "marker_rate_severe": round(sum(marker_fn(r) for r in severe) / len(severe), 4)}
    rec = {}
    for label in LABELS:
        c = [(gt_pos(r, label), is_pos(r, label)) for r in clean if (r.get("parsed") or {}).get("parse_success")]
        s = [(gt_pos(r, label), is_pos(r, label)) for r in severe if (r.get("parsed") or {}).get("parse_success")]
        cr = sum(1 for y, p in c if y and p) / sum(1 for y, _ in c) if sum(1 for y, _ in c) else None
        sr = sum(1 for y, p in s if y and p) / sum(1 for y, _ in s) if sum(1 for y, _ in s) else None
        rec[label] = {"clean": round(cr, 4) if cr is not None else None,
                      "severe": round(sr, 4) if sr is not None else None,
                      "delta": round(sr - cr, 4) if cr is not None and sr is not None else None}
    out["recall"] = rec
    per = {}
    for deg in DEGRADATIONS:
        d = [r for r in severe if r.get("condition") == deg]
        if not d:
            continue
        m = sum(marker_fn(r) for r in d) / len(d)
        pairs = [(gt_pos(r, l), is_pos(r, l)) for r in d for l in LABELS
                 if (r.get("parsed") or {}).get("parse_success")]
        recall = sum(1 for y, p in pairs if y and p) / sum(1 for y, _ in pairs) if pairs else None
        per[deg] = {"marker_rate": round(m, 4), "severe_recall": round(recall, 4) if recall is not None else None}
    out["per_degradation"] = per
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict = {}
    summary["_definitions"] = {
        "n": "parsed prediction rows read from the file (one row per study and cell), "
             "not a count of studies",
        "recall": "unpaired, definite-status recall over every parsed reply, with all "
                  "uncertain/not_mentioned slots counted as negative; the paper's "
                  "paired estimates (1,394 studies, cluster bootstrap, BH-FDR) are in "
                  "deep_thinking_paired.csv and tab:r2label",
        "qwen3_thinking.n": "usable cells after the 8,192-token sample (9,401 of "
                            "10,542 = 89.2%); at the 5,120-token budget alone the "
                            "count is 9,381 (89.0%), see budget_sensitivity.csv",
    }

    # 1. MedGemma-1.5 schema ablation
    mg15_root = ROOT / "outputs" / "ablation_mg15" / "medgemma_15_4b_it"
    mg15 = {}
    if mg15_root.exists():
        for cell in sorted(mg15_root.glob("*/[ob]*")):
            variant, condition = cell.parent.name, cell.name
            if condition not in {"original", "blank"}:
                continue
            mg15.setdefault(variant, {})[condition] = summarize_reliance(
                cell / "image_reliance_predictions.jsonl")
    summary["mg15_ablation"] = mg15

    # 2. Paraphrase (and the existing v0 baseline for the same model)
    para = {}
    for root in [ROOT / "outputs" / "paraphrase_mg4" / "medgemma_4b_it",
                 ROOT / "outputs" / "matrix" / "nih_eval" / "medgemma_4b_it" / "ablation"]:
        if not root.exists():
            continue
        for cell in sorted(root.glob("*/[ob]*")):
            variant, condition = cell.parent.name, cell.name
            if condition not in {"original", "blank"}:
                continue
            p = cell / "image_reliance_predictions.jsonl"
            if p.exists():
                para.setdefault(variant, {})[condition] = summarize_reliance(p)
    summary["paraphrase"] = para

    # 3. DenseNet baseline
    mpath = ROOT / "outputs" / "densenet_baseline" / "metrics.json"
    summary["densenet"] = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else {}

    # 4. Qwen3-VL-8B-Thinking native <think> grid (clean + severe)
    think = ROOT / "outputs" / "matrix" / "nih_eval" / "qwen3_vl_8b_thinking" / "shift" / "shift_predictions.jsonl"
    summary["qwen3_thinking"] = summarize_shift(think, marker_think_pair)

    # Reference: MedGemma-1.5 native marker on the original grid
    ref = ROOT / "outputs" / "matrix" / "nih_eval" / "medgemma_15_4b_it" / "shift" / "shift_predictions.jsonl"
    summary["medgemma15_native"] = summarize_shift(ref, marker_unused94)

    (OUT / "round2_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    flat = []
    for model, cells in [("MedGemma-1.5", mg15), ("MedGemma-4B paraphrase", para)]:
        for variant, conds in cells.items():
            for condition, m in conds.items():
                if not m:
                    continue
                flat.append({"model": model, "variant": variant, "condition": condition,
                             "n": m["n"], "parse_rate": m["parse_rate"],
                             "contract_violation_rate": m["contract_violation_rate"],
                             "no_finding_positive_rate": m["no_finding_positive_rate"]})
    if flat:
        import csv
        with open(OUT / "round2_flat.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0]))
            w.writeheader()
            w.writerows(flat)

    print(json.dumps({
        "mg15_variants": {k: list(v) for k, v in mg15.items()},
        "paraphrase_variants": {k: list(v) for k, v in para.items()},
        "densenet_macro_auroc": (summary.get("densenet", {}).get("eval_1506", {})
                                 .get("macro_pathology_auroc")),
        "qwen3_marker": {k: summary["qwen3_thinking"].get(k)
                          for k in ["marker_rate_clean", "marker_rate_severe"]},
        "medgemma15_marker": {k: summary["medgemma15_native"].get(k)
                              for k in ["marker_rate_clean", "marker_rate_severe"]},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
