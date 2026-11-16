#!/usr/bin/env python
"""108 - Fixed-rule rerun analysis (bucket-2 item 080).

The post-hoc check in the paper applied a fixed 0.5 cut to the probabilities the
models had already verbalised.  That removes the discrete-status artefact but
not the possibility that the probability scale itself moved.  This analysis uses
a *new* run in which the model is asked for probabilities only
(`q_probs_only`); the positive/negative status is then assigned by us at a fixed
0.5 threshold.  Recall and positive rate are therefore measured under a decision
rule that cannot move.

Inputs:
  outputs/fixed_rule/<model>/shift/shift_predictions.jsonl

Output:
  outputs/analysis_round2/fixed_rule_rerun.csv

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\108_fixed_rule_rerun.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "fixed_rule"
MAN = ROOT / "data" / "manifests" / "nih_eval_1506.csv"
OUT = ROOT / "outputs" / "analysis_round2" / "fixed_rule_rerun.csv"

PATHOLOGY = ["atelectasis", "cardiomegaly", "consolidation", "edema",
             "pleural_effusion", "pneumonia", "pneumothorax"]
MODELS = ["internvl3_5_8b", "qwen3_vl_8b_instruct"]
THRESHOLD = 0.5


def load_truth() -> dict[str, dict]:
    d = pd.read_csv(MAN, low_memory=False)
    return {str(r["image_id"]): {l: str(r.get(f"label_{l}") or "").lower() == "positive"
                                 for l in PATHOLOGY}
            for _, r in d.iterrows()}


def has_prob(rec: dict, label: str) -> bool:
    f = (rec.get("parsed") or {}).get("findings") or {}
    v = (f.get(label) or {}).get("probability_positive")
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def prob(rec: dict, label: str) -> float:
    f = (rec.get("parsed") or {}).get("findings") or {}
    v = (f.get(label) or {}).get("probability_positive")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return 0.0
    v = float(v)
    return v if 0.0 <= v <= 1.0 else 0.0


def main() -> int:
    truth = load_truth()
    rows = []
    for model in MODELS:
        path = RUN / model / "shift_predictions.jsonl"
        if not path.exists():
            print(f"skip {model}: no predictions")
            continue
        by_cell: dict[tuple[str, str], dict[str, dict]] = {}
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if not (r.get("parsed") or {}).get("parse_success"):
                    continue
                key = (str(r.get("condition")), str(r.get("severity")))
                by_cell.setdefault(key, {})[str(r.get("image_id"))] = r
        clean = by_cell.get(("clean", "none")) or by_cell.get(("clean", "clean"))
        if not clean:
            print(f"skip {model}: no clean cell")
            continue
        for (cond, sev), cell in sorted(by_cell.items()):
            if (cond, sev) == ("clean", "none"):
                continue
            common = sorted(set(clean) & set(cell))
            completeness = float(np.mean([
                np.mean([1.0 if has_prob(cell[i], l) else 0.0 for l in PATHOLOGY])
                for i in common
            ])) if common else 0.0
            if completeness < 0.5:
                print(f"skip {model} {cond}/{sev}: probability completeness "
                      f"{completeness:.2f} -- the contract collapsed, so a fixed "
                      f"0.5 rule is not identifiable")
                continue
            signal = float(np.mean([
                np.mean([1.0 if prob(cell[i], l) > 0.0 else 0.0 for l in PATHOLOGY])
                for i in common
            ])) if common else 0.0
            if signal < 0.05:
                print(f"skip {model} {cond}/{sev}: probabilities are degenerate "
                      f"(nonzero rate {signal:.4f}) -- a fixed 0.5 rule is not "
                      f"identifiable")
                continue
            # Paired, so differential coverage cannot create the change.
            for label in PATHOLOGY:
                y = [1 if truth[i][label] else 0 for i in common]
                pc = [1 if prob(clean[i], label) >= THRESHOLD else 0 for i in common]
                ps = [1 if prob(cell[i], label) >= THRESHOLD else 0 for i in common]
                den = sum(y)
                if den == 0:
                    continue
                rows.append({
                    "model": model, "condition": cond, "severity": sev,
                    "label": label, "n_paired": len(common), "n_positive": den,
                    "recall_clean": round(sum(yy * pp for yy, pp in zip(y, pc)) / den, 4),
                    "recall_severe": round(sum(yy * pp for yy, pp in zip(y, ps)) / den, 4),
                    "pos_rate_clean": round(float(np.mean(pc)), 4),
                    "pos_rate_severe": round(float(np.mean(ps)), 4),
                    "prob_completeness": round(completeness, 4),
                    "threshold": THRESHOLD,
                })
    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    if len(df):
        summary = (df.groupby(["model", "condition", "severity"])
                     .agg(recall_clean=("recall_clean", "mean"),
                          recall_severe=("recall_severe", "mean"),
                          pos_rate_clean=("pos_rate_clean", "mean"),
                          pos_rate_severe=("pos_rate_severe", "mean"))
                     .reset_index())
        print(summary.to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
