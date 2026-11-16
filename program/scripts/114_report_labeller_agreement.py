#!/usr/bin/env python
"""114 - Does the Open-i model ranking survive a validated report labeller?

Compares the rule-based omission/hallucination counts (script 42, reported in
the paper) with the CheXbert counts on the same 947 studies (script 109).  Both
are slot-level decisions over the same label set, so a rank correlation is
meaningful even though the absolute counts differ with the labeller's
sensitivity.

Output:
  outputs/analysis_chexbert/labeller_comparison.csv
  outputs/analysis_chexbert/labeller_comparison.json

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\114_report_labeller_agreement.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
GEO = ROOT / "outputs" / "geographic"
CHEX = ROOT / "outputs" / "analysis_chexbert" / "openi_chexbert_metrics.csv"
OUT = ROOT / "outputs" / "analysis_chexbert"

MODELS = {
    "medgemma_4b_it": "MedGemma-4B",
    "medgemma_15_4b_it": "MedGemma-1.5",
    "internvl3_5_8b": "InternVL3.5",
    "qwen3_vl_8b_instruct": "Qwen3-VL",
    "qwen25_vl_7b_instruct": "Qwen2.5-VL",
    "chexagent_8b": "CheXagent",
}


def main() -> int:
    rows = []
    for slug, name in MODELS.items():
        p = GEO / slug / "geographic_shift_summary.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))["report"]
        rows.append({"model": slug, "model_name": name,
                     "rule_omissions": d["overall_omission_count"],
                     "rule_hallucinations": d["overall_hallucination_count"]})
    rule = pd.DataFrame(rows)
    chex = pd.read_csv(CHEX)[["model", "omissions", "hallucinations"]]
    chex = chex.rename(columns={"omissions": "chex_omissions",
                                "hallucinations": "chex_hallucinations"})
    df = rule.merge(chex, on="model", how="inner")

    summary = {"n_models": len(df)}
    for col in ["omissions", "hallucinations"]:
        rho, p = spearmanr(df[f"rule_{col}"], df[f"chex_{col}"])
        summary[f"spearman_{col}"] = {"rho": round(float(rho), 3), "p": round(float(p), 4)}
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "labeller_comparison.csv", index=False, encoding="utf-8-sig")
    (OUT / "labeller_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
