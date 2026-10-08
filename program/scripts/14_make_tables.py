#!/usr/bin/env python
"""Generate paper-level CSV/Markdown tables from Stage 2 outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent
OUT = PAPER / "tables" / "stage2"
OUT.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str):
    csv = OUT / f"{name}.csv"
    df.to_csv(csv, index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_string(index=False)
    (OUT / f"{name}.md").write_text(md + "\n", encoding="utf-8")
    print("wrote", csv)


def table_image_reliance():
    rows = []
    for model, path in [
        ("MedGemma", ROOT / "outputs/reliance/medgemma_300/image_reliance_summary.json"),
        ("Qwen2.5-VL", ROOT / "outputs/reliance/qwen_300/image_reliance_summary.json"),
    ]:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for cond, m in data.get("conditions", {}).items():
            rows.append({
                "model": model, "condition": cond,
                "image_sensitivity": m.get("image_sensitivity"),
                "mean_flip_count": m.get("mean_flip_count"),
                "mean_abs_prob_diff": m.get("mean_abs_prob_diff"),
            })
    if rows:
        save_table(pd.DataFrame(rows), "table01_image_reliance")


def table_shift():
    for model, name in [
        ("medgemma_4b_it_100", "table02_shift_medgemma"),
        ("qwen25_vl_7b_instruct_100", "table03_shift_qwen"),
    ]:
        path = ROOT / "outputs/shift_benchmark" / model / "analysis/shift_metrics_by_condition.csv"
        if path.exists():
            df = pd.read_csv(path)
            save_table(df, name)


def table_rrg():
    path = ROOT / "outputs/rrg/medgemma/analysis/rrg_report_metrics_by_label.csv"
    if path.exists():
        df = pd.read_csv(path)
        save_table(df, "table04_rrg_report_errors")
    path2 = ROOT / "outputs/rrg/medgemma/analysis/rrg_hallucination_summary.json"
    if path2.exists():
        data = json.loads(path2.read_text(encoding="utf-8"))
        save_table(pd.DataFrame([data]), "table05_rrg_summary")


def table_bootstrap():
    for model, name in [
        ("medgemma_4b_it_100", "table06_bootstrap_medgemma"),
        ("qwen25_vl_7b_instruct_100", "table07_bootstrap_qwen"),
    ]:
        path = ROOT / "outputs/shift_benchmark" / model / "statistics_macro/paired_clean_vs_shift.csv"
        if path.exists():
            save_table(pd.read_csv(path), name)


def table_main_performance():
    rows = []
    for model, name in [("medgemma_4b_it_1000", "MedGemma"), ("qwen25_vl_7b_instruct_1000", "Qwen2.5-VL")]:
        path = ROOT / "outputs/overnight" / model / "metrics/metrics_by_label.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for split, g in df.groupby("split"):
            row = {
                "model": name,
                "split": split,
                "macro_auroc": float(np.nanmean(g["auroc"])),
                "macro_ece": float(np.nanmean(g["ece"])),
                "macro_brier": float(np.nanmean(g["brier"])),
                "macro_aurc": float(np.nanmean(g["aurc"])),
                "mean_coverage_at_5pct_risk": float(np.nanmean(g["coverage_at_5pct_risk"])),
            }
            # Where the risk score behind macro_aurc comes from. No model in this
            # matrix emits its own `confidence` (null in 26,462 of 26,462 rows
            # measured 2026-10-06), so the score is derived from the probability,
            # and for a finding with a status but no probability it is imputed at
            # 0.5. That share is a property of the table, so it belongs IN the
            # table instead of in a comment -- see docs/50 section 2.10.
            if "conf_source" in df.columns:
                row["conf_source"] = "|".join(
                    sorted(set(g["conf_source"].dropna().astype(str))))
                row["n_conf_imputed"] = int(g["n_conf_imputed"].fillna(0).sum()) \
                    if "n_conf_imputed" in df.columns else 0
            rows.append(row)
    if rows:
        save_table(pd.DataFrame(rows), "table08_main_performance_calibration")


def main():
    table_image_reliance()
    table_shift()
    table_rrg()
    table_bootstrap()
    table_main_performance()
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
