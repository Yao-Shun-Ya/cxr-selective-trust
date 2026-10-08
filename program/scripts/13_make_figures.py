#!/usr/bin/env python
"""Generate paper-level figures from Stage 2 outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent
OUT = PAPER / "figures" / "stage2"
OUT.mkdir(parents=True, exist_ok=True)
# The paper does not include figures from OUT -- it includes them from
# papers/main_v2/figures/. The generator wrote only to OUT, so the two drifted:
# when the ECE bin convention and the calibration universe were corrected, OUT got
# the new figure at 16:23 while the paper kept rendering the stale 01:35 copy, and
# nothing anywhere said so. Figures reachable by a paper are written to both
# locations now, so a regeneration cannot leave the manuscript behind.
PAPER_FIG_DIR = PAPER / "papers" / "main_v2" / "figures"


sns.set_theme(style="whitegrid", context="talk")
SEV_ORDER = ["mild", "moderate", "severe"]


def save(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=300)
        # also refresh the copy the manuscript actually embeds, when this figure is
        # one the manuscript uses
        if PAPER_FIG_DIR.is_dir() and (PAPER_FIG_DIR / f"{name}.pdf").exists():
            fig.savefig(PAPER_FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    target = "OUT + papers/main_v2/figures" if \
        (PAPER_FIG_DIR / f"{name}.pdf").exists() else "OUT"
    print("wrote", OUT / name, f"({target})")


def load_image_reliance():
    rows = []
    for model, path in [
        ("MedGemma", ROOT / "outputs/reliance/medgemma_300/image_reliance_summary.json"),
        ("Qwen2.5-VL", ROOT / "outputs/reliance/qwen_300/image_reliance_summary.json"),
    ]:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for condition, m in data.get("conditions", {}).items():
            rows.append({
                "model": model,
                "condition": condition,
                "image_sensitivity": m.get("image_sensitivity"),
                "mean_flip_count": m.get("mean_flip_count"),
                "mean_abs_prob_diff": m.get("mean_abs_prob_diff"),
            })
    return pd.DataFrame(rows)


def fig_image_reliance():
    df = load_image_reliance()
    if df.empty:
        print("skip image reliance: no data"); return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order = ["blank", "blur", "shuffle"]
    sns.barplot(data=df, x="condition", y="image_sensitivity", hue="model", order=order, ax=axes[0])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Image sensitivity")
    axes[0].set_ylabel("fraction of images with any finding flip")
    sns.barplot(data=df, x="condition", y="mean_abs_prob_diff", hue="model", order=order, ax=axes[1])
    axes[1].set_title("Mean absolute probability change")
    axes[1].set_ylabel("mean |ΔP|")
    for ax in axes:
        ax.legend_.set_title("")
    save(fig, "fig02_image_reliance_main")


def load_shift_model(model_dir: str):
    d = ROOT / "outputs/shift_benchmark" / model_dir / "analysis"
    cond = pd.read_csv(d / "shift_metrics_by_condition.csv") if (d / "shift_metrics_by_condition.csv").exists() else pd.DataFrame()
    sens = pd.read_csv(d / "shift_image_sensitivity.csv") if (d / "shift_image_sensitivity.csv").exists() else pd.DataFrame()
    return cond, sens


def plot_severity_metric(metric_col, ylabel, name, models=("medgemma_4b_it_100", "qwen25_vl_7b_instruct_100")):
    model_names = {"medgemma_4b_it_100": "MedGemma", "qwen25_vl_7b_instruct_100": "Qwen2.5-VL"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, model_dir in zip(axes, models):
        cond, _ = load_shift_model(model_dir)
        if cond.empty:
            ax.set_visible(False); continue
        for shift, g in cond.groupby("condition"):
            if shift == "clean":
                continue
            g = g.set_index("severity").reindex(SEV_ORDER).reset_index()
            ax.plot(g["severity"], g[metric_col], marker="o", label=shift)
        clean = cond[cond["condition"] == "clean"]
        if not clean.empty:
            ax.axhline(float(clean[metric_col].iloc[0]), color="black", linestyle="--", linewidth=1, label="clean")
        ax.set_title(model_names.get(model_dir, model_dir))
        ax.set_xlabel("severity")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9, ncol=2)
    save(fig, name)


def fig_shift_metrics():
    plot_severity_metric("macro_auroc", "macro AUROC", "fig03_shift_auroc_severity")
    plot_severity_metric("macro_ece", "macro ECE", "fig04_shift_ece_severity")


def fig_shift_sensitivity():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, model_dir in zip(axes, ["medgemma_4b_it_100", "qwen25_vl_7b_instruct_100"]):
        _, sens = load_shift_model(model_dir)
        if sens.empty:
            ax.set_visible(False); continue
        for shift, g in sens.groupby("condition"):
            g = g.set_index("severity").reindex(SEV_ORDER).reset_index()
            ax.plot(g["severity"], g["image_sensitivity"], marker="o", label=shift)
        ax.set_ylim(0, 1.05)
        ax.set_title("MedGemma" if "med" in model_dir else "Qwen2.5-VL")
        ax.set_xlabel("severity")
        ax.set_ylabel("image sensitivity")
        ax.legend(fontsize=9, ncol=2)
    save(fig, "fig05_shift_image_sensitivity")


def fig_risk_coverage():
    fig, ax = plt.subplots(figsize=(7, 6))
    for model, label in [("medgemma_4b_it_1000", "MedGemma"), ("qwen25_vl_7b_instruct_1000", "Qwen2.5-VL")]:
        path = ROOT / "outputs/overnight" / model / "calibration_selective/risk_coverage_curve.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ax.plot(df["coverage"], df["risk"], label=label)
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (error rate among accepted)")
    ax.set_title("Risk–coverage on NIH internal_test")
    ax.legend()
    save(fig, "fig06_risk_coverage")


def fig_rrg():
    path = ROOT / "outputs/rrg/medgemma/analysis/rrg_report_metrics_by_label.csv"
    if not path.exists():
        print("skip RRG: no data"); return
    df = pd.read_csv(path)
    df = df.melt(id_vars=["label"], value_vars=["hallucination_rate", "omission_rate"],
                 var_name="metric", value_name="rate")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df, x="label", y="rate", hue="metric", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    ax.set_ylabel("rate")
    ax.set_title("CheXpert-RRG report-level errors (rule_based_v0)")
    save(fig, "fig07_rrg_report_errors")


def fig_shift_heatmap():
    path = ROOT / "outputs/shift_benchmark/medgemma_4b_it_100/analysis/shift_metrics_by_condition.csv"
    if not path.exists():
        print("skip heatmap: no data"); return
    df = pd.read_csv(path)
    piv = df.pivot(index="condition", columns="severity", values="macro_auroc")
    piv = piv[[c for c in SEV_ORDER if c in piv.columns]]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(piv, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.4, vmax=0.8, ax=ax)
    ax.set_title("MedGemma macro AUROC by shift × severity")
    save(fig, "fig08_shift_macro_auroc_heatmap")


def main():
    fig_image_reliance()
    fig_shift_metrics()
    fig_shift_sensitivity()
    fig_risk_coverage()
    fig_rrg()
    fig_shift_heatmap()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
