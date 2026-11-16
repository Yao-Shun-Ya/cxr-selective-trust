#!/usr/bin/env python
"""92 - Figures for the benchmark-engineering paper (CPU only).

Reads the frozen coverage census / run accounting, plus the measured batch
sweep recorded in docs/46, and writes three PDF+PNG figures:

  fig1_calls.pdf     call budget by model and stage family
  fig2_coverage.pdf  parse / all-definite coverage heatmap
  fig3_batch.pdf     measured throughput vs batch size, three models

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\92_make_bench_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "papers" / "bench_v1" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
CENSUS = ROOT / "outputs" / "analysis_sensitivity" / "coverage_census.csv"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MODEL_LABEL = {
    "medgemma_4b_it": "MedGemma-4B",
    "medgemma_15_4b_it": "MedGemma-1.5",
    "qwen25_vl_7b_instruct": "Qwen2.5-VL",
    "qwen3_vl_8b_instruct": "Qwen3-VL",
    "internvl3_5_8b": "InternVL3.5",
    "chexagent_8b": "CheXagent",
    "llava_med_v1_5_7b": "LLaVA-Med",
}


def fig_calls() -> None:
    d = pd.read_csv(CENSUS, encoding="utf-8-sig")
    d["family"] = d["stage"].str.split("/").str[0]
    d["layer"] = d["layer"].map({"nih_eval": "NIH", "padchest": "PadChest", "openi": "Open-i"})
    pivot = d.pivot_table(index="model", columns="layer", values="calls",
                          aggfunc="sum", fill_value=0)
    pivot = pivot.reindex([m for m in MODEL_LABEL if m in pivot.index])
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    left = np.zeros(len(pivot))
    colors = {"NIH": "#0072B2", "PadChest": "#E69F00", "Open-i": "#009E73"}
    for layer in ["NIH", "PadChest", "Open-i"]:
        vals = pivot[layer].to_numpy()
        ax.barh([MODEL_LABEL[m] for m in pivot.index], vals, left=left,
                color=colors[layer], label=layer, height=0.62)
        left += vals
    ax.set_xlabel("Frozen model calls")
    ax.set_xlim(0, 66000)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.38),
              fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Call budget by model and layer", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_calls.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_calls.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_coverage() -> None:
    d = pd.read_csv(CENSUS, encoding="utf-8-sig")
    d = d[d["structured_rows"] > 0].copy()
    d["stage_label"] = d["layer"].str.replace("nih_eval", "NIH") + "/" + d["stage"]
    # Keep the NIH shift grid and the two native-shift layers; aggregate others.
    d = d[d["stage"].isin(["shift", "reliance", "reliance_bf16"])]
    d["col"] = d["layer"].map({"nih_eval": "NIH", "padchest": "PadChest", "openi": "Open-i"}) + " " + d["stage"]
    mat = d.pivot_table(index="model", columns="col", values="all_definite_rate", aggfunc="mean")
    mat = mat.reindex([m for m in MODEL_LABEL if m in mat.index])
    fig, ax = plt.subplots(figsize=(4.8, 2.5))
    im = ax.imshow(mat.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels([MODEL_LABEL[m] for m in mat.index], fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.to_numpy()[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.55 else "black", fontsize=6.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("All-definite rate", fontsize=7)
    ax.set_title("Schema compliance is not contract coverage")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_coverage.pdf")
    fig.savefig(OUT / "fig2_coverage.png", dpi=300)
    plt.close(fig)


def fig_batch() -> None:
    # Measured on RTX 4080 SUPER 32 GB, docs/46 Table S1.
    data = {
        "MedGemma-4B bf16": ([16, 32, 64, 96], [0.53, 0.47, 0.40, 0.88]),
        "Qwen2.5-VL bf16": ([4, 8, 16, 20], [1.59, 1.31, 1.49, 3.17]),
        "CheXagent-8B fp16": ([2, 4, 6], [13.43, 7.14, 117.11]),
    }
    colors = {"MedGemma-4B bf16": "#0072B2", "Qwen2.5-VL bf16": "#E69F00",
              "CheXagent-8B fp16": "#D55E00"}
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    for name, (xs, ys) in data.items():
        ax.plot(xs, ys, marker="o", ms=4, lw=1.4, color=colors[name], label=name)
        # mark the last point (the collapse)
        ax.annotate("collapse", (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(-4, 8), fontsize=6.5, color=colors[name], ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Seconds per image (log scale)")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Throughput is not monotone in batch size")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_batch.pdf")
    fig.savefig(OUT / "fig3_batch.png", dpi=300)
    plt.close(fig)


def main() -> int:
    fig_calls()
    fig_coverage()
    fig_batch()
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
