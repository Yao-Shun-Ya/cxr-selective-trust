#!/usr/bin/env python
"""Generate Figure 9 (selective trust method) and Figure 10 (clinical workflow)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent
OUT = PAPER / "figures" / "stage2"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, color, fontsize=12, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                       linewidth=1.4, edgecolor="#222222", facecolor=color)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, linewidth=1.5, color="#222222"))


def figure_method():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis("off")

    c = {"data": "#DCE9F7", "model": "#D9F0E3", "branch": "#FFF2CC",
         "trust": "#F8D7DA", "decision": "#E2D9F3", "monitor": "#E8F5E9"}

    box(ax, 0.3, 7.7, 2.4, 0.9, "Medical image\n(CXR)", c["data"], bold=True)
    box(ax, 3.3, 7.7, 2.7, 0.9, "Vision–language\nmodel", c["model"], bold=True)
    box(ax, 6.8, 7.7, 3.2, 0.9, "Structured prediction\nfindings + probabilities", c["branch"])

    box(ax, 0.8, 5.6, 2.6, 0.9, "Self-consistency\nprompt / sampling", c["branch"])
    box(ax, 4.2, 5.6, 2.6, 0.9, "Image perturbation\nconsistency", c["branch"])
    box(ax, 7.6, 5.6, 2.8, 0.9, "Image reliance\nblank / blur / shuffle", c["branch"])
    box(ax, 11.0, 5.6, 2.5, 0.9, "Shift / OOD\nscore", c["branch"])

    box(ax, 2.2, 3.6, 3.0, 0.9, "Calibration\nsource-val fitted", c["trust"])
    box(ax, 6.5, 3.6, 3.4, 0.9, "Uncertainty /\nreliability aggregation", c["trust"])
    box(ax, 10.6, 3.6, 2.8, 0.9, "Shift-aware\ntrust score", c["trust"], bold=True)

    box(ax, 1.6, 1.6, 3.0, 0.9, "Accept\nAI decision", c["decision"], bold=True)
    box(ax, 5.4, 1.6, 3.2, 0.9, "Clarify / Escalate\nadditional evidence", c["decision"], bold=True)
    box(ax, 9.4, 1.6, 3.2, 0.9, "Abstain\nhuman review", c["decision"], bold=True)

    box(ax, 5.2, 0.2, 3.6, 0.7, "Deferral feedback / audit log", c["monitor"], fontsize=11)

    # arrows
    arrow(ax, 2.7, 8.15, 3.3, 8.15)
    arrow(ax, 6.0, 8.15, 6.8, 8.15)
    arrow(ax, 8.4, 7.7, 2.1, 6.5)
    arrow(ax, 8.4, 7.7, 5.5, 6.5)
    arrow(ax, 8.4, 7.7, 9.0, 6.5)
    arrow(ax, 8.4, 7.7, 12.2, 6.5)
    arrow(ax, 2.1, 5.6, 3.7, 4.5)
    arrow(ax, 5.5, 5.6, 8.2, 4.5)
    arrow(ax, 9.0, 5.6, 8.2, 4.5)
    arrow(ax, 12.2, 5.6, 12.0, 4.5)
    arrow(ax, 10.6, 4.05, 10.6, 4.05)  # placeholder
    arrow(ax, 9.9, 3.6, 3.1, 2.5)
    arrow(ax, 9.9, 3.6, 7.0, 2.5)
    arrow(ax, 9.9, 3.6, 11.0, 2.5)
    arrow(ax, 3.1, 1.6, 7.0, 0.9)
    arrow(ax, 7.0, 1.6, 7.0, 0.9)
    arrow(ax, 11.0, 1.6, 7.0, 0.9)

    ax.text(7.0, 8.75, "Shift-aware selective trust method", ha="center", va="center",
            fontsize=17, fontweight="bold")
    ax.text(7.0, 0.05, "Uncertainty, calibration, shift, and visual-evidence reliance jointly determine accept / clarify / abstain.",
            ha="center", va="center", fontsize=11, style="italic")
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"fig09_selective_trust_method.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def figure_clinical():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    c = {"low": "#D9F0E3", "mid": "#FFF2CC", "high": "#F8D7DA", "human": "#E2D9F3"}

    box(ax, 0.3, 3.0, 2.4, 0.9, "CXR study", c["low"], bold=True)
    box(ax, 3.2, 3.0, 3.0, 0.9, "VLM risk estimation", c["mid"], bold=True)

    box(ax, 7.0, 5.2, 2.6, 0.9, "Low risk\nAI auto-read", c["low"], bold=True)
    box(ax, 7.0, 3.0, 2.6, 0.9, "Intermediate risk\nAI + human review", c["mid"], bold=True)
    box(ax, 7.0, 0.8, 2.6, 0.9, "High risk\nimmediate human read", c["high"], bold=True)
    box(ax, 10.3, 3.0, 2.3, 0.9, "Radiologist\nworklist", c["human"], bold=True)

    arrow(ax, 2.7, 3.45, 3.2, 3.45)
    arrow(ax, 6.2, 3.6, 7.0, 5.6)
    arrow(ax, 6.2, 3.45, 7.0, 3.45)
    arrow(ax, 6.2, 3.3, 7.0, 1.25)
    arrow(ax, 9.6, 3.45, 10.3, 3.45)
    arrow(ax, 9.6, 5.2, 10.3, 3.5)
    arrow(ax, 9.6, 1.25, 10.3, 3.0)

    ax.text(6.5, 6.6, "Selective clinical deployment workflow", ha="center", va="center",
            fontsize=17, fontweight="bold")
    ax.text(6.5, 0.15, "Low-risk cases are automated; uncertain or high-risk cases are deferred to human readers.",
            ha="center", va="center", fontsize=11, style="italic")
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"fig10_clinical_workflow.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    figure_method()
    figure_clinical()
    print("wrote figures 9 and 10 to", OUT)
