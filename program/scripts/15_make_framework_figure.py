#!/usr/bin/env python
"""Generate Figure 1: shift-aware selective trust framework schematic."""

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

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.axis("off")

colors = {
    "data": "#DCE9F7",
    "model": "#D9F0E3",
    "branch": "#FFF2CC",
    "trust": "#F8D7DA",
    "decision": "#E2D9F3",
}

def box(x, y, w, h, text, color, fontsize=12, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                       linewidth=1.4, edgecolor="#333333", facecolor=color)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal", wrap=True)

def arrow(x1, y1, x2, y2, color="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, linewidth=1.5, color=color))

# Row 1
box(0.4, 6.5, 2.4, 1.0, "Medical image\n(chest X-ray)", colors["data"], bold=True)
box(3.8, 6.5, 2.6, 1.0, "Vision–language\nmodel", colors["model"], bold=True)

# Row 2 branches
box(1.0, 4.7, 2.2, 1.0, "Prediction\nstructured findings", colors["branch"])
box(4.0, 4.7, 2.4, 1.0, "Uncertainty\nconsistency", colors["branch"])
box(7.3, 4.7, 2.4, 1.0, "Image reliance\nblank / blur / shuffle", colors["branch"])

# Row 3
box(3.2, 3.0, 2.6, 1.0, "Calibration\nsource-val fitted", colors["trust"])
box(7.0, 3.0, 3.0, 1.0, "Shift-aware\nreliability score", colors["trust"], bold=True)

# Row 4
box(2.2, 1.2, 2.6, 1.0, "Accept\nAI decision", colors["decision"], bold=True)
box(7.4, 1.2, 2.8, 1.0, "Abstain\nhuman review", colors["decision"], bold=True)

# Arrows
arrow(2.8, 7.0, 3.8, 7.0)
arrow(5.1, 6.5, 2.1, 5.7)
arrow(5.1, 6.5, 5.2, 5.7)
arrow(5.1, 6.5, 8.5, 5.7)
arrow(2.1, 4.7, 4.5, 4.0)
arrow(5.2, 4.7, 5.0, 4.0)
arrow(8.5, 4.7, 7.9, 4.0)
arrow(5.8, 3.0, 7.0, 3.5)
arrow(8.5, 3.0, 4.8, 2.2)
arrow(8.5, 3.0, 8.8, 2.2)

ax.text(6.5, 7.55, "Shift-aware Selective Trust Framework", ha="center", va="center",
        fontsize=16, fontweight="bold")
ax.text(6.5, 0.45, "Shift-aware calibration + uncertainty + visual-evidence reliance",
        ha="center", va="center", fontsize=12, style="italic")

for ext in ["png", "pdf"]:
    fig.savefig(OUT / f"fig01_framework.{ext}", bbox_inches="tight", dpi=300)
plt.close(fig)
print("wrote", OUT / "fig01_framework.png")
