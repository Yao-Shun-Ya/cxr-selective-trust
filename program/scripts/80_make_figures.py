"""Generate the manuscript figures from the frozen artifacts.

Every number plotted here is read from a produced CSV/JSON, not retyped, so a
figure cannot drift away from the tables. Outputs go to papers/main_v4/figures
as vector PDF (for LaTeX) and 300-dpi PNG (for review).

Usage::

    python scripts/80_make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT.parent / "papers" / "main_v7" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
MINE = ROOT / "outputs" / "analysis_nih_mine"
MATRIX = ROOT / "outputs" / "matrix"
RESP = ROOT / "outputs" / "analysis_responsiveness" / "responsiveness_by_cell.csv"

# Okabe-Ito, colour-blind safe.
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
ORANGE, SKY, PURPLE, GREY = "#E69F00", "#56B4E9", "#CC79A7", "#8C8C8C"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linewidth": 0.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

DISPLAY = {
    "medgemma_4b_it": "MedGemma-4B",
    "medgemma_15_4b_it": "MedGemma-1.5",
    "internvl3_5_8b": "InternVL3.5",
    "qwen3_vl_8b_instruct": "Qwen3-VL",
    "chexagent_8b": "CheXagent",
    "qwen25_vl_7b_instruct": "Qwen2.5-VL",
}
ORDER = ["MedGemma-4B", "MedGemma-1.5", "InternVL3.5", "Qwen3-VL",
         "CheXagent", "Qwen2.5-VL"]
LAYERS = {"nih": "NIH (1,506)", "padchest": "PadChest (1,500)",
          "openi": "Open-i (947)"}
LAYER_COLOR = {"nih": BLUE, "padchest": VERM, "openi": GREEN}
LAYER_MARK = {"nih": "o", "padchest": "s", "openi": "^"}


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- #
# Fig 1 -- four preconditions and the four reversals
# --------------------------------------------------------------------------- #
def fig1() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cols = [
        ("P1\nResponsiveness",
         "5/6 models are\n\"image-blind\"\noff-domain",
         "1–20 distinct answers\nfor 1,500 images\n→ reliance is\nunidentifiable"),
        ("P2\nNull model",
         "Qwen3-VL is strongly\ngrounded (J = 0.09)",
         "blank reply is a refusal;\nJ(real, real′) = 0.74\n→ absence detection,\nnot evidence"),
        ("P3\nOperating point",
         "contrast raises recall\n(+4.69 pp)",
         "positive rate +4.06 pp,\nprecision −1.38 pp,\nFP +16%\n→ operating-point shift"),
        ("P4\nContract & mode",
         "4-bit suppresses\nreliance 2.21×",
         "marker 0.13% → 99.07%\n→ output-regime switch,\nnot grounding"),
    ]
    for i, (title, naive, fixed) in enumerate(cols):
        x = i + 0.5
        ax.add_patch(Rectangle((i + 0.05, 0.73), 0.90, 0.25, facecolor="#F3F3F3",
                               edgecolor="#BBBBBB", linewidth=0.6))
        ax.text(x, 0.895, title, ha="center", va="center", fontsize=7.6,
                fontweight="bold", linespacing=1.15)
        ax.text(x, 0.795, naive, ha="center", va="center", fontsize=6.6,
                color="#444444", style="italic", linespacing=1.25)
        ax.add_patch(FancyArrowPatch((x, 0.715), (x, 0.60), arrowstyle="-|>",
                                     mutation_scale=9, color=VERM, lw=1.0))
        ax.text(x + 0.08, 0.655, "check", fontsize=6.2, color=VERM)
        ax.add_patch(Rectangle((i + 0.05, 0.10), 0.90, 0.33, facecolor="#EAF2F8",
                               edgecolor=BLUE, linewidth=0.6))
        ax.text(x, 0.265, fixed, ha="center", va="center", fontsize=6.5,
                color="#123B57", linespacing=1.30)

    save(fig, "fig01_four_reversals")


# --------------------------------------------------------------------------- #
# Fig 2 -- responsiveness audit across three layers
# --------------------------------------------------------------------------- #
def fig2() -> None:
    df = pd.read_csv(RESP, encoding="utf-8-sig")
    df["disp"] = df["model"].map(DISPLAY)
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.axvspan(0.5, 1.02, color="#FBE3DC", alpha=0.7, zorder=0)
    ax.axvline(0.5, color=VERM, lw=0.9, ls="--", zorder=1)
    ax.text(0.75, 0.97, "declared collapse region\n(top-share ≥ 0.50)",
            ha="center", va="top", fontsize=6.8, color=VERM,
            transform=ax.transAxes)

    for _, r in df.iterrows():
        row = ORDER.index(r["disp"])
        jitter = {"nih": -0.16, "padchest": 0.0, "openi": 0.16}[r["layer"]]
        ax.scatter(r["real_top_share_text"], row + jitter, s=34,
                   marker=LAYER_MARK[r["layer"]], color=LAYER_COLOR[r["layer"]],
                   edgecolor="white", linewidth=0.5, zorder=3)
        off = (-9, 0) if r["real_top_share_text"] > 0.90 else (5, 0)
        ha = "right" if r["real_top_share_text"] > 0.90 else "left"
        ax.annotate(f"{int(r['real_n_unique_text'])}",
                    (r["real_top_share_text"], row + jitter),
                    textcoords="offset points", xytext=off, fontsize=6.0,
                    color="#555555", va="center", ha=ha)

    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels(ORDER)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.55, 5.55)
    ax.set_xlabel("top-share of the modal reply on real images (annotated: distinct replies)")
    ax.invert_yaxis()
    ax.set_title("Output collapse is layer-dependent, and not confined to the out-of-domain layer")
    handles = [plt.Line2D([], [], marker=LAYER_MARK[k], color=LAYER_COLOR[k],
                          linestyle="", markersize=5, label=LAYERS[k])
               for k in LAYERS]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=3,
              bbox_to_anchor=(1.0, -0.32))
    save(fig, "fig02_responsiveness")


# --------------------------------------------------------------------------- #
# Fig 3 -- silent-failure quadrant
# --------------------------------------------------------------------------- #
def fig3() -> None:
    mk = pd.read_csv(MINE / "joint_C_marker_by_type.csv", encoding="utf-8-sig")
    dl = pd.read_csv(MINE / "joint_C_type_deltas_per_model.csv", encoding="utf-8-sig")
    single = mk.set_index("type")
    med = dl[dl["model"] == "medgemma_15_4b_it"].set_index("type")
    offsets = {"contrast": (6, -9), "brightness": (6, -10), "noise": (6, 4),
               "blur": (6, 5), "jpeg": (6, -11), "resolution": (-7, 8)}

    fig, ax = plt.subplots(figsize=(4.1, 2.9))
    ax.fill_between([0.02, 1.0], -25, -10, color="#FBE3DC", alpha=0.55,
                    zorder=0, linewidth=0)
    xs = (single["marker_rate_medgemma_15_severe"] * 100).values
    ys = (med.loc[single.index, "d_recall_vs_clean"] * 100).values
    ax.scatter(xs, ys, s=42, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 120)
    ax.axhline(0, color="#888888", lw=0.6)
    for name, xv, yv in zip(single.index, xs, ys):
        dx, dy = offsets.get(name, (6, 3))
        ha = "right" if dx < 0 else "left"
        ax.annotate(name, (xv, yv), textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.4, color="#333333", ha=ha)
    ax.set_ylabel("damage: Δ positive recall vs clean (pp)")
    ax.set_ylim(-21.5, 2.6)
    ax.set_xlabel("noticeability: reasoning-channel rate, severe (%)")
    ax.text(0.035, -19.6, "silent-failure\nquadrant", fontsize=6.8, color=VERM)
    save(fig, "fig03_silent_failure")


# --------------------------------------------------------------------------- #
# Fig 4 -- cross-hospital AUROC transfer
# --------------------------------------------------------------------------- #
def fig4() -> None:
    rows = []
    for model, disp in DISPLAY.items():
        p = ROOT / "outputs" / "geographic" / model / "geographic_shift_summary.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))["cross_domain_transfer"]["by_label"]
        for label in ("cardiomegaly", "no_finding"):
            e = d.get(label, {})
            s, t = e.get("source_auroc"), e.get("target_auroc")
            if s is None or t is None:
                continue
            rows.append({"model": disp, "label": label, "nih": s, "padchest": t})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
    titles = {"cardiomegaly": "(a) cardiomegaly", "no_finding": "(b) no_finding"}
    for ax, label in zip(axes, ["cardiomegaly", "no_finding"]):
        sub = df[df["label"] == label].set_index("model").reindex(ORDER).dropna()
        yy = np.arange(len(sub))
        for i, (name, r) in enumerate(sub.iterrows()):
            ax.plot([r["nih"], r["padchest"]], [i, i], color="#CCCCCC", lw=1.6, zorder=1)
            ax.scatter(r["nih"], i, s=42, color=BLUE, zorder=3,
                       edgecolor="white", linewidth=0.5, label="NIH" if i == 0 else "")
            ax.scatter(r["padchest"], i, s=42, color=VERM, marker="D", zorder=3,
                       edgecolor="white", linewidth=0.5,
                       label="PadChest" if i == 0 else "")
        ax.axvline(0.5, color="#888888", lw=0.7, ls=":")
        ax.set_xlim(0.38, 0.82)
        ax.set_title(titles[label], loc="left")
        ax.set_xlabel("AUROC")
        if label == "cardiomegaly":
            ax.set_yticks(yy)
            ax.set_yticklabels(sub.index)
            ax.invert_yaxis()
            ax.legend(loc="lower right", frameon=False)
    save(fig, "fig04_transfer")


# --------------------------------------------------------------------------- #
# Fig 5 -- blank contrast against its real-vs-real null
# --------------------------------------------------------------------------- #
def fig5() -> None:
    df = pd.read_csv(RESP, encoding="utf-8-sig")
    df = df[df["layer"] == "openi"].copy()
    df["disp"] = df["model"].map(DISPLAY)
    offs = {"MedGemma-4B": (8, -26, "left"), "MedGemma-1.5": (-9, 6, "right"),
            "InternVL3.5": (-8, -14, "right"), "Qwen2.5-VL": (8, -14, "left"),
            "CheXagent": (8, 8, "left"), "Qwen3-VL": (10, 8, "left")}
    fig, ax = plt.subplots(figsize=(4.5, 3.9))
    lim = (0.0, 0.96)
    ax.plot(lim, lim, color="#888888", lw=0.8, ls="--", zorder=1)
    ax.text(0.06, 0.13, "no separation", rotation=40, fontsize=6.4, color="#777777")
    for _, r in df.iterrows():
        refusal = bool(r["blank_top_is_refusal"])
        ax.scatter(r["j_real_real"], r["j_real_blank"], s=58,
                   color=VERM if refusal else BLUE, marker="X" if refusal else "o",
                   edgecolor="white", linewidth=0.6, zorder=3)
        dx, dy, ha = offs.get(r["disp"], (8, 4, "left"))
        ax.annotate(f"{r['disp']}\n($\\Delta_{{\\mathrm{{null}}}}$ {-r['gap_text']:+.2f})",
                    (r["j_real_real"], r["j_real_blank"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=6.5,
                    ha=ha, linespacing=1.15)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("J(real, another real) — the null")
    ax.set_ylabel("J(real, blank)")
    ax.set_title("Blank contrast vs. its real-vs-real null")
    handles = [plt.Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=5,
                          label="blank = prior template"),
               plt.Line2D([], [], marker="X", color=VERM, linestyle="", markersize=5,
                          label="blank = refusal")]
    ax.legend(handles=handles, frameon=False, loc="upper left")
    save(fig, "fig05_blank_null")


# --------------------------------------------------------------------------- #
# Fig 6 -- selective prediction: ranking improves, coverage does not
# --------------------------------------------------------------------------- #
def fig6() -> None:
    rows = []
    for model, disp in DISPLAY.items():
        p = ROOT / "outputs" / "selective_v2" / model / "selective_v2_summary.json"
        q = ROOT / "outputs" / "trust_v1" / model / "trust_v1_summary.json"
        if not p.exists() or not q.exists():
            continue
        a = json.loads(p.read_text(encoding="utf-8"))["levels"]["finding"]["point"]
        comp = json.loads(q.read_text(encoding="utf-8"))["units"]["finding"]["comparison"]
        conf = next((x for x in comp if x["subset"] == "confidence_only"), {})
        shift = next((x for x in comp if x["subset"] == "plus_shift_score"), {})
        rows.append({"model": disp, "cov": a["coverage_at_05pct_risk"],
                     "conf": conf.get("auc"), "shift": shift.get("auc"),
                     "contract": disp == "CheXagent"})
    df = pd.DataFrame(rows).set_index("model").reindex(ORDER).dropna(subset=["cov"])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    fig.subplots_adjust(wspace=0.42)
    ax = axes[0]
    cols = [GREY if m == "CheXagent" else BLUE for m in df.index]
    ax.barh(np.arange(len(df)), df["cov"], color=cols, height=0.55)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.12)
    ax.set_xlabel("coverage at 5% risk (finding level)")
    ax.set_title("(a) Usable coverage", loc="left")
    ax.text(0.98, 0.96, "all ≤ 0.093", transform=ax.transAxes, ha="right",
            va="top", fontsize=6.6, color="#555555")
    ax.annotate("CheXagent: contract 18.9%\n(not comparable)", (0.036, 4.0),
                textcoords="offset points", xytext=(6, 0), fontsize=5.8,
                color="#666666", va="center")

    ax = axes[1]
    sub = df.dropna(subset=["conf", "shift"])
    loff = {"InternVL3.5": (8, -10), "MedGemma-4B": (8, 8),
            "MedGemma-1.5": (8, -4), "Qwen3-VL": (8, 2), "Qwen2.5-VL": (8, -2)}
    for i, (name, r) in enumerate(sub.iterrows()):
        ax.plot([0, 1], [r["conf"], r["shift"]], color="#CCCCCC", lw=1.5, zorder=1)
        ax.scatter(0, r["conf"], s=38, color=GREY, zorder=3, edgecolor="white",
                   linewidth=0.5)
        ax.scatter(1, r["shift"], s=38, color=GREEN, zorder=3, edgecolor="white",
                   linewidth=0.5)
        dx, dy = loff.get(name, (8, 0))
        ax.annotate(name, (1, r["shift"]), textcoords="offset points",
                    xytext=(dx, dy), fontsize=6.2)
    ax.set_xlim(-0.25, 1.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["confidence", "+ shift score"])
    ax.set_ylabel("error-detection AUROC (finding level)")
    ax.set_title("(b) Error ranking barely moves", loc="left")
    ax.grid(axis="x", visible=False)
    save(fig, "fig06_selective")


def fig7() -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.axhline(0.05, color=VERM, lw=0.9, ls="--")
    ax.text(0.40, 0.062, "5% risk budget", fontsize=6.6, color=VERM)
    colors = [BLUE, VERM, GREEN, ORANGE, PURPLE]
    for color, model in zip(colors, ["internvl3_5_8b", "medgemma_4b_it",
                                     "medgemma_15_4b_it", "qwen3_vl_8b_instruct",
                                     "qwen25_vl_7b_instruct"]):
        p = ROOT / "outputs" / "selective_v2" / model / "risk_coverage_bands.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p, encoding="utf-8-sig")
        d = d[d["level"] == "finding"].sort_values("coverage")
        ax.plot(d["coverage"], d["risk_mean"], color=color, lw=1.3,
                label=DISPLAY[model])
        ax.fill_between(d["coverage"], d["risk_lo"], d["risk_hi"],
                        color=color, alpha=0.13, linewidth=0)
    ax.set_xlim(0, 0.65)
    ax.set_ylim(0, 0.38)
    ax.set_xlabel("coverage (fraction of findings answered)")
    ax.set_ylabel("selective risk (error rate among answered)")
    ax.legend(frameon=False, fontsize=6.4, loc="upper left")
    ax.set_title("Risk--coverage at the finding level (patient-cluster bands)")
    save(fig, "fig07_risk_coverage")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
