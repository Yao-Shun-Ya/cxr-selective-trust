"""Sensitivity analyses requested by the manuscript review.

Five checks, all CPU-only and all reading frozen predictions:

1. ``threshold_sweep`` -- does the collapse verdict depend on the descriptive
   rule (top-share >= 0.5, unique <= 5)? Sweep 0.30-0.80 and 1-20.
2. ``common_labels``  -- PadChest lacks pneumonia; recompute the cross-domain
   AUROC loss on the seven labels both datasets actually annotate, and report
   per-label prevalence on each side.
3. ``null_pairing``   -- is the real-vs-real Jaccard null sensitive to how real
   reports are paired? Compare random pairs, length-matched pairs, and a
   bootstrap interval for the random-pair null.
4. ``bonferroni``     -- widen the transfer delta intervals for the full family
   of estimable model-label cells and report which remain below zero.
5. ``flip_matrix``    -- case-level clean -> contrast/severe transitions for
   InternVL3.5, separating lost true positives from new false positives.

Usage::

    python scripts/81_sensitivity_analyses.py
"""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

RESP = ROOT / "outputs" / "analysis_responsiveness" / "responsiveness_by_cell.csv"
GEO = ROOT / "outputs" / "geographic"
MAN_NIH = ROOT / "data" / "manifests" / "nih_eval_1732.csv"
MAN_PC = ROOT / "data" / "manifests" / "padchest_shift.csv"
MAN_OI = ROOT / "data" / "manifests" / "openi_rrg.csv"

MODELS = ["medgemma_4b_it", "medgemma_15_4b_it", "internvl3_5_8b",
          "qwen3_vl_8b_instruct", "chexagent_8b", "qwen25_vl_7b_instruct"]
LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax", "no_finding"]
SHARED = [l for l in LABELS if l != "pneumonia"]


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", (text or "").lower()))


def jaccard(a: str, b: str) -> float:
    A, B = tokens(a), tokens(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


# --------------------------------------------------------------------------- #
def threshold_sweep() -> pd.DataFrame:
    df = pd.read_csv(RESP, encoding="utf-8-sig")
    rows = []
    for top in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        for uniq in [1, 3, 5, 10, 20]:
            for layer, g in df.groupby("layer"):
                flag = (g["real_top_share_text"] >= top) | (g["real_n_unique_text"] <= uniq)
                rows.append({
                    "top_threshold": top,
                    "unique_threshold": uniq,
                    "layer": layer,
                    "n_collapsed": int(flag.sum()),
                    "n_cells": int(len(g)),
                    "models_collapsed": ";".join(sorted(g.loc[flag, "model"])),
                })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "threshold_sweep.csv", index=False, encoding="utf-8-sig")

    base = ((df["real_top_share_text"] >= 0.50) | (df["real_n_unique_text"] <= 5))
    stable = []
    for (top, uniq), g in tab.groupby(["top_threshold", "unique_threshold"]):
        changed = []
        for layer, gg in g.groupby("layer"):
            src = df[df["layer"] == layer]
            b = ((src["real_top_share_text"] >= 0.50) | (src["real_n_unique_text"] <= 5)).sum()
            changed.append(int(gg["n_collapsed"].iloc[0]) != int(b))
        stable.append({"top": top, "unique": uniq, "layers_changed": int(sum(changed))})
    pd.DataFrame(stable).to_csv(OUT / "threshold_sweep_stability.csv",
                                index=False, encoding="utf-8-sig")
    return tab


# --------------------------------------------------------------------------- #
def common_labels() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        p = GEO / model / "geographic_shift_summary.json"
        if not p.exists():
            continue
        by = json.loads(p.read_text(encoding="utf-8"))["cross_domain_transfer"]["by_label"]
        deltas = [by[l]["delta"] for l in SHARED
                  if l in by and by[l].get("delta") is not None]
        chance = [l for l in SHARED
                  if l in by and by[l].get("target_auroc") is not None
                  and abs(by[l]["target_auroc"] - 0.5) < 1e-9]
        rows.append({
            "model": model,
            "n_shared_labels": len(deltas),
            "mean_delta_shared": float(np.mean(deltas)) if deltas else None,
            "min_delta_shared": float(np.min(deltas)) if deltas else None,
            "n_at_chance": len(chance),
            "pneumonia_present": "pneumonia" in by,
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "common_labels.csv", index=False, encoding="utf-8-sig")

    prev = []
    nih = list(csv.DictReader(open(MAN_NIH, encoding="utf-8-sig")))
    nih = [r for r in nih if r["split"] in {"internal_test", "source_val"}]
    pc = list(csv.DictReader(open(MAN_PC, encoding="utf-8-sig")))
    pc = [r for r in pc if r["split"] == "internal_test"]
    for l in LABELS:
        def rate(rs, d):
            vals = [r.get(f"label_{l}") for r in rs]
            pos = sum(1 for v in vals if v == "positive")
            den = sum(1 for v in vals if v in {"positive", "negative"})
            return (pos / den if den else None), den
        rn, nn = rate(nih, "nih")
        rp, npc = rate(pc, "padchest")
        prev.append({"label": l, "nih_rate": rn, "nih_den": nn,
                     "padchest_rate": rp, "padchest_den": npc})
    pd.DataFrame(prev).to_csv(OUT / "label_prevalence.csv", index=False,
                              encoding="utf-8-sig")
    return tab


# --------------------------------------------------------------------------- #
def null_pairing() -> pd.DataFrame:
    rows = []
    man = {r["image_id"]: r for r in csv.DictReader(open(MAN_OI, encoding="utf-8-sig"))}
    for model in MODELS:
        p = ROOT / "outputs" / "matrix" / "openi" / model / "shift" / "rrg_predictions.jsonl"
        if not p.exists():
            continue
        real = {}
        blank = {}
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if r["condition"] == "original":
                real[r["image_id"]] = (r.get("raw_output") or "").strip()
            elif r["condition"] == "blank":
                blank[r["image_id"]] = (r.get("raw_output") or "").strip()
        ids = sorted(set(real) & set(blank))
        rng = random.Random(0)

        # 1. random pairs (discard self-pairs)
        rand_pairs = []
        for _ in range(400):
            a, b = rng.choice(ids), rng.choice(ids)
            if a != b:
                rand_pairs.append((a, b))
        j_rand = np.mean([jaccard(real[a], real[b]) for a, b in rand_pairs])

        # 2. length-matched pairs: same decile of reference report length
        lens = {i: len((man.get(i, {}).get("reference_text") or "").split()) for i in ids}
        order = sorted(ids, key=lambda i: lens[i])
        deciles = np.array_split(order, 10)
        lm = []
        for _ in range(400):
            bucket = rng.choice(deciles)
            if len(bucket) < 2:
                continue
            a, b = rng.sample(list(bucket), 2)
            lm.append((a, b))
        j_len = np.mean([jaccard(real[a], real[b]) for a, b in lm])

        # 3. bootstrap interval on the random-pair null
        boot = []
        for _ in range(1000):
            pick = [rand_pairs[rng.randrange(len(rand_pairs))] for _ in range(len(rand_pairs))]
            boot.append(np.mean([jaccard(real[a], real[b]) for a, b in pick]))
        lo, hi = np.percentile(boot, [2.5, 97.5])

        j_rb = np.mean([jaccard(real[i], blank[i]) for i in ids])
        rows.append({
            "model": model, "n": len(ids),
            "j_real_blank": j_rb,
            "null_random": j_rand,
            "null_length_matched": j_len,
            "null_random_lo": lo, "null_random_hi": hi,
            "gap_random": j_rb - j_rand,
            "gap_length_matched": j_rb - j_len,
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "null_pairing.csv", index=False, encoding="utf-8-sig")
    return tab


# --------------------------------------------------------------------------- #
def bonferroni() -> pd.DataFrame:
    raw = []
    for model in MODELS:
        p = GEO / model / "geographic_shift_summary.json"
        if not p.exists():
            continue
        boot = json.loads(p.read_text(encoding="utf-8"))["cross_domain_transfer"].get("bootstrap", {})
        for label, e in boot.items():
            lo, hi = e.get("delta_ci_conservative", [None, None])
            if lo is None:
                continue
            point = e["target"]["point"] - e["source"]["point"]
            half = (hi - lo) / 2.0
            se = half / 1.959963984540054
            raw.append({"model": model, "label": label, "delta": point,
                        "ci95_lo": lo, "ci95_hi": hi, "se": se})
    # Family = every estimable model-label cell actually reported, not a
    # hand-picked count; this keeps the correction at least as strict as the
    # number of tests a reader can count in the table.
    n_tests = len(raw)
    z_b = NormalDist().inv_cdf(1.0 - 0.025 / n_tests)
    rows = []
    for r in raw:
        lo_b = r["delta"] - z_b * r["se"]
        hi_b = r["delta"] + z_b * r["se"]
        rows.append({
            "model": r["model"], "label": r["label"], "delta": r["delta"],
            "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"],
            "ci_bonf_lo": lo_b,
            "ci_bonf_hi": hi_b,
            "z_bonferroni": z_b,
            "n_tests": n_tests,
            "excludes_zero_bonf": bool(hi_b < 0 or lo_b > 0),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "transfer_bonferroni.csv", index=False, encoding="utf-8-sig")
    return tab


# --------------------------------------------------------------------------- #
def flip_matrix() -> pd.DataFrame:
    p = ROOT / "outputs" / "matrix" / "nih_eval" / "internvl3_5_8b" / "shift" / "shift_predictions.jsonl"
    clean, sev = {}, {}
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        if r["condition"] == "clean":
            clean[r["image_id"]] = r
        elif r["condition"] == "contrast" and r["severity"] == "severe":
            sev[r["image_id"]] = r
    rows = []
    for label in LABELS:
        t = {"gt_pos_kept": 0, "gt_pos_lost": 0, "gt_pos_gained": 0,
             "gt_neg_corrected": 0, "gt_neg_new_fp": 0, "gt_neg_stable": 0}
        for iid in set(clean) & set(sev):
            c = clean[iid]["parsed"]["findings"].get(label, {})
            s = sev[iid]["parsed"]["findings"].get(label, {})
            cp, sp = c.get("status") == "positive", s.get("status") == "positive"
            if cp is None or sp is None:
                continue
            gt = clean[iid]["ground_truth"].get(f"label_{label}")
            if gt == "positive":
                if cp and sp:
                    t["gt_pos_kept"] += 1
                elif cp and not sp:
                    t["gt_pos_lost"] += 1
                elif not cp and sp:
                    t["gt_pos_gained"] += 1
            if gt == "negative":
                if cp and not sp:
                    t["gt_neg_corrected"] += 1
                elif not cp and sp:
                    t["gt_neg_new_fp"] += 1
                elif not cp and not sp:
                    t["gt_neg_stable"] += 1
        rows.append({"label": label, **t})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "flip_matrix_internvl_contrast_severe.csv", index=False,
               encoding="utf-8-sig")
    return tab


if __name__ == "__main__":
    print("== threshold sweep ==")
    print(threshold_sweep().tail(10).to_string(index=False))
    print("\n== common labels ==")
    print(common_labels().to_string(index=False))
    print("\n== null pairing ==")
    print(null_pairing().round(3).to_string(index=False))
    print("\n== bonferroni ==")
    print(bonferroni().round(3).to_string(index=False))
    print("\n== flip matrix ==")
    print(flip_matrix().to_string(index=False))
