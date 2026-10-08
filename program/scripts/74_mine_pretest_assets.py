#!/usr/bin/env python
"""74 — deep-mine the 100-image pre-run for paper assets beyond docs/50 sec 7.

Read-only over ``outputs/pretest`` (never writes into it). Emits:

    outputs/state/pretest_mined_assets.json

All numbers use the SAME primitives the pipeline ships
(``trust.probability_vector`` / ``status_vector``), so no new metric convention.

Assets:
  A. image-reliance spectrum   VRS (mean |dP| vs original) + status-flip,
                               per perturbation, per model and precision.
  B. degradation dose-response macro accuracy, positive-recall, and the
                               parse-failure rate, per 19 shift cells.
  C. no_finding conservatism   P(model reports "no finding") per severity.

Usage:
    .venv-vl\\Scripts\\python.exe scripts\\74_mine_pretest_assets.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
from cxr_trust.evaluation import trust  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRETEST = ROOT / "outputs" / "pretest" / "nih_eval"
OUT = ROOT / "outputs" / "state" / "pretest_mined_assets.json"

LABELS = trust.DEFAULT_LABELS
PATH = LABELS[:7]  # pathology labels, excluding no_finding
GT_COLS = [f"label_{lb}" for lb in LABELS]

CONDITIONS = ["blur", "brightness", "contrast", "jpeg", "noise", "resolution"]
SEVERITIES = ["none", "mild", "moderate", "severe"]
CELLS = (["clean__none"] +
         [f"{c}__{s}" for c in CONDITIONS for s in SEVERITIES[1:]])


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def model_dirs():
    return sorted(p.name for p in PRETEST.iterdir() if p.is_dir())


def rel_paths(model: str):
    d = PRETEST / model
    return {
        "reliance": d / "reliance" / "image_reliance_predictions.jsonl",
        "reliance_bf16": d / "reliance_bf16" / "image_reliance_predictions.jsonl",
        "shift": d / "shift" / "shift_predictions.jsonl",
    }


def gt_binary(row):
    g = row.get("ground_truth") or {}
    return np.asarray([1.0 if g.get(c) == "positive" else 0.0 if g.get(c) == "negative" else np.nan
                       for c in GT_COLS], dtype=float)


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else None


def _vrs(orig, pert):
    base = trust.probability_vector(orig, LABELS)
    if pert is None or np.all(np.isnan(base)):
        return None
    other = trust.probability_vector(pert, LABELS)
    both = ~np.isnan(base) & ~np.isnan(other)
    if not both.any():
        return None
    return float(np.mean(np.abs(base[both] - other[both])))


def _flip(orig, pert):
    base = trust.status_vector(orig, LABELS)
    if pert is None:
        return None
    other = trust.status_vector(pert, LABELS)
    both = ~np.isnan(base) & ~np.isnan(other)
    if not both.any():
        return None
    return float(np.mean(base[both] != other[both]))


# --------------------------------------------------------------------------- #
def asset_reliance():
    spectrum = {}
    for model in model_dirs():
        for tag, key in (("reliance", "reliance"), ("reliance_bf16", "reliance_bf16")):
            p = rel_paths(model)[key]
            if not p.exists():
                continue
            by_img = defaultdict(dict)
            for r in read_jsonl(p):
                c = r.get("condition")
                if c in ("original", "blank", "blur", "shuffle"):
                    by_img[r["image_id"]][c] = r.get("parsed") or {}

            vrs = {c: [] for c in ("blank", "blur", "shuffle")}
            flip = {c: [] for c in ("blank", "blur", "shuffle")}
            n = 0
            for img, conds in by_img.items():
                orig = conds.get("original")
                if not orig:
                    continue
                n += 1
                for c in ("blank", "blur", "shuffle"):
                    v = _vrs(orig, conds.get(c))
                    if v is not None:
                        vrs[c].append(v)
                    f = _flip(orig, conds.get(c))
                    if f is not None:
                        flip[c].append(f)
            all_vrs = _mean([x for c in ("blank", "blur", "shuffle") for x in vrs[c]]) if False else {
                k: _mean(v) for k, v in vrs.items()}
            spectrum.setdefault(model, {})[tag] = {
                "n_images": n,
                "vrs": all_vrs,
                "status_flip": {k: _mean(v) for k, v in flip.items()},
            }
    return spectrum


# --------------------------------------------------------------------------- #
def asset_dose_response():
    out = {}
    for model in model_dirs():
        p = rel_paths(model)["shift"]
        if not p.exists():
            continue
        rows = read_jsonl(p)
        cells = {c: {"S": [], "G": [], "n_attempted": 0, "n_parse_ok": 0} for c in CELLS}
        for r in rows:
            cell = f"{r.get('condition')}__{r.get('severity')}"
            if cell not in cells:
                continue
            cells[cell]["n_attempted"] += 1
            parsed = r.get("parsed") or {}
            if not parsed.get("parse_success"):
                continue
            cells[cell]["n_parse_ok"] += 1
            cells[cell]["S"].append(trust.status_vector(parsed, LABELS))
            cells[cell]["G"].append(gt_binary(r))

        out[model] = {}
        for cell in CELLS:
            c = cells[cell]
            n_att = c["n_attempted"]
            if not c["S"]:
                out[model][cell] = {
                    "n_attempted": n_att, "n_parse_ok": 0, "parse_fail_rate": 1.0,
                    "macro_acc": None, "path_acc": None, "path_recall": None,
                }
                continue
            S = np.vstack(c["S"])
            G = np.vstack(c["G"])
            def per_label_acc(subset):
                accs = []
                for i in subset:
                    m = ~np.isnan(S[:, i]) & ~np.isnan(G[:, i])
                    if m.sum():
                        accs.append(float(np.mean(S[m, i] == G[m, i])))
                return float(np.mean(accs)) if accs else None
            def path_recall():
                rec = []
                for i in range(7):
                    m = ~np.isnan(S[:, i]) & ~np.isnan(G[:, i])
                    pos = m & (G[:, i] == 1.0)
                    if pos.sum():
                        rec.append(float(np.mean(S[pos, i] == 1.0)))
                return float(np.mean(rec)) if rec else None
            out[model][cell] = {
                "n_attempted": n_att,
                "n_parse_ok": int(c["n_parse_ok"]),
                "parse_fail_rate": round(1.0 - c["n_parse_ok"] / n_att, 4) if n_att else None,
                "macro_acc": per_label_acc(range(8)),
                "path_acc": per_label_acc(range(7)),
                "path_recall": path_recall(),
            }
    return out


# --------------------------------------------------------------------------- #
def asset_no_finding_drift():
    out = {}
    for model in model_dirs():
        p = rel_paths(model)["shift"]
        if not p.exists():
            continue
        sev = {s: {"definite": 0, "positive": 0} for s in SEVERITIES}
        for r in read_jsonl(p):
            parsed = r.get("parsed") or {}
            if not parsed.get("parse_success"):
                continue
            s = r.get("severity")
            if s not in sev:
                continue
            st = ((parsed.get("findings") or {}).get("no_finding") or {}).get("status")
            if st in ("positive", "negative"):
                sev[s]["definite"] += 1
                if st == "positive":
                    sev[s]["positive"] += 1
        out[model] = {
            s: {"definite": v["definite"],
                "no_finding_positive_rate": (v["positive"] / v["definite"]) if v["definite"] else None}
            for s, v in sev.items()
        }
    return out


def main():
    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": "outputs/pretest",
        "labels": LABELS,
        "cells": CELLS,
        "image_reliance_spectrum": asset_reliance(),
        "degradation_dose_response": asset_dose_response(),
        "no_finding_conservatism_drift": asset_no_finding_drift(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    f = lambda x: "  .  " if x is None else f"{x:.3f}"
    f2 = lambda x: "  . " if x is None else f"{x:.2f}"

    print("A. image-reliance spectrum (VRS = mean |dP| vs original; higher = reads the image)")
    print(f"{'model':22} {'tag':14} {'n':>4} {'blank':>7} {'blur':>7} {'shuf':>7} {'flip_b':>7} {'flip_l':>7} {'flip_s':>7}")
    for model, tags in sorted(result["image_reliance_spectrum"].items()):
        for tag, e in tags.items():
            v, fl = e["vrs"], e["status_flip"]
            print(f"{model:22} {tag:14} {e['n_images']:>4} {f(v['blank']):>7} {f(v['blur']):>7} "
                  f"{f(v['shuffle']):>7} {f(fl['blank']):>7} {f(fl['blur']):>7} {f(fl['shuffle']):>7}")

    print()
    print("B. dose-response: per cell  n_attempted / n_parse_ok / parse_fail / macro_acc / path_acc / path_recall")
    for model, cells in sorted(result["degradation_dose_response"].items()):
        print(f"\n  {model}")
        for cell in CELLS:
            c = cells[cell]
            print(f"    {cell:22} {c['n_attempted']:>3}/{c['n_parse_ok']:<3} fail={f(c['parse_fail_rate']):>6} "
                  f"acc={f2(c['macro_acc']):>5} pacc={f2(c['path_acc']):>5} prec={f2(c['path_recall']):>5}")

    print()
    print("C. no_finding conservatism drift  P(model says 'no finding') by severity")
    for model, sev in sorted(result["no_finding_conservatism_drift"].items()):
        vals = [f(sev[s]["no_finding_positive_rate"]) for s in SEVERITIES]
        print(f"  {model:22} none={vals[0]:>6} mild={vals[1]:>6} mod={vals[2]:>6} severe={vals[3]:>6}")

    print()
    print(f"ledger -> {OUT}")


if __name__ == "__main__":
    main()