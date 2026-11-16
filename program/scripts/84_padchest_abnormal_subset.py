#!/usr/bin/env python
"""Responsiveness on the disease-positive PadChest subset.

Why this script exists
----------------------
The full PadChest layer is dominated by normal studies (17.7% carry at least
one disease label), so a reviewer can ask whether the five-model output
collapse is just the correct ``always normal'' prior.  This script repeats the
audit on the 265 studies with at least one positive disease label, using the
same normalization, collapse rule, and null as scripts/78_responsiveness_audit.py.

Output: outputs/analysis_sensitivity/padchest_disease_subset.csv
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis_sensitivity"
MATRIX = ROOT / "outputs" / "matrix"
MAN = ROOT / "data" / "manifests" / "padchest_shift.csv"

MODELS = [
    "medgemma_4b_it", "medgemma_15_4b_it", "internvl3_5_8b",
    "qwen3_vl_8b_instruct", "chexagent_8b", "qwen25_vl_7b_instruct",
]


def load_audit():
    spec = importlib.util.spec_from_file_location(
        "aud", ROOT / "scripts" / "78_responsiveness_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    aud = load_audit()
    man = pd.read_csv(MAN)
    disease = [c for c in man.columns
               if c.startswith("label_") and c != "label_no_finding"]
    ids = set(man.loc[(man[disease] == "positive").any(axis=1), "image_id"].astype(str))

    rows = []
    for model in MODELS:
        data = aud.read_layer(MATRIX / "padchest" / model / "shift" /
                              "image_reliance_predictions.jsonl")
        real = {k: v for k, v in data.get("original", {}).items() if str(k) in ids}
        blank = {k: v for k, v in data.get("blank", {}).items() if str(k) in ids}
        if not real:
            continue
        texts = [r.get("raw_output") or "" for r in real.values()]
        status = [aud.status_vector(r.get("parsed") or {}) for r in real.values()]
        div = aud.diversity(texts, status)
        con = aud.contrasts(real, blank)
        rows.append({
            "model": model,
            "n_studies": len(real),
            "n_unique_text": div["n_unique_text"],
            "top_share_text": round(div["top_share_text"], 6),
            "n_unique_status": div["n_unique_status"],
            "collapsed": bool(div["n_unique_text"] <= 5 or div["top_share_text"] >= 0.50),
            "gap_text": round(con["gap_text"], 6),
            "gap_prob": (round(con["gap_prob"], 6)
                         if con.get("gap_prob") is not None else None),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "padchest_disease_subset.csv",
                              index=False, encoding="utf-8-sig")
    for r in rows:
        print(f"{r['model']:26s} n={r['n_studies']} unique={r['n_unique_text']:3d} "
              f"top={r['top_share_text']:.3f} collapsed={r['collapsed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
