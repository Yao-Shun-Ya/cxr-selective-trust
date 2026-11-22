"""Summarise the controlled input-geometry sweep.

Reads ``outputs/geometry/<model>/edge<e>/shift_predictions.jsonl`` and reports,
per model and edge, macro AUROC over the seven pathology labels, the P1 change
rate against clean, parse rate and the realised sent size. The question is
whether the cross-model ordering is stable when the client-side long edge is
fixed, not whether any single edge is best.

    python scripts/129_analyze_geometry.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis_geometry"
LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax"]


def _vec(rec: dict) -> tuple:
    parsed = rec.get("parsed") or {}
    if not parsed.get("parse_success"):
        return tuple(["unparsed"] * len(LABELS))
    f = parsed.get("findings") or {}
    return tuple((f.get(l) or {}).get("status", "not_mentioned") for l in LABELS)


def _d(a: tuple, b: tuple) -> float:
    pa = {i for i, s in enumerate(a) if s == "positive"}
    pb = {i for i, s in enumerate(b) if s == "positive"}
    if not pa and not pb:
        return 0.0
    return 1.0 - len(pa & pb) / len(pa | pb)


def _auroc(rows: list[dict]) -> float:
    y = np.array([[r["ground_truth"][f"label_{l}"] == "positive" for l in LABELS]
                  for r in rows])
    p = np.array([[(r["parsed"].get("findings", {}).get(l) or {}).get(
        "probability_positive") or 0.0 for l in LABELS] for r in rows])
    vals = []
    for j in range(len(LABELS)):
        if len(np.unique(y[:, j])) > 1:
            vals.append(roc_auc_score(y[:, j], p[:, j]))
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    recs = []
    for path in sorted((ROOT / "outputs" / "geometry").glob("*/*/shift_predictions.jsonl")):
        model, edge_dir = path.parts[-3], path.parts[-2]
        edge = int(edge_dir.replace("edge", ""))
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        clean = {r["image_id"]: _vec(r) for r in rows if r["condition"] == "clean"}
        rec = {"model": model, "edge": edge, "n": len(clean),
               "sent_w_median": float(np.median([r.get("sent_w", np.nan) for r in rows])),
               "sent_h_median": float(np.median([r.get("sent_h", np.nan) for r in rows])),
               "parse": float(np.mean([bool((r.get("parsed") or {}).get("parse_success"))
                                       for r in rows])),
               "auroc_clean": _auroc([r for r in rows if r["condition"] == "clean"])}
        for cond in ("contrast", "noise", "resolution"):
            sub = [r for r in rows if r["condition"] == cond and r["severity"] == "severe"]
            rec[f"p1_{cond}"] = float(np.mean(
                [_d(clean[r["image_id"]], _vec(r)) > 0 for r in sub
                 if r["image_id"] in clean])) if sub else float("nan")
            rec[f"auroc_{cond}"] = _auroc(sub) if sub else float("nan")
        recs.append(rec)
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(recs).sort_values(["model", "edge"])
    df.to_csv(OUT / "geometry_summary.csv", index=False)
    print(df.to_string(index=False))
    print(f"wrote {OUT/'geometry_summary.csv'}")


if __name__ == "__main__":
    main()
