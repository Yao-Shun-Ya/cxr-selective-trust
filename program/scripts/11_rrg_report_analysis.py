#!/usr/bin/env python
"""Basic rule-based report-level hallucination + image-reliance analysis for RRG."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

FINDING_KEYWORDS = {
    "atelectasis": ["atelectasis", "collapse"],
    "cardiomegaly": ["cardiomegaly", "enlarged cardiac", "cardiac silhouette is enlarged"],
    "consolidation": ["consolidation"],
    "edema": ["edema", "pulmonary vascular congestion", "vascular congestion"],
    "pleural_effusion": ["pleural effusion", "effusion"],
    "pneumonia": ["pneumonia", "infection"],
    "pneumothorax": ["pneumothorax"],
    "no_finding": ["no acute", "normal", "clear", "unremarkable"],
}

NEGATIONS = [
    "no ", "without ", "negative for ", "no evidence of ", "free of ",
    "resolved", "not ", "absent", "no significant",
]


def evidence_status(text: str, finding: str) -> str:
    if not isinstance(text, str):
        return "not_mentioned"
    t = " " + re.sub(r"\s+", " ", text.lower()) + " "
    for kw in FINDING_KEYWORDS.get(finding, [finding]):
        for m in re.finditer(re.escape(kw), t):
            start = max(0, m.start() - 60)
            window = t[start:m.start()]
            if any(neg in window for neg in NEGATIONS):
                return "absent"
            return "present"
    return "not_mentioned"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path(args.predictions).read_text(encoding="utf-8").splitlines() if l.strip()]
    records = []
    for obj in rows:
        if not (obj.get("parsed") or {}).get("parse_success"):
            continue
        ref = obj.get("reference_text", "")
        for label, fp in (obj["parsed"].get("findings") or {}).items():
            ev = evidence_status(ref, label)
            status = fp.get("status")
            records.append({
                "image_id": obj["image_id"],
                "task": obj.get("task"),
                "condition": obj.get("condition"),
                "label": label,
                "model_status": status,
                "evidence_status": ev,
                "hallucination": int(status == "positive" and ev != "present"),
                "negation_failure": int(status == "positive" and ev == "absent"),
                "omission": int(status == "negative" and ev == "present"),
                "uncertain": int(status == "uncertain"),
            })
    df = pd.DataFrame(records)
    if df.empty:
        (out_dir / "rrg_summary.json").write_text(json.dumps({"error": "no parse-success rows"}), encoding="utf-8")
        print("no parse-success rows"); return

    per_label = df.groupby("label").agg(
        n=("label", "size"),
        hallucination_rate=("hallucination", "mean"),
        negation_failure_rate=("negation_failure", "mean"),
        omission_rate=("omission", "mean"),
        uncertain_rate=("uncertain", "mean"),
    ).reset_index()
    per_label.to_csv(out_dir / "rrg_report_metrics_by_label.csv", index=False)

    summary = {
        "evidence_method": "rule_based_v0",
        "n_rows": int(len(df)),
        "n_images": int(df["image_id"].nunique()),
        "overall_hallucination_rate": float(df["hallucination"].mean()),
        "overall_negation_failure_rate": float(df["negation_failure"].mean()),
        "overall_omission_rate": float(df["omission"].mean()),
        "overall_uncertain_rate": float(df["uncertain"].mean()),
    }
    (out_dir / "rrg_hallucination_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Image reliance: original vs blank if available
    orig = df[df["condition"] == "original"][["image_id", "label", "model_status"]].rename(columns={"model_status": "orig_status"})
    blank = df[df["condition"] == "blank"][["image_id", "label", "model_status"]].rename(columns={"model_status": "blank_status"})
    if not blank.empty:
        merged = orig.merge(blank, on=["image_id", "label"], how="inner")
        merged["flip"] = (merged["orig_status"] != merged["blank_status"]).astype(int)
        merged.to_csv(out_dir / "rrg_image_reliance_per_pair.csv", index=False)
        sens = {
            "n_pairs": int(len(merged)),
            "status_flip_rate": float(merged["flip"].mean()),
        }
    else:
        sens = {"n_pairs": 0, "status_flip_rate": None, "note": "no blank condition in predictions"}
    (out_dir / "rrg_image_reliance_summary.json").write_text(json.dumps(sens, indent=2), encoding="utf-8")
    print(json.dumps({"report": summary, "image_reliance": sens}, indent=2))


if __name__ == "__main__":
    main()
