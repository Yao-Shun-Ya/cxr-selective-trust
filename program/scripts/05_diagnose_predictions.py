#!/usr/bin/env python
"""Diagnose prediction distributions: status counts, probability ranges, degenerate patterns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_rows(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        findings = obj.get("findings") or {}
        gt = obj.get("ground_truth") or {}
        for label, fp in findings.items():
            rows.append({
                "image_id": obj.get("image_id"),
                "dataset": obj.get("dataset"),
                "split": obj.get("split"),
                "label": label,
                "status": fp.get("status"),
                "prob": fp.get("probability_positive"),
                "conf": fp.get("confidence"),
                "gt": gt.get(f"label_{label}"),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = load_rows(Path(args.predictions))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_label = df.groupby("label").agg(
        n=("label", "size"),
        positive=("status", lambda s: (s == "positive").sum()),
        negative=("status", lambda s: (s == "negative").sum()),
        uncertain=("status", lambda s: (s == "uncertain").sum()),
        not_mentioned=("status", lambda s: (s == "not_mentioned").sum()),
        prob_mean=("prob", "mean"),
        prob_min=("prob", "min"),
        prob_max=("prob", "max"),
        prob_nunique=("prob", "nunique"),
    ).reset_index()
    by_label.to_csv(out_dir / "diagnosis_by_label.csv", index=False)

    # Image-level pattern: how many unique labels are predicted positive?
    per_image = df[df["status"] == "positive"].groupby("image_id").size().rename("n_positive_labels")
    pattern = per_image.value_counts().sort_index().rename_axis("n_positive_labels").reset_index(name="n_images")
    pattern.to_csv(out_dir / "diagnosis_positive_label_count.csv", index=False)

    summary = {
        "n_predictions": int(len(df)),
        "n_images": int(df["image_id"].nunique()),
        "status_counts": df["status"].value_counts().to_dict(),
        "labels_with_any_positive": int((by_label["positive"] > 0).sum()),
        "labels_all_negative": int((by_label["negative"] == by_label["n"]).sum()),
        "prob_unique_values": sorted([float(x) for x in df["prob"].dropna().unique()])[:50],
    }
    (out_dir / "diagnosis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
