#!/usr/bin/env python
"""110 - Compare greedy outputs across batch sizes (determinism audit).

Greedy decoding is described as deterministic, but a GPU kernel may reduce a
batch differently from a single row, so the last bits of a logit can differ and
flip a token.  This script compares two or more prediction files produced on the
*same* fixed manifest at different batch sizes and reports:

  * exact raw-output agreement,
  * parsed-status agreement per label,
  * contract/parse disagreement.

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\110_compare_determinism.py `
      outputs/determinism/medgemma_4b_it/bs1/image_reliance_predictions.jsonl `
      outputs/determinism/medgemma_4b_it/bs4/image_reliance_predictions.jsonl `
      outputs/determinism/medgemma_4b_it/bs8/image_reliance_predictions.jsonl `
      --out outputs/determinism/medgemma_4b_it/determinism.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax", "no_finding"]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def read(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[str(r.get("image_id"))] = r
    return out


def status(r: dict, label: str) -> str:
    f = (r.get("parsed") or {}).get("findings") or {}
    return str((f.get(label) or {}).get("status") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    runs = [read(p) for p in args.preds]
    common = set.intersection(*(set(r) for r in runs))
    if not common:
        raise SystemExit("no common image ids across prediction files")

    rows = []
    for iid in sorted(common):
        base = runs[0][iid]
        row = {"image_id": iid}
        for k in range(1, len(runs)):
            other = runs[k][iid]
            row[f"exact_vs_{k}"] = int(norm(base.get("raw_output")) == norm(other.get("raw_output")))
            row[f"parse_vs_{k}"] = int(bool((base.get("parsed") or {}).get("parse_success")) ==
                                       bool((other.get("parsed") or {}).get("parse_success")))
            agree = sum(status(base, l) == status(other, l) for l in LABELS)
            row[f"status8_vs_{k}"] = agree
        rows.append(row)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    summary = {"n_images": len(df), "runs": [str(p) for p in args.preds]}
    for k in range(1, len(runs)):
        summary[f"exact_rate_vs_{k}"] = round(float(df[f"exact_vs_{k}"].mean()), 4)
        summary[f"parse_rate_vs_{k}"] = round(float(df[f"parse_vs_{k}"].mean()), 4)
        summary[f"mean_labels_agree_vs_{k}"] = round(float(df[f"status8_vs_{k}"].mean()), 3)
    out_json = args.out.with_suffix(".json")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
