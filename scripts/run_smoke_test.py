#!/usr/bin/env python
"""Run a small smoke test for one model on a manifest.

This does not download data or models. For real models, implement the wrapper
in src/medvlm_trust/models and provide the checkpoint/API credentials.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medvlm_trust.inference.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from src.medvlm_trust.inference.schema import parse_vlm_output  # noqa: E402
from src.medvlm_trust.models.registry import get_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--model", default="dummy")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail-rate", type=float, default=0.0)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest).head(args.limit).copy()
    images_root = Path(args.images_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(args.model, seed=args.seed, fail_rate=args.fail_rate)
    prompt = SYSTEM_PROMPT + "\n\n" + build_user_prompt()

    raw_records = []
    prediction_records = []
    for _, row in manifest.iterrows():
        image_path = images_root / row["image_relpath"]
        raw = model.generate(image_path, prompt)
        parsed = parse_vlm_output(raw)
        raw_records.append({
            "image_id": row.get("image_id"),
            "image_relpath": row.get("image_relpath"),
            "raw_output": raw,
        })
        rec = parsed.to_dict()
        rec.update({
            "image_id": row.get("image_id"),
            "dataset": row.get("dataset"),
            "split": row.get("split"),
            "label_ground_truth": {
                k: row[k] for k in row.index if str(k).startswith("label_")
            },
        })
        prediction_records.append(rec)

    with (out_dir / "raw_outputs.jsonl").open("w", encoding="utf-8") as f:
        for r in raw_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in prediction_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(prediction_records)
    failures = sum(1 for r in prediction_records if not r["parse_success"])
    summary = {
        "model": args.model,
        "n": n,
        "parse_failures": failures,
        "parse_failure_rate": failures / n if n else 0.0,
    }
    (out_dir / "parse_failure_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
