"""Export matrix predictions into the flat schema that 18/19 consume.

Why an adapter instead of pointing 18/19 at the matrix files
------------------------------------------------------------
The three-layer runners write one row per (image, condition) with the parsed
answer nested under ``parsed.findings`` and **no** ground-truth column for the
reliance runner. ``18_selective_metrics_v2.py`` and
``19_accept_clarify_abstain.py`` expect the older shape:

    {"image_id": ..., "split": ..., "findings": {...}, "ground_truth": {...}}

so pointing them straight at ``image_reliance_predictions.jsonl`` yields
``findings = {}`` for every row, the loader skips all of them, and the script
prints "[error] no usable predictions" (18) or silently writes an empty summary
(19). That is a schema mismatch, not a data problem -- and it is exactly the
silent-empty-output class this project has already been bitten by.

The adapter reads the ``clean`` cell of ``shift_predictions.jsonl`` because that
is the only NIH file that carries BOTH the parsed answer and the ground truth
in one row. Condition ``clean`` is the undegraded in-domain image, which is the
correct evaluation set for selective prediction.

Output: ``<out-dir>/<model>/predictions_clean.jsonl``.

Usage::

    python scripts/79_export_selective_inputs.py \
        --matrix-root outputs/matrix --output-dir outputs/analysis_selective
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
                  "pleural_effusion", "pneumonia", "pneumothorax", "no_finding"]

MODELS = [
    "medgemma_4b_it",
    "medgemma_15_4b_it",
    "internvl3_5_8b",
    "qwen3_vl_8b_instruct",
    "chexagent_8b",
    "qwen25_vl_7b_instruct",
]


def export_one(src: Path, dst: Path) -> dict:
    n_in = n_out = n_parse = 0
    by_split: dict[str, int] = {}
    slots = {"positive": 0, "negative": 0, "uncertain": 0,
             "not_mentioned": 0, "missing": 0}
    images_all_definite = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fh, \
            dst.open("w", encoding="utf-8") as out:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            n_in += 1
            if str(obj.get("condition")) != "clean":
                continue
            parsed = obj.get("parsed") or {}
            findings = parsed.get("findings") or {}
            gt = obj.get("ground_truth") or {}
            if not findings or not gt:
                continue
            if any(gt.get(f"label_{lb}") not in {"positive", "negative"}
                   for lb in findings):
                # A partial label set would make 18/19 silently drop the image.
                continue
            # Contract coverage: what fraction of label slots carries a definite
            # answer. CheXagent emits `not_mentioned` for ~79% of slots, so 18
            # (which skips labels with no probability) and 19 (which maps every
            # non-positive status to negative) end up measuring different things
            # -- 0.183 vs 1.000 image error for the same model. Recorded here so
            # no selective metric can be read without its denominator.
            all_def = True
            for lb in DEFAULT_LABELS:
                st = (findings.get(lb) or {}).get("status")
                key = st if st in slots else "missing"
                slots[key] += 1
                if key != "positive" and key != "negative":
                    all_def = False
            images_all_definite += int(all_def)
            rec = {
                "image_id": obj.get("image_id"),
                "split": obj.get("split"),
                "findings": findings,
                "ground_truth": gt,
                "parse_success": bool(parsed.get("parse_success")),
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1
            n_parse += int(rec["parse_success"])
            by_split[str(rec.get("split"))] = by_split.get(str(rec.get("split")), 0) + 1
    total = sum(slots.values())
    definite = slots["positive"] + slots["negative"]
    return {"matrix_cells": n_in, "exported": n_out, "parse_ok": n_parse,
            "by_split": by_split, "label_slots": slots,
            "definite_rate": (round(definite / total, 4) if total else None),
            "images_all_definite": images_all_definite,
            "images_all_definite_rate": (round(images_all_definite / n_out, 4)
                                         if n_out else None)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix-root", default=str(ROOT / "outputs" / "matrix"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()

    matrix = Path(args.matrix_root)
    out_dir = Path(args.output_dir)
    manifest: dict[str, dict] = {}
    for model in args.models:
        src = matrix / "nih_eval" / model / "shift" / "shift_predictions.jsonl"
        if not src.exists():
            print(f"[--] {model}: no shift_predictions.jsonl")
            continue
        dst = out_dir / model / "predictions_clean.jsonl"
        stats = export_one(src, dst)
        manifest[model] = {"source": str(src), "output": str(dst), **stats}
        print(f"{model:24s} exported {stats['exported']:5d} of "
              f"{stats['matrix_cells']:6d} cells  parse_ok={stats['parse_ok']}  "
              f"splits={stats['by_split']}")
        if stats["exported"] == 0:
            print(f"[BAD ] {model}: exported zero rows -- 18/19 would run on nothing")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_dir / 'export_manifest.json'}")
    return 0 if all(v["exported"] > 0 for v in manifest.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
