"""Self-contained demo: proves the CLI works without any private artifacts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from .census import census
from .io import read_jsonl
from .responsiveness import responsiveness
from .selective import selective_audit

LABELS = ["a", "b", "c"]


def _row(image_id: str, condition: str, statuses: list[str],
         parse: bool = True) -> dict:
    findings = {label: {"status": status}
                for label, status in zip(LABELS, statuses)}
    return {
        "image_id": image_id,
        "condition": condition,
        "severity": "none" if condition == "clean" else "severe",
        "parsed": {"parse_success": parse, "findings": findings},
    }


def run_demo() -> dict:
    with tempfile.TemporaryDirectory(prefix="cxr_audit_demo_") as tmp:
        tmp_path = Path(tmp)
        rows = [
            _row("img1", "clean", ["positive", "negative", "not_mentioned"]),
            _row("img2", "clean", ["positive", "positive", "negative"]),
            _row("img3", "clean", ["negative", "negative", "negative"]),
            _row("img4", "clean", ["negative", "negative", "negative"], parse=False),
            _row("img1", "contrast", ["positive", "negative", "not_mentioned"]),
            _row("img2", "contrast", ["negative", "positive", "negative"]),
            _row("img3", "contrast", ["negative", "negative", "negative"]),
            _row("img4", "contrast", ["negative", "negative", "negative"], parse=False),
        ]
        pred_path = tmp_path / "predictions.jsonl"
        pred_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                             encoding="utf-8")
        duplicate_rows = [rows[0], rows[0], rows[1]]
        dup_path = tmp_path / "duplicates.jsonl"
        dup_path.write_text("\n".join(json.dumps(row) for row in duplicate_rows) + "\n",
                            encoding="utf-8")

        items = []
        for i in range(10):
            items.append({"p_calibrated": 0.9, "conf": 0.9, "error": 0.0,
                          "patient_id": f"p{i}"})
            items.append({"p_calibrated": 0.6, "conf": 0.6, "error": 1.0,
                          "patient_id": f"p{i}"})
        item_path = tmp_path / "items.csv"
        pd.DataFrame(items).to_csv(item_path, index=False)

        return {
            "census": census(read_jsonl(pred_path)),
            "duplicate_census": census(read_jsonl(dup_path)),
            "responsiveness": responsiveness(read_jsonl(pred_path), LABELS),
            "selective": selective_audit(pd.read_csv(item_path)),
        }
