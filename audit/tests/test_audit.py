"""Unit tests for the CPU-only audit CLI."""

from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd

from cxr_audit.census import census
from cxr_audit.responsiveness import responsiveness
from cxr_audit.selective import selective_audit


def _row(image_id: str, condition: str, statuses: list[str],
         parse: bool = True) -> dict:
    return {
        "image_id": image_id,
        "condition": condition,
        "severity": "none" if condition == "clean" else "severe",
        "parsed": {
            "parse_success": parse,
            "findings": {label: {"status": status}
                         for label, status in zip(["a", "b", "c"], statuses)},
        },
    }


def test_census_counts_duplicates_and_parse_failures() -> None:
    rows = [
        _row("img1", "clean", ["positive", "negative", "negative"]),
        _row("img1", "clean", ["positive", "negative", "negative"]),
        _row("img2", "clean", ["negative", "negative", "negative"], parse=False),
    ]
    result = census(rows)
    assert result["rows"] == 3
    assert result["unique_keys"] == 2
    assert result["duplicate_rows"] == 1
    assert result["duplicate_groups"] == 1
    assert abs(result["parse_rate"] - 2 / 3) < 1e-9


def test_responsiveness_detects_collapse_and_change() -> None:
    rows = [
        _row("img1", "clean", ["positive", "negative", "negative"]),
        _row("img2", "clean", ["positive", "negative", "negative"]),
        _row("img3", "clean", ["negative", "negative", "negative"]),
        _row("img1", "contrast", ["positive", "negative", "negative"]),
        _row("img2", "contrast", ["negative", "negative", "negative"]),
        _row("img3", "contrast", ["negative", "negative", "negative"]),
    ]
    result = responsiveness(rows, ["a", "b", "c"])
    assert result["distinct_clean_vectors"] == 2
    assert abs(result["modal_share"] - 2 / 3) < 1e-9
    assert result["n_shift_matched"] == 3
    assert abs(result["p1_change_rate"] - 1 / 3) < 1e-9


def test_selective_confidence_entropy_margin() -> None:
    frame = pd.DataFrame([
        {"p_calibrated": 0.9, "conf": 0.9, "error": 0.0},
        {"p_calibrated": 0.9, "conf": 0.9, "error": 0.0},
        {"p_calibrated": 0.6, "conf": 0.6, "error": 1.0},
        {"p_calibrated": 0.6, "conf": 0.6, "error": 1.0},
    ])
    result = selective_audit(frame)
    assert result["n_items"] == 4
    assert result["signals"]["confidence"]["error_detection_auroc"] == 1.0
    assert 0.0 <= result["signals"]["entropy"]["coverage_at_5pct_risk"] <= 1.0


def test_cli_demo_runs_without_data() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "cxr_audit.cli", "demo"],
        capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["census"]["rows"] == 8
    assert "responsiveness" in payload
    assert "selective" in payload
