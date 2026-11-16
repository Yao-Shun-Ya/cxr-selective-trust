"""CPU-only tests for the authoritative ``cxr_trust`` package.

Run from ``program/``::

    .venv-vl\\Scripts\\python.exe -m pytest -q tests

The legacy parser tests at repository root cover the old ``src/medvlm_trust``
package; these cover the code path that actually produced the paper's frozen
predictions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.cxr_trust.evaluation.metrics import auroc, expected_calibration_error
from src.cxr_trust.inference.durable import DurableJsonl, read_jsonl
from src.cxr_trust.inference.schema import PRIMARY_LABELS, parse_vlm_output


def test_authoritative_parser_primary_labels():
    parsed = parse_vlm_output('{"findings": {"atelectasis": "positive"}}')
    assert parsed.parse_success
    assert set(parsed.findings) == set(PRIMARY_LABELS)
    assert parsed.findings["atelectasis"].status == "positive"
    assert parsed.findings["cardiomegaly"].status == "not_mentioned"


def test_authoritative_parser_rejects_non_json():
    parsed = parse_vlm_output("The image is uninterpretable.")
    assert not parsed.parse_success
    assert parsed.parse_error == "no_json_object_found"


def test_metrics_chance_and_perfect():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert np.isnan(auroc([0, 0], [0.1, 0.2]))
    assert expected_calibration_error([0, 1], [0.0, 1.0]) == 0.0


def test_durable_append_resume_and_replacement(tmp_path: Path):
    path = tmp_path / "pred.jsonl"

    def key(r):
        return (r["image_id"], r["condition"])

    with DurableJsonl(path, key=key) as out:
        assert out.needs("a", "clean")
        out.write_many([
            {"image_id": "a", "condition": "clean", "error": ""},
            {"image_id": "b", "condition": "clean", "error": ""},
        ])
        assert not out.needs("a", "clean")
        assert out.stats()["records_on_disk"] == 2

    # Resume: nothing new, no duplicates.
    with DurableJsonl(path, key=key) as out:
        assert not out.needs("a", "clean")
        assert out.stats()["records_on_disk"] == 2

    # A failed retry is replaced in place, not appended beside the success.
    with DurableJsonl(path, key=key) as out:
        out.write_many([{"image_id": "a", "condition": "clean", "error": "boom"}])
        assert not out.needs("a", "clean")  # retry_failed=True -> not done
        out.write_many([{"image_id": "a", "condition": "clean", "error": ""}])
        assert out.stats()["records_on_disk"] == 2
    assert len(read_jsonl(path)) == 2


def test_durable_truncates_torn_tail(tmp_path: Path):
    path = tmp_path / "pred.jsonl"

    def key(r):
        return (r["image_id"], r["condition"])

    with DurableJsonl(path, key=key) as out:
        out.write_many([{"image_id": "a", "condition": "clean", "error": ""}])
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"image_id": "b", "condition": "cle')  # torn write

    with DurableJsonl(path, key=key) as out:
        assert out.stats()["truncated_bytes_on_open"] > 0
        assert out.needs("b", "clean")
        out.write_many([{"image_id": "b", "condition": "clean", "error": ""}])
    rows = read_jsonl(path)
    assert [r["image_id"] for r in rows] == ["a", "b"]
