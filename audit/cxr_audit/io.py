"""Input/output helpers shared by the audit commands.

The audit tool deliberately knows only two formats:

* JSONL prediction rows with ``image_id``, ``condition``, ``severity`` and a
  ``parsed`` object (the format written by the project's runners);
* a CSV of finding-level items with ``p_calibrated``, ``conf``, ``error`` and
  ``patient_id`` (the format written by ``18_selective_metrics_v2.py``).

No model, image or network access is required.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

DEFAULT_LABELS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
    "no_finding",
]


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a UTF-8 JSONL file, tolerating a BOM and blank lines."""
    p = Path(path)
    rows = []
    with p.open("r", encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: invalid JSON ({exc})") from exc
    return rows


def status_vector(record: dict, labels: Iterable[str]) -> tuple[str, ...]:
    """Return the contract status vector for one prediction row.

    A parse failure is carried as the token ``unparsed`` rather than dropped:
    dropping it would make a contract failure look like a missing cell.
    """
    parsed = record.get("parsed") or {}
    findings = parsed.get("findings") or {}
    if not parsed.get("parse_success"):
        return tuple(["unparsed"] * len(list(labels)))
    return tuple((findings.get(label) or {}).get("status", "not_mentioned")
                 for label in labels)


def positive_set(vector: tuple[str, ...]) -> set[int]:
    return {i for i, status in enumerate(vector) if status == "positive"}


def vector_change(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """Jaccard distance between two status vectors' positive sets."""
    pa, pb = positive_set(a), positive_set(b)
    if not pa and not pb:
        return 0.0
    return 1.0 - len(pa & pb) / len(pa | pb)
