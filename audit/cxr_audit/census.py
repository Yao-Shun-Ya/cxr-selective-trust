"""Denominator audit: rows, unique keys, duplicates and parse rate."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

DEFAULT_KEY_FIELDS = ("image_id", "condition", "severity")


def census(rows: Sequence[dict],
           key_fields: Iterable[str] = DEFAULT_KEY_FIELDS) -> dict:
    """Recompute a prediction file's denominator from the file itself.

    ``duplicate_rows`` is the number of rows that would have to be removed to
    make the key unique; ``duplicate_groups`` is the number of keys involved.
    """
    keys = [tuple(str(row.get(field, "")) for field in key_fields) for row in rows]
    counts = Counter(keys)
    duplicate_groups = [{"key": list(key), "rows": n}
                        for key, n in counts.items() if n > 1]
    duplicate_groups.sort(key=lambda item: (-item["rows"], item["key"]))
    parsed = sum(1 for row in rows
                 if bool((row.get("parsed") or {}).get("parse_success")))
    conditions = Counter(str(row.get("condition", "")) for row in rows)
    return {
        "rows": len(rows),
        "unique_keys": len(counts),
        "duplicate_rows": sum(n - 1 for n in counts.values() if n > 1),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_examples": duplicate_groups[:5],
        "parse_rate": (parsed / len(rows)) if rows else 0.0,
        "conditions": dict(sorted(conditions.items())),
    }
