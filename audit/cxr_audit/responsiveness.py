"""Responsiveness audit: does the model produce variable answers at all?"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from .io import status_vector, vector_change


def responsiveness(rows: Sequence[dict], labels: Iterable[str],
                   clean: str = "clean", shift: str = "contrast",
                   severity: str | None = "severe") -> dict:
    """Distinct/modal status vectors on clean inputs and P1 change under a shift.

    A low distinct count or a modal share near 1 means there is no variation to
    explain: reliance is *unidentifiable*, not zero. P1 is the share of images
    whose status vector changes under the shift relative to clean.
    """
    labels = list(labels)
    clean_rows = [row for row in rows if row.get("condition") == clean]
    if not clean_rows:
        raise ValueError(f"no rows with condition={clean!r}")
    clean_vectors = {str(row["image_id"]): status_vector(row, labels)
                     for row in clean_rows}
    counts = Counter(clean_vectors.values())
    modal_vector, modal_n = counts.most_common(1)[0]

    shift_rows = [row for row in rows if row.get("condition") == shift]
    if severity is not None:
        shift_rows = [row for row in shift_rows
                      if str(row.get("severity", "")) == severity]
    changed = [vector_change(clean_vectors[str(row["image_id"])],
                             status_vector(row, labels)) > 0
               for row in shift_rows if str(row["image_id"]) in clean_vectors]
    changes = [vector_change(clean_vectors[str(row["image_id"])],
                             status_vector(row, labels))
               for row in shift_rows if str(row["image_id"]) in clean_vectors]

    return {
        "clean_condition": clean,
        "shift_condition": shift,
        "severity": severity,
        "n_clean": len(clean_rows),
        "n_shift_matched": len(changed),
        "distinct_clean_vectors": len(counts),
        "modal_share": modal_n / len(clean_rows),
        "modal_vector": list(modal_vector),
        "parse_rate_clean": sum(
            1 for row in clean_rows
            if bool((row.get("parsed") or {}).get("parse_success"))) / len(clean_rows),
        "p1_change_rate": (sum(changed) / len(changed)) if changed else None,
        "mean_jaccard_change": (sum(changes) / len(changes)) if changes else None,
    }
