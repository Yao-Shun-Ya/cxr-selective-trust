"""Label-space definitions and normalization helpers."""

from __future__ import annotations

from typing import Any

PRIMARY_LABELS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
    "no_finding",
]

CHEXPERT_COLUMNS = {
    "atelectasis": "Atelectasis",
    "cardiomegaly": "Cardiomegaly",
    "consolidation": "Consolidation",
    "edema": "Edema",
    "pleural_effusion": "Pleural Effusion",
    "pneumonia": "Pneumonia",
    "pneumothorax": "Pneumothorax",
    "no_finding": "No Finding",
}

NIH_LABEL_MAP = {
    "atelectasis": "atelectasis",
    "cardiomegaly": "cardiomegaly",
    "effusion": "pleural_effusion",
    "infiltration": "lung_opacity",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "consolidation": "consolidation",
    "edema": "edema",
    "no finding": "no_finding",
}

PADCHEST_PRIMARY_MAP = {
    # These keys are initial candidates and MUST be manually verified against
    # the actual PadChest label vocabulary before full experiments.
    "atelectasis": "atelectasis",
    "cardiomegaly": "cardiomegaly",
    "consolidation": "consolidation",
    "edema": "edema",
    "pulmonary edema": "edema",
    "pleural effusion": "pleural_effusion",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "no finding": "no_finding",
    "normal": "no_finding",
}


def normalize_chexpert_cell(value: Any) -> str:
    """Map a CheXpert label cell to a status string."""
    if value is None:
        return "not_mentioned"
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return "not_mentioned"
    if text in {"1", "1.0", "positive", "yes", "true"}:
        return "positive"
    if text in {"0", "0.0", "negative", "no", "false"}:
        return "negative"
    if text in {"-1", "-1.0", "uncertain"}:
        return "uncertain"
    return "uncertain"


def normalize_nih_findings(value: Any) -> dict[str, str]:
    """Convert NIH pipe-separated finding labels to our primary statuses."""
    statuses = {label: "negative" for label in PRIMARY_LABELS}
    if value is None:
        return statuses
    text = str(value).strip().lower()
    if not text or text in {"nan", "none"}:
        return statuses
    findings = [x.strip() for x in text.split("|") if x.strip()]
    if "no finding" in findings:
        statuses["no_finding"] = "positive"
    for finding in findings:
        mapped = NIH_LABEL_MAP.get(finding)
        if mapped in statuses:
            statuses[mapped] = "positive"
    return statuses


def map_padchest_labels(labels: list[str]) -> dict[str, str]:
    """Map PadChest labels to primary statuses.

    This is an initial mapping scaffold. The mapping must be manually reviewed
    against the actual PadChest label vocabulary.
    """
    statuses = {label: "negative" for label in PRIMARY_LABELS}
    normalized = {x.strip().lower() for x in labels if x and x.strip()}
    for raw in normalized:
        mapped = PADCHEST_PRIMARY_MAP.get(raw)
        if mapped in statuses:
            statuses[mapped] = "positive"
    return statuses
