"""Structured output schema and parser for CXR VLM predictions.

Design goals:
- one stable JSON interface across CheXagent, GPT-4o, and Qwen2.5-VL;
- keep parse failures explicit;
- support per-finding probability and confidence;
- support later calibration / selective prediction / image-reliance scoring.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PRIMARY_LABELS: List[str] = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
    "no_finding",
]

ALLOWED_STATUSES = {"positive", "negative", "uncertain", "not_mentioned"}


@dataclass
class FindingPrediction:
    label: str
    status: str
    probability_positive: Optional[float] = None
    confidence: Optional[float] = None


@dataclass
class ParsedPrediction:
    findings: Dict[str, FindingPrediction] = field(default_factory=dict)
    rationale: str = ""
    image_reliance_claim: Optional[str] = None
    parse_success: bool = False
    parse_error: str = ""
    raw_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parse_success": self.parse_success,
            "parse_error": self.parse_error,
            "findings": {
                label: {
                    "status": fp.status,
                    "probability_positive": fp.probability_positive,
                    "confidence": fp.confidence,
                }
                for label, fp in self.findings.items()
            },
            "rationale": self.rationale,
            "image_reliance_claim": self.image_reliance_claim,
        }


def _clamp_probability(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= x <= 1.0:
        return x
    return None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from a model response.

    Handles code fences and surrounding explanation text.
    Returns None if no valid JSON object is found.
    """
    if not text:
        return None

    cleaned = text.strip()
    # Remove common markdown fences.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Direct parse first.
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: find the first balanced {...} block.
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    return None
    return None


def parse_vlm_output(raw_output: str, labels: Optional[List[str]] = None) -> ParsedPrediction:
    """Parse a VLM response into a strict ParsedPrediction.

    Parse failure is explicit. Missing labels are recorded as not_mentioned
    rather than silently dropped.
    """
    labels = labels or PRIMARY_LABELS
    result = ParsedPrediction(raw_output=raw_output)
    obj = extract_json_object(raw_output)
    if obj is None:
        result.parse_error = "no_json_object_found"
        return result

    findings_obj = obj.get("findings")
    if not isinstance(findings_obj, dict):
        result.parse_error = "findings_missing_or_not_object"
        return result

    conf_obj = obj.get("confidence")
    if conf_obj is not None and not isinstance(conf_obj, dict):
        result.parse_error = "confidence_not_object"
        return result
    conf_obj = conf_obj or {}

    prob_obj = obj.get("probability_positive")
    if prob_obj is not None and not isinstance(prob_obj, dict):
        result.parse_error = "probability_positive_not_object"
        return result
    prob_obj = prob_obj or {}

    parsed_findings: Dict[str, FindingPrediction] = {}
    for label in labels:
        raw_status = findings_obj.get(label, "not_mentioned")
        status = str(raw_status).strip().lower().replace(" ", "_")
        if status not in ALLOWED_STATUSES:
            status = "uncertain"
        parsed_findings[label] = FindingPrediction(
            label=label,
            status=status,
            probability_positive=_clamp_probability(prob_obj.get(label)),
            confidence=_clamp_probability(conf_obj.get(label)),
        )

    result.findings = parsed_findings
    result.rationale = str(obj.get("rationale", "") or "")
    claim = obj.get("image_reliance_claim")
    result.image_reliance_claim = str(claim) if claim is not None else None
    result.parse_success = True
    return result


def parse_failure_rate(records: List[ParsedPrediction]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if not r.parse_success) / len(records)
