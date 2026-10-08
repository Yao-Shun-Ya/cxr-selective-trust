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


# CheXagent writes its JSON with HTML line breaks instead of newlines in a
# minority of cells (16 of 1,900 shift rows on the 100-image pre-run). The object
# is otherwise complete, so ``<br>`` is a spelling of whitespace, not a different
# answer. The rule names no model and runs on every model's raw output.
_HTML_BREAK_RE = re.compile(r"<br\s*/?>|</br\s*>", re.IGNORECASE)


def normalise_model_text(text: str) -> str:
    """Map HTML line-break tags to newlines before JSON extraction.

    Applied uniformly to every model's raw output. On this study it changes
    CheXagent's 16 shift rows and nothing else -- verified by re-parsing every
    prediction file (``scripts/69_reparse_predictions.py``).
    """
    if not text:
        return text
    return _HTML_BREAK_RE.sub("\n", text)


def _balanced_object_spans(text: str):
    """Yield ``(start, end)`` for every brace-balanced ``{...}`` group, left to right.

    Groups are located independently, so one that never closes (a truncated
    thought trace) cannot hide a well-formed object that follows it. This is the
    difference from a plain ``find("{")`` + count: that stops at the first group
    and gives up if the group is malformed, which is exactly what lost the one
    remaining medgemma_15 precision row (docs/50 section 2.12).
    """
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        j = i
        while j < n:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j < n:
            yield i, j + 1
            i = j + 1
        else:
            # Unterminated group; resume one character later so a later, complete
            # object is still reachable.
            i += 1


def _legacy_first_object(text: str) -> Optional[Dict[str, Any]]:
    """The pre-2026-10-08 rule, kept verbatim as the fallback path.

    It stops at the first ``{`` and gives up if that group does not parse. The
    fixed rule in :func:`extract_json_object` runs first and can only turn a
    failure into a success; falling back to this exact rule is what keeps a reply
    that already failed on its original ``parse_error``. That matters because the
    stored labels have to stay comparable across the fix, and
    ``scripts/69_reparse_predictions.py`` asserts zero relabels.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return obj if isinstance(obj, dict) else None
    return None


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the model's structured answer from a response.

    The answer is the first JSON object -- the whole reply first, then each
    balanced ``{...}`` group left to right -- that parses AND carries a
    ``findings`` mapping. Requiring ``findings`` is what makes the search safe: a
    group that fails to parse, or that parses to something without the contract's
    key (a draft in a thought trace, an example in prose), is skipped rather than
    ending the search. Handles HTML line breaks and code fences.

    Returns None if no such object is found; the legacy rule then decides, so a
    reply that already failed keeps its original failure label.
    """
    if not text:
        return None

    cleaned = normalise_model_text(text).strip()
    # Remove common markdown fences.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    for candidate in [cleaned]:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(obj, dict) and "findings" in obj:
                return obj

    for start, end in _balanced_object_spans(cleaned):
        try:
            obj = json.loads(cleaned[start:end])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "findings" in obj:
            return obj

    return _legacy_first_object(cleaned)


def _match_label_key(label: str, obj: dict):
    """Find a canonical label key in a possibly misspelled model output."""
    if label in obj:
        return label
    # case/space normalizations
    normalized = {str(k).strip().lower().replace(" ", "_"): k for k in obj}
    if label in normalized:
        return normalized[label]
    # fuzzy matching for quantization/7B spelling errors
    try:
        from difflib import get_close_matches
        matches = get_close_matches(label, list(obj.keys()), n=1, cutoff=0.55)
        return matches[0] if matches else None
    except Exception:
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
        status_key = _match_label_key(label, findings_obj)
        prob_key = _match_label_key(label, prob_obj) if prob_obj else None
        conf_key = _match_label_key(label, conf_obj) if conf_obj else None
        raw_status = findings_obj.get(status_key, "not_mentioned") if status_key else "not_mentioned"
        status = str(raw_status).strip().lower().replace(" ", "_")
        if status not in ALLOWED_STATUSES:
            status = "uncertain"
        parsed_findings[label] = FindingPrediction(
            label=label,
            status=status,
            probability_positive=_clamp_probability(prob_obj.get(prob_key)) if prob_key else None,
            confidence=_clamp_probability(conf_obj.get(conf_key)) if conf_key else None,
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
