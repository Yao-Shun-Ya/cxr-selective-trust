"""Prompt templates for structured CXR VLM inference."""

from __future__ import annotations

from .schema import PRIMARY_LABELS

SYSTEM_PROMPT = """You are a careful chest X-ray interpretation assistant.
You must answer only from the provided chest radiograph and the user's question.
Do not use demographic priors, disease prevalence, or text-only shortcuts.
If the image evidence is insufficient, mark the finding as uncertain.
Return only a single valid JSON object with no markdown and no extra text."""


def build_user_prompt(labels=None, view_position: str | None = None) -> str:
    labels = labels or PRIMARY_LABELS
    label_list = "\n".join(f"- {x}" for x in labels)
    view_note = f"\nKnown view position: {view_position}\n" if view_position else ""
    return f"""Evaluate the chest radiograph for the following findings:

{label_list}

{view_note}
Definitions:
- positive: visual evidence supports the finding.
- negative: visual evidence argues against the finding.
- uncertain: evidence is ambiguous or insufficient.
- not_mentioned: the finding was not assessable or not addressed.

Return a single JSON object with this exact structure:
{{
  "findings": {{
    "atelectasis": "positive | negative | uncertain | not_mentioned",
    "cardiomegaly": "positive | negative | uncertain | not_mentioned",
    "...": "same for all listed labels"
  }},
  "probability_positive": {{
    "atelectasis": 0.0,
    "cardiomegaly": 0.0,
    "...": "same for all listed labels"
  }},
  "confidence": {{
    "atelectasis": 0.0,
    "cardiomegaly": 0.0,
    "...": "same for all listed labels"
  }},
  "rationale": "short image-grounded explanation",
  "image_reliance_claim": "visual evidence used, or reason for uncertainty"
}}

Rules:
- `probability_positive[label]` is P(finding is present | image), in [0,1].
- `confidence[label]` is confidence in the assigned `findings` status, in [0,1].
- If findings[label] is positive, confidence should be close to probability_positive.
- If findings[label] is negative, confidence should be close to 1 - probability_positive.
- Do not omit any label.
- Do not output markdown fences.
"""


SIMPLE_SYSTEM_PROMPT = """You are a careful chest X-ray interpretation assistant.
Answer only from the image. Return one valid JSON object and no other text."""


def build_simple_prompt(labels=None) -> str:
    labels = labels or PRIMARY_LABELS
    label_json = ", ".join(f'"{x}": "positive|negative|uncertain"' for x in labels)
    prob_json = ", ".join(f'"{x}": 0.0' for x in labels)
    return f"""Assess these chest X-ray findings:
{", ".join(labels)}

Return exactly this JSON shape:
{{
  "findings": {{{label_json}}},
  "probability_positive": {{{prob_json}}}
}}

Rules:
- positive = finding present; negative = absent; uncertain = not clear.
- probability_positive is P(finding present | image), between 0 and 1.
- Do not add markdown, rationale, or text outside JSON.
"""
