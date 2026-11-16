"""Prompt variants for the Qwen image-invariance ablation.

Research question
-----------------
Qwen2.5-VL-7B predicts ``no_finding`` for 988/1000 NIH studies and its
predictions barely change when the image is blanked, blurred or shuffled
(image sensitivity 0.003).  Two explanations are possible:

1. **Language prior** -- the model answers from the prevalence of "normal" in
   its text distribution and essentially ignores the pixels.
2. **Prompt artefact** -- the prompt invites a cheap answer, so the model never
   engages with the image.

The ablation separates these by holding the model and the data fixed and
changing only how the question is asked.  Every variant must therefore be run
with an **original/blank pair** on the same images: image reliance is the
dependent variable, not accuracy.

Design constraints
------------------
* Every variant returns the *same* parseable structure, so one parser and one
  set of metrics apply to all of them.
* Variants differ only in what they ask for, never in the output contract.
* ``V4`` is the decisive test: if a prompt that explicitly normalises "normal"
  as a rare answer still yields ~0 image sensitivity, the invariance is a
  property of the model, not of the wording.

Usage
-----
::

    from cxr_trust.inference.prompts_ablation import PROMPT_VARIANTS, build_prompt

    system, user = build_prompt("v3_per_label_binary")
"""

from __future__ import annotations

from .schema import PRIMARY_LABELS

# --------------------------------------------------------------------------- #
# shared output contract (identical for every variant)
# --------------------------------------------------------------------------- #
_JSON_CONTRACT = """Return exactly one JSON object:
{
  "findings": {"<label>": "positive|negative|uncertain"},
  "probability_positive": {"<label>": 0.0}
}
No markdown, no text outside the JSON."""

_SYSTEM = """You are a careful chest X-ray interpretation assistant.
Answer only from the provided radiograph. Do not use demographic priors,
disease prevalence, or text-only shortcuts.
Return only a single valid JSON object."""

_STATUS_RULE = ("positive = visual evidence present; negative = visual evidence "
                "absent; uncertain = image evidence ambiguous or insufficient.")


def _finding_block() -> str:
    return "\n".join(f"- {x}" for x in PRIMARY_LABELS)


# --------------------------------------------------------------------------- #
# V0 -- baseline, mirrors the current simple prompt
# --------------------------------------------------------------------------- #
V0_SYSTEM = _SYSTEM
V0_USER = f"""Assess these chest X-ray findings:
{", ".join(PRIMARY_LABELS)}

{_STATUS_RULE}

{_JSON_CONTRACT}"""


# --------------------------------------------------------------------------- #
# V1 -- per-label binary, one finding per call is simulated by asking for an
#        explicit positive/negative decision with no "uncertain" escape hatch
#        presented first (uncertain is still allowed, but must be justified)
# --------------------------------------------------------------------------- #
V1_SYSTEM = _SYSTEM
V1_USER = f"""For each of the following findings, decide whether it is present
in this chest radiograph. Give a definite positive or negative answer unless the
image is genuinely uninterpretable, in which case answer uncertain.

Findings to decide, one by one:
{_finding_block()}

Answer each finding independently; do not let a normal appearance of one region
decide another.

{_STATUS_RULE}

{_JSON_CONTRACT}"""


# --------------------------------------------------------------------------- #
# V2 -- few-shot: anchor with one abnormal and one normal worked example
# --------------------------------------------------------------------------- #
V2_SYSTEM = _SYSTEM
V2_USER = f"""Assess these chest X-ray findings:
{", ".join(PRIMARY_LABELS)}

Worked examples of the required reasoning granularity (illustrative only, not
from this image):

Example A -- cardiomegaly present: the cardiac silhouette is enlarged relative
to the thorax, so cardiomegaly is positive and the other findings are judged
independently.
Example B -- no acute abnormality: all structures appear within normal limits,
so every disease finding is negative and no_finding is positive.

Now apply the same granularity to the radiograph you were given. In particular,
check each region separately (lungs, pleura, cardiac silhouette, mediastinum,
bones) before deciding, and only answer no_finding when you have verified that
each disease finding is absent.

{_STATUS_RULE}

{_JSON_CONTRACT}"""


# --------------------------------------------------------------------------- #
# V3 -- report-first: elicit a narrative, then the structured labels
# --------------------------------------------------------------------------- #
V3_SYSTEM = _SYSTEM
V3_USER = f"""Describe the chest radiograph in this image, then label it.

Step 1: report what you actually see, region by region (lungs, pleura, cardiac
silhouette, mediastinum, bones). Describe observable features only.

Step 2: derive the structured labels from that description. Every label must be
justified by something you described in step 1. If you described no evidence
for a finding, that finding is negative; if the region was not assessable, it
is uncertain.

Findings to label:
{_finding_block()}

Put the step-1 description in the "rationale" field and the step-2 labels in
"findings".

Return exactly one JSON object:
{{
  "findings": {{"<label>": "positive|negative|uncertain"}},
  "probability_positive": {{"<label>": 0.0}},
  "rationale": "the region-by-region description from step 1"
}}
No markdown, no text outside the JSON."""


# --------------------------------------------------------------------------- #
# V4 -- normality-prior pressure: the decisive test for the language-prior
#        hypothesis. Explicitly attacks the "normal is the safe answer" prior.
# --------------------------------------------------------------------------- #
V4_SYSTEM = """You are a radiologist reading a chest X-ray for a symptomatic
patient population. In this population an entirely normal radiograph is the
exception, not the rule: a normal study is uncommon and should only be reported
when you have actively excluded every abnormality.
Answer only from the provided radiograph.
Return only a single valid JSON object."""

V4_USER = f"""Read this chest radiograph carefully.

Important: this image comes from a patient population in which most studies are
NOT normal. Do not default to "no_finding". Before you may answer no_finding,
you must have actively checked and excluded each of the following, and you
should expect to find at least one abnormality in most images:

{_finding_block()}

For each finding, state the specific visual feature you checked and whether it
was present. Only report no_finding = positive if, after that check, every
disease finding is negative.

{_STATUS_RULE}

{_JSON_CONTRACT}"""


# --------------------------------------------------------------------------- #
# R0 -- prompted reasoning channel: ask the model to write a trace inside
#       <think>...</think> before the same JSON contract.  Used only for the
#       second-model marker probe; it is a *prompted* channel, not an intrinsic
#       one, and must not be compared to MedGemma-1.5's native marker without
#       saying so.
# --------------------------------------------------------------------------- #
R0_SYSTEM = """You are a careful chest X-ray interpretation assistant.
Before answering, reason about the image inside <think> ... </think>.
Do not put JSON inside the think block. After </think>, return only the JSON object.
Answer only from the provided radiograph."""
R0_USER = f"""Assess these chest X-ray findings:
{", ".join(PRIMARY_LABELS)}

First write a short image-grounded reasoning trace inside <think>...</think>.
Then return exactly one JSON object after </think>:
{_JSON_CONTRACT}"""


# --------------------------------------------------------------------------- #
# P1/P2 -- paraphrase controls. Same JSON contract and same label set as V0,
# only the wording changes. Used to test whether the schema result is a
# wording artefact.
# --------------------------------------------------------------------------- #
P1_SYSTEM = _SYSTEM
P1_USER = f"""Review this chest radiograph and decide the status of each
finding below. Use only what is visible in the image.

Findings:
{_finding_block()}

{_STATUS_RULE}

{_JSON_CONTRACT}"""

P2_SYSTEM = _SYSTEM
P2_USER = f"""Act as a second reader on this chest X-ray. For every finding
listed, state whether the visual evidence supports it.

Checklist:
{_finding_block()}

Decide each item independently; do not let one normal region decide another.

{_STATUS_RULE}

{_JSON_CONTRACT}"""


Q_SYSTEM = _SYSTEM
Q_USER = f"""For each of the following chest X-ray findings, give only a
probability between 0.0 and 1.0 that the finding is present. Do not decide a
positive/negative status: the decision rule is applied outside the model, so the
same probabilities can be read at any fixed threshold. Use "uncertain" in the
status map as a placeholder; only the probabilities are scored.

Findings:
{_finding_block()}

Return exactly one JSON object:
{{
  "findings": {{"<label>": "uncertain"}},
  "probability_positive": {{"<label>": 0.0}}
}}
No markdown, no text outside the JSON."""


PROMPT_VARIANTS = {
    "v0_simple": (V0_SYSTEM, V0_USER),
    "v1_per_label_binary": (V1_SYSTEM, V1_USER),
    "v2_few_shot": (V2_SYSTEM, V2_USER),
    "v3_report_first": (V3_SYSTEM, V3_USER),
    "v4_normality_pressure": (V4_SYSTEM, V4_USER),
    "r0_think_json": (R0_SYSTEM, R0_USER),
    "p1_paraphrase_short": (P1_SYSTEM, P1_USER),
    "p2_paraphrase_second_reader": (P2_SYSTEM, P2_USER),
    "q_probs_only": (Q_SYSTEM, Q_USER),
}

# variant -> short description used in reports
VARIANT_NOTES = {
    "v0_simple": "Baseline, matches the existing simple prompt.",
    "v1_per_label_binary": "Forces a definite decision per finding, region by region.",
    "v2_few_shot": "Adds one abnormal and one normal worked example.",
    "v3_report_first": "Elicits a region-by-region narrative before labelling.",
    "v4_normality_pressure": "Explicitly removes the 'normal is the safe answer' prior.",
    "r0_think_json": "Prompted <think> trace before JSON; for the second-model marker probe only.",
    "p1_paraphrase_short": "V0 semantics, shorter wording, same JSON contract.",
    "p2_paraphrase_second_reader": "V0 semantics, second-reader wording, same JSON contract.",
    "q_probs_only": "Probabilities only, no status; the decision rule is applied outside the model.",
}


def build_prompt(variant: str):
    """Return ``(system_prompt, user_prompt)`` for a named variant."""
    if variant not in PROMPT_VARIANTS:
        raise KeyError(f"unknown prompt variant {variant!r}; "
                       f"known: {sorted(PROMPT_VARIANTS)}")
    return PROMPT_VARIANTS[variant]


def list_variants():
    return [{"variant": k, "note": VARIANT_NOTES.get(k, "")} for k in PROMPT_VARIANTS]
