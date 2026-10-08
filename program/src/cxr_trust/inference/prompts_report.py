"""Free-text report-generation prompts for the Open-i external-validation layer.

Why a separate prompt module
----------------------------
The NIH and PadChest layers ask for a *structured verdict* -- eight findings,
each positive/negative/uncertain with a probability -- because their questions
are about calibration, selective risk and image reliance, and those need a
number per finding.

Open-i answers a different question: whether the safety framework survives on
**free-text report generation**, where the failure mode is not a mis-calibrated
probability but an asserted statement with no support in the image (hallucination)
or a missed finding (omission). Scoring that requires the model to produce a
report, and requires the reference report as the comparison target.

So this is deliberately NOT ``build_simple_prompt``. Two properties matter:

1. **Style-matched to the references.** The Open-i ``findings_section`` text is
   terse, impersonal clinical prose ("The cardiac silhouette and mediastinum size
   are within normal limits. There is no pulmonary edema."). A model asked for
   markdown headings and bullet lists produces output that no lexical or judged
   comparison can align with the reference, which would manufacture omission and
   hallucination rates out of a formatting mismatch.
2. **No structured escape hatch.** The model is told to write prose and nothing
   else. If it emits JSON anyway the record still parses the structured schema,
   but that is recorded as a style violation rather than silently accepted.

The judge used to extract findings from both the reference and the generated
text is the same judged labeller used by ``20_rrg_labeler_eval.py`` -- one
labeller for both sides of the comparison, so a systematic labelling bias cancels
instead of masquerading as a hallucination.
"""

from __future__ import annotations

REPORT_SYSTEM_PROMPT = """You are a radiologist dictating the FINDINGS section \
of a chest radiograph report.

Write plain clinical prose in the style of a terse radiology report. Describe only \
what you can actually see in the image. Do not use markdown, headings, bullet \
points, numbered lists, or JSON. Do not add a greeting, a disclaimer, or a \
signature. Do not mention that you are an AI."""


def build_report_prompt() -> str:
    """The user turn for report generation (constant across studies)."""
    return """Write the FINDINGS section for this chest radiograph.

Cover these findings explicitly, saying for each whether it is present or absent:
atelectasis, cardiomegaly, consolidation, edema, pleural effusion, pneumonia, \
pneumothorax, and whether the study is otherwise normal.

Rules:
- Write continuous prose, 2 to 4 sentences, in the register of a clinical report.
- State each finding as a definite assertion; do not hedge with "may" or "possible".
- If the image is uninterpretable, say so explicitly instead of describing findings.
- Describe only what is visible. Do not describe findings you cannot support."""


# A second reference prompt, used only to measure how much of the hallucination
# rate is a prompt artefact. It is strictly terse: if a model's hallucination rate
# depends on which of the two prompts it was given, the result is about the prompt,
# and the paper has to say so.
def build_terse_report_prompt() -> str:
    return """Dictate the FINDINGS section of a chest radiograph report in one to \
three sentences of clinical prose, then stop. No markdown, no lists, no JSON."""
