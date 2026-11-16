"""Modality-adapted prompts for the cross-modal arm.

Experimental control: the output contract (JSON shape, status vocabulary,
probability field) is byte-identical to the CXR arm; only the anatomy noun and
the label vocabulary change. Any difference observed across modalities is then
attributable to the image, not to the question.

Frozen in docs/58 section 13 before any cross-modal data were downloaded. Do not
edit without appending a deviation record to docs/58 section 13.10.
"""

from __future__ import annotations

from typing import Dict, List

MODALITY_NOUN: Dict[str, str] = {
    "mri": "brain MRI slice",
    "oct": "retinal OCT B-scan",
    "derm": "dermoscopic skin-lesion photograph",
}

# Label vocabularies. Human-readable forms are used rather than the dataset's
# directory acronyms: the acronyms are metadata conventions, not clinical terms,
# and a model that has never seen "akiec" would be tested on vocabulary rather
# than on the image.
MODALITY_LABELS: Dict[str, List[str]] = {
    "mri": ["meningioma", "glioma", "pituitary_tumor"],
    "oct": [
        "choroidal_neovascularization",
        "diabetic_macular_edema",
        "drusen",
        "normal",
    ],
    "derm": [
        "actinic_keratosis",
        "basal_cell_carcinoma",
        "benign_keratosis",
        "dermatofibroma",
        "melanoma",
        "melanocytic_nevus",
        "vascular_lesion",
    ],
}

_STATUS_RULE = ("positive = visual evidence present; negative = visual evidence "
                "absent; uncertain = image evidence ambiguous or insufficient.")

_JSON_CONTRACT = """Return exactly one JSON object:
{
  "findings": {"<label>": "positive|negative|uncertain"},
  "probability_positive": {"<label>": 0.0}
}
No markdown, no text outside the JSON."""


def labels_for(modality: str) -> List[str]:
    try:
        return list(MODALITY_LABELS[modality])
    except KeyError as exc:  # pragma: no cover - caller is a CLI choice list
        raise KeyError(f"unknown modality {modality!r}; "
                       f"known: {sorted(MODALITY_LABELS)}") from exc


def _system(modality: str) -> str:
    return (f"You are a careful {MODALITY_NOUN[modality]} interpretation assistant.\n"
            "Answer only from the provided image. Do not use demographic priors,\n"
            "disease prevalence, or text-only shortcuts.\n"
            # Deliberately NOT adding "do not reason first": measured on the
            # seven-label dermoscopy prompt, the instruction does not suppress
            # MedGemma-1.5's <unused94>thought trace and lowers the 256-token
            # parse rate from 31/44 to 13/44. The trace is a property of the
            # checkpoint, so the arm keeps the CXR contract byte-identical and
            # pays for it with a 1,024-token budget instead.
            "Return only a single valid JSON object.")


def build_prompt(modality: str, schema: str = "v0"):
    """Return ``(system_prompt, user_prompt)``.

    ``v0`` and ``v1`` mirror ``prompts_ablation.V0_*`` / ``V1_*`` with the
    anatomy noun and label list substituted. Nothing else is reworded, so the
    CXR-vs-modality contrast is not confounded by prompt phrasing.
    """
    labels = labels_for(modality)
    noun = MODALITY_NOUN[modality]

    if schema == "v0":
        user = (f"Assess these {noun} findings:\n"
                f"{', '.join(labels)}\n\n"
                f"{_STATUS_RULE}\n\n"
                f"{_JSON_CONTRACT}")
        return _system(modality), user

    if schema == "v1":
        block = "\n".join(f"- {x}" for x in labels)
        user = (f"For each of the following findings, decide whether it is present\n"
                f"in this {noun}. Give a definite positive or negative answer unless the\n"
                "image is genuinely uninterpretable, in which case answer uncertain.\n\n"
                f"Findings to decide, one by one:\n{block}\n\n"
                "Answer each finding independently; do not let a normal appearance of one "
                "region decide another.\n\n"
                f"{_STATUS_RULE}\n\n"
                f"{_JSON_CONTRACT}")
        return _system(modality), user

    raise KeyError(f"unknown schema {schema!r}; expected 'v0' or 'v1'")


def full_prompt(modality: str, schema: str = "v0") -> str:
    system, user = build_prompt(modality, schema)
    return system + "\n\n" + user
