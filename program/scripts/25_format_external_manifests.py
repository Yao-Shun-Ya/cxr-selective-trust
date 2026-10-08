"""Adapters: external datasets -> the manifest contract the stage scripts expect.

The stage scripts (``06_image_reliance.py``, ``08_shift_benchmark.py``,
``18_selective_metrics_v2.py``) do not read the raw dataset CSVs; they read a
manifest with an exact contract:

    image_id, image_relpath, dataset, patient_id, split,
    label_<8 standard labels>

and every label must be the string ``"positive"`` / ``"negative"`` -- not 0/1,
because the label matrix is built with ``df[c].astype(str) == "positive"``.

The first version of this adapter broke three of those requirements at once, and
the failure was invisible: it wrote 0/1 labels, dropped ``patient_id`` and
``split`` entirely, and mapped PadChest's ``Normal`` to the substring ``normal``
against a lowercased label string, so any label containing "normal" -- including
``Normal`` itself -- lit up ``no_finding``. It also used plain substring
matching, which silently turns "no pleural effusion" into a positive
``pleural_effusion``.

PadChest labels are free text, so this adapter does negation-aware phrase
matching instead: a finding counts as positive only when a phrase naming it
appears without a leading negation ("no", "without", "sin", "absence of").

Usage::

    python scripts/25_format_external_manifests.py
    python scripts/25_format_external_manifests.py --report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# The 8 labels the paper reports.
STD_LABELS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "pleural_effusion", "pneumonia", "pneumothorax", "no_finding",
]

# PadChest free-text label -> standard label. Order matters only for reporting.
PADCHEST_MAP = {
    "atelectasis": ["atelectasis", "laminar atelectasis", "subsegmental atelectasis",
                    "linear atelectasis"],
    "cardiomegaly": ["cardiomegaly", "enlarged cardiac", "cardiac enlargement"],
    "consolidation": ["consolidation", "alveolar pattern", "airspace opacit",
                      "infiltrates", "increased density"],
    "edema": ["edema", "pulmonary congestion", "congestion"],
    "pleural_effusion": ["pleural effusion", "effusion"],
    "pneumonia": ["pneumonia", "bronchopneumonia"],
    "pneumothorax": ["pneumothorax"],
}

NEGATIONS = ("no ", "no_", "without ", "sin ", "absence of ", "negative for ",
             "ruled out", "not ")

# PadChest "Normal" is an explicit statement of normality, not the absence of a
# single finding, so it is handled separately from the negation logic.
NORMAL_PHRASES = ("normal", "no findings", "unremarkable", "within normal limits")


def _has_positive(text: str, phrases) -> bool:
    """True when a phrase occurs and is not negated in the same clause."""
    for phrase in phrases:
        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx < 0:
                break
            window = text[max(0, idx - 24):idx]
            if not any(neg in window for neg in NEGATIONS):
                return True
            start = idx + len(phrase)
    return False


def build_padchest(out_csv: Path) -> pd.DataFrame | None:
    in_csv = ROOT / "data" / "padchest" / "padchest_manifest.csv"
    if not in_csv.exists():
        print(f"[skip] padchest: {in_csv} not found")
        return None
    df = pd.read_csv(in_csv)
    if df.empty:
        print("[skip] padchest: manifest is empty")
        return None

    out = pd.DataFrame()
    out["image_id"] = df["image_id"].astype(str)
    out["image_relpath"] = df["image_id"].apply(lambda x: f"images/{x}")
    out["dataset"] = "padchest"
    # PatientID is a 128-bit hash rendered as text; pandas would read it as a
    # float and destroy it, so keep it as a string.
    out["patient_id"] = df["PatientID"].astype(str)
    out["view"] = ""

    # PadChest is the geographic/hospital shift set: every image is out-of-domain
    # relative to NIH, so the whole set is the target domain, and no NIH row ever
    # appears here. `internal_test` marks it for the analysis scripts, which
    # select that split for evaluation and never fit anything on it.
    out["split"] = "internal_test"

    lower = df["label"].fillna("").astype(str).str.lower()
    for lb in STD_LABELS:
        out[f"label_{lb}"] = "negative"

    for lb, phrases in PADCHEST_MAP.items():
        mask = lower.apply(lambda t: _has_positive(t, phrases))
        out.loc[mask, f"label_{lb}"] = "positive"

    normal_mask = lower.apply(lambda t: _has_positive(t, NORMAL_PHRASES))
    out.loc[normal_mask, "label_no_finding"] = "positive"

    out.to_csv(out_csv, index=False)
    print(f"padchest: {len(out)} rows -> {out_csv}")
    return out


def build_openi(out_csv: Path) -> pd.DataFrame | None:
    """Open-i is a report-generation set: it has reference text, not labels.

    The shift/reliance stages still need a well-formed manifest, so the label
    columns are filled with the honest value for "not annotated" rather than a
    guess. Saying ``not_mentioned`` would be wrong too -- nothing was looked at.
    ``unknown`` is not one of the four statuses the parser accepts, so it can
    never be mistaken for a real negative. Analysis on Open-i is the LLM-as-judge
    report path (``20_rrg_labeler_eval.py``), which reads ``reference_text``.
    """
    in_csv = ROOT / "data" / "open-i" / "openi_manifest.csv"
    if not in_csv.exists():
        print(f"[skip] open-i: {in_csv} not found")
        return None
    df = pd.read_csv(in_csv)
    if df.empty:
        print("[skip] open-i: manifest is empty")
        return None

    out = pd.DataFrame()
    out["image_id"] = df["image_id"].astype(str)
    out["image_relpath"] = df["image_id"].apply(lambda x: f"images/{x}")
    out["dataset"] = "openi"
    out["patient_id"] = df["image_id"].astype(str)
    out["view"] = ""
    out["split"] = "external_report"
    out["reference_text"] = df["findings_section"].fillna("")
    out["default_prompt"] = df.get("default_prompt", pd.Series([""] * len(df))).fillna("")
    for lb in STD_LABELS:
        out[f"label_{lb}"] = "unknown"

    out.to_csv(out_csv, index=False)
    print(f"open-i: {len(out)} rows -> {out_csv}")
    return out


def report(df: pd.DataFrame | None, name: str) -> dict:
    if df is None:
        return {}
    body = {
        "dataset": name,
        "images": int(len(df)),
        "unique_patients": int(df["patient_id"].nunique()),
        "splits": df["split"].value_counts().to_dict(),
    }
    body["positives"] = {
        lb: int((df[f"label_{lb}"] == "positive").sum()) for lb in STD_LABELS
    }
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="print the composition table")
    args = ap.parse_args()

    manifests = ROOT / "data" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    pad = build_padchest(manifests / "padchest_shift.csv")
    oi = build_openi(manifests / "openi_rrg.csv")

    if args.report:
        for name, df in (("padchest", pad), ("openi", oi)):
            body = report(df, name)
            if body:
                print(json.dumps(body, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
