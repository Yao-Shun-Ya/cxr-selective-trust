"""Build the deterministic probe manifests (step 2).

Profiles
    full_multimodal  9,400 calls (CXR 4,000 + three non-CXR modalities 5,400)
    lite             2,700 calls

Sampling is at patient / lesion level with a fixed seed, and every manifest is
written with a SHA-256 lock so a later run can prove which rows it consumed.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common import MANIFEST_DIR, ROOT, ensure_dirs, sha256_file

SEED = 42
PROFILES = {
    "full_multimodal": {"nih": 400, "padchest": 500, "padchest_pos": 265,
                        "openi": 300, "cross": 300},
    "lite": {"nih": 200, "padchest": 155, "padchest_pos": 80,
             "openi": 100, "cross": 55},
}
PATHOLOGY = ["atelectasis", "cardiomegaly", "consolidation", "edema",
             "pleural_effusion", "pneumonia", "pneumothorax"]


def _patient_sample(df: pd.DataFrame, n: int, pos_mask: pd.Series | None,
                    pos_n: int | None, seed: int) -> pd.DataFrame:
    """One image per patient, split into a positive and a negative stratum."""
    df = df.drop_duplicates("patient_id")
    rng = __import__("numpy").random.default_rng(seed)
    if pos_mask is None:
        idx = rng.choice(df.index.to_numpy(), size=min(n, len(df)), replace=False)
        return df.loc[sorted(idx)]
    pos = df[pos_mask.reindex(df.index).fillna(False)]
    neg = df[~pos_mask.reindex(df.index).fillna(False)]
    pos_n = min(pos_n or 0, len(pos), n)
    neg_n = min(n - pos_n, len(neg))
    take_pos = pos.sample(n=pos_n, random_state=seed)
    take_neg = neg.sample(n=neg_n, random_state=seed)
    return pd.concat([take_pos, take_neg]).sort_values("image_id")


def build_cxr(profile: str) -> pd.DataFrame:
    p = PROFILES[profile]
    nih = pd.read_csv(ROOT / "data" / "manifests" / "nih_eval_1506.csv")
    nih_pos = nih["label_pneumonia"].eq("positive")
    nih_s = _patient_sample(nih, p["nih"],
                            nih_pos, p["nih"] // 2, SEED)
    rows = []
    for r in nih_s.itertuples(index=False):
        gt = {c: getattr(r, c) for c in nih.columns if c.startswith("label_")}
        for cond in ("clean", "contrast_severe", "noise_severe",
                     "resolution_severe"):
            rows.append(_row("cxr", "nih", r.image_id, r.image_relpath, cond,
                             "v0", r.patient_id, "patient", r.split, "", gt))
        for schema in ("v0", "v1"):
            rows.append(_row("cxr", "nih", r.image_id, r.image_relpath,
                             "blank_gray", schema, r.patient_id, "patient",
                             r.split, "", gt))

    pad = pd.read_csv(ROOT / "data" / "manifests" / "padchest_shift.csv")
    pad_pos = pad[[f"label_{x}" for x in PATHOLOGY]].eq("positive").any(axis=1)
    pad_s = _patient_sample(pad, p["padchest"], pad_pos,
                            p["padchest_pos"], SEED)
    for r in pad_s.itertuples(index=False):
        gt = {c: getattr(r, c) for c in pad.columns if c.startswith("label_")}
        for cond in ("native", "blank_gray"):
            rows.append(_row("cxr", "padchest", r.image_id, r.image_relpath,
                             cond, "v0", r.patient_id, "patient", r.split, "", gt))

    oi = pd.read_csv(ROOT / "data" / "manifests" / "openi_rrg.csv")
    oi_s = oi.sample(n=min(p["openi"], len(oi)), random_state=SEED)
    for r in oi_s.itertuples(index=False):
        gt = {"reference_text": r.reference_text}
        for cond in ("real", "blank_gray"):
            rows.append(_row("openi", "openi", r.image_id, r.image_relpath,
                             cond, "report", r.patient_id, "patient", r.split,
                             "", gt))
    return pd.DataFrame(rows)


def build_cross(profile: str) -> pd.DataFrame:
    n = PROFILES[profile]["cross"]
    rows = []
    for modality in ("mri", "oct", "derm"):
        man = pd.read_csv(ROOT / "data" / "manifests" / f"crossmodal_{modality}.csv")
        unit = "split_unit"
        # OCT ships no patient identifier, so each image is its own unit.
        man[unit] = man[unit].fillna(man["image_id"]).replace("", pd.NA).fillna(
            man["image_id"])
        # MRI has 233 patients only: two slices per patient are allowed to reach
        # 300 images. Derm stays at one image per lesion to avoid near-duplicates.
        per_unit = 2 if modality == "mri" else 1
        cand = man.groupby(unit, dropna=False).head(per_unit)
        picked = cand.sample(n=min(n, len(cand)), random_state=SEED)
        for r in picked.itertuples(index=False):
            gt = {"label": r.label}
            conds = [("clean", "v0"), ("contrast_severe", "v0"),
                     ("noise_severe", "v0"), ("blank_gray", "v0"),
                     ("blank_gray", "v1"), ("blank_phase", "v0")]
            for cond, schema in conds:
                rows.append(_row(modality, modality, r.image_id, r.image_relpath,
                                 cond, schema, r.split_unit,
                                 "image" if modality == "oct" else
                                 ("lesion" if modality == "derm" else "patient"),
                                 "", r.label, gt))
    return pd.DataFrame(rows)


def _row(cohort, dataset, image_id, relpath, condition, schema,
         cluster_id, cluster_level, split, label, gt) -> dict:
    sev = condition.split("_", 1)[1] if "_" in condition else "none"
    return {
        "probe_id": f"{cohort}|{image_id}|{condition}|{schema}",
        "cohort": cohort,
        "modality": "cxr" if cohort == "cxr" else cohort,
        "dataset": dataset,
        "image_id": str(image_id),
        "image_relpath": str(relpath),
        "condition": condition,
        "severity": sev,
        "schema": schema,
        "cluster_id": str(cluster_id),
        "cluster_level": cluster_level,
        "split": str(split),
        "label": str(label),
        "ground_truth_json": json.dumps(gt, ensure_ascii=False),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full_multimodal",
                    choices=list(PROFILES))
    args = ap.parse_args()
    ensure_dirs()

    df = pd.concat([build_cxr(args.profile), build_cross(args.profile)],
                   ignore_index=True)
    df = df.drop_duplicates("probe_id").sort_values("probe_id")
    path = MANIFEST_DIR / f"probe_{args.profile}.csv"
    df.to_csv(path, index=False)

    lock = {
        "profile": args.profile,
        "rows": int(len(df)),
        "by_cohort": df["cohort"].value_counts().to_dict(),
        "by_condition": df["condition"].value_counts().to_dict(),
        "seed": SEED,
        "sha256": sha256_file(path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (MANIFEST_DIR / f"probe_{args.profile}.lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8")
    print(json.dumps(lock, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
