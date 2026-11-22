"""NIH ChestX-ray14 data preparation utilities."""

from __future__ import annotations

import random
import tarfile

from tqdm import tqdm
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

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

NIH_TO_PRIMARY = {
    "atelectasis": "atelectasis",
    "cardiomegaly": "cardiomegaly",
    "consolidation": "consolidation",
    "edema": "edema",
    "effusion": "pleural_effusion",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "no finding": "no_finding",
    "infiltration": "lung_opacity",
    "mass": "mass_nodule",
    "nodule": "mass_nodule",
    "emphysema": "emphysema",
    "fibrosis": "fibrosis",
    "pleural_thickening": "pleural_thickening",
    "hernia": "hernia",
}

SECONDARY_LABELS = [
    "lung_opacity",
    "mass_nodule",
    "emphysema",
    "fibrosis",
    "pleural_thickening",
    "hernia",
]


def load_metadata(metadata_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_csv)
    required = {"Image Index", "Finding Labels", "Patient ID", "View Position"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"NIH metadata missing columns: {sorted(missing)}")
    return df


def _parse_findings(value) -> dict[str, str]:
    status = {x: "negative" for x in PRIMARY_LABELS + SECONDARY_LABELS}
    text = "" if value is None else str(value).strip().lower()
    if not text or text == "nan":
        return status
    found = [x.strip() for x in text.split("|") if x.strip()]
    if "no finding" in found:
        status["no_finding"] = "positive"
    for f in found:
        mapped = NIH_TO_PRIMARY.get(f)
        if mapped:
            status[mapped] = "positive"
    return status


def build_manifest(
    metadata_csv: str | Path,
    train_val_list: str | Path,
    test_list: str | Path,
    output_csv: str | Path,
    pilot_size: Optional[int] = None,
    val_ratio: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    df = load_metadata(metadata_csv)
    train_val = set(Path(train_val_list).read_text(encoding="utf-8").splitlines())
    test = set(Path(test_list).read_text(encoding="utf-8").splitlines())

    df = df.copy()
    df["image_id"] = df["Image Index"].astype(str)
    df["patient_id"] = df["Patient ID"].astype(str)
    df["view"] = df["View Position"].astype(str)
    df["image_relpath"] = df["image_id"].apply(lambda x: f"images/{x}")

    parsed = df["Finding Labels"].apply(_parse_findings)
    for label in PRIMARY_LABELS + SECONDARY_LABELS:
        df[f"label_{label}"] = parsed.apply(lambda d: d[label])

    # Official test list is used as the internal held-out test set.
    df["split"] = "internal_train"
    df.loc[df["image_id"].isin(test), "split"] = "internal_test"

    # Split train_val patients into internal_train / source_val.
    tv = df[df["image_id"].isin(train_val)].copy()
    patients = sorted(tv["patient_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = int(round(len(patients) * val_ratio))
    val_patients = set(patients[:n_val])
    df.loc[df["image_id"].isin(train_val) & df["patient_id"].isin(val_patients), "split"] = "source_val"

    if pilot_size is not None and pilot_size < len(df):
        # Stratify by number of positive primary labels.
        pos_count = (df[[f"label_{x}" for x in PRIMARY_LABELS]] == "positive").sum(axis=1)
        df["_strata"] = pos_count.astype(str)
        parts = []
        for _, group in df.groupby("_strata", sort=False):
            take = max(1, int(round(pilot_size * len(group) / len(df))))
            take = min(take, len(group))
            parts.append(group.sample(n=take, random_state=seed))
        df = pd.concat(parts).drop(columns=["_strata"]).head(pilot_size).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    out_cols = [
        "image_id", "image_relpath", "dataset", "patient_id", "split", "view",
        *[f"label_{x}" for x in PRIMARY_LABELS + SECONDARY_LABELS],
    ]
    df["dataset"] = "nih"
    out = df[out_cols]
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def iter_archives(images_archive_dir: str | Path) -> Iterable[Path]:
    root = Path(images_archive_dir)
    files = sorted([p for p in root.glob("*.tar.gz") if "images_" in p.name])
    # Normalize the browser-download filename.
    weird = root / "attachment; filename=images_001.tar.gz"
    if weird.exists():
        # Keep it but treat as an archive named images_001 internally.
        files = [p for p in files if p.name != weird.name] + [weird]
    return files


def extract_images(
    images_archive_dir: str | Path,
    output_dir: str | Path,
    image_ids: Optional[set[str]] = None,
    max_images: Optional[int] = None,
) -> int:
    """Extract images from NIH tar.gz archives with a progress bar."""
    archive_dir = Path(images_archive_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    archives = list(iter_archives(archive_dir))
    wanted = set(image_ids) if image_ids else None
    extracted = 0
    total = len(wanted) if wanted is not None else None
    pbar = tqdm(total=total, desc="Extracting NIH images", unit="img", dynamic_ncols=True)
    for archive in archives:
        pbar.set_description(f"Extracting {archive.name[:32]}")
        with tarfile.open(archive, "r:gz") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                name = Path(member.name).name
                if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                if wanted is not None and name not in wanted:
                    continue
                target = out_root / "images" / name
                if target.exists():
                    extracted += 1
                    pbar.update(1)
                else:
                    member.name = f"images/{name}"
                    tf.extract(member, path=out_root)
                    extracted += 1
                    pbar.update(1)
                if max_images is not None and extracted >= max_images:
                    pbar.close()
                    return extracted
                if wanted is not None and extracted >= len(wanted):
                    pbar.close()
                    return extracted
    pbar.close()
    return extracted
