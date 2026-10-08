"""Manifest builders for CXR datasets.

These functions assume the dataset has already been downloaded and that the
user has the official label CSV files. They do not download data.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .labels import (
    CHEXPERT_COLUMNS,
    PRIMARY_LABELS,
    map_padchest_labels,
    normalize_chexpert_cell,
    normalize_nih_findings,
)


def _image_index(images_root: Path) -> dict[str, Path]:
    """Return filename -> path for common image suffixes."""
    suffixes = {".png", ".jpg", ".jpeg"}
    index: dict[str, Path] = {}
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            index.setdefault(p.name.lower(), p)
    return index


def _patient_from_path(path: str) -> str:
    m = re.search(r"(patient\d+)", str(path), flags=re.IGNORECASE)
    return m.group(1).lower() if m else "unknown"


def _study_from_path(path: str) -> str:
    m = re.search(r"(s\d+)", Path(str(path)).name, flags=re.IGNORECASE)
    return m.group(1).lower() if m else "unknown"


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def _make_group_splits(
    df: pd.DataFrame,
    group_col: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.Series:
    groups = sorted(df[group_col].astype(str).unique())
    rng = pd.Series(groups).sample(frac=1.0, random_state=seed).tolist()
    n = len(rng)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    test_groups = set(rng[:n_test])
    val_groups = set(rng[n_test : n_test + n_val])
    split = []
    for g in df[group_col].astype(str):
        if g in test_groups:
            split.append("external_test")
        elif g in val_groups:
            split.append("source_val")
        else:
            split.append("internal_train")
    return pd.Series(split, index=df.index)


def _pilot_sample(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    if not n or n >= len(df):
        return df.reset_index(drop=True)
    # Stratify by number of positive primary labels and no_finding status.
    status_cols = [f"label_{x}" for x in PRIMARY_LABELS if f"label_{x}" in df.columns]
    pos_count = (df[status_cols] == "positive").sum(axis=1) if status_cols else 0
    df = df.copy()
    df["_strata"] = pos_count.astype(str)
    parts = []
    for _, group in df.groupby("_strata", sort=False):
        take = max(1, int(round(n * len(group) / len(df))))
        take = min(take, len(group))
        parts.append(group.sample(n=take, random_state=seed))
    out = pd.concat(parts).drop(columns=["_strata"]).head(n).reset_index(drop=True)
    return out


def _finalize_columns(df: pd.DataFrame, dataset: str, images_root: Path) -> pd.DataFrame:
    out = pd.DataFrame()
    out["image_relpath"] = df["_image_path"].apply(lambda p: Path(p).relative_to(images_root).as_posix())
    out["image_id"] = df["_image_path"].apply(lambda p: Path(p).name)
    out["dataset"] = dataset
    out["patient_id"] = df["_patient_id"].astype(str)
    out["study_id"] = df["_study_id"].astype(str)
    out["split"] = df["split"].astype(str)
    out["view"] = df.get("view", "unknown")
    out["view_detail"] = df.get("view_detail", "unknown")
    for label in PRIMARY_LABELS:
        out[f"label_{label}"] = df[f"label_{label}"].astype(str)
    out["image_hash"] = out["image_relpath"].apply(_stable_hash)
    return out


def build_chexpert_manifest(
    labels_csv: str | Path,
    images_root: str | Path,
    output_csv: str | Path,
    pilot_size: Optional[int] = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    labels_csv = Path(labels_csv)
    images_root = Path(images_root)
    df = pd.read_csv(labels_csv)
    if "Path" not in df.columns:
        raise ValueError("CheXpert labels CSV must contain a 'Path' column.")

    index = _image_index(images_root)
    df["_image_path"] = df["Path"].apply(lambda x: index.get(Path(str(x)).name.lower()))
    df = df[df["_image_path"].notna()].copy()
    if df.empty:
        raise FileNotFoundError("No CheXpert images matched label CSV filenames.")

    for label, col in CHEXPERT_COLUMNS.items():
        if col in df.columns:
            df[f"label_{label}"] = df[col].apply(normalize_chexpert_cell)
        else:
            df[f"label_{label}"] = "not_mentioned"

    df["_patient_id"] = df["Path"].apply(_patient_from_path)
    df["_study_id"] = df["Path"].apply(_study_from_path)
    df["view"] = df.get("Frontal/Lateral", "unknown")
    df["view_detail"] = df.get("AP/PA", "unknown")
    df["split"] = _make_group_splits(df, "_patient_id", val_ratio, test_ratio, seed)
    out = _finalize_columns(df, "chexpert", images_root)
    out = _pilot_sample(out, pilot_size, seed)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def build_nih_manifest(
    labels_csv: str | Path,
    images_root: str | Path,
    output_csv: str | Path,
    pilot_size: Optional[int] = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    labels_csv = Path(labels_csv)
    images_root = Path(images_root)
    df = pd.read_csv(labels_csv)
    image_col = "Image Index" if "Image Index" in df.columns else None
    if image_col is None:
        raise ValueError("NIH labels CSV must contain an 'Image Index' column.")
    index = _image_index(images_root)
    df["_image_path"] = df[image_col].apply(lambda x: index.get(Path(str(x)).name.lower()))
    df = df[df["_image_path"].notna()].copy()
    if df.empty:
        raise FileNotFoundError("No NIH images matched label CSV filenames.")

    parsed = df["Finding Labels"].apply(normalize_nih_findings) if "Finding Labels" in df.columns else None
    for label in PRIMARY_LABELS:
        if parsed is not None:
            df[f"label_{label}"] = parsed.apply(lambda d: d[label])
        else:
            df[f"label_{label}"] = "not_mentioned"

    df["_patient_id"] = df["Patient ID"].astype(str) if "Patient ID" in df.columns else df[image_col].apply(_patient_from_path)
    df["_study_id"] = "unknown"
    df["view"] = df.get("View Position", "unknown")
    df["view_detail"] = "unknown"
    df["split"] = _make_group_splits(df, "_patient_id", val_ratio, test_ratio, seed)
    out = _finalize_columns(df, "nih", images_root)
    out = _pilot_sample(out, pilot_size, seed)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def build_padchest_manifest(
    labels_csv: str | Path,
    images_root: str | Path,
    output_csv: str | Path,
    image_id_col: str = "ImageID",
    labels_col: str = "Labels",
    view_col: str = "View",
    pilot_size: Optional[int] = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    labels_csv = Path(labels_csv)
    images_root = Path(images_root)
    df = pd.read_csv(labels_csv)
    if image_id_col not in df.columns:
        raise ValueError(f"PadChest labels CSV must contain '{image_id_col}'.")
    index = _image_index(images_root)
    df["_image_path"] = df[image_id_col].apply(lambda x: index.get(Path(str(x)).name.lower()))
    df = df[df["_image_path"].notna()].copy()
    if df.empty:
        raise FileNotFoundError("No PadChest images matched label CSV filenames.")

    def parse_labels(value) -> dict[str, str]:
        if value is None or str(value).lower() in {"nan", "none", ""}:
            return map_padchest_labels([])
        raw = [x.strip() for x in re.split(r"[|,;]", str(value)) if x.strip()]
        return map_padchest_labels(raw)

    parsed = df[labels_col].apply(parse_labels) if labels_col in df.columns else None
    for label in PRIMARY_LABELS:
        if parsed is not None:
            df[f"label_{label}"] = parsed.apply(lambda d: d[label])
        else:
            df[f"label_{label}"] = "not_mentioned"

    df["_patient_id"] = (
        df["PatientID"].astype(str)
        if "PatientID" in df.columns
        else df[image_id_col].apply(_patient_from_path)
    )
    df["_study_id"] = "unknown"
    df["view"] = df.get(view_col, "unknown")
    df["view_detail"] = "unknown"
    df["split"] = _make_group_splits(df, "_patient_id", val_ratio, test_ratio, seed)
    out = _finalize_columns(df, "padchest", images_root)
    out = _pilot_sample(out, pilot_size, seed)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
