"""CheXpert-plus-RRG report-generation subset utilities."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image


def load_table(parquet_path: str | Path) -> pd.DataFrame:
    return pq.read_table(parquet_path).to_pandas()


def _decode_image(image_field: Any) -> Image.Image:
    if isinstance(image_field, dict):
        if image_field.get("bytes") is not None:
            return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB")
        if image_field.get("path"):
            return Image.open(image_field["path"]).convert("RGB")
    if isinstance(image_field, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_field)).convert("RGB")
    raise TypeError(f"Unsupported image field type: {type(image_field)}")


def export_section(
    parquet_path: str | Path,
    output_dir: str | Path,
    section_name: str,
    text_column: str,
) -> pd.DataFrame:
    root = Path(output_dir)
    image_dir = root / "images" / section_name
    image_dir.mkdir(parents=True, exist_ok=True)
    df = load_table(parquet_path)
    rows = []
    for i, row in df.iterrows():
        image = _decode_image(row["main_image"])
        image_id = f"{section_name}_{i:05d}"
        rel = f"images/{section_name}/{image_id}.png"
        image.save(root / rel)
        rows.append({
            "image_id": image_id,
            "image_relpath": rel,
            "dataset": "chexpert_plus_rrg",
            "split": "external_report",
            "task": section_name,
            "reference_text": row.get(text_column, ""),
            "default_prompt": row.get("default_prompt", ""),
        })
    out = pd.DataFrame(rows)
    out_path = root / f"{section_name}_manifest.csv"
    out.to_csv(out_path, index=False)
    return out


def export_all(rrg_root: str | Path, output_dir: str | Path) -> pd.DataFrame:
    rrg_root = Path(rrg_root)
    out_dir = Path(output_dir)
    findings = export_section(
        rrg_root / "findings section" / "valid-00000-of-00001.parquet",
        out_dir,
        "findings_section",
        "findings_section",
    )
    impression = export_section(
        rrg_root / "impression section" / "valid-00000-of-00001.parquet",
        out_dir,
        "impression_section",
        "findings_section",  # the parquet text column is named findings_section in both files
    )
    return pd.concat([findings, impression], ignore_index=True)
