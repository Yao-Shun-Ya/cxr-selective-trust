"""Pre-flight validation for the cross-modal arm (run before a multi-day queue).

Checks a covering slice rather than the full cross product: every model appears
at least once, every modality appears at least once, both schemas appear, and one
combination includes a real-vs-real pair so the P2 path is exercised.

For each combination it records the parse rate, seconds per cell, the peak VRAM
reported by the runner, and one raw reply for eyeballing. Nothing here is a
scientific result; it is the evidence that the thing will survive a week.

    python scripts/125_validate_crossmodal.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "crossmodal_validate"

PY_MODERN = ROOT / ".venv-modern" / "Scripts" / "python.exe"
PY_CHEX = ROOT / ".venv-chexagent" / "Scripts" / "python.exe"
# InternVL's remote code needs the older transformers in .venv-vl.
PY_VL = ROOT / ".venv-vl" / "Scripts" / "python.exe"

# (model, modality, schema, batch, vram_budget_gb, interpreter)
COMBOS = [
    ("medgemma_15_4b_it", "mri", "v0", 8, 14, PY_MODERN),
    ("medgemma_15_4b_it", "oct", "v0", 8, 14, PY_MODERN),
    ("medgemma_15_4b_it", "derm", "v0", 8, 14, PY_MODERN),
    ("medgemma_15_4b_it", "derm", "v1", 8, 14, PY_MODERN),
    ("internvl3_5_8b", "derm", "v0", 2, 26, PY_VL),
    ("qwen3_vl_8b_instruct", "oct", "v0", 2, 26, PY_MODERN),
    ("chexagent_8b", "derm", "v0", 1, 26, PY_CHEX),
]


def subset(manifest: Path) -> pd.DataFrame:
    """Two rows: one image and its real' partner.

    Two rather than three because the expensive case (MedGemma-1.5 on the
    seven-label dermoscopy prompt) runs ~16 s/cell, and the third row only adds
    a second class for eyeballing. The partner is kept so the P2 path is live.
    """
    df = pd.read_csv(manifest)
    first = df.iloc[0]
    partner = df[df["image_id"] == first["pair_id"]].iloc[0]
    return pd.DataFrame([first, partner]).drop_duplicates("image_id")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, modality, schema, bs, budget, py in COMBOS:
        tag = f"{modality}_{schema}_{model}"
        man = OUT / "manifests" / f"{tag}.csv"
        man.parent.mkdir(parents=True, exist_ok=True)
        subset(ROOT / "data" / "manifests" / f"crossmodal_{modality}.csv").to_csv(
            man, index=False)
        outdir = OUT / tag
        cmd = [str(py), str(ROOT / "scripts" / "121_crossmodal_benchmark.py"),
               "--model", model, "--modality", modality, "--schema", schema,
               "--manifest", str(man), "--output-dir", str(outdir),
               "--batch-size", str(bs), "--max-new-tokens", "1024",
               "--vram-budget-gb", str(budget)]
        print(f"[run] {tag}", flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        log = proc.stdout + "\n" + proc.stderr
        (OUT / f"{tag}.log").write_text(log, encoding="utf-8")

        path = outdir / "shift_predictions.jsonl"
        cells = parsed = 0
        secs, comp, sample = [], [], ""
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                cells += 1
                parsed += bool((obj.get("parsed") or {}).get("parse_success"))
                secs.append(float(obj.get("elapsed_sec") or 0))
                comp.append(float(obj.get("label_completeness") or 0))
                if obj["condition"] == "clean" and not sample:
                    sample = (obj.get("raw_output") or "")[:220].replace("\n", " ")

        rows.append({
            "model": model, "modality": modality, "schema": schema,
            "exit": proc.returncode, "cells": cells, "parsed": parsed,
            "parse_rate": parsed / cells if cells else 0.0,
            "label_completeness": (sum(comp) / len(comp)) if comp else 0.0,
            "sec_per_cell": sum(secs) / len(secs) if secs else float("nan"),
            "vram": next((ln.strip() for ln in log.splitlines()
                          if ln.startswith("[vram] peak")), ""),
            "sample": sample,
        })
        mean_sec = (sum(secs) / len(secs)) if secs else float("nan")
        print(f"    exit={proc.returncode} cells={cells} "
              f"parsed={parsed} {mean_sec:.2f}s/cell", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "validation.csv", index=False)
    print("\n" + df[["model", "modality", "schema", "exit", "cells", "parsed",
                     "parse_rate", "label_completeness",
                     "sec_per_cell"]].to_string(index=False))
    print("\n--- one clean reply per combination ---")
    for r in rows:
        print(f"\n[{r['model']} | {r['modality']} | {r['schema']}] {r['sample'][:200]}")
    return 0 if (df["exit"] == 0).all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
