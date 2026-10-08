"""Inference runner for structured CXR VLM outputs with progress + resume."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from tqdm import tqdm

from .schema import parse_vlm_output


def _existing_image_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ids.add(str(json.loads(line).get("image_id")))
        except Exception:
            continue
    return ids


def run_manifest(
    model,
    manifest_csv: str | Path,
    images_root: str | Path,
    output_dir: str | Path,
    prompt: str,
    limit: Optional[int] = None,
    transform: Optional[Callable[[Path], Path]] = None,
    resume: bool = False,
    max_new_tokens: int = 256,
) -> dict:
    manifest = pd.read_csv(manifest_csv)
    if limit:
        manifest = manifest.head(limit)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_root = Path(images_root)

    raw_path = out_dir / "raw_outputs.jsonl"
    pred_path = out_dir / "predictions.jsonl"
    done = _existing_image_ids(raw_path) if resume else set()
    if resume and done:
        before = len(manifest)
        manifest = manifest[~manifest["image_id"].astype(str).isin(done)].reset_index(drop=True)
        print(f"[resume] skipping {before - len(manifest)} already-processed images", flush=True)
    if len(manifest) == 0:
        summary = {"n": 0, "parse_failures": 0, "parse_failure_rate": 0.0, "resumed": resume}
        (out_dir / "parse_failure_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary

    mode = "a" if resume else "w"
    failures = 0
    model_name = getattr(model, "name", "model")
    pbar = tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc=f"{model_name} | {Path(manifest_csv).stem}",
        unit="img",
        dynamic_ncols=True,
    )
    progress_log_path = out_dir / "progress.log"
    progress_json_path = out_dir / "progress.json"
    start_all = time.time()
    with progress_log_path.open("a" if resume else "w", encoding="utf-8") as progress_f,          raw_path.open(mode, encoding="utf-8") as raw_f, pred_path.open(mode, encoding="utf-8") as pred_f:
        for _, row in pbar:
            src = images_root / row["image_relpath"]
            image_path = transform(src) if transform else src
            start = time.time()
            try:
                raw = model.generate(image_path, prompt, max_new_tokens=max_new_tokens)
                error = ""
            except Exception as exc:
                raw = ""
                error = f"generation_error: {exc}"
            elapsed = time.time() - start

            parsed = parse_vlm_output(raw)
            if error:
                parsed.parse_success = False
                parsed.parse_error = error
            if not parsed.parse_success:
                failures += 1

            raw_f.write(json.dumps({
                "image_id": row.get("image_id"),
                "image_relpath": row.get("image_relpath"),
                "raw_output": raw,
                "error": error,
                "elapsed_sec": elapsed,
            }, ensure_ascii=False) + "\n")
            raw_f.flush()

            rec = parsed.to_dict()
            rec.update({
                "image_id": row.get("image_id"),
                "dataset": row.get("dataset"),
                "split": row.get("split"),
                "ground_truth": {k: row[k] for k in row.index if str(k).startswith("label_")},
            })
            pred_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pred_f.flush()

            done_n = pbar.n
            elapsed_total = time.time() - start_all
            eta = elapsed_total / done_n * (len(manifest) - done_n) if done_n else None
            stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            base = f"[{stamp}] {done_n}/{len(manifest)} image_id={row.get('image_id')} elapsed={elapsed:.2f}s failures={failures}"
            if eta is not None:
                base += f" eta_sec={eta:.0f}"
            progress_f.write(base + chr(10))
            progress_f.flush()
            progress_json_path.write_text(json.dumps({
                "model": model_name,
                "manifest": str(manifest_csv),
                "total": int(len(manifest)),
                "done": int(done_n),
                "failures": int(failures),
                "elapsed_sec": round(elapsed_total, 2),
                "eta_sec": round(eta, 2) if eta is not None else None,
                "last_image_id": row.get("image_id"),
                "last_elapsed_sec": round(elapsed, 2),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }, indent=2), encoding="utf-8")
            pbar.set_postfix(
                fail=failures,
                fail_rate=f"{failures / (done_n or 1):.3f}",
                last=f"{elapsed:.1f}s",
            )

    summary = {
        "n": int(len(manifest)),
        "parse_failures": int(failures),
        "parse_failure_rate": failures / len(manifest),
        "resumed": resume,
    }
    (out_dir / "parse_failure_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary
