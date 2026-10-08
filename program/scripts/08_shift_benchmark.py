#!/usr/bin/env python
"""Shift benchmark: clean + blur/noise/JPEG/brightness/contrast/resolution.

Stores one JSONL row per image x condition x severity. Use --resume to continue.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.inference.durable import DurableJsonl  # noqa: E402
from src.cxr_trust.inference.prompts import SIMPLE_SYSTEM_PROMPT, build_simple_prompt  # noqa: E402
from src.cxr_trust.inference.schema import parse_vlm_output  # noqa: E402
from src.cxr_trust.models.registry import MODEL_NAMES, get_model  # noqa: E402

# Test seam: when RUNNER_STUB=1 the model factory is replaced AFTER all imports have
# run. It must be installed here and not from sitecustomize, because src/ has no
# __init__.py, so "src.cxr_trust.*" and "cxr_trust.*" are two separate module
# objects -- a patch applied before this import would miss this one and the runner
# would quietly load a real checkpoint while the test believed it was stubbed.
if os.environ.get("RUNNER_STUB") == "1":  # pragma: no cover - test-only path
    import stub_model

    stub_model.install()



def _shift_key(rec: dict) -> tuple:
    return (str(rec.get("image_id")), str(rec.get("condition")),
            str(rec.get("severity")))


# Only "this wrapper cannot batch" is recoverable by going one image at a time.
# A driver, allocator or CUDA failure must propagate so the stage exits non-zero
# rather than becoming a file of per-row error strings with exit code 0.
_BATCH_REJECTION_ERRORS = (TypeError, ValueError, AttributeError, AssertionError,
                           NotImplementedError, KeyError, IndexError)

DEFAULT_SEVERITIES = {
    "blur": {"mild": 0.5, "moderate": 1.0, "severe": 2.0},
    "noise": {"mild": 0.02, "moderate": 0.05, "severe": 0.10},
    "jpeg": {"mild": 70, "moderate": 40, "severe": 20},
    "brightness": {"mild": 0.8, "moderate": 1.2, "severe": 1.5},
    "contrast": {"mild": 0.7, "moderate": 1.3, "severe": 1.6},
    "resolution": {"mild": 0.5, "moderate": 0.25, "severe": 0.125},
}


def apply_shift(image: Image.Image, shift: str, severity: str) -> Image.Image:
    image = image.convert("RGB")
    if shift == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(DEFAULT_SEVERITIES["blur"][severity])))
    if shift == "noise":
        sigma = float(DEFAULT_SEVERITIES["noise"][severity])
        arr = np.asarray(image, dtype=np.float32) / 255.0
        rng = np.random.default_rng(0)
        arr = np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 1)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if shift == "jpeg":
        quality = int(DEFAULT_SEVERITIES["jpeg"][severity])
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if shift == "brightness":
        return ImageEnhance.Brightness(image).enhance(float(DEFAULT_SEVERITIES["brightness"][severity]))
    if shift == "contrast":
        return ImageEnhance.Contrast(image).enhance(float(DEFAULT_SEVERITIES["contrast"][severity]))
    if shift == "resolution":
        factor = float(DEFAULT_SEVERITIES["resolution"][severity])
        w, h = image.size
        small = image.resize((max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)
    raise ValueError(f"unknown shift: {shift}")


def _select_rows(manifest: pd.DataFrame, split, seed: int) -> pd.DataFrame:
    """Restrict a manifest to the evaluation splits, then SHUFFLE.

    ``--limit`` takes the first N rows of the file, which is NOT a sample: on
    nih_shared_5000.csv the rows are ordered by image_id, so the first 300 rows
    hold only 138 evaluation images and 12 pneumonia positives. Worse, the file
    also carries 3,268 ``internal_train`` rows, so a large ``--limit`` silently
    evaluates on training data.

    ``--split`` filters explicitly; shuffling turns ``--limit`` into an
    arbitrary but reproducible subset of the evaluation splits instead of
    whatever happens to sort first.
    """
    if split:
        wanted = [s.strip() for s in str(split).split(",") if s.strip()]
        if "split" not in manifest.columns:
            print(f"[warn] --split {split} ignored: manifest has no 'split' column")
        else:
            before = len(manifest)
            manifest = manifest[manifest["split"].isin(wanted)].copy()
            print(f"split filter {wanted}: {before} -> {len(manifest)} images")
    if len(manifest):
        manifest = manifest.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return manifest


def load_done(path: Path) -> set[tuple[str, str, str]]:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if (obj.get("parsed") or {}).get("parse_success"):
                done.add((str(obj.get("image_id")), str(obj.get("condition")), str(obj.get("severity"))))
        except Exception:
            continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--shifts", nargs="+", default=["blur", "noise", "jpeg", "brightness", "contrast", "resolution"])
    parser.add_argument("--severities", nargs="+", default=["mild", "moderate", "severe"])
    parser.add_argument("--split", default=None,
                        help="comma-separated splits to keep, e.g. internal_test,source_val; "
                             "without it every row in the manifest is used")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the manifest shuffle that --limit subsamples")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()

    model = get_model(
        args.model,
        root=ROOT,
        device_map="auto",
        load_in_4bit=args.load_in_4bit,
    )
    model.load()
    prompt = SIMPLE_SYSTEM_PROMPT + chr(10) * 2 + build_simple_prompt()

    if os.environ.get("RUNNER_STUB") == "1":
        # Fail loudly if the stub did not take over: a silently un-stubbed test run
        # would load a real checkpoint and still "pass".
        assert type(model).__name__ == "_StubVLM", (
            "RUNNER_STUB=1 but get_model returned "
            + type(model).__name__ + "; the stub did not patch this import path")


    manifest = pd.read_csv(args.manifest)
    manifest = _select_rows(manifest, args.split, args.seed)
    if args.limit:
        manifest = manifest.head(args.limit)
    print(f"manifest: {len(manifest)} images"
          f"{' (split=' + args.split + ')' if args.split else ' (all splits)'}", flush=True)
    images_root = Path(args.images_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shift_predictions.jsonl"

    jobs = []
    for _, row in manifest.iterrows():
        jobs.append((row, "clean", "none", None))
        for shift in args.shifts:
            for severity in args.severities:
                jobs.append((row, shift, severity, DEFAULT_SEVERITIES[shift][severity]))

    # Crash-safe writing. The old version opened the file with
    # `"a" if done else "w"` and only called flush(): a kill could lose the tail,
    # a torn last line survived, and -- because `done` was keyed on parse_success
    # -- a file containing only failed rows would be opened "w" and wiped. This
    # stage is the longest of the whole study (19 cells per image), so it is the
    # one that most needs to survive an interruption.
    with DurableJsonl(out_path, key=_shift_key) as out:
        # A shift cell is keyed by (image, condition, severity), not the two-part
        # (image, condition) of the other runners, so membership is tested against
        # the key set directly rather than through the two-argument helper.
        pending = [j for j in jobs
                   if (str(j[0]["image_id"]), str(j[1]), str(j[2])) not in out.done]
        # Group the work by condition before batching it.
        #
        # A batch runs until its LONGEST member finishes, so a mixed batch costs
        # what its slowest cell costs. That is invisible while every cell emits
        # about the same number of tokens, and very visible once a model's output
        # length depends on the condition: MedGemma-1.5 opens a reasoning trace on
        # the hard conditions (blank 100/100, resolution/severe 68/100) and needs
        # the 1024-token budget for them. Measured on this runner with the jobs in
        # their natural order (image-major), a batch of 64 spanned ~3.4 images and
        # therefore almost always contained a reasoning cell: 3.30 s/cell for a
        # stage whose 256-token version measured 0.825 -- 4x, paid by cells that
        # mostly answer in 170 tokens.
        #
        # Sorting by (condition, severity) makes each batch homogeneous, so only
        # the batches that really contain long replies pay for them. Row order in
        # the output file changes; nothing else does (the durable key does not
        # depend on order, and every analysis groups by condition).
        pending.sort(key=lambda j: (str(j[1]), str(j[2])))
        print(f"resume: {len(jobs) - len(pending)} of {len(jobs)} cells already durable, "
              f"{len(pending)} to run", flush=True)
        if not pending:
            print("Nothing to do.")
        pbar = tqdm(total=len(pending),
                    desc=f"{args.model} | shift (bs={args.batch_size})",
                    unit="img", dynamic_ncols=True)
        interrupted = None
        for i in range(0, len(pending), args.batch_size):
            chunk = pending[i:i + args.batch_size]
            images = []
            for row, condition, severity, param in chunk:
                image = Image.open(images_root / row["image_relpath"]).convert("RGB")
                if condition != "clean":
                    image = apply_shift(image, condition, severity)
                images.append(image)

            prompts = [prompt] * len(chunk)

            start = time.time()
            try:
                if args.batch_size > 1 and getattr(model, "generate_batch", None):
                    try:
                        raws = model.generate_batch(images, prompts, max_new_tokens=args.max_new_tokens)
                        errs = [""] * len(raws)
                    except _BATCH_REJECTION_ERRORS as exc:
                        # Only "this wrapper cannot batch" is recoverable; a driver
                        # or allocator failure propagates so the stage exits non-zero
                        # instead of being recorded as a file of error strings.
                        print(f"[warn] batching rejected ({type(exc).__name__}: {exc}); "
                              f"falling back to one image at a time", flush=True)
                        raws, errs = [], []
                        for img in images:
                            raws.append(model.generate(img, prompt, max_new_tokens=args.max_new_tokens))
                            errs.append("")
                else:
                    raws, errs = [], []
                    for img in images:
                        raws.append(model.generate(img, prompt, max_new_tokens=args.max_new_tokens))
                        errs.append("")
            except BaseException as exc:  # noqa: BLE001  (incl. MemoryError)
                interrupted = exc
                break

            elapsed = time.time() - start
            time_per_img = elapsed / len(chunk)

            rows = []
            for j, (row, condition, severity, param) in enumerate(chunk):
                raw = raws[j]
                error = errs[j]
                parsed = parse_vlm_output(raw) if not error else parse_vlm_output("")
                rows.append({
                    "image_id": str(row["image_id"]),
                    "dataset": str(row.get("dataset", "nih")),
                    "split": str(row.get("split", "")),
                    "condition": condition,
                    "severity": severity,
                    "param": param,
                    "raw_output": raw,
                    "error": error,
                    "elapsed_sec": time_per_img,
                    "parsed": parsed.to_dict(),
                    "ground_truth": {k: row[k] for k in row.index if str(k).startswith("label_")},
                })
            out.write_many(rows)      # append + fsync + durable byte marker
            pbar.update(len(chunk))
            pbar.set_postfix(sec_per_img=f"{time_per_img:.1f}s")

        pbar.close()
        stats = out.stats()

    if interrupted is not None:
        print(f"[INTERRUPTED] {type(interrupted).__name__}: {interrupted}", flush=True)
        print(f"[INTERRUPTED] {stats['records_on_disk']} cells are durable in "
              f"{out_path.name}. Re-run the same command to continue.", flush=True)
        return 1

    print(f"done; wrote {out_path} ({stats['records_on_disk']} cells, "
          f"{stats['appended_this_run']} added this run, "
          f"{stats['rows_collapsed_on_open']} superseded rows collapsed)")
    return 0


if __name__ == "__main__":
    # raise SystemExit(main()) and NOT main(): the interrupted path returns 1 and
    # the orchestrator decides whether this stage counts as complete from the exit
    # code. Discarding it would make a truncated stage look like a success.
    raise SystemExit(main())
