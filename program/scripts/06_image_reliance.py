#!/usr/bin/env python
"""Image-reliance experiment: original vs blank vs strong blur vs shuffled image.

Example:
  python scripts/06_image_reliance.py \
    --model medgemma_4b_it \
    --manifest data/manifests/nih_pilot_1000.csv \
    --images-root data/nih/images \
    --output-dir outputs/reliance/medgemma_100 \
    --limit 100 \
    --load-in-4bit
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image, ImageFilter
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



def _reliance_key(rec: dict) -> tuple:
    return (str(rec.get("image_id")), str(rec.get("condition")))


# Only "this wrapper cannot batch" is recoverable by going one image at a time.
# A driver, allocator or CUDA failure must propagate so the stage exits non-zero
# instead of becoming a file of per-row error strings with exit code 0.
_BATCH_REJECTION_ERRORS = (TypeError, ValueError, AttributeError, AssertionError,
                           NotImplementedError, KeyError, IndexError)


def load_done(path: Path) -> set[tuple[str, str]]:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if (obj.get("parsed") or {}).get("parse_success"):
                done.add((str(obj.get("image_id")), str(obj.get("condition"))))
        except Exception:
            continue
    return done


def _save_atomic(image, path: Path) -> None:
    """Write a PNG via temp file + os.replace.

    A plain ``save()`` is not atomic: an interruption leaves a truncated PNG that
    looks like a finished file ever after. That matters here because the blank and
    blur controls are regenerated material -- a corrupt control image silently
    becomes the "image" the model is shown.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".",
                               suffix=".tmp")
    os.close(fd)
    try:
        image.save(tmp, format="PNG")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def make_conditions(manifest: pd.DataFrame, images_root: Path, out_dir: Path, seed: int = 0):
    rng = random.Random(seed)
    image_ids = manifest["image_id"].astype(str).tolist()
    # One pass to build id -> relpath. The previous version filtered the whole
    # dataframe (`manifest["image_id"].astype(str) == shuffled_id`) once per image,
    # which is O(n^2) with a full string cast each time -- about 1-3 minutes of pure
    # CPU for 1,506 images, paid again on every stage start. The dict changes
    # nothing semantically: rng is consumed in the same order, so the same partner
    # is chosen for the same image.
    relpath_of = dict(zip(image_ids, manifest["image_relpath"].astype(str).tolist()))
    records = []
    for idx, row in manifest.iterrows():
        image_id = str(row["image_id"])
        src = images_root / row["image_relpath"]
        orig = Image.open(src).convert("RGB")
        blank = Image.new("RGB", orig.size, (128, 128, 128))
        blur = orig.filter(ImageFilter.GaussianBlur(radius=8))
        others = [x for x in image_ids if x != image_id]
        shuffled_id = rng.choice(others) if others else image_id
        shuffled_path = images_root / relpath_of.get(shuffled_id, row["image_relpath"])
        condition_paths = {
            "original": src,
            "blank": out_dir / "images" / "blank" / f"{image_id}.png",
            "blur": out_dir / "images" / "blur" / f"{image_id}.png",
            "shuffle": shuffled_path,
        }
        # Regenerate only what is missing. The controls are ~106 s of PNG encoding
        # for 1,506 images, paid again on every stage start -- a resume that has
        # already built them would spend that time rewriting identical pixels.
        #
        # Skipping is only safe because _save_atomic makes the write atomic: with a
        # plain save(), an interruption could leave a truncated PNG that this
        # existence check would then trust for the rest of the run. 43 always skipped
        # and had exactly that hazard; this file always rewrote and paid the cost.
        # Both now skip, both atomically.
        if not condition_paths["blank"].exists():
            _save_atomic(blank, condition_paths["blank"])
        if not condition_paths["blur"].exists():
            _save_atomic(blur, condition_paths["blur"])
        records.append({"image_id": image_id, "original_relpath": row["image_relpath"],
                        "conditions": condition_paths})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--conditions", nargs="+", default=["original", "blank", "blur", "shuffle"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="internal_test,source_val",
                        help="comma-separated splits to keep (default: the evaluation "
                             "splits); pass 'all' to use every manifest row")
    parser.add_argument("--all-splits", action="store_true",
                        help="explicitly allow training rows; equivalent to --split all")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()

    # Greedy decoding is deterministic in exact arithmetic, but batch-dependent
    # kernels can still change the last bits of a logit. Seeding torch makes the
    # run reproducible for a fixed batch size; the batch-size comparison in
    # scripts/110_greedy_determinism.py measures what remains.
    import torch
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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


    if args.all_splits:
        args.split = "all"
    if str(args.split).lower() == "all":
        args.split = None

    manifest = pd.read_csv(args.manifest)
    if args.split and "split" in manifest.columns:
        wanted = [s.strip() for s in args.split.split(",") if s.strip()]
        before = len(manifest)
        manifest = manifest[manifest["split"].isin(wanted)].copy()
        print(f"split filter {wanted}: {before} -> {len(manifest)} images", flush=True)
    elif args.split:
        print(f"[warn] --split {args.split} ignored: manifest has no 'split' column")
    # Shuffle BEFORE the limit: --limit takes the first N rows of the file, which
    # on nih_shared_5000.csv is ordered by image_id and therefore is not a sample.
    if len(manifest):
        manifest = manifest.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.limit:
        manifest = manifest.head(args.limit)
    manifest = manifest.copy()
    print(f"manifest: {len(manifest)} images", flush=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = make_conditions(manifest, Path(args.images_root), out_dir, seed=args.seed)
    # Condition-major, for the same reason 08 sorts its pending list: a batch runs
    # until its LONGEST member finishes, and with image-major order every batch
    # mixes `blank` (where MedGemma-1.5 opens a 1024-token reasoning trace) with
    # `original` (where it answers in ~170 tokens). Homogeneous batches mean only
    # the blank batches pay for the blank behaviour.
    jobs = [(r, c) for c in args.conditions for r in records if c in r["conditions"]]
    out_path = out_dir / "image_reliance_predictions.jsonl"

    # Crash-safe writing (see src/cxr_trust/inference/durable.py). The previous
    # version opened "a" if done else "w" and only flushed: a kill could lose the
    # tail, a torn line survived, and a file holding only failed rows would be
    # reopened with "w" and destroyed.
    with DurableJsonl(out_path, key=_reliance_key) as out:
        pending = [(r, c) for r, c in jobs if out.needs(r["image_id"], c)]
        print(f"resume: {len(jobs) - len(pending)} of {len(jobs)} cells already durable, "
              f"{len(pending)} to run", flush=True)
        pbar = tqdm(total=len(pending),
                    desc=f"{args.model} | image-reliance (bs={args.batch_size})",
                    unit="img", dynamic_ncols=True)
        interrupted = None
        for i in range(0, len(pending), args.batch_size):
            chunk = pending[i:i + args.batch_size]
            images = []
            for rec, condition in chunk:
                image_path = rec["conditions"][condition]
                images.append(Image.open(image_path).convert("RGB"))

            prompts = [prompt] * len(chunk)

            start = time.time()
            try:
                if args.batch_size > 1 and getattr(model, "generate_batch", None):
                    try:
                        raws = model.generate_batch(images, prompts, max_new_tokens=args.max_new_tokens)
                        errs = [""] * len(raws)
                    except _BATCH_REJECTION_ERRORS as exc:
                        # Only "this wrapper cannot batch" is recoverable; driver and
                        # allocator failures propagate so the stage exits non-zero.
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
            for j, (rec, condition) in enumerate(chunk):
                raw = raws[j]
                err = errs[j]
                image_path = rec["conditions"][condition]
                parsed = parse_vlm_output(raw) if not err else parse_vlm_output("")
                rows.append({
                    "image_id": rec["image_id"],
                    "condition": condition,
                    "image_path": str(image_path),
                    "raw_output": raw,
                    "error": err,
                    "elapsed_sec": time_per_img,
                    "parsed": parsed.to_dict(),
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

    # Summarise
    #
    # This block used to require parse_success and then call
    # per_df.groupby("condition"). When every reply failed to parse, per_df was an
    # empty frame with no columns and groupby raised `KeyError: 'condition'` --
    # which is exactly how the reliance stage of the first full run died, and the
    # traceback pointed at pandas rather than at the real cause (all-empty output).
    # Two changes: failures are counted per condition without skipping the image,
    # and a lexical-overlap fallback keeps the summary meaningful when the model
    # answers in prose instead of the JSON contract.
    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_key = {(r["image_id"], r["condition"]): r for r in rows}
    per_image = []
    condition_parse_failures: dict[str, int] = {c: 0 for c in args.conditions}
    condition_generation_failures: dict[str, int] = {c: 0 for c in args.conditions}

    for rec in records:
        for condition in args.conditions:
            cur = by_key.get((rec["image_id"], condition))
            if cur and cur.get("error"):
                condition_generation_failures[condition] = \
                    condition_generation_failures.get(condition, 0) + 1
            elif cur and not cur["parsed"].get("parse_success"):
                condition_parse_failures[condition] = \
                    condition_parse_failures.get(condition, 0) + 1

        orig = by_key.get((rec["image_id"], "original"))
        if not orig:
            continue
        for condition in args.conditions:
            if condition == "original":
                continue
            cur = by_key.get((rec["image_id"], condition))
            if not cur:
                continue
            of = orig["parsed"].get("findings", {}) or {}
            cf = cur["parsed"].get("findings", {}) or {}
            flips = 0
            prob_diffs = []
            for label in of:
                if of[label].get("status") != (cf.get(label) or {}).get("status"):
                    flips += 1
                op = of[label].get("probability_positive")
                cp = (cf.get(label) or {}).get("probability_positive")
                if op is not None and cp is not None:
                    prob_diffs.append(abs(float(op) - float(cp)))

            a = (orig.get("raw_output") or "").strip()
            b = (cur.get("raw_output") or "").strip()
            ta = set(re.findall(r"[a-z]{3,}", a.lower()))
            tb = set(re.findall(r"[a-z]{3,}", b.lower()))
            jaccard = (len(ta & tb) / len(ta | tb)) if (ta or tb) else None

            per_image.append({
                "image_id": rec["image_id"],
                "condition": condition,
                "structured_available": int(bool(prob_diffs)),
                "flip_count": flips if (of and cf) else None,
                "flip_any": int(flips > 0) if (of and cf) else None,
                "mean_abs_prob_diff": (sum(prob_diffs) / len(prob_diffs)) if prob_diffs else None,
                "text_jaccard": jaccard,
                "identical_text": int(bool(a) and a == b),
            })

    per_df = pd.DataFrame(per_image)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_df.to_csv(out_dir / "image_reliance_per_image.csv", index=False)

    summary: dict[str, Any] = {
        "model": args.model,
        "n_images": len(records),
        "cells_on_disk": len(by_key),
        "parse_failures_by_condition": condition_parse_failures,
        "generation_failures_by_condition": condition_generation_failures,
        "conditions": {},
    }
    if per_df.empty:
        summary["note"] = ("no original/condition pairs were available to compare: "
                           "every cell either failed to generate or was not run")
        summary["contrast_source"] = None
    else:
        summary["contrast_source"] = ("probability_vector"
                                      if per_df["structured_available"].any()
                                      else "text_overlap")
        for condition, g in per_df.groupby("condition"):
            numeric = g["mean_abs_prob_diff"].dropna()
            jac = g["text_jaccard"].dropna()
            structured = g["structured_available"].sum()
            summary["conditions"][str(condition)] = {
                "n_valid_pairs": int(len(g)),
                "n_structured": int(structured),
                "image_sensitivity": (float(g["flip_any"].dropna().mean())
                                      if g["flip_any"].notna().any() else None),
                "mean_flip_count": (float(g["flip_count"].dropna().mean())
                                    if g["flip_count"].notna().any() else None),
                "mean_abs_prob_diff": float(numeric.mean()) if len(numeric) else None,
                "mean_text_jaccard": float(jac.mean()) if len(jac) else None,
                "pct_identical_text": round(100.0 * float(g["identical_text"].mean()), 2),
            }
    (out_dir / "image_reliance_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    # raise SystemExit and NOT a bare main(): the interrupted path returns 1, and
    # the orchestrator reads the exit code to decide whether the stage is complete.
    raise SystemExit(main())
