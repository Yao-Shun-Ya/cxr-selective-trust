#!/usr/bin/env python
"""PadChest geographic-shift runner: the cross-hospital layer of the 6K design.

Why PadChest gets its own runner instead of the 19-cell shift matrix
-------------------------------------------------------------------
``08_shift_benchmark.py`` applies six *synthetic* degradations (blur, noise,
JPEG, brightness, contrast, resolution) at three severities each. That is the
right instrument for the question "how much artificial image degradation does a
model tolerate before its selective signal collapses", which is the NIH layer's
question.

It is the wrong instrument for PadChest. PadChest's scientific value is that it
is a **different hospital in a different country**: different scanners, different
acquisition parameters, different population and different label vocabulary.
That difference is already present in the pixels. Re-applying a Gaussian blur on
top of it answers a synthetic-robustness question that NIH already answers, and
burns roughly 75% of the compute budget doing so. The authoritative 6K plan
(``docs/41``) states this role explicitly: "External Geographic/Hospital Shift.

So this runner measures the *native* shift:

``original``
    the PadChest radiograph as published -- already out-of-domain relative to the
    NIH evaluation set, with no artificial perturbation added.
``blank``
    the same study with the pixels replaced by flat grey. The ``original`` minus
    ``blank`` gap is the out-of-domain image-reliance signal: a model that keeps
    its answers identical on a blank image was never reading the foreign domain
    either, and that is exactly the failure this study is about.

Why a blank control rather than a synthetic grid
------------------------------------------------
Image reliance is a *contrast*, so it needs two conditions with everything else
held constant. One control is enough to measure it, and it costs 2 cells/image
instead of 19. That is the whole point of the per-layer design: 1,500 images x 2
conditions instead of 1,500 x 19, on every one of seven models.

Output
------
Deliberately byte-compatible with ``06_image_reliance.py``, so
``18_selective_metrics_v2.py``, ``19_accept_clarify_abstain.py`` and
``42_geographic_shift_analysis.py`` consume it with no changes:

    <output-dir>/image_reliance_predictions.jsonl
    <output-dir>/image_reliance_per_image.csv
    <output-dir>/image_reliance_summary.json
    <output-dir>/blank_images/<image_id>            (the control, kept for audit)

Crash safety
------------
Appends go through ``DurableJsonl``: every batch is fsynced and a byte-offset
sidecar is written atomically, so a power cut loses at most the in-flight batch
and the next run resumes without redoing finished work and without keeping a
torn row. See ``src/cxr_trust/inference/durable.py``.

Usage::

    python scripts/43_run_padchest_shift.py \\
        --model medgemma_4b_it \\
        --manifest data/manifests/padchest_shift.csv \\
        --images-root data/padchest \\
        --output-dir outputs/matrix/padchest/medgemma_4b_it \\
        --conditions original blank \\
        --batch-size 16 --load-in-4bit

    # smoke: one image, print the verdict
    python scripts/43_run_padchest_shift.py --model medgemma_4b_it --limit 1 \\
        --manifest data/manifests/padchest_shift.csv --images-root data/padchest \\
        --output-dir /tmp/pc_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.inference.durable import DurableJsonl, read_jsonl  # noqa: E402
from src.cxr_trust.inference.prompts import SIMPLE_SYSTEM_PROMPT, build_simple_prompt  # noqa: E402
from src.cxr_trust.inference.schema import parse_vlm_output  # noqa: E402
from src.cxr_trust.models.registry import MODEL_NAMES, get_model  # noqa: E402

# Test seam: see 44_run_openi_report.py for why the stub is installed here rather
# than from sitecustomize (dual module identity because src/ has no __init__.py).
if os.environ.get("RUNNER_STUB") == "1":  # pragma: no cover - test-only path
    import stub_model
    stub_model.install()


BLANK_RGB = (128, 128, 128)
DEFAULT_CONDITIONS = ["original", "blank"]
FINDING_LABELS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "pleural_effusion", "pneumonia", "pneumothorax", "no_finding",
]


def record_key(rec: dict) -> tuple:
    return (str(rec.get("image_id")), str(rec.get("condition")))


def _tokens(text: str) -> set[str]:
    """Word set used for the text-overlap fallback contrast."""
    return {t for t in re.findall(r"[a-z]{3,}", (text or "").lower())}


def _save_atomic(image, path: Path) -> None:
    """Write a PNG via temp file + os.replace.

    This runner skips regeneration with ``if not dst.exists()``, which is right for
    a multi-day job but turns a plain ``save()`` into a permanent hazard: an
    interruption during the write leaves a truncated PNG, and the existence check
    then treats that corrupt control image as finished work for the rest of the run.
    Atomic replace makes the file either absent or complete.
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


def prepare_conditions(manifest: pd.DataFrame, images_root: Path, out_dir: Path,
                       conditions: list[str]) -> list[dict]:
    """Build the job list, materialising the blank control images on disk.

    The control is written out rather than generated in memory so that what the
    model was actually shown can be inspected afterwards -- a blank image that is
    silently a black frame or the wrong size is a real and previously-hit bug
    class.
    """
    records: list[dict] = []
    blank_dir = out_dir / "blank_images"
    if "blank" in conditions:
        blank_dir.mkdir(parents=True, exist_ok=True)

    for _, row in manifest.iterrows():
        image_id = str(row["image_id"])
        rel = str(row["image_relpath"])
        src = images_root / rel
        cond: dict[str, str] = {}
        if "original" in conditions:
            if not src.exists():
                print(f"[warn] missing image, skipped: {src}")
                continue
            cond["original"] = str(src)
        if "blank" in conditions:
            dst = blank_dir / f"{image_id}.png"
            if not dst.exists():
                if not src.exists():
                    print(f"[warn] cannot build blank control, missing {src}")
                    continue
                with Image.open(src) as im:
                    _save_atomic(Image.new("RGB", im.size, BLANK_RGB), dst)
            cond["blank"] = str(dst)
        records.append({"image_id": image_id, "conditions": cond,
                        "dataset": str(row.get("dataset", "padchest")),
                        "patient_id": str(row.get("patient_id", ""))})
    return records


# Errors that mean "this wrapper cannot batch", as opposed to "the hardware or
# the allocator failed". Only the former is safe to retry one image at a time.
# Anything else must propagate: swallowing a CUDA/driver/OOM error here turns a
# hard failure into a file full of per-row error strings and an exit code of 0,
# which is exactly the silent-failure shape this project has already been bitten
# by (see the InternVL 'Image' object bug in 第一次跑.md).
_BATCH_REJECTION_ERRORS = (TypeError, ValueError, AttributeError, AssertionError,
                           NotImplementedError, KeyError, IndexError)


def run_batch(model, images: list, prompt: str, max_new_tokens: int) -> tuple[list[str], list[str]]:
    """Generate for a chunk, falling back to serial only on batch rejection."""
    raws: list[str] = []
    errs: list[str] = []
    if len(images) > 1 and getattr(model, "generate_batch", None):
        try:
            raws = list(model.generate_batch(images, [prompt] * len(images),
                                             max_new_tokens=max_new_tokens))
            return raws, [""] * len(raws)
        except _BATCH_REJECTION_ERRORS as exc:
            print(f"[warn] batching rejected by this wrapper "
                  f"({type(exc).__name__}: {exc}); falling back to one image at a "
                  f"time", flush=True)
        # Any other exception propagates to the caller, which stops the run with a
        # non-zero exit code so the stage is not recorded as complete.
    for img in images:
        raws.append(model.generate(img, prompt, max_new_tokens=max_new_tokens))
        errs.append("")
    return raws, errs


def summarise(records: list[dict], by_key: dict, conditions: list[str],
              out_dir: Path, model_name: str) -> dict:
    """Per-image contrast against the original plus a run-level summary.

    The contrast is computed from the **probability vector** when the reply parsed
    into the structured schema, and from **status flips** when it did not. The
    first version of this function required ``parse_success`` and therefore
    produced an empty summary for any model that answers in prose -- which is not
    a hypothetical: the Open-i layer asks for prose on purpose, and a model can
    ignore the JSON contract on PadChest too. A summary that goes quietly empty
    when the model changes register is the same class of failure as the silent
    empty rows this project already lost a run to.
    """
    per_image: list[dict] = []
    failures: dict[str, int] = {c: 0 for c in conditions}
    structured_ok: dict[str, int] = {c: 0 for c in conditions}

    for rec in records:
        for condition in conditions:
            cur = by_key.get((rec["image_id"], condition))
            if not cur:
                continue
            if cur.get("error"):
                failures[condition] = failures.get(condition, 0) + 1
            elif (cur.get("parsed") or {}).get("parse_success"):
                structured_ok[condition] = structured_ok.get(condition, 0) + 1

        orig = by_key.get((rec["image_id"], "original"))
        if not orig:
            continue
        for condition in conditions:
            if condition == "original":
                continue
            cur = by_key.get((rec["image_id"], condition))
            if not cur:
                continue

            of = (orig.get("parsed") or {}).get("findings", {}) or {}
            cf = (cur.get("parsed") or {}).get("findings", {}) or {}

            # Structured path: how far did the probabilities move?
            diffs = []
            for label in of:
                op = of[label].get("probability_positive")
                cp = (cf.get(label) or {}).get("probability_positive")
                if op is not None and cp is not None:
                    diffs.append(abs(float(op) - float(cp)))

            # Text path: does the reply still look like the same assessment?
            # Without the structured schema, the cheapest honest signal is
            # lexical overlap of the two replies: a model that ignores the image
            # writes nearly the same sentence for the radiograph and for a grey
            # square.
            a = (orig.get("raw_output") or "").strip()
            b = (cur.get("raw_output") or "").strip()
            ta, tb = _tokens(a), _tokens(b)
            jaccard = (len(ta & tb) / len(ta | tb)) if (ta or tb) else None

            flips = sum(1 for label in of
                        if of[label].get("status") != (cf.get(label) or {}).get("status"))

            per_image.append({
                "image_id": rec["image_id"],
                "condition": condition,
                "structured_available": int(bool(diffs)),
                "flip_count": flips if of and cf else None,
                "mean_abs_prob_diff": (sum(diffs) / len(diffs)) if diffs else None,
                "text_jaccard": jaccard,
                "identical_text": int(bool(a) and a == b),
                "original_empty": int(not a),
                "condition_empty": int(not b),
            })

    per_df = pd.DataFrame(per_image)
    if not per_df.empty:
        per_df.to_csv(out_dir / "image_reliance_per_image.csv", index=False)

    # Coverage must be over the cells this run actually ATTEMPTED, not over the
    # manifest. On 2026-10-21 an interrupted run (17 of 3,000 cells, all parsed)
    # printed "original 1.1%, blank 0.0% of replies parsed" because the
    # denominator was the planned 1,500 -- a parse rate of 17/17 was reported as
    # 17/1500 and would have been read as a model failure. Incompleteness is
    # already expressed by `cells_on_disk` versus n_images x len(conditions), so
    # the two must not be conflated. Conditions with zero attempted cells get
    # None (not run yet) rather than 0.0 (all replies broken).
    attempted = {c: sum(1 for rec in records
                        if (str(rec.get("image_id")), c) in by_key)
                 for c in conditions}
    coverage = {c: (structured_ok.get(c, 0) / attempted[c]
                    if attempted.get(c) else None)
                for c in conditions}
    # The contrast SOURCE must be decided by COVERAGE, not by "at least one reply
    # parsed". The old rule (`if structured_ok.get("original")`) labelled
    # padchest/qwen25 `probability_vector` at 323/1500 = 21.5% parse and then
    # published `mean_abs_prob_diff = 0.0` -- a number computed on 257 of 1500
    # pairs -- as the layer's image-reliance figure. Read as "this model never
    # moves its probabilities", it actually meant "17% of the images were
    # measured and the other 83% could not be". The text path (lexical overlap of
    # the two replies) is computed for every pair and is the honest fallback;
    # that is what `contrast_source` exists to express. 0.95 is the same
    # threshold the coverage guard uses elsewhere in this project.
    contrast_source = ("probability_vector"
                       if (coverage.get("original") or 0.0) >= 0.95
                       else "text_overlap")
    low = [c for c in conditions
           if coverage.get(c) is not None and coverage[c] < 0.95]

    summary: dict[str, Any] = {
        "model": model_name,
        "layer": "padchest_geographic_shift",
        "design": "native cross-hospital shift + blank control (no synthetic grid)",
        "n_images": len(records),
        "conditions_run": conditions,
        "generation_failures_by_condition": failures,
        "structured_parse_ok_by_condition": structured_ok,
        "n_attempted_by_condition": attempted,
        "structured_coverage_by_condition": {
            k: (round(v, 4) if v is not None else None)
            for k, v in coverage.items()},
        "cells_on_disk": len(by_key),
        "contrast_source": contrast_source,
        "contrast_source_note": (
            "probability_vector is used only at >=95% structured coverage on the "
            "original condition; below that the text-overlap path is reported "
            "instead, because a probability mean over a minority of images reads "
            "as a layer-level finding it is not."),
        "coverage_below_threshold": low,
        "by_condition": {},
    }
    if low:
        print("\n" + "!" * 78)
        print("STRUCTURED-COVERAGE WARNING: "
              + ", ".join(f"{c} {coverage[c]:.1%} of attempted replies parsed"
                          for c in low)
              + ". Reporting text_overlap, and NOT a"
              " probability-vector reliance number, for this layer."
              f" (attempted so far: {attempted})")
        print("!" * 78)
    if not per_df.empty:
        for condition, g in per_df.groupby("condition"):
            numeric = g["mean_abs_prob_diff"].dropna()
            jac = g["text_jaccard"].dropna()
            summary["by_condition"][str(condition)] = {
                "n": int(len(g)),
                "n_structured": int(g["structured_available"].sum()),
                # The denominator of every number above. `mean_abs_prob_diff`
                # lives on n_prob_pairs, not on n: they differ whenever some
                # replies did not parse.
                "n_prob_pairs": int(g["mean_abs_prob_diff"].notna().sum()),
                "n_flip_pairs": int(g["flip_count"].notna().sum()),
                "mean_flip_count": (float(g["flip_count"].dropna().mean())
                                    if g["flip_count"].notna().any() else None),
                "mean_abs_prob_diff": float(numeric.mean()) if len(numeric) else None,
                "mean_text_jaccard": float(jac.mean()) if len(jac) else None,
                "pct_identical_text": round(100.0 * float(g["identical_text"].mean()), 2),
                "pct_condition_empty": round(100.0 * float(g["condition_empty"].mean()), 2),
            }
    (out_dir / "image_reliance_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    ap.add_argument("--manifest", default="data/manifests/padchest_shift.csv")
    ap.add_argument("--images-root", default="data/padchest")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of images; applied after the shuffle")
    ap.add_argument("--split", default=None, help="comma-separated splits to keep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    manifest = pd.read_csv(ROOT / args.manifest if not Path(args.manifest).is_absolute()
                           else args.manifest)
    if args.split and "split" in manifest.columns:
        wanted = [s.strip() for s in args.split.split(",") if s.strip()]
        before = len(manifest)
        manifest = manifest[manifest["split"].isin(wanted)].copy()
        print(f"split filter {wanted}: {before} -> {len(manifest)} images", flush=True)
    # Shuffle before limiting: --limit takes the first N rows, which is not a
    # sample, and the manifest is ordered.
    if len(manifest):
        manifest = manifest.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.limit:
        manifest = manifest.head(args.limit)
    print(f"manifest: {len(manifest)} images; conditions={args.conditions}", flush=True)
    if manifest.empty:
        print("[error] nothing to do: the manifest selection is empty")
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "image_reliance_predictions.jsonl"

    records = prepare_conditions(manifest, Path(args.images_root), out_dir,
                                 args.conditions)
    all_jobs = [(r, c) for c in args.conditions for r in records
                if c in r["conditions"]]
    # Condition-major on purpose: a batch runs until its LONGEST member finishes,
    # so mixing `blank` cells (where MedGemma-1.5 opens a 1024-token reasoning
    # trace) with `original` cells (~170 tokens) makes every batch pay the slow
    # one. See the same note in 06_image_reliance.py and docs/50 section 2.8.
    if not all_jobs:
        print("[error] no runnable jobs (check --images-root and the manifest paths)")
        return 2

    model = get_model(args.model, root=ROOT, device_map="auto",
                      load_in_4bit=args.load_in_4bit)
    model.load()
    prompt = SIMPLE_SYSTEM_PROMPT + chr(10) * 2 + build_simple_prompt()

    with DurableJsonl(out_path, key=record_key) as out:
        pending = [j for j in all_jobs if out.needs(j[0]["image_id"], j[1])]
        skipped = len(all_jobs) - len(pending)
        print(f"resume: {skipped} of {len(all_jobs)} cells already durable, "
              f"{len(pending)} to run", flush=True)

        pbar = tqdm(total=len(pending), desc=f"{args.model} | padchest (bs={args.batch_size})",
                    unit="img", dynamic_ncols=True)
        interrupted = None
        for i in range(0, len(pending), args.batch_size):
            chunk = pending[i:i + args.batch_size]
            images, meta = [], []
            for rec, condition in chunk:
                path = rec["conditions"][condition]
                try:
                    with Image.open(path) as im:
                        images.append(im.convert("RGB"))
                    meta.append((rec, condition, path))
                except Exception as exc:  # noqa: BLE001
                    out.write({
                        "image_id": rec["image_id"], "condition": condition,
                        "image_path": path, "raw_output": "", "error":
                        f"image_load_failed: {type(exc).__name__}: {exc}",
                        "elapsed_sec": 0.0,
                        "parsed": parse_vlm_output("").to_dict(),
                        "dataset": rec.get("dataset", "padchest"),
                        "patient_id": rec.get("patient_id", ""),
                    })
                    pbar.update(1)
            if not images:
                continue

            t0 = time.time()
            try:
                raws, errs = run_batch(model, images, prompt, args.max_new_tokens)
            except BaseException as exc:  # noqa: BLE001
                # A hard failure (driver error, power-cut-the-process, OOM) must
                # NOT look like success: the orchestrator decides whether to mark
                # this stage complete from the exit code, and everything fsynced
                # so far is already durable, so the honest thing is to stop with
                # a non-zero code and say how to continue.
                pbar.close()
                interrupted = exc
                break
            per_img = (time.time() - t0) / len(images)

            rows = []
            for (rec, condition, path), raw, err in zip(meta, raws, errs):
                parsed = parse_vlm_output(raw) if not err else parse_vlm_output("")
                rows.append({
                    "image_id": rec["image_id"],
                    "condition": condition,
                    "image_path": path,
                    "raw_output": raw,
                    "error": err,
                    "elapsed_sec": per_img,
                    "parsed": parsed.to_dict(),
                    "dataset": rec.get("dataset", "padchest"),
                    "patient_id": rec.get("patient_id", ""),
                })
            out.write_many(rows)          # append + fsync + durable marker
            pbar.update(len(chunk))
            pbar.set_postfix(sec_per_img=f"{per_img:.1f}s")
        pbar.close()
        stats = out.stats()

    by_key = {(str(r.get("image_id")), str(r.get("condition"))): r
              for r in read_jsonl(out_path)}
    summary = summarise(records, by_key, args.conditions, out_dir, args.model)
    summary["durable"] = stats
    (out_dir / "image_reliance_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if interrupted is not None:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\n[INTERRUPTED] {type(interrupted).__name__}: {interrupted}", flush=True)
        print(f"[INTERRUPTED] {summary['cells_on_disk']} cells are durable in "
              f"{out_path.name}. Re-run the same command to continue.", flush=True)
        return 1

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
