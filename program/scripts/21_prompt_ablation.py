#!/usr/bin/env python
"""Prompt ablation with original/blank pairs -- the Qwen image-invariance test.

What this script does
---------------------
Runs one (model x prompt-variant x image-condition) cell and writes a JSONL of
records in the same shape as ``06_image_reliance.py``, so the existing reliance
analysis can be pointed at it unchanged.

The design point is the **original/blank pair**. Accuracy alone cannot tell a
model that reads the image apart from one that recites a prior: a language-prior
model and a genuinely good model can both score well on a normal-heavy test set.
Image reliance is what separates them, so every variant here is run against both
the real radiograph and a flat gray image, on exactly the same studies.

Cell layout::

    <output-root>/<model>/<variant>/<condition>/
        image_reliance_predictions.jsonl
        raw_outputs.jsonl
        progress.json

Partial cells are resumable: images already parsed successfully are skipped.

Usage
-----
Single cell::

    python scripts/21_prompt_ablation.py \
        --model qwen25_vl_7b_instruct \
        --variant v4_normality_pressure \
        --condition original \
        --manifest data/manifests/nih_pilot_1000.csv \
        --images-root data/nih/images \
        --output-root outputs/prompt_ablation \
        --limit 100 --load-in-4bit

All variants x both conditions for one model, one model load (recommended)::

    python scripts/21_prompt_ablation.py \
        --model qwen25_vl_7b_instruct \
        --variant all --condition original blank \
        --manifest data/manifests/nih_pilot_1000.csv \
        --images-root data/nih/images \
        --output-root outputs/prompt_ablation \
        --limit 100 --load-in-4bit --batch-size 16

List the available variants without running anything::

    python scripts/21_prompt_ablation.py --list-variants

Batching
--------
``--batch-size`` was added after the 7-model x 100-image pre-run measured this
stage at **8.28 s/cell** against 0.63 s/cell for the batched shift stage -- a 13x
penalty, and ~38 extra hours over the full run. The cause was structural, not
numerical: the loop called ``model.generate`` once per image while every other
runner in the project went through ``generate_batch``.

The batch size cannot simply be copied from the shift stage, because the five
variants differ a lot in how much they emit (``v3_report_first`` asks for a
region-by-region narrative before the labels, and ``v4_normality_pressure`` asks
for a per-finding justification), and the padding cost of a batch is set by its
longest member. Measure instead of guessing -- see the numbers in
``docs/50_resume_prerun_findings.md`` §2.6.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.inference.durable import DurableJsonl  # noqa: E402
from src.cxr_trust.inference.prompts_ablation import (  # noqa: E402
    PROMPT_VARIANTS,
    VARIANT_NOTES,
    build_prompt,
    list_variants,
)
from src.cxr_trust.inference.schema import extract_json_object, parse_vlm_output  # noqa: E402
from src.cxr_trust.models.registry import MODEL_NAMES, get_model  # noqa: E402

# Test seam: when RUNNER_STUB=1 the model factory is replaced AFTER all imports have
# run. It must be installed here and not from sitecustomize, because src/ has no
# __init__.py, so "src.cxr_trust.*" and "cxr_trust.*" are two separate module
# objects -- a patch applied before this import would miss this one and the runner
# would quietly load a real checkpoint while the test believed it was stubbed.
if os.environ.get("RUNNER_STUB") == "1":  # pragma: no cover - test-only path
    import stub_model

    stub_model.install()


CONDITIONS = ("original", "blank")

# Only "this wrapper cannot batch" is recoverable by going one image at a time.
# A driver, allocator or CUDA failure must propagate so the stage exits non-zero
# instead of becoming a file of per-row error strings with exit code 0. This is
# the same contract 06/08/43/44 use; the classes are repeated rather than
# imported so a change in one runner can never silently widen another's net.
_BATCH_REJECTION_ERRORS = (TypeError, ValueError, AttributeError, AssertionError,
                           NotImplementedError, KeyError, IndexError)


def contract_compliance(parsed: dict, raw: str) -> dict:
    """Measure how well a response honoured the output contract.

    This matters because a variant that suppresses the JSON contract can look
    "safe" while actually destroying information.  A missing label becomes
    ``not_mentioned`` and a missing probability becomes ``None``; downstream
    analysis drops ``None`` probabilities, so an under-specified response
    silently shrinks the evaluation set rather than raising an error.

    Two independent violations are counted, because they can occur separately:

    * ``n_labels_defaulted`` -- labels the model never gave a status for;
    * ``n_probs_missing``    -- labels absent from the top-level
      ``probability_positive`` object (these become ``None`` and are then
      dropped by the metrics code).

    ``contract_violation`` is True when either occurs.
    """
    findings = parsed.get("findings") or {}
    n_total = len(findings)
    n_stated = sum(1 for v in findings.values()
                   if (v or {}).get("status") not in (None, "not_mentioned"))

    prob_obj = {}
    try:
        obj = extract_json_object(raw)
        if isinstance(obj, dict):
            candidate = obj.get("probability_positive")
            if isinstance(candidate, dict):
                prob_obj = candidate
    except Exception:
        prob_obj = {}

    def usable(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False
        return 0.0 <= float(v) <= 1.0

    n_probs = sum(1 for k in findings if usable(prob_obj.get(k)))
    n_defaulted = n_total - n_stated
    n_probs_missing = n_total - n_probs
    return {
        "n_labels_total": n_total,
        "n_labels_stated": n_stated,
        "n_labels_defaulted": n_defaulted,
        "n_probs_valid": n_probs,
        "n_probs_missing": n_probs_missing,
        "contract_violation": bool(n_defaulted > 0 or n_probs_missing > 0),
    }


def load_done(path: Path) -> set[str]:
    """image_ids already parsed successfully in this cell."""
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if (obj.get("parsed") or {}).get("parse_success"):
            done.add(str(obj.get("image_id")))
    return done


def condition_image(row, images_root: Path, condition: str, blank_cache: Path) -> Path:
    """Resolve the image path for one condition.

    ``original`` uses the manifest relpath; ``blank`` materialises a flat gray
    image of identical size (same convention as ``06_image_reliance.py``) so that
    image dimensions are not a confound.
    """
    src = images_root / row["image_relpath"]
    if condition == "original":
        return src
    if condition == "blank":
        dst = blank_cache / f"{row['image_id']}.png"
        if not dst.exists():
            blank_cache.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", Image.open(src).convert("RGB").size, (128, 128, 128)).save(dst)
        return dst
    raise ValueError(f"unknown condition {condition!r}")


def _ablation_key(rec: dict) -> tuple:
    """One identity per model call: image x variant x condition.

    The variant and condition must be in the key. A single cell file happens to
    hold only one (variant, condition) today, but the key is what decides whether
    an existing row is reused or replaced, and leaving the cell out means the same
    image appearing in two cells could be mistaken for already-done work.
    """
    return (str(rec.get("image_id")), str(rec.get("variant")),
            str(rec.get("condition")))


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON via temp file + os.replace so a reader never sees it half-written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def run_cell(model, model_name: str, variant: str, condition: str, manifest: pd.DataFrame,
             images_root: Path, cell_dir: Path, max_new_tokens: int, resume: bool,
             batch_size: int = 1) -> dict:
    """Run one (variant, condition) cell.

    Crash safety
    ------------
    Both outputs go through ``DurableJsonl``: append, then ``fsync``, then an
    atomic byte-offset sidecar. The previous version called ``flush()`` only, so on
    Windows the tail could sit in the file cache and a power cut lost it; a torn
    final line was silently swallowed by ``load_done`` and then concatenated onto
    by the next append, corrupting two rows; and because the resume set contains
    only rows that parsed, a cell whose replies all failed to parse would be
    reopened with mode ``"w"`` and wiped.

    ``raw_outputs.jsonl`` is kept as a separate durable stream because it is the
    audit trail: it records what the model actually emitted, including replies the
    structured parser rejected. Losing it would make a parse failure
    unreproducible. Its resume key carries the variant and condition so the same
    image appearing in two cells cannot collide.
    """
    cell_dir.mkdir(parents=True, exist_ok=True)
    pred_path = cell_dir / "image_reliance_predictions.jsonl"
    raw_path = cell_dir / "raw_outputs.jsonl"
    legacy_progress_path = cell_dir / "progress.json"
    blank_cache = cell_dir / "images" / "blank"

    system, user = build_prompt(variant)
    prompt = system + chr(10) * 2 + user

    if not resume:
        # An explicit non-resume run means "redo this cell from scratch". With the
        # durable writer the way to express that is to remove the existing outputs,
        # not to open them with mode "w", because the writer owns the file.
        for path in (pred_path, raw_path):
            for sidecar in (path, Path(str(path) + ".progress.json")):
                if sidecar.exists():
                    sidecar.unlink()

    with DurableJsonl(pred_path, key=_ablation_key) as out, \
            DurableJsonl(raw_path, key=_ablation_key) as raw_out:
        todo = [row for _, row in manifest.iterrows()
                if out.needs(row.get("image_id"), variant, condition)]
        skipped = len(manifest) - len(todo)
        if skipped:
            print(f"[resume] {variant}/{condition}: {skipped} already durable, "
                  f"{len(todo)} to run", flush=True)
        if not todo:
            return {"variant": variant, "condition": condition, "n": 0, "skipped": True}

        failures = 0
        start_all = time.time()
        step = max(1, int(batch_size))
        desc = f"{model_name} | {variant} | {condition} (bs={step})"
        pbar = tqdm(total=len(todo), desc=desc, unit="img", dynamic_ncols=True)
        interrupted = None
        done_n = 0
        try:
            for i in range(0, len(todo), step):
                chunk = todo[i:i + step]
                # Images are opened once and handed to the model as PIL objects:
                # every wrapper's generate_batch accepts either, and the blank
                # control cannot be re-derived inside the model.
                image_paths = [condition_image(row, images_root, condition, blank_cache)
                               for row in chunk]
                images = [Image.open(p).convert("RGB") for p in image_paths]
                prompts = [prompt] * len(chunk)

                start = time.time()
                if step > 1 and getattr(model, "generate_batch", None):
                    try:
                        raws = list(model.generate_batch(images, prompts,
                                                         max_new_tokens=max_new_tokens))
                        if len(raws) != len(chunk):
                            # ValueError is in _BATCH_REJECTION_ERRORS, so a
                            # mis-sized batch is treated as "this wrapper cannot
                            # batch" and redone serially rather than misaligned:
                            # pairing raws[j] with the wrong image would be a
                            # silent, unrecoverable corruption of the cell.
                            raise ValueError(
                                f"generate_batch returned {len(raws)} outputs "
                                f"for {len(chunk)} inputs")
                        errs = [""] * len(raws)
                    except _BATCH_REJECTION_ERRORS as exc:
                        print(f"[warn] batching rejected ({type(exc).__name__}: {exc}); "
                              f"falling back to one image at a time", flush=True)
                        raws, errs = [], []
                        for img in images:
                            raws.append(model.generate(img, prompt,
                                                       max_new_tokens=max_new_tokens))
                            errs.append("")
                else:
                    raws, errs = [], []
                    for img in images:
                        raws.append(model.generate(img, prompt,
                                                   max_new_tokens=max_new_tokens))
                        errs.append("")
                elapsed = time.time() - start
                # One call per image is what the cost model counts, so the batch's
                # wall clock is divided by its size: a cell then means the same
                # thing whether it was produced alone or inside a batch of 16.
                per_image = elapsed / len(chunk)

                raw_rows, pred_rows = [], []
                for j, row in enumerate(chunk):
                    raw = raws[j]
                    error = errs[j]
                    parsed = parse_vlm_output(raw) if not error else parse_vlm_output("")
                    if not parsed.parse_success:
                        failures += 1
                    raw_rows.append({
                        "image_id": row.get("image_id"),
                        "image_relpath": row.get("image_relpath"),
                        "variant": variant,
                        "condition": condition,
                        "raw_output": raw,
                        "error": error,
                        "elapsed_sec": per_image,
                        "batch_size": len(chunk),
                    })
                    pred_rows.append({
                        "image_id": row.get("image_id"),
                        "condition": condition,
                        "variant": variant,
                        "image_path": str(image_paths[j]),
                        "raw_output": raw,
                        "error": error,
                        "elapsed_sec": per_image,
                        "batch_size": len(chunk),
                        "parsed": parsed.to_dict(),
                        "contract": contract_compliance(parsed.to_dict(), raw),
                    })

                # Both files are written together: the audit trail and the parsed
                # record describe the same model call, so they must never disagree
                # about how many calls happened.
                raw_out.write_many(raw_rows)
                out.write_many(pred_rows)

                done_n += len(chunk)
                pbar.update(len(chunk))

                # A human-readable progress note, rewritten atomically. It is not
                # what resume depends on any more -- the durable byte offset is --
                # so it is written opportunistically and its failure is not fatal.
                elapsed_total = time.time() - start_all
                eta = (elapsed_total / done_n * (len(todo) - done_n))
                _write_json_atomic(legacy_progress_path, {
                    "model": model_name, "variant": variant, "condition": condition,
                    "total": int(len(todo)), "done": int(done_n),
                    "failures": int(failures),
                    "batch_size": int(step),
                    "elapsed_sec": round(elapsed_total, 2),
                    "eta_sec": round(eta, 2),
                    "last_image_id": chunk[-1].get("image_id"),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "durable_bytes": out._offset,
                })
                pbar.set_postfix(fail=failures, sec_per_img=f"{per_image:.2f}s")
        except BaseException as exc:  # noqa: BLE001  (incl. MemoryError)
            interrupted = exc
        finally:
            pbar.close()
        stats = out.stats()

    if interrupted is not None:
        print(f"[INTERRUPTED] {type(interrupted).__name__}: {interrupted}", flush=True)
        print(f"[INTERRUPTED] {stats['records_on_disk']} cells are durable in "
              f"{pred_path.name}. Re-run the same command to continue.", flush=True)
        return {"variant": variant, "condition": condition,
                "n": 0, "interrupted": str(interrupted), "durable": stats}

    # contract-compliance roll-up for this cell
    n_records = n_violations = n_defaulted = n_probs_missing = 0
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        c = obj.get("contract") or {}
        if not c:
            continue
        n_records += 1
        n_violations += int(bool(c.get("contract_violation")))
        n_defaulted += int(c.get("n_labels_defaulted") or 0)
        n_probs_missing += int(c.get("n_probs_missing") or 0)

    summary = {
        "model": model_name, "variant": variant, "condition": condition,
        "n_total_records": n_records,
        "n_this_run": int(len(todo)),
        "batch_size": int(step),
        "sec_per_cell": round((time.time() - start_all) / len(todo), 3) if len(todo) else 0.0,
        "parse_failures": int(failures),
        "parse_failure_rate": failures / len(todo) if len(todo) else 0.0,
        "contract_violation_rate": (n_violations / n_records) if n_records else 0.0,
        "mean_labels_defaulted": (n_defaulted / n_records) if n_records else 0.0,
        "mean_probs_missing": (n_probs_missing / n_records) if n_records else 0.0,
        "elapsed_sec": round(time.time() - start_all, 2),
    }
    (cell_dir / "parse_failure_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-variants", action="store_true",
                        help="print the available prompt variants and exit")
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--variant", nargs="+", default=["v0_simple"],
                        help="one or more variants, or 'all'")
    parser.add_argument("--condition", nargs="+", default=["original", "blank"],
                        choices=list(CONDITIONS))
    parser.add_argument("--manifest", default="data/manifests/nih_pilot_1000.csv")
    parser.add_argument("--images-root", default="data/nih/images")
    parser.add_argument("--output-root", default="outputs/prompt_ablation")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--split", default=None,
                        help="comma-separated splits to keep, e.g. internal_test,source_val")
    parser.add_argument("--seed", type=int, default=0)
    # The standalone layout is <root>/<model>/<variant>/<condition>, which is right
    # when one model is run by hand. The orchestrator already puts each model in its
    # own directory, so it passes --flat-output and avoids
    # <root>/<model>/ablation/<model>/<variant>/<condition> with the model twice.
    parser.add_argument("--flat-output", action="store_true",
                        help="write cells directly under --output-root instead of "
                             "<root>/<model>/...")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    # 8 is a conservative default for the models this stage is actually run on
    # (medgemma / qwen); models whose measured sweet spot is smaller must pass
    # their own value -- the orchestrator does.
    parser.add_argument("--batch-size", type=int, default=8,
                        help="images per generate_batch call (1 = the pre-batching "
                             "serial path; see the Batching note in this file's "
                             "docstring for measured sizes)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--attn-implementation", default=None,
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="override the attention backend; needed when the "
                             "default SDPA kernel raises 'attn_bias_ptr is not "
                             "correctly aligned' on a particular prompt length")
    args = parser.parse_args()

    if args.list_variants:
        for row in list_variants():
            print(f"{row['variant']:24s} {row['note']}")
        return 0

    if not args.model:
        parser.error("--model is required unless --list-variants is used")

    variants = list(PROMPT_VARIANTS) if "all" in args.variant else args.variant
    unknown = [v for v in variants if v not in PROMPT_VARIANTS]
    if unknown:
        parser.error(f"unknown variants {unknown}; known: {sorted(PROMPT_VARIANTS)}")

    # Escape hatch for the torch 2.5.1 efficient-attention kernel, which raises
    # "p.attn_bias_ptr is not correctly aligned" for some prompt lengths (the
    # v3_report_first cells hit it on their first batch and wrote zero rows).
    # Disabling that kernel leaves flash / math, which carry the same maths.
    if os.environ.get("CXR_DISABLE_MEM_EFFICIENT_SDPA") == "1":
        import torch
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        print("[sdpa] memory-efficient attention disabled", flush=True)

    model_kwargs = dict(root=ROOT, device_map="auto",
                        load_in_4bit=args.load_in_4bit)
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
        print(f"[attn] requested backend: {args.attn_implementation}", flush=True)
    model = get_model(args.model, **model_kwargs)
    model.load()

    if os.environ.get("RUNNER_STUB") == "1":
        # Fail loudly if the stub did not take over: a silently un-stubbed test run
        # would load a real checkpoint and still "pass".
        assert type(model).__name__ == "_StubVLM", (
            "RUNNER_STUB=1 but get_model returned "
            + type(model).__name__ + "; the stub did not patch this import path")


    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    manifest = pd.read_csv(manifest_path)
    if args.split and "split" in manifest.columns:
        wanted = [s.strip() for s in args.split.split(",") if s.strip()]
        before = len(manifest)
        manifest = manifest[manifest["split"].isin(wanted)].copy()
        print(f"split filter {wanted}: {before} -> {len(manifest)} images", flush=True)
    elif args.split:
        print(f"[warn] --split {args.split} ignored: manifest has no 'split' column")
    # Shuffle BEFORE the limit: --limit takes the first N rows of the file, which
    # is not a sample (and can include training rows).
    if len(manifest):
        manifest = manifest.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    manifest = manifest.head(args.limit).copy()
    print(f"manifest: {len(manifest)} images", flush=True)
    images_root = Path(args.images_root)

    print(f"model={args.model} limit={len(manifest)} variants={variants} "
          f"conditions={args.condition} batch_size={args.batch_size}")
    print(f"cells to run: {len(variants) * len(args.condition)}")

    # One place decides the model subdirectory, so the cells and the run summary
    # can never disagree about where the output lives.
    model_root = Path(args.output_root) if args.flat_output \
        else Path(args.output_root) / args.model

    results = []
    interrupted_cells = []
    for variant in variants:
        for condition in args.condition:
            cell_dir = model_root / variant / condition
            summary = run_cell(model, args.model, variant, condition, manifest,
                               images_root, cell_dir, args.max_new_tokens, args.resume,
                               batch_size=args.batch_size)
            results.append(summary)
            if summary.get("interrupted"):
                interrupted_cells.append(f"{variant}/{condition}")
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    (model_root / "ablation_run_summary.json").write_text(
        json.dumps({"model": args.model, "limit": args.limit, "cells": results,
                    "interrupted_cells": interrupted_cells},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {model_root}")

    if interrupted_cells:
        # A stage that stopped early must not look like a completed one: the
        # orchestrator decides from the exit code, and it also checks the output
        # for error rows before marking this stage done.
        print(f"[INTERRUPTED] {len(interrupted_cells)} cell(s) did not finish: "
              f"{', '.join(interrupted_cells)}", flush=True)
        print("[INTERRUPTED] re-run the same command to continue; every finished "
              "cell is already durable", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
