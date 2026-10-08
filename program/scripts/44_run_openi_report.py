#!/usr/bin/env python
"""Open-i external report-generation runner: the third layer of the 6K design.

What question this layer answers
--------------------------------
NIH asks "how much synthetic degradation before the selective signal collapses".
PadChest asks "does the framework survive a different hospital". Open-i asks the
third and hardest one: **does it survive free-text report generation, where the
failure mode is hallucinated content rather than a mis-calibrated probability.**

Why it is not a classification run
----------------------------------
An Open-i study has a reference *report* (``findings_section``) and no per-finding
ground-truth labels -- the manifest's label columns carry ``unknown``, which is
the honest value, because nothing was annotated. Scoring it with the eight-label
AUROC machinery would compute a metric against a column of unknowns. So the
primary output is the generated report text, and the metrics come from comparing
it to the reference:

* **omission** -- findings the reference asserts that the generation does not.
* **hallucination** -- findings the generation asserts that the reference does not.

Both require a labeller applied to *both* texts, so that a systematic labelling
bias cancels instead of inflating one side. That is
``42_geographic_shift_analysis.py`` (``report_metrics``), which reuses the same
keyword+negation evidence rule as ``11_rrg_report_analysis.py``; the judged
labeller in ``20_rrg_labeler_eval.py`` is the upgrade path for that evidence rule.

The ``blank`` control
---------------------
Every study is also run with the pixels replaced by flat grey. On a
report-generation task this is the sharpest image-reliance probe available: a
model that produces a fluent, plausible report from a grey rectangle is
generating from its language prior, and the overlap between the real and blank
reports quantifies exactly how much of the output was ever image-driven.

Crash safety
------------
Appends go through ``DurableJsonl`` (append -> fsync -> atomic byte-offset
marker), so an interruption loses at most the in-flight batch, a torn final line
is truncated away on reopen, and failed rows are retried rather than becoming
permanent holes. See ``src/cxr_trust/inference/durable.py``.

Usage::

    python scripts/44_run_openi_report.py \\
        --model medgemma_4b_it \\
        --manifest data/manifests/openi_rrg.csv \\
        --images-root data/open-i \\
        --output-dir outputs/matrix/openi/medgemma_4b_it \\
        --conditions original blank \\
        --batch-size 8 --load-in-4bit

    # smoke: one study, both conditions
    python scripts/44_run_openi_report.py --model medgemma_4b_it --limit 1 \\
        --output-dir /tmp/oi_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
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
from src.cxr_trust.inference.prompts_report import (  # noqa: E402
    REPORT_SYSTEM_PROMPT,
    build_report_prompt,
    build_terse_report_prompt,
)
from src.cxr_trust.inference.schema import parse_vlm_output  # noqa: E402
from src.cxr_trust.models.registry import MODEL_NAMES, get_model  # noqa: E402

# Test seam: when RUNNER_STUB=1 the model factory is replaced after all imports
# have run. It must happen here rather than in sitecustomize, because src/ has no
# __init__.py and therefore "src.cxr_trust.*" and "cxr_trust.*" are two separate
# module objects -- a patch applied before this import would miss this one.
if os.environ.get("RUNNER_STUB") == "1":  # pragma: no cover - test-only path
    import stub_model
    stub_model.install()


BLANK_RGB = (128, 128, 128)
DEFAULT_CONDITIONS = ["original", "blank"]

# Markdown / list syntax in the output is a prompt-compliance failure, not a
# finding. It is counted rather than silently stripped, because "the model
# produced something the reference format cannot be compared against" is a result.
_FORMAT_MARKERS = re.compile(r"(^\s*[-*#]\s)|(^\s*\d+\.\s)|(^```)|(\*\*)|(^\s*\{)")


def record_key(rec: dict) -> tuple:
    return (str(rec.get("image_id")), str(rec.get("condition")))


def looks_structured(text: str) -> bool:
    return bool(_FORMAT_MARKERS.search(text or ""))


def build_prompts(variant: str) -> tuple[str, str]:
    if variant == "terse":
        return REPORT_SYSTEM_PROMPT, build_terse_report_prompt()
    if variant == "full":
        return REPORT_SYSTEM_PROMPT, build_report_prompt()
    raise ValueError(f"unknown prompt variant {variant!r}; known: full, terse")


def prepare_conditions(manifest: pd.DataFrame, images_root: Path, out_dir: Path,
                       conditions: list[str]) -> list[dict]:
    records: list[dict] = []
    blank_dir = out_dir / "blank_images"
    if "blank" in conditions:
        blank_dir.mkdir(parents=True, exist_ok=True)
    for _, row in manifest.iterrows():
        image_id = str(row["image_id"])
        src = images_root / str(row["image_relpath"])
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
                    Image.new("RGB", im.size, BLANK_RGB).save(dst)
            cond["blank"] = str(dst)
        records.append({
            "image_id": image_id,
            "conditions": cond,
            "dataset": str(row.get("dataset", "openi")),
            "patient_id": str(row.get("patient_id", "")),
            "reference_text": str(row.get("reference_text", "") or ""),
        })
    return records


# See 43_run_padchest_shift.py: only "this wrapper cannot batch" is recoverable.
_BATCH_REJECTION_ERRORS = (TypeError, ValueError, AttributeError, AssertionError,
                           NotImplementedError, KeyError, IndexError)


def run_batch(model, images: list, system_prompt: str, user_prompt: str,
              max_new_tokens: int) -> tuple[list[str], list[str]]:
    """Generate for a chunk, falling back to serial only on batch rejection.

    Wrappers build their own prompt scaffolding, so the system and user turns are
    joined the way ``06``/``08`` do: system text, a blank line, then the user turn.
    """
    prompt = system_prompt + chr(10) * 2 + user_prompt
    raws: list[str] = []
    errs: list[str] = []
    if len(images) > 1 and getattr(model, "generate_batch", None):
        try:
            raws = list(model.generate_batch(images, [prompt] * len(images),
                                             max_new_tokens=max_new_tokens))
            return raws, [""] * len(raws)
        except _BATCH_REJECTION_ERRORS as exc:
            print(f"[warn] batching rejected by this wrapper "
                  f"({type(exc).__name__}: {exc}); falling back to one study at a "
                  f"time", flush=True)
    for img in images:
        raws.append(model.generate(img, prompt, max_new_tokens=max_new_tokens))
        errs.append("")
    return raws, errs


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z]{3,}", (text or "").lower())}


def summarise(records: list[dict], by_key: dict, conditions: list[str],
              variant: str, out_dir: Path, model_name: str) -> dict[str, Any]:
    """Lexical overlap plus format compliance.

    The lexical number is a *diagnostic*, not the reported metric: the reported
    omission/hallucination rates come from the judged labeller in script 42. Its
    job here is to be cheap enough to compute during the run and sharp enough to
    catch a broken run immediately -- a case where original and blank outputs are
    identical, or where every output is empty.
    """
    blank_stats: dict[str, Any] = {"n": 0, "mean_jaccard": None,
                                   "identical_to_original": 0, "example": ""}
    format_violations = 0
    empty_outputs = 0
    generation_failures = 0
    overlaps: list[float] = []

    for rec in records:
        orig = by_key.get((rec["image_id"], "original"))
        blank = by_key.get((rec["image_id"], "blank"))
        if orig:
            if orig.get("error"):
                generation_failures += 1
            if not (orig.get("raw_output") or "").strip():
                empty_outputs += 1
            if orig.get("format_violation") or looks_structured(orig.get("raw_output") or ""):
                format_violations += 1
        if not (orig and blank):
            continue
        a = (orig.get("raw_output") or "").strip()
        b = (blank.get("raw_output") or "").strip()
        if not a or not b:
            continue
        ta, tb = _tokens(a), _tokens(b)
        if ta or tb:
            overlaps.append(len(ta & tb) / max(1, len(ta | tb)))
        blank_stats["n"] += 1
        if a == b:
            blank_stats["identical_to_original"] += 1
            if not blank_stats["example"]:
                blank_stats["example"] = a[:200]

    if overlaps:
        blank_stats["mean_jaccard"] = sum(overlaps) / len(overlaps)
    if blank_stats["n"]:
        blank_stats["pct_identical_to_original"] = round(
            100.0 * blank_stats["identical_to_original"] / blank_stats["n"], 2)

    summary: dict[str, Any] = {
        "model": model_name,
        "layer": "openi_external_report",
        "design": "free-text report generation + blank control",
        "prompt_variant": variant,
        "n_studies": len(records),
        "conditions_run": conditions,
        "cells_on_disk": len(by_key),
        "generation_failures": generation_failures,
        "empty_outputs": empty_outputs,
        "format_violations": format_violations,
        "blank_control": blank_stats,
        "note": ("omission and hallucination rates are produced by "
                 "42_geographic_shift_analysis.py, which labels BOTH the reference "
                 "and the generation with the same evidence rule; its judged "
                 "upgrade path is 20_rrg_labeler_eval.py"),
    }
    (out_dir / "report_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    ap.add_argument("--manifest", default="data/manifests/openi_rrg.csv")
    ap.add_argument("--images-root", default="data/open-i")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--prompt-variant", default="full", choices=["full", "terse"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default=None, help="comma-separated splits to keep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=192,
                    help="a findings section is short; 192 is generous and keeps "
                         "the image-reliance contrast honest")
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    mpath = Path(args.manifest)
    if not mpath.is_absolute():
        mpath = ROOT / mpath
    manifest = pd.read_csv(mpath)
    if "reference_text" not in manifest.columns:
        print("[error] manifest has no reference_text column; this runner needs the "
              "reference report to compare against")
        return 2
    if args.split and "split" in manifest.columns:
        wanted = [s.strip() for s in args.split.split(",") if s.strip()]
        before = len(manifest)
        manifest = manifest[manifest["split"].isin(wanted)].copy()
        print(f"split filter {wanted}: {before} -> {len(manifest)} studies", flush=True)
    if len(manifest):
        manifest = manifest.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    if args.limit:
        manifest = manifest.head(args.limit)
    print(f"manifest: {len(manifest)} studies; conditions={args.conditions}; "
          f"prompt={args.prompt_variant}", flush=True)
    if manifest.empty:
        print("[error] nothing to do: the manifest selection is empty")
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rrg_predictions.jsonl"

    records = prepare_conditions(manifest, Path(args.images_root), out_dir,
                                 args.conditions)
    # Condition-major on purpose: a batch runs until its LONGEST member finishes,
    # so a mixed batch costs what its slowest cell costs. See docs/50 section 2.8.
    all_jobs = [(r, c) for c in args.conditions for r in records if c in r["conditions"]]
    if not all_jobs:
        print("[error] no runnable jobs (check --images-root and the manifest paths)")
        return 2

    system_prompt, user_prompt = build_prompts(args.prompt_variant)

    model = get_model(args.model, root=ROOT, device_map="auto",
                      load_in_4bit=args.load_in_4bit)
    if os.environ.get("RUNNER_STUB") == "1":
        # Fail loudly if the stub did not take over: a silently un-stubbed test
        # run would load a real checkpoint and still "pass".
        assert type(model).__name__ == "_StubVLM", (
            f"RUNNER_STUB=1 but get_model returned {type(model).__name__}; "
            f"the stub did not patch this import path")
    model.load()

    with DurableJsonl(out_path, key=record_key) as out:
        pending = [j for j in all_jobs if out.needs(j[0]["image_id"], j[1])]
        print(f"resume: {len(all_jobs) - len(pending)} of {len(all_jobs)} cells already "
              f"durable, {len(pending)} to run", flush=True)

        pbar = tqdm(total=len(pending), desc=f"{args.model} | open-i report (bs={args.batch_size})",
                    unit="study", dynamic_ncols=True)
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
                        "image_path": path, "dataset": rec.get("dataset", "openi"),
                        "reference_text": rec.get("reference_text", ""),
                        "raw_output": "",
                        "error": f"image_load_failed: {type(exc).__name__}: {exc}",
                        "elapsed_sec": 0.0,
                        "parsed": parse_vlm_output("").to_dict(),
                    })
                    pbar.update(1)
            if not images:
                continue

            t0 = time.time()
            try:
                raws, errs = run_batch(model, images, system_prompt, user_prompt,
                                       args.max_new_tokens)
            except BaseException as exc:  # noqa: BLE001
                # Same contract as 43: an unexpected hard failure exits non-zero
                # so the orchestrator does not record a half-done stage as
                # complete, while everything already fsynced stays usable.
                pbar.close()
                interrupted = exc
                break
            per_img = (time.time() - t0) / len(images)

            rows = []
            for (rec, condition, path), raw, err in zip(meta, raws, errs):
                # The structured parse is kept even though it is not the primary
                # metric: when the model ignores the prose instruction and emits
                # JSON, this is what records it, and it lets the same file feed the
                # existing selective-metric tooling if a reviewer asks for it.
                parsed = parse_vlm_output(raw) if not err else parse_vlm_output("")
                rows.append({
                    "image_id": rec["image_id"],
                    "condition": condition,
                    "image_path": path,
                    "dataset": rec.get("dataset", "openi"),
                    "reference_text": rec.get("reference_text", ""),
                    "prompt_variant": args.prompt_variant,
                    "raw_output": raw,
                    "error": err,
                    "elapsed_sec": per_img,
                    "format_violation": int(looks_structured(raw or "")),
                    "parsed": parsed.to_dict(),
                })
            out.write_many(rows)          # append + fsync + durable marker
            pbar.update(len(chunk))
            pbar.set_postfix(sec_per_study=f"{per_img:.1f}s")
        pbar.close()
        stats = out.stats()

    by_key = {(str(r.get("image_id")), str(r.get("condition"))): r
              for r in read_jsonl(out_path)}
    summary = summarise(records, by_key, args.conditions, args.prompt_variant,
                        out_dir, args.model)
    summary["durable"] = stats
    (out_dir / "report_summary.json").write_text(
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
