#!/usr/bin/env python
"""Run MedGemma on CheXpert-plus-RRG report-generation subset.

Supports original and optional blank-image condition for image reliance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.inference.durable import DurableJsonl  # noqa: E402
from src.cxr_trust.inference.prompts import SIMPLE_SYSTEM_PROMPT, build_simple_prompt  # noqa: E402
from src.cxr_trust.inference.schema import parse_vlm_output  # noqa: E402
from src.cxr_trust.models.registry import get_model  # noqa: E402

# Test seam: when RUNNER_STUB=1 the model factory is replaced AFTER all imports have
# run. It must be installed here and not from sitecustomize, because src/ has no
# __init__.py, so "src.cxr_trust.*" and "cxr_trust.*" are two separate module
# objects -- a patch applied before this import would miss this one and the runner
# would quietly load a real checkpoint while the test believed it was stubbed.
if os.environ.get("RUNNER_STUB") == "1":  # pragma: no cover - test-only path
    import stub_model

    stub_model.install()



def _rrg_key(rec: dict) -> tuple:
    return (str(rec.get("image_id")), str(rec.get("condition")))


# Only "this wrapper cannot batch" is recoverable by going one image at a time;
# a driver or allocator failure must propagate so the stage exits non-zero.
_BATCH_REJECTION_ERRORS = (TypeError, ValueError, AttributeError, AssertionError,
                           NotImplementedError, KeyError, IndexError)


def load_done(path: Path) -> set[tuple[str, str]]:
    """Superseded by the durable writer's byte-offset sidecar; kept for reference."""
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
            pass
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medgemma_4b_it", choices=["medgemma_4b_it", "qwen25_vl_7b_instruct", "chexagent_8b", "qwen3_vl_8b_instruct", "internvl3_5_8b", "llava_med_v1_5_7b", "medgemma_15_4b_it"])
    parser.add_argument("--manifest", required=True, help="Path to RRG manifest CSV (e.g. data/manifests/openi_rrg.csv)")
    parser.add_argument("--images-root", required=True, help="Path to images directory (e.g. data/open-i)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--blank-limit", type=int, default=0, help="also run blank image for first N images")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text(encoding="utf-8"))
    spec = cfg["primary_models"][args.model]
    model = get_model(args.model, model_path=ROOT / spec["path"], device_map=spec.get("device_map", "auto"),
                      load_in_4bit=args.load_in_4bit or spec.get("load_in_4bit", False),
                      max_memory=spec.get("max_memory"))
    model.load()
    prompt = SIMPLE_SYSTEM_PROMPT + chr(10) * 2 + build_simple_prompt()

    if os.environ.get("RUNNER_STUB") == "1":
        # Fail loudly if the stub did not take over: a silently un-stubbed test run
        # would load a real checkpoint and still "pass".
        assert type(model).__name__ == "_StubVLM", (
            "RUNNER_STUB=1 but get_model returned "
            + type(model).__name__ + "; the stub did not patch this import path")


    manifest = pd.read_csv(args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rrg_predictions.jsonl"

    jobs = []
    for idx, row in manifest.iterrows():
        jobs.append((idx, row, "original"))
        if idx < args.blank_limit:
            jobs.append((idx, row, "blank"))

    # Crash-safe writing (src/cxr_trust/inference/durable.py). The old version
    # opened `"a" if done else "w"` and only flushed. Three ways that lost data:
    # a kill dropped the OS-buffered tail; a torn final line was swallowed by
    # load_done and then concatenated onto by the next append, corrupting two
    # rows; and because `done` counted only rows that parsed, a file containing
    # nothing but failed rows looked empty -- so the next run opened it with mode
    # "w" and destroyed every row it held.
    with DurableJsonl(out_path, key=_rrg_key) as out:
        pending = [j for j in jobs if out.needs(j[1]["image_id"], j[2])]
        print(f"resume: {len(jobs) - len(pending)} of {len(jobs)} cells already durable, "
              f"{len(pending)} to run", flush=True)

        pbar = tqdm(total=len(pending),
                    desc=f"{args.model} | RRG (bs={args.batch_size})",
                    unit="img", dynamic_ncols=True)
        interrupted = None
        for i in range(0, len(pending), args.batch_size):
            chunk = pending[i:i + args.batch_size]
            images = []
            try:
                for idx, row, condition in chunk:
                    # Context manager: a multi-day loop that leaks one file handle
                    # per image eventually hits the handle limit, and the failure
                    # lands in the middle of a batch.
                    with Image.open(Path(args.images_root) / row["image_relpath"]) as im:
                        image = im.convert("RGB")
                    if condition == "blank":
                        image = Image.new("RGB", image.size, (128, 128, 128))
                    images.append(image)
            except Exception as exc:  # noqa: BLE001
                out.write({
                    "image_id": chunk[0][1]["image_id"],
                    "condition": chunk[0][2],
                    "error": f"image_load_failed: {type(exc).__name__}: {exc}",
                    "raw_output": "", "elapsed_sec": 0.0,
                    "parsed": parse_vlm_output("").to_dict(),
                })
                pbar.update(1)
                continue

            prompts = [prompt] * len(chunk)

            start = time.time()
            try:
                if args.batch_size > 1 and getattr(model, "generate_batch", None):
                    try:
                        raws = model.generate_batch(images, prompts, max_new_tokens=args.max_new_tokens)
                        errs = [""] * len(raws)
                    except _BATCH_REJECTION_ERRORS as exc:
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
            for j, (idx, row, condition) in enumerate(chunk):
                raw = raws[j]
                err = errs[j]
                parsed = parse_vlm_output(raw) if not err else parse_vlm_output("")
                rows.append({
                    "image_id": row["image_id"],
                    "dataset": Path(args.manifest).stem,
                    "task": row.get("task", ""),
                    "condition": condition,
                    "reference_text": row.get("reference_text", ""),
                    "default_prompt": row.get("default_prompt", ""),
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

    print(f"done; wrote {out_path} ({stats['records_on_disk']} cells, "
          f"{stats['appended_this_run']} added this run)")
    return 0


if __name__ == "__main__":
    # raise SystemExit, not a bare main(): the interrupted path returns 1 and the
    # caller decides completion from the exit code.
    raise SystemExit(main())
