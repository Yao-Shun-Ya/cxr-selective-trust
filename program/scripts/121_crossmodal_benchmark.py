"""Cross-modal shift benchmark (protocol: docs/58 section 13).

Same 19-cell shift grid, same parser, same durable writer and the same
``apply_shift`` function object as the CXR arm (imported from
``cxr_trust.inference.shifts``, which ``08_shift_benchmark.py`` also uses), plus
three per-image blank cells that the P2 null needs.

    python scripts/121_crossmodal_benchmark.py \
        --model medgemma_15_4b_it --modality derm --schema v0

Resume safety: every stage directory carries a ``provenance.json`` fingerprint
(prompt, label set, manifest, runner source). Re-running with a different prompt
or manifest now refuses instead of silently reusing rows produced under the old
configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cxr_trust.inference.durable import DurableJsonl  # noqa: E402
from src.cxr_trust.inference.prompts_crossmodal import (  # noqa: E402
    full_prompt,
    labels_for,
)
from src.cxr_trust.inference.schema import (  # noqa: E402
    extract_json_object,
    parse_vlm_output,
)
from src.cxr_trust.inference.shifts import (  # noqa: E402
    BLANK_KINDS,
    DEFAULT_SEVERITIES,
    apply_shift,
    make_blank,
)
from src.cxr_trust.models.registry import MODEL_NAMES, get_model  # noqa: E402

# Which unit the bootstrap must resample on. OCT ships no patient identifier, so
# claiming a patient-level cluster would be false; the analysis reads this field
# rather than guessing from the modality name.
CLUSTER_LEVEL = {"mri": "patient", "derm": "lesion", "oct": "image"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _key(rec: dict) -> tuple:
    return (str(rec.get("image_id")), str(rec.get("condition")),
            str(rec.get("severity")))


def _provenance(path: Path, prompt: str, labels: list[str],
                manifest: Path) -> tuple[dict, str]:
    prov = {
        "prompt_sha256": _sha(prompt),
        "labels_sha256": _sha(json.dumps(labels)),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    prov_id = _sha(json.dumps(prov, sort_keys=True))[:16]
    return prov, prov_id


def _guard_provenance(out_dir: Path, prov: dict) -> None:
    """Refuse to resume a stage produced under a different configuration.

    Without this, changing one prompt word and re-running would keep every row
    already on disk and add new rows beside them, so a single result file could
    contain two experiments with no way to tell them apart.
    """
    path = out_dir / "provenance.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != prov:
            diff = {k: (previous.get(k), prov.get(k))
                    for k in prov if previous.get(k) != prov.get(k)}
            raise SystemExit(
                f"provenance mismatch in {out_dir}: {diff}. "
                "Move the old stage directory away or use a new --output-dir; "
                "refusing to mix configurations.")
    else:
        path.write_text(json.dumps(prov, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--modality", required=True, choices=["mri", "oct", "derm"])
    ap.add_argument("--schema", default="v0", choices=["v0", "v1"])
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--images-root", default=str(ROOT / "data"))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all rows; a positive value takes the first N "
                         "(pilots only, never a reported cell)")
    ap.add_argument("--shifts", nargs="+", default=list(DEFAULT_SEVERITIES))
    ap.add_argument("--severities", nargs="+", default=["mild", "moderate", "severe"])
    ap.add_argument("--no-blanks", action="store_true")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--vram-budget-gb", type=float, default=None,
                    help="hard GPU budget; the loader fails instead of paging "
                         "VRAM into host RAM when the model does not fit")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    manifest_path = (Path(args.manifest) if args.manifest
                     else ROOT / "data" / "manifests" / f"crossmodal_{args.modality}.csv")
    manifest = pd.read_csv(manifest_path)
    if args.limit:
        manifest = manifest.head(args.limit)
    print(f"manifest {manifest_path.name}: {len(manifest)} images", flush=True)

    out_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "outputs" / "crossmodal" / args.modality / args.schema / args.model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "shift_predictions.jsonl"

    labels = labels_for(args.modality)
    prompt = full_prompt(args.modality, args.schema)

    prov, prov_id = _provenance(out_path.parent, prompt, labels, manifest_path)
    _guard_provenance(out_dir, prov)
    print(f"provenance {prov_id}", flush=True)

    import torch
    torch.manual_seed(0)
    kwargs = {}
    if args.vram_budget_gb:
        # BaseVLM documents this form as "refuse to offload": a model that does
        # not fit raises immediately instead of being paged over PCIe.
        kwargs["max_memory"] = {0: f"{args.vram_budget_gb:g}GiB", "cpu": "0GiB"}
    model = get_model(args.model, root=ROOT, device_map="auto",
                      load_in_4bit=args.load_in_4bit, **kwargs)
    model.load()
    print(f"loaded {args.model}", flush=True)
    print(f"[prompt] {len(prompt)} chars, {len(labels)} labels, "
          f"schema={args.schema}", flush=True)
    import torch as _t
    if _t.cuda.is_available():
        print(f"[vram] after_load_alloc={_t.cuda.memory_allocated()/2**30:.1f}GiB", flush=True)

    blanks = [] if args.no_blanks else list(BLANK_KINDS)
    jobs = []
    for _, row in manifest.iterrows():
        jobs.append((row, "clean", "none", None))
        for shift in args.shifts:
            for severity in args.severities:
                jobs.append((row, shift, severity, DEFAULT_SEVERITIES[shift][severity]))
        for kind in blanks:
            jobs.append((row, kind, "none", None))

    images_root = Path(args.images_root)
    cluster_level = CLUSTER_LEVEL[args.modality]
    with DurableJsonl(out_path, key=_key) as out:
        pending = [j for j in jobs
                   if (str(j[0]["image_id"]), str(j[1]), str(j[2])) not in out.done]
        pending.sort(key=lambda j: (str(j[1]), str(j[2])))
        print(f"resume: {len(jobs) - len(pending)} of {len(jobs)} durable, "
              f"{len(pending)} to run", flush=True)

        pbar = tqdm(total=len(pending),
                    desc=f"{args.model}|{args.modality}|{args.schema}",
                    unit="img", dynamic_ncols=True)
        interrupted = None
        step = max(1, args.batch_size)
        for i in range(0, len(pending), step):
            chunk = pending[i:i + step]
            images = []
            for row, condition, severity, _ in chunk:
                img = Image.open(images_root / row["image_relpath"]).convert("RGB")
                if condition in BLANK_KINDS:
                    seed = int(hashlib.sha1(
                        f"{row['image_id']}|{condition}|20261114".encode()
                    ).hexdigest()[:8], 16)
                    img = make_blank(img, condition, seed)
                elif condition != "clean":
                    img = apply_shift(img, condition, severity)
                images.append(img)
            prompts = [prompt] * len(chunk)

            start = time.time()
            try:
                if args.batch_size > 1 and getattr(model, "generate_batch", None):
                    try:
                        raws = model.generate_batch(images, prompts,
                                                    max_new_tokens=args.max_new_tokens)
                    except (TypeError, ValueError, AttributeError, NotImplementedError):
                        raws = [model.generate(im, prompt,
                                               max_new_tokens=args.max_new_tokens)
                                for im in images]
                else:
                    raws = [model.generate(im, prompt, max_new_tokens=args.max_new_tokens)
                            for im in images]
            except BaseException as exc:  # noqa: BLE001
                interrupted = exc
                break

            elapsed = time.time() - start
            rows = []
            for j, (row, condition, severity, param) in enumerate(chunk):
                raw = raws[j]
                parsed = parse_vlm_output(raw, labels=labels)
                # ``parse_success`` only requires a JSON object that carries a
                # ``findings`` key, so a reply that names ONE label and omits the
                # rest still counts as parsed. Measured on this arm, that is
                # common: CheXagent echoed the literal placeholder ``<label>``
                # and Qwen3-VL returned a single key. Completeness is therefore
                # recorded separately, otherwise P4 would report a contract that
                # the replies do not actually satisfy.
                obj = extract_json_object(raw) or {}
                findings_obj = obj.get("findings") if isinstance(obj, dict) else None
                present = 0
                if isinstance(findings_obj, dict):
                    norm = {str(k).strip().lower().replace(" ", "_")
                            for k in findings_obj}
                    present = sum(1 for lab in labels if lab in norm)
                rows.append({
                    "image_id": str(row["image_id"]),
                    "model": args.model,
                    "modality": args.modality,
                    "schema": args.schema,
                    "condition": condition,
                    "severity": severity,
                    "param": param,
                    "cluster_id": str(row.get("split_unit") or row["image_id"]),
                    "cluster_level": cluster_level,
                    "provenance": prov_id,
                    "labels_present": present,
                    "label_completeness": present / len(labels),
                    "raw_output": raw,
                    "elapsed_sec": elapsed / len(chunk),
                    "parsed": parsed.to_dict(),
                    "ground_truth": {"label": str(row["label"])},
                })
            out.write_many(rows)
            pbar.update(len(chunk))
            pbar.set_postfix(sec_per_img=f"{elapsed / max(1, len(chunk)):.1f}s")
        pbar.close()
        stats = out.stats()

    if interrupted is not None:
        print(f"[INTERRUPTED] {type(interrupted).__name__}: {interrupted}", flush=True)
        print(f"[INTERRUPTED] {stats['records_on_disk']} durable; re-run to continue.",
              flush=True)
        return 1
    print(f"done; {stats['records_on_disk']} cells "
          f"({stats['appended_this_run']} added this run)")
    import torch as _t
    if _t.cuda.is_available():
        print(f"[vram] peak_alloc={_t.cuda.max_memory_allocated()/2**30:.1f}GiB "
              f"peak_reserved={_t.cuda.max_memory_reserved()/2**30:.1f}GiB",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
