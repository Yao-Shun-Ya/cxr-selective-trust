"""Step 3: adaptive-limit async runner (AIMD) with a single durable writer.

Concurrency is a control loop, not a constant: +1 after 50 consecutive successes
(cap 10), halved on 429/502/503/timeout (floor 2) with exponential backoff. Every
response is funnelled through one queue into one ``DurableJsonl`` writer, so no
two coroutines ever touch the file.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
from pathlib import Path

import httpx
import pandas as pd
from PIL import Image

from common import (
    API_KEY, BASE_URL, JPEG_QUALITY, MAX_EDGE, MAX_NEW_TOKENS, MANIFEST_DIR, MODEL, OUT_DIR,
    Request, build_condition_image, call_api, encode_image, ensure_dirs,
    parse_response, payload_for, prompt_for, prompt_fingerprint, resolve_image,
    setup_logging, sha256_file,
)
from src.cxr_trust.inference.durable import DurableJsonl

# Measured: with 8 concurrent TLS handshakes the gateway resets a third of the
# connections, and halving on every hard failure pinned the effective concurrency
# near 2 while recovery needed 50 consecutive successes. Start at 6, recover at
# +1 per 20, and decay by 25% instead of 50%.
START, LO, HI = 6, 2, 10
SUCCESS_STREAK = 20


class AIMD:
    def __init__(self, start=START, lo=LO, hi=HI):
        self.limit = start
        self.lo, self.hi = lo, hi
        self.active = 0
        self.streak = 0
        self.cond = asyncio.Condition()

    async def acquire(self):
        async with self.cond:
            while self.active >= self.limit:
                await self.cond.wait()
            self.active += 1

    async def release(self, ok: bool):
        async with self.cond:
            self.active -= 1
            if ok:
                self.streak += 1
                if self.streak >= SUCCESS_STREAK and self.limit < self.hi:
                    self.limit += 1
                    self.streak = 0
            else:
                self.streak = 0
                self.limit = max(self.lo, int(self.limit * 0.75))
            self.cond.notify()


def _guard_provenance(prov: dict) -> None:
    path = OUT_DIR / "provenance.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != prov:
            diff = {k: (old.get(k), prov.get(k)) for k in prov
                    if old.get(k) != prov.get(k)}
            raise SystemExit(f"provenance mismatch: {diff}. Move {OUT_DIR} away "
                             "or use a new model slug; refusing to mix runs.")
    else:
        path.write_text(json.dumps(prov, indent=2), encoding="utf-8")


async def run(args) -> int:
    ensure_dirs()
    log = setup_logging("03_run_api_probe")
    if not API_KEY:
        log.error("OPENAI_API_KEY is empty")
        return 2

    man_path = Path(args.manifest) if args.manifest else (
        MANIFEST_DIR / f"probe_{args.profile}.csv")
    rows = pd.read_csv(man_path).to_dict("records")
    if args.dry_run:
        rows = rows[:20]
    log.info("manifest=%s rows=%d model=%s", man_path.name, len(rows), args.model)

    prov = {
        "model": args.model,
        "manifest_sha256": sha256_file(man_path),
        "max_edge": MAX_EDGE,
        "jpeg_quality": JPEG_QUALITY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompts": {
            f"{c}|{s}": prompt_fingerprint(c, s)
            for c, schemas in {
                "cxr": ("v0", "v1"), "mri": ("v0", "v1"),
                "oct": ("v0", "v1"), "derm": ("v0", "v1"),
                "openi": ("report",),
            }.items()
            for s in schemas
        },
        "runner_sha256": sha256_file(Path(__file__)),
    }
    _guard_provenance(prov)
    prov_id = __import__("hashlib").sha256(
        json.dumps(prov, sort_keys=True).encode()).hexdigest()[:16]

    out_path = OUT_DIR / ("dry_run_predictions.jsonl" if args.dry_run
                          else "predictions.jsonl")
    gate = AIMD()
    queue: asyncio.Queue = asyncio.Queue()
    done = {"n": 0, "ok": 0, "err": 0, "t0": time.time()}
    state = {"total": 0}
    STOP = object()

    async def consumer(sink):
        while True:
            item = await queue.get()
            if item is STOP:
                break
            sink.write_many([item])
            done["n"] += 1
            if not item.get("error"):
                done["ok"] += 1
            else:
                done["err"] += 1
            if done["n"] % 50 == 0:
                rate = done["n"] / max(1e-9, time.time() - done["t0"])
                left = max(0, state["total"] - done["n"])
                log.info("progress %d/%d ok=%d err=%d %.2f/s limit=%d eta=%.1fmin",
                         done["n"], state["total"], done["ok"], done["err"],
                         rate, gate.limit, (left / rate / 60) if rate else -1)

    async def worker(row):
        probe_id = row["probe_id"]
        started = time.perf_counter()
        try:
            system, user, labels = prompt_for(row["cohort"], row["schema"])
            img = build_condition_image(
                Image.open(resolve_image(row, row["cohort"])),
                row["condition"], row["image_id"])
            b64, w, h, nbytes = encode_image(img)
            req = Request(probe_id=probe_id, cohort=row["cohort"],
                          schema=row["schema"], condition=row["condition"],
                          image_b64=b64, system=system, user=user,
                          labels=labels, meta={})
            payload = payload_for(req, args.model)
            await gate.acquire()
            ok = False
            try:
                body, meta = await call_api(payload, log=log,
                                            probe_id=probe_id)
                ok = True
            finally:
                await gate.release(ok)
            parsed = parse_response(body, labels, row["cohort"])
            record = _record(row, parsed, meta, prov_id, w, h, nbytes,
                             time.perf_counter() - started, "", args.model)
        except Exception as exc:  # noqa: BLE001
            record = _record(row, {"content": "", "finish_reason": "",
                                   "parsed": {"parse_success": False,
                                              "parse_error": "api_error",
                                              "findings": {}},
                                   "labels_present": 0, "label_completeness": 0.0,
                                   "usage": {}, "model_returned": ""},
                             {"attempts": -1, "http_status": -1, "latency_s": 0},
                             prov_id, 0, 0, 0,
                             time.perf_counter() - started,
                             f"{type(exc).__name__}: {exc}", args.model)
        await queue.put(record)

    interrupted = False
    with DurableJsonl(out_path, key=lambda r: (str(r["probe_id"]),)) as sink:
        pending = [r for r in rows if (r["probe_id"],) not in sink.done]
        state["total"] = len(pending)
        log.info("resume: %d of %d already durable, %d to run",
                 len(rows) - len(pending), len(rows), len(pending))
        cons = asyncio.create_task(consumer(sink))
        BATCH = 200
        for i in range(0, len(pending), BATCH):
            chunk = pending[i:i + BATCH]
            try:
                await asyncio.gather(*(worker(r) for r in chunk))
            except asyncio.CancelledError:
                interrupted = True
                break
        await queue.put(STOP)
        await cons
        stats = sink.stats()

    summary = {
        "model": args.model, "manifest": man_path.name,
        "rows_target": len(rows), "run_this_session": done["n"],
        "ok": done["ok"], "errors": done["err"],
        "durable_total": stats["records_on_disk"],
        "wall_sec": round(time.time() - done["t0"], 1),
        "final_limit": gate.limit, "provenance": prov_id,
        "interrupted": interrupted,
    }
    (OUT_DIR / ("dry_run_summary.json" if args.dry_run else "run_summary.json")
     ).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("summary: %s", json.dumps(summary))
    return 0 if summary["errors"] == 0 else 1


def _record(row, parsed, meta, prov_id, w, h, nbytes, elapsed, error,
            model: str) -> dict:
    return {
        "probe_id": row["probe_id"],
        "model": model,
        "cohort": row["cohort"],
        "modality": row["modality"],
        "dataset": row["dataset"],
        "image_id": row["image_id"],
        "image_relpath": row["image_relpath"],
        "condition": row["condition"],
        "severity": row["severity"],
        "schema": row["schema"],
        "split": row["split"],
        "cluster_id": row["cluster_id"],
        "cluster_level": row["cluster_level"],
        "sent_w": w, "sent_h": h, "sent_bytes": nbytes,
        "raw_output": parsed["content"],
        "finish_reason": parsed["finish_reason"],
        "model_returned": parsed["model_returned"],
        "parsed": parsed["parsed"],
        "labels_present": parsed["labels_present"],
        "label_completeness": parsed["label_completeness"],
        "reasoning_chars": parsed.get("reasoning_chars", 0),
        "usage": parsed["usage"],
        "attempts": meta["attempts"], "http_status": meta["http_status"],
        "latency_s": round(meta["latency_s"], 3),
        "elapsed_sec": round(elapsed, 3),
        "error": error,
        "provenance": prov_id,
        "ground_truth": json.loads(row["ground_truth_json"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="full_multimodal")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dry-run", action="store_true")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
