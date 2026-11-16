"""Shared plumbing for the frontier-API probe.

Isolation contract (see papers/参考/外部商用api接口加调试.md):
  * every file this package writes lives under ``outputs/api_probe/<model_slug>/``;
  * existing manifests, state files and prediction files are read-only inputs;
  * image construction, prompts and the parser are imported from the CXR /
    cross-modal code paths rather than re-implemented, so the API arm measures
    the same object as the local arm.

The API key is read from the environment and is never written to disk.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from src.cxr_trust.inference import prompts_ablation as PA  # noqa: E402
from src.cxr_trust.inference.prompts import PRIMARY_LABELS  # noqa: E402
from src.cxr_trust.inference.prompts_crossmodal import (  # noqa: E402
    MODALITY_LABELS,
    build_prompt as build_crossmodal_prompt,
)
from src.cxr_trust.inference.prompts_report import (  # noqa: E402
    REPORT_SYSTEM_PROMPT,
    build_report_prompt,
)
from src.cxr_trust.inference.shifts import (  # noqa: E402
    DEFAULT_SEVERITIES,
    apply_shift,
    make_blank,
)

MODEL = os.environ.get("OPENAI_MODEL", "deepseek-v4.1-flash")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://amr-link.open-design.ai/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "")

MAX_EDGE = 1024
JPEG_QUALITY = 95
MAX_NEW_TOKENS = 8192
SEED = 42

# The NIH manifest's relpath is "images/<file>" but the files sit one level
# deeper than for PadChest/Open-i, so its root carries the extra "images".
IMAGE_ROOTS = {
    "nih": ROOT / "data" / "nih" / "images",
    "padchest": ROOT / "data" / "padchest",
    "openi": ROOT / "data" / "open-i",
}

# cohort -> (labels, prompt builder)
CROSS_MODAL = {"mri", "oct", "derm"}


def slug(name: str = MODEL) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


OUT_DIR = ROOT / "outputs" / "api_probe" / slug()
MANIFEST_DIR = OUT_DIR / "manifests"


def ensure_dirs() -> None:
    for d in (OUT_DIR, MANIFEST_DIR):
        d.mkdir(parents=True, exist_ok=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def setup_logging(name: str) -> logging.Logger:
    ensure_dirs()
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(OUT_DIR / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def append_event(event: dict[str, Any]) -> None:
    """One JSON object per line, appended by the single writer only."""
    ensure_dirs()
    with open(OUT_DIR / "events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# image handling
# --------------------------------------------------------------------------- #
def resolve_image(row: dict, cohort: str) -> Path:
    """Resolve an image path. CXR is two datasets, so the dataset decides the root."""
    rel = str(row["image_relpath"])
    if cohort in CROSS_MODAL:
        # cross-modal relpaths already carry their modality folder (brain_mri/...)
        return ROOT / "data" / rel
    dataset = str(row.get("dataset") or cohort)
    if dataset == "openi":
        return IMAGE_ROOTS["openi"] / rel
    return IMAGE_ROOTS[dataset] / rel


def encode_image(img: Image.Image) -> tuple[str, int, int, int]:
    """JPEG q=95, long edge <= 1024. Returns (b64, w, h, bytes)."""
    w, h = img.size
    scale = min(1.0, MAX_EDGE / max(w, h))
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                         Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=JPEG_QUALITY)
    raw = buf.getvalue()
    return base64.b64encode(raw).decode("ascii"), img.size[0], img.size[1], len(raw)


def build_condition_image(img: Image.Image, condition: str, image_id: str) -> Image.Image:
    img = img.convert("RGB")
    if condition in ("clean", "native", "real"):
        return img
    if condition in ("blank_gray", "blank_phase", "blank_shuffle"):
        seed = int(hashlib.sha1(f"{image_id}|{condition}|20261114".encode())
                   .hexdigest()[:8], 16)
        return make_blank(img, condition, seed)
    shift, _, severity = condition.partition("_")
    return apply_shift(img, shift, severity)


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #
def prompt_for(cohort: str, schema: str) -> tuple[str, str, list[str]]:
    """Return (system, user, labels)."""
    if cohort == "openi":
        return REPORT_SYSTEM_PROMPT, build_report_prompt(), list(PRIMARY_LABELS)
    if cohort in CROSS_MODAL:
        system, user = build_crossmodal_prompt(cohort, schema)
        return system, user, list(MODALITY_LABELS[cohort])
    if schema == "v1":
        return PA.V1_SYSTEM, PA.V1_USER, list(PRIMARY_LABELS)
    return PA.V0_SYSTEM, PA.V0_USER, list(PRIMARY_LABELS)


def prompt_fingerprint(cohort: str, schema: str) -> dict:
    system, user, labels = prompt_for(cohort, schema)
    return {
        "cohort": cohort, "schema": schema,
        "system_sha256": sha256_text(system),
        "user_sha256": sha256_text(user),
        "labels_sha256": sha256_text(json.dumps(labels)),
    }


# --------------------------------------------------------------------------- #
# request construction / response handling
# --------------------------------------------------------------------------- #
@dataclass
class Request:
    probe_id: str
    cohort: str
    schema: str
    condition: str
    image_b64: str
    system: str
    user: str
    labels: list[str]
    meta: dict


def payload_for(req: Request, model: str = MODEL) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": req.system},
            {"role": "user", "content": [
                {"type": "text", "text": req.user},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{req.image_b64}"}},
            ]},
        ],
        "temperature": 0,
        "top_p": 1,
        # Measured: at 1,024 the reasoning model spends the whole budget on its
        # hidden trace and returns ``choices: []``. At 4,096 one CXR cell in 16
        # still exhausted the budget, so the cap is 8,192. A typical call stops
        # near 2,900 tokens, so the higher cap costs nothing unless it is needed.
        "max_tokens": MAX_NEW_TOKENS,
    }


class Retryable(Exception):
    pass


def _post_sync(url: str, headers: dict, payload: dict,
               timeout: float) -> tuple[int, bytes]:
    """Blocking POST via the standard library.

    httpx/httpcore completes its TLS handshake with this relay by sending an
    ALPN offer the gateway answers with a bare EOF (``EOF occurred in violation
    of protocol``), while both curl and urllib negotiate fine. Since urllib is
    stdlib and this arm is I/O-bound rather than CPU-bound, running it in a
    thread is the smaller risk than carrying a client that cannot connect.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


async def call_api(payload: dict, max_retries: int = 5,
                   log: logging.Logger | None = None,
                   probe_id: str = "") -> tuple[dict, dict]:
    """POST with exponential backoff. Returns (response_json, attempt_meta)."""
    url = BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}",
               "Content-Type": "application/json"}
    delay = 3.0
    last: Exception | None = None
    for attempt in range(1, max_retries + 1):
        started = time.perf_counter()
        try:
            status, raw = await __import__("asyncio").to_thread(
                _post_sync, url, headers, payload, 180.0)
            elapsed = time.perf_counter() - started
            if status in (429, 502, 503, 504):
                raise Retryable(f"HTTP {status}")
            if status >= 400:
                raise RuntimeError(f"HTTP {status}: {raw[:200]!r}")
            return json.loads(raw), {"attempts": attempt, "http_status": status,
                                     "latency_s": elapsed}
        except (Retryable, urllib.error.URLError, TimeoutError, ConnectionError,
                OSError) as exc:
            last = exc
            if log:
                log.warning("retry %s attempt %d/%d: %s: %r",
                            probe_id, attempt, max_retries,
                            type(exc).__name__, str(exc)[:160])
            if attempt == max_retries:
                break
            await __import__("asyncio").sleep(delay + random.uniform(0, 1))
            delay *= 2
    raise RuntimeError(f"exhausted {max_retries} attempts: "
                       f"{type(last).__name__}: {last}")


def parse_response(body: dict, labels: list[str], cohort: str) -> dict:
    from src.cxr_trust.inference.schema import (
        extract_json_object,
        parse_vlm_output,
    )
    reasoning_chars = 0
    try:
        choices = body.get("choices") or []
        if not choices:
            # Reasoning model exhausted max_tokens before emitting a visible
            # answer. This is a measured failure mode, not a transport error:
            # usage is present and must be kept.
            content, finish = "", "empty_choices"
        else:
            choice = choices[0]
            msg = choice.get("message") or {}
            content = msg.get("content") or ""
            reasoning_chars = len(msg.get("reasoning_content") or "")
            finish = choice.get("finish_reason")
    except Exception:
        content, finish = "", "malformed_response"

    if cohort == "openi":
        parsed = {"parse_success": False, "parse_error": "prose_expected",
                  "findings": {}, "rationale": "", "image_reliance_claim": None}
        present = 0
    else:
        pp = parse_vlm_output(content, labels=labels)
        parsed = pp.to_dict()
        obj = extract_json_object(content) or {}
        fo = obj.get("findings") if isinstance(obj, dict) else None
        present = 0
        if isinstance(fo, dict):
            norm = {str(k).strip().lower().replace(" ", "_") for k in fo}
            present = sum(1 for lab in labels if lab in norm)

    usage = body.get("usage") or {}
    details = usage.get("completion_tokens_details") or {}
    return {
        "content": content,
        "finish_reason": finish,
        "parsed": parsed,
        "labels_present": present,
        "label_completeness": present / len(labels) if labels else 0.0,
        "reasoning_chars": reasoning_chars,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        "model_returned": body.get("model", ""),
    }
