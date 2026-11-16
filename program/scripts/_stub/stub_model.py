"""A stand-in VLM that satisfies the wrapper interface WITHOUT loading weights.

Injected by ``41_test_runner_resume.py`` through the ``PYTHONPATH`` site hooks
(``sitecustomize.py`` -> this module -> ``sitecustomize.py`` again, so it also
runs after ``site`` has finished).

Why a stub is the right instrument here: the bugs this test hunts are in the
*runner's* I/O and resume logic, not in the model. Loading a real 4B checkpoint
would add minutes and gigabytes while testing the same lines. The stub keeps the
exact call contract the real wrappers honour -- ``generate(image, prompt,
max_new_tokens=...)`` and ``generate_batch(images, prompts, max_new_tokens=...)``
-- and returns a deterministic reply derived from the image, so the test can also
assert that a resumed row was not silently regenerated with different content.

``CRASH_AFTER_BATCHES`` makes the stub raise on the Nth *batch* call. Because the
counter lives in a file, the crash point survives across the separate runner
processes the test launches: run 1 crashes after batch N, run 2 resumes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _counter_path() -> Path:
    env = os.environ.get("STUB_COUNTER_FILE")
    return Path(env) if env else Path("stub_counter.json")


# The counter file is shared across the runner processes a single test case
# launches, so it accumulates. Crash points must therefore be counted RELATIVE to
# the start of each process: with an absolute count, a test that asked for "fail on
# batch 2" of the second run was actually comparing against batch 9, failed before
# writing anything, and reported zero partial rows -- a test bug that looked like a
# durability bug. The baseline is captured on the process's first call, which is
# exactly what "this run's Nth batch" means.
_BASELINE: int | None = None


def _bump() -> int:
    """Increment the shared call counter; return how many calls THIS process made."""
    global _BASELINE
    p = _counter_path()
    data = {"batch_calls": 0}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {"batch_calls": 0}
    total = int(data.get("batch_calls", 0)) + 1
    data["batch_calls"] = total
    p.write_text(json.dumps(data), encoding="utf-8")
    if _BASELINE is None:
        _BASELINE = total - 1
    return total - _BASELINE


def _reply_for(image) -> str:
    """Deterministic, image-derived reply in the style the Open-i runner expects."""
    try:
        digest = hashlib.md5(image.tobytes()).hexdigest()[:8]
        size = image.size
    except Exception:  # noqa: BLE001
        digest, size = "drawfail", (0, 0)
    return (f"The cardiac silhouette is within normal limits. No focal consolidation "
            f"or pleural effusion is identified. Study signature {digest}.")


def _crash_limit() -> int:
    """Read the crash point at CALL time, not import time.

    The env var is set by the test harness that launches this process; reading it
    once at import would freeze whatever the parent had.
    """
    return int(os.environ.get("CRASH_AFTER_BATCHES", "0") or 0)


def _maybe_hard_fail(batch_index: int) -> None:
    """Raise a non-recoverable error at batch N when asked.

    ``RuntimeError`` is deliberately NOT in the runners' "this wrapper cannot
    batch" set, so it must propagate and take the process down with a non-zero
    exit code. That is the distinction the tests exist to protect: a driver or
    allocator failure must never be recorded as a file full of per-row error
    strings with an exit code of 0.
    """
    at = int(os.environ.get("STUB_HARD_FAIL_AT", "0") or 0)
    if at and batch_index >= at:
        raise RuntimeError(f"STUB_HARD_FAILURE: simulated driver error at batch {batch_index}")


class _StubVLM:
    name = "stub_vlm"
    supports_batching = True

    def __init__(self, model_name: str = "medgemma_4b_it", **kwargs) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def generate(self, image, prompt: str, max_new_tokens: int = 512) -> str:
        n = _bump()
        _maybe_hard_fail(n)
        limit = _crash_limit()
        if limit and n > limit:
            raise RuntimeError(f"STUB_CRASH: simulated power cut on call {n}")
        return _reply_for(image)

    def generate_batch(self, images, prompts, max_new_tokens: int = 512, chunk=None):
        n = _bump()
        _maybe_hard_fail(n)
        limit = _crash_limit()
        if limit and n > limit:
            raise RuntimeError(f"STUB_CRASH: simulated power cut on batch {n}")
        return [_reply_for(img) for img in images]


def install() -> None:
    """Patch ``get_model`` in the registry module the runner actually imported.

    There are TWO import paths to the same file, and they are separate module
    objects with separate globals:

        src.cxr_trust.models.registry        <- what every script/ file uses
        cxr_trust.models.registry            <- what sets root=<program>/src

    The first version of this stub patched only the second, so the runner used
    the real factory and silently loaded a multi-GB checkpoint -- the test looked
    like it was exercising a stub while it was exercising MedGemma. Both names
    are patched here, and any module already imported under either name is fixed
    in place.
    """
    import importlib
    import sys as _sys

    def _stub_get_model(name, *args, **kwargs):
        return _StubVLM(name)

    patched: list[str] = []
    for modname in ("cxr_trust.models.registry", "src.cxr_trust.models.registry"):
        mod = _sys.modules.get(modname)
        if mod is None:
            try:
                mod = importlib.import_module(modname)
            except Exception:  # noqa: BLE001
                continue
        if mod is not None and getattr(mod, "get_model", None) is not _stub_get_model:
            mod.get_model = _stub_get_model
            patched.append(modname)

    # Any module that already did `from ... import get_model` holds its own
    # reference; rebind those too so an early import cannot slip past.
    for modname, mod in list(_sys.modules.items()):
        if mod is None or modname.startswith("stub_model"):
            continue
        if getattr(mod, "get_model", None) is not None and \
                getattr(getattr(mod, "get_model"), "__module__", "").endswith("registry"):
            try:
                mod.get_model = _stub_get_model
                patched.append(modname)
            except Exception:  # noqa: BLE001
                pass

    message = "stub model installed for: " + (", ".join(sorted(set(patched))) or "NOTHING")
    marker = os.environ.get("STUB_REPORT")
    if marker:
        with open(marker, "a", encoding="utf-8") as fh:
            fh.write(message + "\n")
    print(f"[stub] {message}", flush=True)
