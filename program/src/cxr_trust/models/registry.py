"""Model registry -- one entry per model, all served through a common interface.

Two wrapper families
--------------------
``legacy``
    The hand-written wrappers used to produce the published 4-bit results
    (``medgemma.py``, ``qwen25vl.py``, ``chexagent.py``). These are kept
    byte-for-byte so prior results stay reproducible, and they remain the
    reference for the original three models.

``generic``
    ``GenericVLM`` (``generic.py``): any modern checkpoint that loads through
    ``AutoProcessor`` + ``AutoModelForImageTextToText``. Adds true batch
    generation and lets a 4-bit / bf16 precision switch be a constructor
    argument instead of a separate code path.

Where each model lives
----------------------
Paths are shown as they exist on disk. Older checkpoints sit in
``models/<family>/<name>``; the 2026 additions sit flat in ``models/<name>``.
``MODEL_PATHS`` below is the single source of truth -- if a directory moves,
change it here only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import image_pixel_hints

# --------------------------------------------------------------------------- #
# on-disk locations (relative to program/)
#
# NOTE: the 2026 consolidation flattened every checkpoint into models/<name>/.
# The pre-2026 paths (models/medgemma/MedGemma-4B-IT etc.) no longer exist.
# --------------------------------------------------------------------------- #
MODEL_PATHS: dict[str, str] = {
    # --- the 8 models the study actually uses ---
    "medgemma_4b_it": "models/MedGemma-4B-IT",
    "qwen25_vl_7b_instruct": "models/Qwen2.5-VL-7B-Instruct",
    # fp16 copy of the shipped F32 checkpoint, produced by
    # scripts/62_convert_chexagent_fp16.py. The original stays on disk; this path
    # is the one that loads. Reason: torch_dtype is silently ignored for this
    # remote-code class, so the fp32 weights (31.15 GiB) landed on a 32 GiB card
    # and paged through the Windows driver -- see docs/50 section 2.2 and the
    # header of that script.
    "chexagent_8b": "models/CheXagent-8b-fp16",
    "medgemma_15_4b_it": "models/medgemma-1.5-4b-it",
    "qwen3_vl_8b_instruct": "models/Qwen3-VL-8B-Instruct",
    "qwen3_vl_8b_thinking": "models/Qwen3-VL-8B-Thinking",
    "internvl3_5_8b": "models/InternVL3_5-8B",
    "llava_med_v1_5_7b": "models/llava-med-v1.5-mistral-7b-hf",
    # Pre-quantized 4-bit (pack-quantized / compressed-tensors, 17.88 GB) so the
    # 30B MoE fits a 32 GB card. The bf16 original was 57.9 GB, did not fit, and
    # has been deleted (87.7 GB reclaimed along with the other unused models).
    "qwen3_vl_30b_a3b_awq4bit": "models/Qwen3-VL-30B-A3B-Instruct-AWQ4bit",
}

# Removed from disk and from the matrix. Kept here so the reasons travel with the
# code rather than living only in a document.
REMOVED_MODELS: dict[str, str] = {
    "chexagent_2_3b":
        "remote code uses the removed transformers API DynamicCache.get_max_length; "
        "the CXR-specialist role is covered by CheXagent-8B",
    "qwen3_vl_30b_a3b_instruct":
        "bf16 weights are 57.9 GB and cannot fit a 32 GB card; superseded by the "
        "pre-quantized qwen3_vl_30b_a3b_awq4bit",
}

# --------------------------------------------------------------------------- #
# model metadata: wrapper, role in the design, and default precision
# --------------------------------------------------------------------------- #
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "medgemma_4b_it": {
        "wrapper": "legacy", "family": "gemma3", "params_b": 4.0,
        "domain": "medical", "role": "anchor: image-dependent reference",
        "precision": "4bit",
    },
    "qwen25_vl_7b_instruct": {
        "wrapper": "legacy", "family": "qwen2_5_vl", "params_b": 7.0,
        "domain": "general", "role": "anchor: image-invariant negative control",
        "precision": "4bit",
    },
    "chexagent_8b": {
        "wrapper": "legacy", "family": "chexagent", "params_b": 8.0,
        "domain": "cxr", "role": "CXR specialist, large (needs .venv-chexagent)",
        "precision": "fp16",
    },
    "medgemma_15_4b_it": {
        "wrapper": "generic", "family": "gemma3", "params_b": 4.0,
        "domain": "medical", "role": "same-family version bump; adds grounding",
        "precision": "bf16",
    },
    "qwen3_vl_8b_instruct": {
        "wrapper": "generic", "family": "qwen3_vl", "params_b": 8.0,
        "domain": "general", "role": "same-family version bump (needs transformers>=4.57)",
        "precision": "bf16",
    },
    "qwen3_vl_8b_thinking": {
        "wrapper": "generic", "family": "qwen3_vl", "params_b": 8.0,
        "domain": "general",
        "role": "second reasoning model: native <think> channel",
        "precision": "bf16",
    },
    "qwen3_vl_30b_a3b_awq4bit": {
        "wrapper": "generic", "family": "qwen3_vl_moe", "params_b": 30.0,
        "domain": "general",
        "role": "scale question; pre-quantized 4-bit so it fits 32 GB",
        # Weights arrive already quantized, so no on-the-fly quantisation is
        # applied. Note the comparability caveat: this model's quantisation
        # scheme differs from the bitsandbytes NF4 used elsewhere in the matrix.
        "precision": "prebuilt4bit",
    },
    "internvl3_5_8b": {
        "wrapper": "internvl", "family": "internvl", "params_b": 8.0,
        "domain": "general", "role": "cross-family generality",
        "precision": "bf16",
    },
    "llava_med_v1_5_7b": {
        "wrapper": "llava", "family": "llava", "params_b": 7.0,
        "domain": "medical", "role": "older medical baseline for contrast",
        "precision": "bf16",
    },
}

MODEL_NAMES = list(MODEL_SPECS)

# Models that must NOT run in the legacy .venv-vl (transformers 4.51.3).
# See docs/39 section on environment forking.
NEEDS_MODERN_TRANSFORMERS = {
    "qwen3_vl_8b_instruct",
    "qwen3_vl_8b_thinking",
    "qwen3_vl_30b_a3b_instruct",
}

# Per-model generation defaults worth centralising.
MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    # Qwen-style processors take explicit pixel hints so every model in the
    # family sees the SAME image. Without a hint the ceiling comes from
    # qwen_vl_utils' own default, and that default is NOT the same in the two
    # environment forks: .venv-vl resolves MAX_PIXELS = 16384*28*28 (12.85 MP),
    # while .venv-modern (qwen3) resolves it as IMAGE_MAX_TOKEN_NUM*factor**2.
    # So the hint is explicit now (base.IMAGE_MAX_PIXELS = 19.27 MP, above the
    # largest image in any of the three layers) instead of inherited.
    #
    # The old `qwen25_vl_7b_instruct: {"max_pixels": 768*28*28}` entry was DEAD
    # CONFIG: `get_model` constructs legacy wrappers with **kwargs only and never
    # applies MODEL_OPTIONS, so that value never reached Qwen25VL. The 0.59 MP cap
    # that broke the PadChest layer lived in qwen25vl.py itself. Removed from here
    # so the registry cannot be mis-read as the thing that controls it.
    "qwen3_vl_8b_instruct": {"attn_implementation": "sdpa", **image_pixel_hints()},
    "qwen3_vl_8b_thinking": {"attn_implementation": "sdpa", **image_pixel_hints()},
    "qwen3_vl_30b_a3b_awq4bit": {"attn_implementation": "sdpa", **image_pixel_hints()},
    "internvl3_5_8b": {"model_cls_name": "AutoModel"},
    "llava_med_v1_5_7b": {},
}


def resolve_path(name: str, root: str | Path) -> Path:
    """Absolute on-disk path for a registry name."""
    if name not in MODEL_PATHS:
        raise KeyError(f"Unknown model {name!r}. Known: {sorted(MODEL_PATHS)}")
    return Path(root) / MODEL_PATHS[name]


def describe(name: str) -> dict[str, Any]:
    spec = dict(MODEL_SPECS[name])
    spec["name"] = name
    spec["path"] = MODEL_PATHS[name]
    spec.update(MODEL_OPTIONS.get(name, {}))
    return spec


def get_model(
    name: str,
    model_path: str | Path | None = None,
    root: str | Path | None = None,
    **kwargs: Any,
):
    """Instantiate a model wrapper.

    ``model_path`` may be given explicitly; otherwise it is resolved from
    ``root`` (defaults to the program directory) via ``MODEL_PATHS``.
    """
    if name not in MODEL_SPECS:
        raise KeyError(f"Unknown model: {name}. Available: {MODEL_NAMES}")
    spec = MODEL_SPECS[name]

    if model_path is None:
        if root is None:
            root = Path(__file__).resolve().parents[3]   # program/
        model_path = resolve_path(name, root)

    if spec["wrapper"] == "legacy":
        if name == "chexagent_8b":
            from .chexagent import CheXagent
            return CheXagent(model_path=model_path, **kwargs)
        if name == "medgemma_4b_it":
            from .medgemma import MedGemma
            return MedGemma(model_path=model_path, **kwargs)
        if name == "qwen25_vl_7b_instruct":
            # ONLY the 2.5 wrapper. Qwen3-VL has a different architecture and must
            # go through GenericVLM: routing it here loads the wrong auto class and
            # fails with "model type qwen3_vl not recognised".
            from .qwen25vl import Qwen25VL
            return Qwen25VL(model_path=model_path, **kwargs)
        raise KeyError(f"legacy wrapper not implemented for {name}")

    if spec["wrapper"] == "chexagent2":
        from .chexagent2 import CheXagent2
        return CheXagent2(model_path=model_path, **kwargs)

    if spec["wrapper"] == "internvl":
        from .internvl import InternVL
        return InternVL(model_path=model_path, **kwargs)

    if spec["wrapper"] == "llava":
        from .llava import LlavaMed
        return LlavaMed(model_path=model_path, **kwargs)

    from .generic import GenericVLM
    opts = dict(MODEL_OPTIONS.get(name, {}))
    opts.update(kwargs)
    # GenericVLM takes no model_cls_name; drop keys it does not understand so a
    # stale option in MODEL_OPTIONS cannot break a load.
    for k in ("model_cls_name",):
        opts.pop(k, None)
    return GenericVLM(model_path=model_path, name=name, **opts)
