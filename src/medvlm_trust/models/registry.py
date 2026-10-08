"""Model registry for local and API VLM wrappers."""

from __future__ import annotations

from .dummy import DummyVLM


def get_model(name: str, **kwargs):
    if name == "dummy":
        return DummyVLM(name="dummy", **kwargs)

    if name in {"chexagent", "chexagent_8b"}:
        raise NotImplementedError(
            "CheXagent wrapper is not implemented yet. "
            "Install transformers/model weights and implement a local wrapper before running."
        )
    if name in {"qwen25_vl", "qwen25_vl_7b"}:
        raise NotImplementedError(
            "Qwen2.5-VL wrapper is not implemented yet. "
            "Install transformers/vLLM and provide a local checkpoint path."
        )
    if name in {"gpt4o", "gpt_4o_2024_11_20"}:
        raise NotImplementedError(
            "GPT-4o API wrapper is not implemented yet. "
            "Provide an approved API client and set credentials outside git."
        )
    raise ValueError(f"unknown model: {name}")
