"""CheXagent-2-3b wrapper (custom remote code + external vision encoder).

Why this needs its own wrapper
------------------------------
CheXagent-2-3b is not a chat-template model:

* it has **no processor file**. ``AutoProcessor`` cannot be instantiated, so the
  generic ``GenericVLM`` path fails outright.
* its ``config.json`` declares
  ``auto_map: {AutoModelForCausalLM: modeling_chexagent.CheXagentForCausalLM}``.
  Despite the ``ForCausalLM`` name this **is** the full VLM — the class subclasses
  ``PhiPreTrainedModel`` and holds the vision tower as ``self.visual``. Loading it
  through ``AutoModelForImageTextToText`` fails, and loading it through
  ``AutoModelForCausalLM`` is the intended route, not a workaround that drops
  vision.
* its tokenizer encodes the image as a **path string** between ``<|img|>`` and
  ``<|/img|>``; ``forward`` decodes those token ids back to paths and calls
  ``visual.encode(paths)``. Images are therefore read from disk *inside* the
  forward pass, which is why an in-memory PIL image is not directly supported and
  why batching needs care.

The vision encoder is a separate checkpoint
-------------------------------------------
``config.visual.vision_model_name_or_path`` points at
``models/XraySigLIP__vit-l-16-siglip-384__webli``. That directory must exist
alongside the model, otherwise the visual tower cannot be built.

Batching
--------
``visual.encode`` genuinely batches (it stacks N images and runs one forward).
What is unverified is whether the *language-side* replacement of image tokens is
positionally correct when several examples are padded into one batch, because
that code decodes ``input_ids`` to recover paths and indexes by example. So:

* ``generate`` is serial by default — correct and simple;
* ``generate_batch`` batches the vision tower while keeping the language step
  per example unless ``--trust-batch`` behaviour is explicitly requested.

Set ``CHEXAGENT2_BATCH=1`` in the environment to enable the experimental true
batch path, and verify equivalence against the serial path before trusting its
numbers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .base import BaseVLM


class CheXagent2(BaseVLM):
    name = "chexagent_2_3b"

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        try:
            self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        except Exception:
            self.generation_config = GenerationConfig()

        quant = self._quant_config()
        kwargs = dict(
            trust_remote_code=True,
            device_map=self._device_map_for(quant),
            torch_dtype=torch.bfloat16,
        )
        if self.max_memory:
            kwargs["max_memory"] = self.max_memory
        if quant is not None:
            kwargs["quantization_config"] = quant
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        self.model.eval()
        self.true_batch = os.environ.get("CHEXAGENT2_BATCH", "0") == "1"
        self.supports_batching = True

    # ------------------------------------------------------------------ #
    def _build_inputs(self, image_paths: Sequence[Path]):
        """One tokenised example per image, image referenced by path."""
        texts = []
        for p in image_paths:
            texts.append(f" USER: <s><|img|>{str(p)}<|/img|> ASSISTANT: <s>")
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        return {k: v.to(self.model.device) for k, v in enc.items() if torch.is_tensor(v)}

    @staticmethod
    def _as_path(image) -> Path:
        if isinstance(image, (str, Path)):
            return Path(image)
        if isinstance(image, Image.Image):
            raise TypeError(
                "CheXagent-2 reads images from disk inside forward(); pass a path, "
                "not a PIL image")
        return Path(str(image))

    def generate(self, image_path, prompt: str, max_new_tokens: int = 512) -> str:
        path = self._as_path(image_path)
        text = f" USER: <s><|img|>{str(path)}<|/img|>{prompt} ASSISTANT: <s>"
        enc = self.tokenizer([text], return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in enc.items() if torch.is_tensor(v)}
        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                max_new_tokens=max_new_tokens,
            )
        return self.tokenizer.decode(gen[0], skip_special_tokens=True)

    def generate_batch(self, images: Sequence, prompts: Sequence[str],
                       max_new_tokens: int = 512,
                       chunk: Optional[int] = None) -> list[str]:
        if len(images) != len(prompts):
            raise ValueError(f"images/prompts length mismatch: {len(images)} vs {len(prompts)}")
        if not images:
            return []

        # Serial is the verified path. The batched path is opt-in because the
        # prompt at the head of this file explains why padding may misalign the
        # per-example image-token replacement.
        if not self.true_batch:
            self.supports_batching = False
            return [self.generate(i, p, max_new_tokens)
                    for i, p in zip(images, prompts)]

        texts = [f" USER: <s><|img|>{str(self._as_path(i))}<|/img|>{p} ASSISTANT: <s>"
                 for i, p in zip(images, prompts)]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                             add_special_tokens=False)
        inputs = {k: v.to(self.model.device) for k, v in enc.items() if torch.is_tensor(v)}
        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                max_new_tokens=max_new_tokens,
            )
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)
