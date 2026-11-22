"""Generic chat-template VLM wrapper -- one code path for every modern model.

Why a generic wrapper
---------------------
Each new model previously needed its own hand-written wrapper, which is why
CheXagent ended up stranded in a second virtualenv. Modern multimodal checkpoints
(CheXagent-2, MedGemma 1.5, Qwen3-VL, InternVL3.5, LLaVA-Med, ...) are all
loadable through the same two entry points:

    AutoProcessor.from_pretrained(...)
    AutoModelForImageTextToText.from_pretrained(...)

so one wrapper covers them, and the differences that genuinely matter
(image size caps, batching quirks) become constructor options instead of new
files.

`BaseVLM` is left untouched: the already-published 4-bit results must stay
reproducible with exactly the code that produced them.

Batch generation
----------------
``generate_batch`` is the throughput win. The old single-image path reloads and
re-encodes one image per call; here a whole chunk of (image, prompt) pairs goes
through the processor and one ``model.generate`` call, so the shift benchmark
stops decoding the same base image 19 separate times.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor

from .base import BaseVLM, ensure_left_padding, release_fragmented_cache

DEFAULT_SYSTEM = "You are a careful chest X-ray interpretation assistant."


# Auto classes to try, in order, when no explicit class is requested.
# ``AutoModel`` is last but essential: checkpoints whose ``auto_map`` declares a
# custom ``AutoModel``/``AutoModelForCausalLM`` entry (InternVL's
# ``internvl_chat`` is one) are not loadable through the image-text classes, and
# the resulting error ("Unrecognized configuration class ... for this kind of
# AutoModel") does not say which class would have worked.
_AUTO_CLASS_CANDIDATES = (
    "AutoModelForImageTextToText",
    "AutoModelForVision2Seq",
    "AutoModelForCausalLM",
    "AutoModel",
)


def _candidate_classes(model_cls_name: Optional[str] = None):
    """Ordered list of model classes to attempt.

    An explicit ``model_cls_name`` wins outright; otherwise the auto classes are
    tried in a fixed order so a failure is a clear "none of these worked" rather
    than depending on which one happened to be first.
    """
    import transformers

    if model_cls_name:
        if not hasattr(transformers, model_cls_name):
            raise AttributeError(
                f"transformers {transformers.__version__} has no {model_cls_name}")
        return [getattr(transformers, model_cls_name)]

    out = []
    for name in _AUTO_CLASS_CANDIDATES:
        cls = getattr(transformers, name, None)
        if cls is not None:
            out.append(cls)
    if not out:
        raise ImportError("no usable AutoModel class found in transformers")
    return out


def _load_model_class(model_cls_name: Optional[str] = None):
    """Backwards-compatible single-class accessor (first candidate)."""
    return _candidate_classes(model_cls_name)[0]


class GenericVLM(BaseVLM):
    """Chat-template VLM wrapper with true batch generation.

    Parameters
    ----------
    name : str
        Registry name; recorded in progress files and provenance.
    min_pixels / max_pixels : int, optional
        Passed to the processor as a ``{"type": "image"}`` hint. Only Qwen-style
        processors honour them, and only when the image arrives as a PIL object
        (not a path). Leave as ``None`` to disable, which is the right default on
        a 32GB card: the old ``max_pixels=768*28*28`` cap was an 8GB workaround
        and would truncate high-resolution shift conditions.
    image_token_patch : str
        Some processors require the image placeholder in the text content.
    """

    def __init__(
        self,
        model_path: str | Path,
        name: str = "generic_vlm",
        device_map: str = "auto",
        load_in_4bit: bool = False,
        max_memory: Optional[dict] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        system_prompt: Optional[str] = DEFAULT_SYSTEM,
        trust_remote_code: bool = True,
        model_cls_name: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(model_path, device_map=device_map, load_in_4bit=load_in_4bit,
                         max_memory=max_memory)
        self.name = name
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.system_prompt = system_prompt
        self.model_cls_name = model_cls_name
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.torch_dtype = torch_dtype or torch.bfloat16
        # filled during load()
        self.supports_batching = True
        self.load_error = ""

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    def load(self) -> None:
        quant = self._quant_config()
        base_kwargs = dict(
            device_map=self._device_map_for(quant),
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
        )
        if self.max_memory:
            base_kwargs["max_memory"] = self.max_memory
        if quant is not None:
            base_kwargs["quantization_config"] = quant

        errors: list[str] = []
        self.model = None
        # Where accelerate may spill weights if no budget fits. Without this,
        # a model that needs offloading fails at generation with
        # "Must provide an `offload_dir` to perform disk offloading".
        #
        # The offload cache tries to create symlinks, which on Windows fails with
        # "WinError 1314: A required privilege is not held by the client" unless
        # Developer Mode or admin rights are available. Setting
        # HF_HUB_DISABLE_SYMLINKS_WARNING is not enough — accelerate is told
        # explicitly via SAFETENSORS_USE_SYMLINKS=0 so it copies instead.
        offload_dir = Path(self.model_path) / ".offload"
        offload_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("SAFETENSORS_USE_SYMLINKS", "0")
        for cls in _candidate_classes(self.model_cls_name):
            kwargs = dict(base_kwargs)
            kwargs["offload_folder"] = str(offload_dir)
            if self.attn_implementation:
                kwargs["attn_implementation"] = self.attn_implementation
            attempts = [kwargs,
                        {k: v for k, v in kwargs.items() if k != "attn_implementation"}]
            for attempt_kwargs in attempts:
                try:
                    self.model = cls.from_pretrained(self.model_path, **attempt_kwargs)
                    self.loaded_via = cls.__name__
                    break
                except Exception as exc:            # noqa: BLE001
                    errors.append(f"{cls.__name__}: {type(exc).__name__}: {str(exc)[:160]}")
            if self.model is not None:
                break
        if self.model is None:
            raise RuntimeError(
                "no auto class could load this checkpoint; tried "
                + " | ".join(errors))

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=self.trust_remote_code)
        self.padding_side = ensure_left_padding(self.processor)
        self.model.eval()

    # ------------------------------------------------------------------ #
    # prompt / input construction
    # ------------------------------------------------------------------ #
    def _image_content(self, image: Image.Image) -> dict:
        item: dict = {"type": "image", "image": image}
        if self.min_pixels:
            item["min_pixels"] = self.min_pixels
        if self.max_pixels:
            item["max_pixels"] = self.max_pixels
        return item

    def _messages(self, image: Image.Image, prompt: str):
        msgs = []
        if self.system_prompt:
            msgs.append({"role": "system",
                         "content": [{"type": "text", "text": self.system_prompt}]})
        msgs.append({"role": "user",
                     "content": [self._image_content(image), {"type": "text", "text": prompt}]})
        return msgs

    @staticmethod
    def _open(image_path) -> Image.Image:
        if isinstance(image_path, Image.Image):
            return image_path.convert("RGB")
        return Image.open(image_path).convert("RGB")

    def _prepare_batch(self, images: Sequence, prompts: Sequence[str]):
        """Tokenise a batch; returns (inputs, batch_size).

        Multiple chat conversations go in as a *list* of conversations, which is
        what every multimodal processor accepts and what yields a genuinely
        padded batch rather than a list of single-item batches.
        """
        convs = [self._messages(self._open(img), p) for img, p in zip(images, prompts)]
        try:
            inputs = self.processor.apply_chat_template(
                convs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True,
            )
        except TypeError:
            # older processor signature without padding
            inputs = self.processor.apply_chat_template(
                convs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
        return inputs

    @staticmethod
    def _input_len(inputs) -> int:
        ids = inputs["input_ids"]
        if ids.dim() == 1:
            return ids.shape[-1]
        # padded batches: the prefill length is the padded width
        return ids.shape[-1]

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #
    def generate(self, image_path, prompt: str, max_new_tokens: int = 512) -> str:
        return self.generate_batch([image_path], [prompt], max_new_tokens=max_new_tokens)[0]

    def generate_batch(self, images: Sequence, prompts: Sequence[str],
                       max_new_tokens: int = 512, chunk: Optional[int] = None) -> list[str]:
        """Generate for several (image, prompt) pairs under one model call per chunk."""
        if len(images) != len(prompts):
            raise ValueError(f"images/prompts length mismatch: {len(images)} vs {len(prompts)}")
        if not images:
            return []

        step = chunk or len(images)
        device = getattr(self.model, "device", None)
        out_texts: list[str] = []
        for start in range(0, len(images), step):
            img_chunk = list(images[start:start + step])
            pr_chunk = list(prompts[start:start + step])
            try:
                inputs = self._prepare_batch(img_chunk, pr_chunk)
            except Exception:
                # Batch tokenisation failed -> fall back to one-by-one so a single
                # awkward processor cannot take down an overnight run.
                self.supports_batching = False
                out_texts.extend(self._generate_one_by_one(img_chunk, pr_chunk, max_new_tokens))
                continue

            if device is None:
                try:
                    device = next(self.model.parameters()).device
                except StopIteration:
                    device = None
            if device is not None:
                inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            padded_len = self._input_len(inputs)
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            gen = generation[:, padded_len:]
            texts = self.processor.batch_decode(gen, skip_special_tokens=True)
            out_texts.extend(texts)
            # Same native-resolution VRAM cliff as the Qwen2.5 wrapper: without
            # this, a 12.85 MP image leaves 28 GiB reserved and the next cell
            # spills to host RAM at ~20x the cost with no OOM (docs/55 section 10).
            release_fragmented_cache()
        return out_texts

    def _generate_one_by_one(self, images, prompts, max_new_tokens: int) -> list[str]:
        texts = []
        for img, pr in zip(images, prompts):
            inputs = self._prepare_batch([img], [pr])
            device = getattr(self.model, "device", None)
            if device is None:
                device = next(self.model.parameters()).device
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            n = self._input_len(inputs)
            with torch.inference_mode():
                generation = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            texts.append(self.processor.batch_decode(
                generation[:, n:], skip_special_tokens=True)[0])
        return texts
