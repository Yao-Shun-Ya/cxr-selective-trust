"""InternVL3.5 wrapper.

Why a dedicated wrapper is needed
---------------------------------
InternVL is not a chat-template model. Its remote code defines a contract that
the generic wrapper cannot satisfy:

1. ``InternVLForConditionalGeneration.generate`` asserts
   ``self.img_context_token_id is not None``. That attribute is set *inside*
   ``chat()`` from the tokenizer's ``<IMG_CONTEXT>`` token, so calling
   ``model.generate(...)`` directly raises a bare ``AssertionError`` with no
   message. This is the failure mode that looks like "generation is broken".
2. The question must be prefixed with ``<image>``, and every ``<image>`` must be
   expanded into
   ``<img>`` + ``<IMG_CONTEXT> * num_image_token * num_patches`` + ``</img>``.
   If it is not expanded, the next assertion (``selected.sum() != 0``) fires.
3. Prompt scaffolding comes from the bundled ``conversation.py`` template
   (``internvl2_5``), not from ``apply_chat_template``.

Transformers version
--------------------
The remote code was written against transformers 4.3x. Under transformers 5.x it
fails with ``'InternVLChatModel' object has no attribute 'all_tied_weights_keys'``
(a removed attribute), so this model runs in ``.venv-vl`` (4.51.3), not
``.venv-modern``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from .base import BaseVLM, ensure_left_padding

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def _load_conversation_module(model_path: Path):
    """Import the checkpoint's own ``conversation.py``.

    It is shipped with the weights but is not installed, so it is loaded by path.
    Using the bundled module (rather than re-implementing the template) keeps the
    prompt format identical to the authors'.
    """
    conv_file = model_path / "conversation.py"
    if not conv_file.exists():
        raise FileNotFoundError(f"{conv_file} not found; cannot build the prompt")
    spec = importlib.util.spec_from_file_location("internvl_conversation", conv_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["internvl_conversation"] = mod
    spec.loader.exec_module(mod)
    return mod


class InternVL(BaseVLM):
    name = "internvl3_5_8b"

    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
        self.supports_batching = True

    def load(self) -> None:
        model_path = Path(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False)
        self.padding_side = ensure_left_padding(self.tokenizer)
        self.conversation = _load_conversation_module(model_path)

        # The repo declares InternVLProcessor but does not ship it, so the
        # normalisation statistics come straight from preprocessor_config.json.
        import json
        pp = json.loads((model_path / "preprocessor_config.json").read_text(encoding="utf-8"))
        self._norm = (torch.tensor(pp["image_mean"], dtype=torch.float32),
                      torch.tensor(pp["image_std"], dtype=torch.float32))
        self._rescale = float(pp.get("rescale_factor", 1.0 / 255.0))
        size = pp.get("size") or {}
        self._tile_size = int(size.get("height", 448))
        self._max_patches = int(pp.get("max_patches") or 12)

        quant = self._quant_config()
        kwargs = dict(trust_remote_code=True, torch_dtype=torch.bfloat16,
                      low_cpu_mem_usage=True,
                      device_map=self._device_map_for(quant))
        if self.max_memory:
            kwargs["max_memory"] = self.max_memory
        if quant is not None:
            kwargs["quantization_config"] = quant
        self.model = AutoModel.from_pretrained(model_path, **kwargs).eval()

        # Required by model.generate(); normally set inside chat().
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id

        # number of visual tokens per 448x448 tile, derived from the model itself
        self.num_image_token = int(getattr(self.model, "num_image_token", 0)) or None
        cfg = getattr(self.model, "config", None)
        if self.num_image_token is None and cfg is not None:
            for attr in ("num_image_token", "num_image_tokens"):
                v = getattr(cfg, attr, None)
                if isinstance(v, int) and v > 0:
                    self.num_image_token = v
                    break
        template_name = getattr(cfg, "template", "internvl2_5")
        self.template_name = template_name

    # ------------------------------------------------------------------ #
    def _build_query(self, prompt: str, num_patches: int) -> str:
        conv = self.conversation.get_conv_template(self.template_name)
        conv.system_message = getattr(self.model.config, "system_message", None) or ""
        conv.append_message(conv.roles[0], "<image>\n" + prompt)
        conv.append_message(conv.roles[1], None)
        query = conv.get_prompt()
        image_tokens = (IMG_START_TOKEN
                        + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                        + IMG_END_TOKEN)
        return query.replace("<image>", image_tokens, 1)

    def _tile(self, image: Image.Image, tile: int = 448, max_patches: int = 12):
        """Split one image into 448x448 tiles, InternVL style.

        The checkpoint declares ``"processor_class": "InternVLProcessor"`` in
        ``processor_config.json`` but **does not ship that class**, so
        ``AutoProcessor`` silently falls back to the tokenizer (verified: it
        returns ``Qwen2TokenizerFast``). Image preprocessing therefore has to be
        done here, otherwise ``processor(images=...)`` raises
        ``ValueError: You need to specify either `text` or `text_target``.

        Tiling rule: choose the aspect-ratio-preserving grid of tiles that needs
        the fewest patches (``1x1`` for square-ish images, elongated grids for
        wide or tall ones), capped by ``max_patches``.
        """
        w, h = image.size
        ratio = w / h

        best = (1, 1)
        for cols in range(1, max_patches + 1):
            for rows in range(1, max_patches + 1):
                if cols * rows > max_patches:
                    continue
                if abs((cols / rows) - ratio) <= 0.1:
                    if cols * rows < best[0] * best[1]:
                        best = (cols, rows)
        cols, rows = best

        target = (cols * tile, rows * tile)
        resized = image.resize(target, Image.BICUBIC)
        arr = torch.from_numpy(np.asarray(resized, dtype=np.float32)).permute(2, 0, 1)
        # normalise with the checkpoint's own statistics
        mean, std = self._norm
        arr = (arr * self._rescale - mean[:, None, None]) / std[:, None, None]

        # Split by explicit slicing. A single reshape/permute is easy to get
        # backwards (rows and columns swap silently and only the shape error
        # reveals it), so the split is written out.
        tiles = []
        for r in range(rows):
            for c in range(cols):
                tiles.append(arr[:, r * tile:(r + 1) * tile, c * tile:(c + 1) * tile])
        return torch.stack(tiles, dim=0).contiguous(), cols * rows

    @staticmethod
    def _as_pil(image):
        """Accept either a path or an already-decoded PIL image.

        The stage scripts disagree about what they pass: the batch sweep
        (``26_batch_sweep.py``) hands over paths, while the shift and reliance
        runners (``08``/``06``) decode the images themselves and hand over PIL
        objects. Assuming paths made ``Image.open(p)`` receive a PIL object and
        fail with ``'Image' object has no attribute 'read'`` -- which the callers
        swallow and retry one image at a time, so every single row of the
        reliance stage was written as an empty string.
        """
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        raise TypeError(f"unsupported image type: {type(image)!r}")

    def _prepare(self, paths):
        """Return tokenised inputs plus pixel_values for a list of images."""
        all_tiles = []
        per_image = []
        queries = []
        for p, prompt in zip(paths, self._prompts):
            image = self._as_pil(p)
            tiles, n = self._tile(image)
            all_tiles.append(tiles)
            per_image.append(n)
            queries.append(self._build_query(prompt, n))

        pixel_values = torch.cat(all_tiles, dim=0)
        enc = self.tokenizer(text=queries, return_tensors="pt", padding=True,
                             add_special_tokens=False)
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "pixel_values": pixel_values,
        }, per_image

    def _num_patches_per_image(self, pixel_values: torch.Tensor) -> list[int]:
        """How many 448x448 tiles each input image produced (kept for clarity)."""
        return [int(pixel_values.shape[0])]

    def generate(self, image_path, prompt: str, max_new_tokens: int = 512) -> str:
        return self.generate_batch([image_path], [prompt],
                                   max_new_tokens=max_new_tokens)[0]

    def generate_batch(self, images: Sequence, prompts: Sequence[str],
                       max_new_tokens: int = 512,
                       chunk: Optional[int] = None) -> list[str]:
        if len(images) != len(prompts):
            raise ValueError(f"images/prompts length mismatch: {len(images)} vs {len(prompts)}")
        if not images:
            return []
        # Serial: each image is tiled independently, so a batch would require
        # per-image grids and variable-length image-token runs. InternVL's own
        # chat() path is per-conversation too, so serial is the correct contract
        # rather than a limitation to work around.
        self.supports_batching = False
        out: list[str] = []
        for img, pr in zip(images, prompts):
            out.append(self._generate_one(img, pr, max_new_tokens))
        return out

    def _generate_one(self, image, prompt: str, max_new_tokens: int) -> str:
        self._prompts = [prompt]
        inputs, per_image = self._prepare([image])
        model_dtype = next(self.model.parameters()).dtype
        moved = {}
        for k, v in inputs.items():
            if not torch.is_tensor(v):
                continue
            v = v.to(self.model.device)
            # pixel_values arrive as float32 from the uint8 source; the vision
            # tower is bf16, so cast them or the first conv raises
            # "Input type (float) and bias type (c10::BFloat16) should be the same".
            if k == "pixel_values":
                v = v.to(model_dtype)
            moved[k] = v
        with torch.inference_mode():
            gen = self.model.generate(
                **moved,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(gen[0], skip_special_tokens=True)
