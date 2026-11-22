"""LLaVA-Med v1.5 (Mistral-7B) wrapper.

Why a dedicated wrapper
-----------------------
LLaVA-1.5 predates HF chat templates (its processor has no ``chat_template``), so
the generic wrapper fails with *"No chat template is set for this processor"*.
Its prompt format is the LLaVA-1.5 convention::

    USER: <image>\n{question} ASSISTANT:

where ``<image>`` is replaced by 576 copies of the image-context token. 576 is
``(336 / 14) ** 2`` — the CLIP ViT-L/14-336 patch grid — and ``mm_patch_merge_type:
flat`` in this checkpoint means no 2x2 pooling, so the count is not halved.

Config provenance
-----------------
This checkpoint shipped a flat ``llava_mistral`` config that no current
transformers recognises. ``scripts/29_fix_llava_med_config.py`` rewrote it into
the nested ``llava`` layout (keeping ``config.json.orig``); this wrapper assumes
that has been done.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from .base import BaseVLM

IMG_TOKEN = "<image>"
PAD_TOKEN = "<pad>"                 # appended after Mistral's 32000 tokens
DEFAULT_NUM_IMAGE_TOKENS = 576      # (336/14)**2, mm_patch_merge_type=flat


class LlavaMed(BaseVLM):
    name = "llava_med_v1_5_7b"

    # Padding uses Mistral's eos row (id 2). It must be a real embedding row:
    # an out-of-range padding_idx trips nn.Embedding's
    # "Padding_idx must be within num_embeddings" assertion at load time.
    PAD_ID = 2

    def load(self) -> None:
        # ---- 1. processor / tokenizer ------------------------------------
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor

        # LlavaProcessor computes the image-token count as
        # (h // patch_size) * (w // patch_size) and patch_size arrives as None for
        # this checkpoint, so any processor call dies with
        # "TypeError: unsupported operand type(s) for //: 'int' and 'NoneType'".
        if getattr(self.processor, "patch_size", None) is None:
            self.processor.patch_size = 14

        # ---- 2. the model ------------------------------------------------
        quant = self._quant_config()
        kwargs = dict(device_map=self._device_map_for(quant),
                      torch_dtype=torch.bfloat16,
                      low_cpu_mem_usage=True)
        if self.max_memory:
            kwargs["max_memory"] = self.max_memory
        if quant is not None:
            kwargs["quantization_config"] = quant
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path, **kwargs).eval()

        # ---- 3. image placeholder token, AFTER the model exists ----------
        # HF's LlavaForConditionalGeneration scans input_ids for
        # config.image_token_index and overwrites those positions with projected
        # vision features. This checkpoint's tokenizer has no <image> token, so the
        # embedding table has no row for that id either; injecting one anyway gives
        # "srcIndex < srcSelectDimSize" from the embedding's index_select.
        # The id is therefore appended and the token embedding resized to match —
        # which is exactly why LLaVA-1.5 shipped vocab 32064 with <image> at 32000.
        self.tokenizer.add_special_tokens({"additional_special_tokens": [IMG_TOKEN]})
        self.image_token_id = int(self.tokenizer.convert_tokens_to_ids(IMG_TOKEN))
        need = self.image_token_id + 1
        cur = int(self.model.get_input_embeddings().weight.shape[0])
        if need > cur:
            self.model.resize_token_embeddings(need)
        self.vocab_size = int(self.model.get_input_embeddings().weight.shape[0])

        self.pad_token_id = self.PAD_ID
        self.tokenizer.pad_token_id = self.PAD_ID

        # keep the config consistent with the resized table
        for attr, val in (("vocab_size", self.vocab_size),
                          ("image_token_index", self.image_token_id),
                          ("pad_token_id", self.pad_token_id)):
            try:
                setattr(self.model.config, attr, val)
            except Exception:
                pass
            try:
                setattr(self.model.config.text_config, attr, val)
            except Exception:
                pass

        # Prefer the config's declared sequence length; it is authoritative for
        # how many vision features the projector will emit.
        for attr in ("image_seq_length", "num_image_tokens"):
            v = getattr(self.model.config, attr, None)
            if isinstance(v, int) and v > 0:
                self.num_image_tokens = v
                break
        else:
            self.num_image_tokens = DEFAULT_NUM_IMAGE_TOKENS

        # A dedicated pad token (32001 in the converted checkpoint). Falling back
        # to EOS is acceptable for generation but the checkpoint defines its own.
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)

    def model_config_or_none(self):
        try:
            import json
            return type("C", (), json.loads(
                (Path(self.model_path) / "config.json").read_text(encoding="utf-8")))()
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _open(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def _build_text(self, prompt: str) -> str:
        """LLaVA-1.5 prompt with a SINGLE ``<image>`` placeholder.

        The placeholder must appear exactly once: ``LlavaProcessor.__call__``
        expands it into ``num_image_tokens`` (576) copies itself, taking the count
        from ``(h // patch_size) * (w // patch_size)``. Pre-expanding it here
        instead multiplies the two (576 x 576 = 331,776 tokens) and the sequence
        blows past the model's maximum length.
        """
        return f"USER: {IMG_TOKEN}\n{prompt} ASSISTANT:"

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
        # Serial: left-padded batches of 576-token image blocks are possible, but
        # a padding mistake here silently misaligns image features, so the
        # verified path is one image per call.
        self.supports_batching = False
        out: list[str] = []
        for img, pr in zip(images, prompts):
            out.append(self._generate_one(img, pr, max_new_tokens))
        return out

    def _generate_one(self, image, prompt: str, max_new_tokens: int) -> str:
        """Tokenise text and encode the image separately, then generate.

        ``LlavaProcessor.__call__`` is deliberately bypassed. It expands the single
        ``<image>`` marker into ``image_seq_length`` copies, but that expansion
        depends on processor state that is fragile for this particular checkpoint
        (``patch_size`` arrives as ``None`` unless ``image_seq_length`` is present
        in ``config.json``). Writing the 576 copies here makes the count explicit
        and independent of the processor, while keeping exactly the same token
        layout the model expects.

        Verified against the reference config of
        ``chaoyinshe/llava-med-v1.5-mistral-7b-hf``: ``image_seq_length: 576``,
        ``vision_feature_layer: -2``, ``vision_feature_select_strategy: default``.
        """
        pil = self._open(image)

        # 1) text. The placeholder is injected by id rather than by tokenising a
        #    literal "<image>": this checkpoint's tokenizer has no such token, so
        #    convert_tokens_to_ids("<image>") returns <unk> and the tokenised text
        #    would contain 0 image positions (measured: "text has 0 image tokens,
        #    model expects 576"). Building the id sequence directly removes the
        #    dependency on the tokenizer's vocabulary entirely.
        n = self.num_image_tokens
        # NO leading BOS, and that is deliberate.
        #
        # The reference LLaVA pipeline (`tokenizer_image_token`) tokenises the first
        # chunk with the tokenizer's defaults, and this tokenizer has
        # `add_bos_token=True`, so the reference prompt is
        # `[<s>] + "USER: " + <image>*576 + "\n{question} ASSISTANT:"`. Adding that
        # `<s>` to THIS converted checkpoint makes the model emit a space and then
        # EOS -- measured on the same image, same prompt:
        #
        #   without BOS : "The chest X-ray findings include atelectasis, ..."
        #   with BOS    : "" (first two generated ids [28705, 2] = space + </s>)
        #
        # and letting LlavaProcessor build the prompt end to end (which also adds
        # BOS) reproduces the empty reply exactly. So the HF conversion of these
        # weights is not the same sequence as the original LLaVA pipeline; the
        # conversion wins, because it is what the weights here were exported for.
        #
        # This is a real trap, not a curiosity: the empty reply is 0.22 s/cell and
        # looks like "the model refused", and every prompt-format theory built on
        # that reading was wrong. The actual pre-run failure was `rope_theta`
        # (scripts/29_fix_llava_med_config.py).
        prefix = self.tokenizer("USER: ", add_special_tokens=False)["input_ids"]
        suffix = self.tokenizer(f"\n{prompt} ASSISTANT:", add_special_tokens=False)["input_ids"]
        ids = prefix + [self.image_token_id] * n + suffix
        input_ids = torch.tensor([ids], dtype=torch.long,
                                 device=self.model.device)
        attention_mask = torch.ones_like(input_ids)

        # 2) image: CLIP preprocessing only
        px = self.image_processor(images=pil, return_tensors="pt")["pixel_values"]

        moved = {"input_ids": input_ids, "attention_mask": attention_mask,
                 "pixel_values": px.to(self.model.device, dtype=torch.bfloat16)}

        prompt_len = input_ids.shape[-1]
        with torch.inference_mode():
            gen = self.model.generate(
                **moved,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.pad_token_id or self.tokenizer.eos_token_id,
            )
        # decode only the continuation; decoding from 0 re-emits the prompt
        return self.tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
