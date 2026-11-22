"""Local MedGemma-4B-IT wrapper."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from .base import BaseVLM, ensure_left_padding


def _load_model_class():
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText
    except Exception:
        from transformers import Gemma3ForConditionalGeneration
        return Gemma3ForConditionalGeneration


class MedGemma(BaseVLM):
    name = "medgemma_4b_it"

    def load(self) -> None:
        model_cls = _load_model_class()
        quant = self._quant_config()
        kwargs = dict(
            device_map=self._device_map_for(quant),
            torch_dtype=torch.bfloat16,
        )
        if self.max_memory:
            kwargs["max_memory"] = self.max_memory
        if quant is not None:
            kwargs["quantization_config"] = quant
        self.model = model_cls.from_pretrained(self.model_path, **kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.padding_side = ensure_left_padding(self.processor)

    def generate(self, image_path: str | Path, prompt: str, max_new_tokens: int = 512) -> str:
        return self.generate_batch([image_path], [prompt], max_new_tokens=max_new_tokens)[0]

    def generate_batch(self, images: Sequence, prompts: Sequence[str], max_new_tokens: int = 512, chunk: Optional[int] = None) -> list[str]:
        batch_messages = []
        for img, prompt in zip(images, prompts):
            image = img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB")
            batch_messages.append([
                {"role": "system", "content": [{"type": "text", "text": "You are a careful chest X-ray interpretation assistant."}]},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]},
            ])
            
        inputs = self.processor.apply_chat_template(
            batch_messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generation = generation[:, input_len:]
        return self.processor.batch_decode(generation, skip_special_tokens=True)
