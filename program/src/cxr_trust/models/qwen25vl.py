"""Local Qwen2.5-VL wrapper."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .base import BaseVLM, ensure_left_padding, image_pixel_hints, release_fragmented_cache


class Qwen25VL(BaseVLM):
    name = "qwen25_vl_7b_instruct"

    def load(self) -> None:
        quant = self._quant_config()
        kwargs = dict(
            device_map=self._device_map_for(quant),
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        if self.max_memory:
            kwargs["max_memory"] = self.max_memory
        if quant is not None:
            kwargs["quantization_config"] = quant
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_path, **kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.padding_side = ensure_left_padding(self.processor)

    def generate(self, image_path: str | Path, prompt: str, max_new_tokens: int = 512) -> str:
        return self.generate_batch([image_path], [prompt], max_new_tokens=max_new_tokens)[0]

    def generate_batch(self, images: Sequence, prompts: Sequence[str], max_new_tokens: int = 512, chunk: Optional[int] = None) -> list[str]:
        from qwen_vl_utils import process_vision_info

        batch_messages = []
        for img, prompt in zip(images, prompts):
            image = img.convert("RGB") if isinstance(img, Image.Image) else Image.open(img).convert("RGB")
            batch_messages.append([{"role": "user", "content": [
                # Shared preprocessing contract (base.IMAGE_MAX_PIXELS = 19.27 MP),
                # NOT the old hard-coded 768*28*28 = 0.59 MP legacy cap. That cap
                # downscaled PadChest by ~10x in area (7.00 MP -> 0.59 MP, ~9% of
                # the pixels kept) and, because the NIH `resolution` cells are
                # built by downsampling then upsampling back to 1024x1024, it also
                # silently re-degraded every resolution cell. See base.py.
                {"type": "image", "image": image, **image_pixel_hints()},
                {"type": "text", "text": prompt},
            ]}])
            
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = self.processor(
            text=texts, images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.05,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
        texts = self.processor.batch_decode(trimmed, skip_special_tokens=True)
        # Native-resolution PadChest cells reserve up to 25--33 GiB and the
        # allocator never gives it back; the next cell then spills to host RAM
        # without raising OOM (see base.release_fragmented_cache).
        release_fragmented_cache()
        return texts
