"""Local CheXagent-8b wrapper (custom remote code, batched)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from .base import BaseVLM, ensure_left_padding


class CheXagent(BaseVLM):
    name = "chexagent_8b"

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.padding_side = ensure_left_padding(self.processor)
        self.generation_config = GenerationConfig.from_pretrained(self.model_path)

        # The shipped generation_config.json says num_beams=5 and generate() below
        # passes it straight through, so this model ran BEAM SEARCH while every
        # other model in the study runs greedy. Measured cost on the 100-image
        # pre-run: 52/400 reliance cells fail, every one of them an infinite digit
        # loop (`999...`/`000...`) that never reaches the JSON contract -- and
        # raising max_new_tokens 256 -> 1024 moves the parse rate by 0.000
        # (scripts/67_measure_chexagent_parse_budget.py). Greedy removes all of it
        # (96/96 parse, ~6x faster) and makes the study's greedy claim true.
        self.generation_config.num_beams = 1
        self.generation_config.do_sample = False

        # CheXagent's tokenizer ships WITHOUT a pad token, so any call with
        # padding=True raises:
        #   "Asking to pad but the tokenizer does not have a padding token"
        # EOS is the conventional substitute and is safe here because the
        # processor returns an attention_mask that masks the padding out.
        # This must happen before the first batched call, not lazily at call time.
        tok = getattr(self.processor, "tokenizer", None)
        if tok is not None and tok.pad_token is None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        if getattr(self.processor, "pad_token", None) is None and tok is not None:
            try:
                self.processor.pad_token = tok.eos_token
            except Exception:
                pass

        quant = self._quant_config()
        kwargs = dict(trust_remote_code=True, torch_dtype=torch.float16)
        # NO device_map by default, and that is the fix -- see the long note below.
        # A caller who supplies max_memory (or a quantisation config) has stated an
        # explicit placement intent, so their device_map is honoured.
        if quant is not None or self.max_memory:
            kwargs["device_map"] = self._device_map_for(quant, pin_when_unquantized=True)
        if self.max_memory:
            kwargs["max_memory"] = self.max_memory
        if quant is not None:
            kwargs["quantization_config"] = quant
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        self.supports_batching = True

        # Why chexagent is loaded on the CPU and then moved, instead of onto the
        # card directly. Each step below was measured on 2026-10-06; the whole
        # chain matters, because each "obvious" fix fails for a different reason.
        #
        # 1. The shipped checkpoint is F32 (31.15 GB). `torch_dtype=torch.float16`
        #    is accepted into the config and then ignored for this remote-code
        #    class: measured 31.15 GiB of float32 parameters with the config
        #    reading `torch.float16`, with and without low_cpu_mem_usage
        #    (scripts/_diag_chexagent_dtype.py).
        # 2. 31.15 GiB of parameters do not fit the ~30.6 GiB that is free on a
        #    32 GiB card, so `device_map={"": 0}` (option A) fills the card and the
        #    driver starts paging VRAM to host RAM: 100% "utilisation", 82 W, one
        #    shard taking 2 minutes.
        # 3. Converting the checkpoint to fp16 on disk (option C,
        #    scripts/62_convert_chexagent_fp16.py) is necessary but not
        #    sufficient: `_load_pretrained_model` calls
        #    `caching_allocator_warmup(..., factor=2)` whenever a device_map is
        #    present, and that pre-allocates ~half of FREE VRAM in one block before
        #    the weights arrive. With a device_map the load therefore still pages,
        #    even though the fp16 weights themselves (15.58 GiB) fit comfortably.
        # 4. With no device_map there is no warmup and no estimation: the weights
        #    are built in host memory (63 GB here) and moved to the card in one
        #    transfer that peaks at the model size and nothing more.
        #
        # `_cast_and_move_to_gpu` then VERIFIES the footprint and raises if the
        # model did not end up where it was supposed to -- because the failure mode
        # of this model is not an exception, it is 16 minutes of paging followed by
        # a driver reset.
        if quant is None and "device_map" not in kwargs:
            self._cast_and_move_to_gpu()

    def _cast_and_move_to_gpu(self, dtype: torch.dtype = torch.float16) -> None:
        """Move the CPU-loaded model to the single GPU and prove it arrived.

        fp16 is not a compromise here: it is what the wrapper always claimed to
        load (`torch_dtype=torch.float16`) and what fits with ~15 GiB to spare. The
        guard exists so that a future mis-load raises immediately instead of
        paging; it is not a precision decision.
        """
        if not torch.cuda.is_available():
            print("[chexagent] CUDA unavailable; leaving the model on the CPU", flush=True)
            return
        current = next(self.model.parameters()).dtype
        if current != dtype:
            self.model = self.model.to(dtype=dtype)
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated() / 2**30
        self.model = self.model.to("cuda")
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() / 2**30
        print(f"[chexagent] dtype {current} -> "
              f"{next(self.model.parameters()).dtype}; GPU resident "
              f"{before:.1f} -> {after:.1f} GiB (fp16 expectation ~15.6)", flush=True)
        if after > 20.0:
            raise RuntimeError(
                f"chexagent occupies {after:.1f} GiB of GPU memory after the "
                f"{dtype} cast; a forward pass in this state pages VRAM through "
                f"the Windows driver instead of computing. Check that the registry "
                f"points at models/CheXagent-8b-fp16 (scripts/62_convert_"
                f"chexagent_fp16.py) -- and do NOT lower precision to work around "
                f"this.")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _open(image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    @staticmethod
    def _text(prompt: str) -> str:
        return f" USER: <s>{prompt} ASSISTANT: <s>"

    def _prepare(self, images, texts):
        """Build model inputs with the 5-D ``pixel_values`` CheXagent requires.

        CheXagent's ``forward`` indexes ``pixel_values.sum(dim=(2, 3, 4))`` and then
        ``pixel_values[image_mask]``, so it expects
        ``(batch, n_images, channels, H, W)`` — one image slot per sample. That
        extra axis is deliberate (it is what lets one sample carry several images),
        not an accident, so a 4-D tensor is wrong and raises
        ``Dimension out of range ... got 4``.

        The upstream ``CheXagentProcessor.__call__`` also produces 5-D output, but
        wraps it as ``[torch.tensor(pv) for pv in ...]`` when already batched,
        which nests one axis too many and breaks any batch above 1. Building the
        layout here keeps the model's contract while making batching correct.
        """
        tok = self.processor.tokenizer
        ip = self.processor.image_processor

        enc = tok(list(texts), return_tensors="pt", padding=True,
                  add_special_tokens=True)
        img = ip(images=list(images), return_tensors="pt")["pixel_values"]
        if isinstance(img, (list, tuple)):
            img = torch.stack([torch.as_tensor(x) for x in img], dim=0)
        pv = torch.as_tensor(img).float()

        # normalise to (n, 3, H, W) ...
        while pv.dim() > 4 and pv.shape[0] == 1:
            pv = pv.squeeze(0)
        if pv.dim() == 3:
            pv = pv.unsqueeze(0)
        if pv.dim() != 4:
            raise RuntimeError(f"unexpected pixel_values shape {tuple(pv.shape)}")

        # ... then add the per-sample image-slot axis the model expects
        pv = pv.unsqueeze(1)

        n_text = enc["input_ids"].shape[0]
        if pv.shape[0] != n_text:
            raise RuntimeError(
                f"image/text batch mismatch: {pv.shape[0]} images vs {n_text} texts")
        inputs = dict(enc)
        inputs["pixel_values"] = pv
        return {k: v.to(self.model.device) for k, v in inputs.items() if torch.is_tensor(v)}

    def generate(self, image_path, prompt: str, max_new_tokens: int = 512) -> str:
        return self.generate_batch([image_path], [prompt], max_new_tokens=max_new_tokens)[0]

    def generate_batch(self, images: Sequence, prompts: Sequence[str],
                       max_new_tokens: int = 512,
                       chunk: Optional[int] = None) -> list[str]:
        if len(images) != len(prompts):
            raise ValueError(f"images/prompts length mismatch: {len(images)} vs {len(prompts)}")
        if not images:
            return []

        step = chunk or len(images)
        out: list[str] = []
        for start in range(0, len(images), step):
            imgs = [self._open(i) for i in images[start:start + step]]
            texts = [self._text(p) for p in prompts[start:start + step]]
            try:
                inputs = self._prepare(imgs, texts)
            except Exception:
                # A processor that cannot batch must not take down an overnight run.
                self.supports_batching = False
                out.extend(
                    self._generate_serial(i, p, max_new_tokens)
                    for i, p in zip(images[start:start + step], prompts[start:start + step])
                )
                continue
            with torch.inference_mode():
                gen = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    max_new_tokens=max_new_tokens,
                )
            out.extend(self.processor.tokenizer.batch_decode(gen, skip_special_tokens=True))
        return out

    def _generate_serial(self, image, prompt: str, max_new_tokens: int) -> str:
        inputs = self._prepare([self._open(image)], [self._text(prompt)])
        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                max_new_tokens=max_new_tokens,
            )
        return self.processor.tokenizer.decode(gen[0], skip_special_tokens=True)
