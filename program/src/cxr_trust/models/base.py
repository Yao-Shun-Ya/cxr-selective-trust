"""Base interface and shared loading helpers for local open-source VLMs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import torch


class BaseVLM(ABC):
    name: str

    def __init__(
        self,
        model_path: str | Path,
        device_map: str = "auto",
        load_in_4bit: bool = False,
        max_memory: Optional[dict] = None,
    ) -> None:
        self.model_path = str(model_path)
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self.max_memory = max_memory
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, image_path: str | Path, prompt: str, max_new_tokens: int = 512) -> str:
        raise NotImplementedError

    def _device_map_for(self, quant_config, pin_when_unquantized: bool = False):
        """Decide where layers go.

        History, because this function caused a real incident: it used to return
        ``{"": 0}`` whenever a quantisation config was present, *discarding*
        ``self.max_memory``. On a 32 GB card that meant a 30B model's 882 shards
        were all pushed at the GPU regardless of the caller's budget; once the
        physical limit was passed, the Windows driver began paging VRAM into host
        RAM, load crawled at ~15 s/shard (~3.7 h projected) at 85 W and 100%
        utilisation, and the desktop became unresponsive. Detection could not
        save it, because the thrash happened during *load*, before any per-batch
        guard could run.

        So an explicit ``max_memory`` now wins. That is what makes "refuse to
        offload" expressible: pass ``{0: "<budget>GiB", "cpu": "0GiB"}`` and a
        model that cannot fit raises immediately instead of silently paging.

        Behaviour is deliberately unchanged when ``max_memory`` is None — the
        previously published 4-bit results were produced that way and must stay
        reproducible.

        ``pin_when_unquantized`` is opt-in and exists for one model. CheXagent-8b
        ships its checkpoint in **F32** (31.15 GB on disk) but asks to load in
        fp16, and ``"auto"`` makes accelerate estimate the model size from the
        *checkpoint metadata* rather than from the requested dtype. It concludes
        ~31 GB will not fit in ~31 GB free, splits the model across GPU and CPU,
        and cannot place the top-level ``query_tokens`` Parameter (a bare
        ``nn.Parameter`` at the model root, not inside a submodule), which is
        exactly the meta-device warning the pre-run recorded. Every forward pass
        then crossed PCIe (38.69 -> 118.05 s/img) until the WDDM watchdog reset
        the driver twice at the 16-minute mark. The fix is not lower precision —
        fp16 already fits with ~15 GB to spare — but telling the loader not to
        estimate. See ``docs/50_resume_prerun_findings.md`` §2.2.
        """
        if quant_config is None:
            if (pin_when_unquantized and not self.max_memory
                    and self.device_map == "auto" and torch.cuda.is_available()):
                # An explicit max_memory still wins: that is the escape hatch
                # ("fail immediately instead of paging") and must keep working.
                return {"": 0}
            return self.device_map
        if self.max_memory:
            # Caller supplied a budget: let accelerate honour it. Do NOT force
            # everything onto the GPU.
            return self.device_map
        if torch.cuda.is_available():
            return {"": 0}
        return self.device_map

    def _device_kwargs(self) -> dict:
        """Assemble the device/quantisation kwargs consistently.

        Keeps every wrapper from re-deriving the same precedence rules and from
        forgetting ``max_memory`` when a quantisation config is present.
        """
        quant = self._quant_config()
        kwargs: dict = {"device_map": self._device_map_for(quant)}
        if self.max_memory:
            kwargs["max_memory"] = self.max_memory
        if quant is not None:
            kwargs["quantization_config"] = quant
        return kwargs

    def _quant_config(self):
        if not self.load_in_4bit:
            return None
        # A checkpoint can already be quantized (e.g. a pack-quantized /
        # compressed-tensors release). Passing a BitsAndBytesConfig on top of that
        # fails with:
        #   "The model is quantized with CompressedTensorsConfig but you are
        #    passing a BitsAndBytesConfig config"
        # so the on-the-fly quantisation is skipped and the pre-quantized weights
        # are used as shipped.
        if self._checkpoint_is_prequantized():
            return None
        try:
            from transformers import BitsAndBytesConfig
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception as exc:
            print(f"[warn] 4-bit loading unavailable, falling back: {exc}")
            return None

    def _checkpoint_is_prequantized(self) -> bool:
        """True when the checkpoint ships its own quantisation config."""
        try:
            import json
            from pathlib import Path
            cfg = Path(self.model_path) / "config.json"
            if not cfg.exists():
                return False
            data = json.loads(cfg.read_text(encoding="utf-8"))
            return isinstance(data.get("quantization_config"), dict)
        except Exception:
            return False


# --------------------------------------------------------------------------- #
# shared image-preprocessing contract
# --------------------------------------------------------------------------- #
# Full write-up, evidence and consequences: docs/55_preprocessing_contract.md.
#
# Every Qwen-processor model (qwen25, qwen3, qwen30b, and medgemma_15 through
# the generic wrapper) must hand the processor the SAME pixel limits, or the
# three layers stop being comparable across models.
#
# What goes wrong without this: ``qwen25vl.py`` hard-coded
# ``max_pixels=768*28*28`` (= 602,112 px, 0.59 MP) -- a legacy 8 GB-card
# workaround. Measured dataset sizes are NIH 1024x1024 = 1.05 MP (all 1,506),
# PadChest mean 7.00 MP / max 15.37 MP, Open-i mean 4.75 MP / max 5.92 MP. So on
# PadChest qwen25 saw ~9% of the pixels while qwen3 -- which passes no hint at
# all and therefore inherits ``qwen_vl_utils``' own default -- saw the native
# image, and qwen25 dropped out of the JSON contract on 78% of PadChest cells
# (323/1500 parseable) while internvl parsed 1500/1500 on the same layer.
#
# The same ceiling also silently re-degraded the NIH ``resolution`` cells. Those
# are built by downsampling to 12.5-50% and upsampling back to 1024x1024, so
# they arrive at the processor already soft; the extra 1.32x linear reduction
# (1.05 MP -> 0.59 MP) is invisible in every table, and it made qwen25's
# resolution curve incomparable with qwen3's.
#
# The ceiling below sits ABOVE the largest image in any of the three layers
# (PadChest 15.37 MP), and it is explicit rather than inherited from
# ``qwen_vl_utils``' default, because that default is not guaranteed identical
# across the two environment forks. VRAM is bounded by BATCH SIZE instead of by
# resolution.
#
# KNOWN RESIDUAL (measured, scripts/_diag_pixel_hint.py): the hint binds for
# processors that honour it (qwen3: grid 224 -> 244, 12,544 -> 14,884 tokens),
# but qwen25's checkpoint ships its own `preprocessor_config.json` ceiling of
# 16384*28*28 = 12.85 MP, which the second resize stage applies anyway -- so on
# PadChest the 100 images larger than that (6.67% of 1,500) still lose 16% of
# their pixels (1.09x linear). NIH (1.05 MP) and Open-i (5.92 MP max) are below
# that ceiling and are untouched. Removing the residual would mean editing the
# checkpoint's declared limit, which is a reproducibility decision, not a bug
# fix, so it is documented in docs/55 rather than done here.
IMAGE_MAX_PIXELS = 24576 * 28 * 28     # 19.27 MP -- above every dataset image
IMAGE_MIN_PIXELS = 256 * 28 * 28       # 0.20 MP -- original floor, never binds


def image_pixel_hints() -> dict:
    """The pixel limits every Qwen-style message must carry."""
    return {"max_pixels": IMAGE_MAX_PIXELS, "min_pixels": IMAGE_MIN_PIXELS}


def release_fragmented_cache(threshold_gib: float = 1.0) -> bool:
    """Return cached-but-unused VRAM to the driver after a call.

    Why this exists: measured on PadChest (``scripts/_diag_vram_cliff.py``,
    2026-10-21), the caching allocator keeps growing its reserved pool with the
    image size and never gives it back. After a 6.99 MP cell torch had 17.5 GiB
    reserved while only 11.2 GiB was live; after an 11.40 MP cell, 28.1 GiB
    reserved with 2.5 GiB free. The next cell (16,384 image tokens) then asked
    for more than the card has, and **Windows did not raise OOM** -- WDDM paged
    VRAM into host RAM and the cell took 269--290 s instead of ~40 s, at 100%
    utilisation and 122 W. That is the silent failure mode this guard closes:
    the run looks healthy, only 20x slower, and the rate table silently rots.

    ``empty_cache`` only drops blocks that are already free, so it cannot touch
    live tensors. It is skipped when fragmentation is small, which keeps the
    fixed-size stages (NIH, InternVL, CheXagent, MedGemma) on their fast path.

    Returns True when a release actually happened.
    """
    if not torch.cuda.is_available():
        return False
    try:
        reserved = torch.cuda.memory_reserved()
        allocated = torch.cuda.memory_allocated()
    except Exception:  # noqa: BLE001 - a broken CUDA context must not kill inference
        return False
    if (reserved - allocated) / 2**30 < threshold_gib:
        return False
    torch.cuda.empty_cache()
    return True


def ensure_left_padding(processor) -> str:
    """Force left padding on a decoder-only processor's tokenizer.

    ``padding=True`` pads on the RIGHT by default, and then the first generated
    token is conditioned on a pad position, so every short member of a
    variable-length batch is decoded from garbage. The NIH layer never exposed
    this: its images are uniformly 1024x1024 and the synthetic ``resolution``
    cells are upsampled back to 1024x1024, so those batches were equal-length
    and no pad was ever inserted. PadChest and Open-i have variable native
    sizes, and transformers says so out loud -- "A decoder-only architecture is
    being used, but right-padding was detected!".

    Accepts either a processor (its ``.tokenizer`` is used) or a bare tokenizer
    -- InternVL and CheXagent keep a tokenizer without a processor object.

    Returns the side that was set, so the caller can print it into provenance.
    """
    tok = getattr(processor, "tokenizer", None) or processor
    if tok is not None and hasattr(tok, "padding_side"):
        tok.padding_side = "left"
        return tok.padding_side
    return "unknown"
