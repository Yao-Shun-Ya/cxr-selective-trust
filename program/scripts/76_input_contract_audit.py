"""Audit the actual input contract each VLM sees: source pixels -> pixels handed
to the processor -> image tokens -> batched input length -> pad tokens, plus the
padding side in force.

Why this exists
---------------
The three evaluation layers carry images of very different native size (NIH
1024x1024 = 1.05 MP, PadChest mean 7.00 MP / max 15.37 MP, Open-i mean 4.75 MP /
max 5.92 MP). Two independent defects silently changed how much of an image each
model actually saw:

  * ``qwen25vl.py`` hard-coded ``max_pixels = 768*28*28`` (0.59 MP), a legacy 8 GB
    workaround. Replaced by the shared ``base.IMAGE_MAX_PIXELS`` (19.27 MP) +
    ``base.IMAGE_MIN_PIXELS`` handed over via ``image_pixel_hints()``.
  * right-padding by default. ``base.ensure_left_padding(processor)`` is now called
    from every wrapper's ``load()``.

This script measures the post-fix state without ever loading weights: it uses only
``AutoProcessor`` / ``AutoTokenizer`` and the wrappers' own preprocessing helpers,
on the CPU. For each model x layer it takes the smallest / median / largest image
by pixel count (full scan, not a random sample), reports the pipeline numbers, then
runs one 2-image batch to see whether padding is actually exercised.

Run it in both environment forks -- qwen3 needs transformers 5.x (.venv-modern),
everyone else .venv-vl (4.51.3):

  .\.venv-vl\Scripts\python.exe scripts\76_input_contract_audit.py \
      --models medgemma_4b_it,medgemma_15_4b_it,qwen25_vl_7b_instruct,chexagent_8b,internvl3_5_8b,llava_med_v1_5_7b
  .\.venv-modern\Scripts\python.exe scripts\76_input_contract_audit.py \
      --models qwen3_vl_8b_instruct

Results are MERGED into outputs/state/input_contract_audit.json so the two runs
produce one combined file.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import torch  # noqa: F401  (CPU only; never a model weight)
from PIL import Image

PROGRAM = Path(__file__).resolve().parents[1]           # program/
SRC = PROGRAM / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cxr_trust.models.base import ensure_left_padding, image_pixel_hints  # noqa: E402
from cxr_trust.models.registry import (  # noqa: E402
    MODEL_OPTIONS, MODEL_SPECS, resolve_path,
)

PROMPT = "Describe the findings in this chest X-ray."
DEFAULT_OUT = PROGRAM / "outputs" / "state" / "input_contract_audit.json"

# layer -> manifest, image root, splits to keep. image_relpath is joined onto root
# (verified: NIH relpath "images/x.png" under root data/nih/images -> the double
# "images/" is real on disk).
LAYERS: dict[str, dict] = {
    "nih": {
        "manifest": PROGRAM / "data" / "manifests" / "nih_eval_1732.csv",
        "root": PROGRAM / "data" / "nih" / "images",
        "splits": {"internal_test", "source_val"},
    },
    "padchest": {
        "manifest": PROGRAM / "data" / "manifests" / "padchest_shift.csv",
        "root": PROGRAM / "data" / "padchest",
        "splits": {"internal_test"},
    },
    "openi": {
        "manifest": PROGRAM / "data" / "manifests" / "openi_rrg.csv",
        "root": PROGRAM / "data" / "open-i",
        "splits": {"external_report"},
    },
}

ALL_MODELS = [
    "medgemma_4b_it",
    "medgemma_15_4b_it",
    "qwen25_vl_7b_instruct",
    "qwen3_vl_8b_instruct",
    "chexagent_8b",
    "internvl3_5_8b",
    "llava_med_v1_5_7b",
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.open(img).convert("RGB")


def _read_json(path: Path) -> dict:
    try:
        return json.loads((path / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _preproc(path: Path) -> dict:
    try:
        return json.loads((path / "preprocessor_config.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _pad_tokens(attn) -> int:
    if attn is None:
        return 0
    return int((attn == 0).sum())


def _count_ids(ids, token_id) -> int:
    if token_id is None:
        return -1
    return int((ids == int(token_id)).sum())


def _left_pad_stats(id_lists: list[list[int]]) -> tuple[int, int]:
    """Left-pad token id lists to the max length; return (input_len, pad_count)."""
    width = max(len(x) for x in id_lists)
    pads = sum(width - len(x) for x in id_lists)
    return width, pads


# --------------------------------------------------------------------------- #
# layer scanning: min / median / max by pixel count
# --------------------------------------------------------------------------- #
def select_layer_images(layer: str) -> dict:
    cfg = LAYERS[layer]
    rows = list(csv.DictReader(cfg["manifest"].open(encoding="utf-8", newline="")))
    kept = []
    for r in rows:
        if r.get("split") not in cfg["splits"]:
            continue
        rel = (r.get("image_relpath") or "").strip()
        if not rel:
            continue
        kept.append(cfg["root"] / rel)

    sized = []
    for p in kept:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:
            continue
        sized.append({"path": str(p), "src_w": int(w), "src_h": int(h), "src_px": int(w) * int(h)})

    if not sized:
        raise RuntimeError(f"no images found for layer {layer}")

    sized.sort(key=lambda d: (d["src_px"], d["path"]))
    n = len(sized)
    picks = {"min": sized[0], "median": sized[n // 2], "max": sized[-1]}
    return {"n_scanned": n, "picks": picks}


# --------------------------------------------------------------------------- #
# per-model adapters. Each builds the processor/tokenizer exactly the wrapper
# does, without calling load(), and returns (single, batch, meta).
# --------------------------------------------------------------------------- #
def _qwen_emit(img, proc, hints, tid):
    """One image through the Qwen VL path; returns the full processor inputs."""
    from qwen_vl_utils import process_vision_info
    msg = [{"role": "user", "content": [
        {"type": "image", "image": _pil(img), **hints},
        {"type": "text", "text": PROMPT},
    ]}]
    text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([msg])
    inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                  padding=True, return_tensors="pt")
    return inputs


def _qwen_from_grid(inputs, proc, tid):
    grid = inputs["image_grid_thw"][0]
    patch = int(getattr(proc.image_processor, "patch_size", 14))
    merge = int(getattr(proc.image_processor, "merge_size", 2))
    t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
    eff = h * patch * w * patch * t
    grid_tokens = (t * h * w) // (merge * merge)
    counted = _count_ids(inputs["input_ids"][0], tid)
    return {
        "effective_px": int(eff),
        "effective_shape": [t, h * patch, w * patch],
        "grid_thw": [t, h, w],
        "img_tokens": counted if counted >= 0 else int(grid_tokens),
        "img_tokens_from_grid": int(grid_tokens),
        "patch_size": patch,
        "merge_size": merge,
        "pixel_values_shape": list(inputs["pixel_values"].shape),
    }


def make_qwen25(model: str, path: Path):
    """Inline copy of qwen25vl.generate_batch's input construction.

    The wrapper's generate_batch only differs by the ``.to(model.device)`` move and
    the ``model.generate`` call, neither of which touches tokenisation, so the same
    lines reproduce the exact inputs. ``image_pixel_hints`` is imported, never
    inlined as a number.
    """
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(path)
    hints = image_pixel_hints()
    tid = _read_json(path).get("image_token_id")

    def single(img):
        return _qwen_from_grid(_qwen_emit(img, proc, hints, tid), proc, tid)

    def batch(imgs):
        from qwen_vl_utils import process_vision_info
        msgs = []
        for i in imgs:
            msgs.append([{"role": "user", "content": [
                {"type": "image", "image": _pil(i), **hints},
                {"type": "text", "text": PROMPT},
            ]}])
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = proc(text=texts, images=image_inputs, videos=video_inputs,
                      padding=True, return_tensors="pt")
        return {"input_len": int(inputs["input_ids"].shape[1]),
                "pad_tokens": _pad_tokens(inputs.get("attention_mask"))}

    meta = {"processor": proc, "pad_target": proc,
            "method": "inline copy of qwen25vl.generate_batch (image_pixel_hints imported from base)",
            "replicated": True,
            "reason": "generate_batch is a single inline block; its tokenisation is copied verbatim "
                      "(only the model.device move and model.generate are omitted)",
            "diagnostics": {
                "pixel_hint": hints,
                # The processor re-resizes with ITS OWN config defaults, so the
                # image-dict hint is not the only ceiling in force.
                "processor_config_max_pixels": _preproc(path).get("max_pixels"),
            }}
    return single, batch, meta


def make_generic(model: str, path: Path, with_hints: bool):
    """Reuse GenericVLM._prepare_batch directly -- it never touches self.model."""
    from transformers import AutoProcessor
    from cxr_trust.models.generic import GenericVLM

    opts = dict(MODEL_OPTIONS.get(model, {})) if with_hints else {}
    w = GenericVLM(model_path=path, name=model, **opts)
    w.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    tid = getattr(w.processor, "image_token_id", None) or _read_json(path).get("image_token_id")

    def _grid(inputs):
        return inputs.get("image_grid_thw")

    def single(img):
        inputs = w._prepare_batch([img], [PROMPT])
        if _grid(inputs) is not None:
            return _qwen_from_grid(inputs, w.processor, tid)
        px = inputs["pixel_values"]
        eff = int(px.shape[-2]) * int(px.shape[-1])
        return {"effective_px": eff,
                "effective_shape": [int(px.shape[-2]), int(px.shape[-1])],
                "img_tokens": _count_ids(inputs["input_ids"][0], tid),
                "pixel_values_shape": list(px.shape)}

    def batch(imgs):
        inputs = w._prepare_batch(list(imgs), [PROMPT] * len(imgs))
        return {"input_len": int(inputs["input_ids"].shape[1]),
                "pad_tokens": _pad_tokens(inputs.get("attention_mask"))}

    meta = {"processor": w.processor, "pad_target": w.processor,
            "method": "GenericVLM._prepare_batch (applied unchanged)" if with_hints
                      else "GenericVLM._prepare_batch (no pixel hints set for this model)",
            "replicated": False,
            "reason": "" if with_hints else "MODEL_OPTIONS has no entry -> min/max_pixels stay None",
            "diagnostics": {
                "pixel_hint": image_pixel_hints() if with_hints else None,
                "processor_config_size": _preproc(path).get("size"),
            }}
    return single, batch, meta


def make_medgemma4b(model: str, path: Path):
    """Legacy MedGemma wrapper has no separable prepare method -> copy its 3 lines."""
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(path)
    tid = getattr(proc, "image_token_id", None) or _read_json(path).get("image_token_id")
    sys_msg = "You are a careful chest X-ray interpretation assistant."

    def _msgs(imgs):
        out = []
        for i in imgs:
            out.append([
                {"role": "system", "content": [{"type": "text", "text": sys_msg}]},
                {"role": "user", "content": [{"type": "image", "image": _pil(i)},
                                             {"type": "text", "text": PROMPT}]},
            ])
        return out

    def single(img):
        inputs = proc.apply_chat_template(_msgs([img]), add_generation_prompt=True,
                                          tokenize=True, return_dict=True,
                                          return_tensors="pt", padding=True)
        px = inputs["pixel_values"]
        return {"effective_px": int(px.shape[-2]) * int(px.shape[-1]),
                "effective_shape": [int(px.shape[-2]), int(px.shape[-1])],
                "img_tokens": _count_ids(inputs["input_ids"][0], tid),
                "pixel_values_shape": list(px.shape)}

    def batch(imgs):
        inputs = proc.apply_chat_template(_msgs(list(imgs)), add_generation_prompt=True,
                                          tokenize=True, return_dict=True,
                                          return_tensors="pt", padding=True)
        return {"input_len": int(inputs["input_ids"].shape[1]),
                "pad_tokens": _pad_tokens(inputs.get("attention_mask"))}

    meta = {"processor": proc, "pad_target": proc,
            "method": "replica of medgemma.generate_batch apply_chat_template call",
            "replicated": True,
            "reason": "legacy wrapper keeps the processor call inline; no reusable prepare method"}
    return single, batch, meta


def make_chexagent(model: str, path: Path):
    """Copy chexagent._prepare's two processor calls (its tail moves to model.device)."""
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(path, trust_remote_code=True)
    tok = proc.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    ip = proc.image_processor
    text = f" USER: <s>{PROMPT} ASSISTANT: <s>"

    def single(img):
        enc = tok([text], return_tensors="pt", add_special_tokens=True)
        pv = ip(images=[_pil(img)], return_tensors="pt")["pixel_values"]
        if isinstance(pv, (list, tuple)):
            pv = torch.stack([torch.as_tensor(x) for x in pv], dim=0)
        pv = torch.as_tensor(pv)
        while pv.dim() > 4 and pv.shape[0] == 1:
            pv = pv.squeeze(0)
        return {"effective_px": int(pv.shape[-2]) * int(pv.shape[-1]),
                "effective_shape": [int(pv.shape[-2]), int(pv.shape[-1])],
                # CheXagent has NO image placeholder in input_ids: the image reaches
                # the (T5) language model through Q-Former cross-attention.
                "img_tokens": 0,
                "img_tokens_note": "no image placeholder token; image injected via cross-attention "
                                   "(32 query tokens, not in input_ids)",
                "pixel_values_shape": list(pv.shape),
                "txt_len": int(enc["input_ids"].shape[1])}

    def batch(imgs):
        enc = tok([text, text], return_tensors="pt", padding=True, add_special_tokens=True)
        return {"input_len": int(enc["input_ids"].shape[1]),
                "pad_tokens": _pad_tokens(enc.get("attention_mask"))}

    meta = {"processor": proc, "pad_target": proc,
            "method": "replica of chexagent._prepare (processor.tokenizer + processor.image_processor)",
            "replicated": True,
            "reason": "chexagent._prepare ends by moving tensors to model.device, so it needs the weights; "
                      "the two processor calls are reproduced and the pad_token=eos fix is applied"}
    return single, batch, meta


def make_internvl(model: str, path: Path):
    """Reuse InternVL._tile / _build_query / _prepare with config-derived attributes."""
    from transformers import AutoTokenizer
    from cxr_trust.models.internvl import (
        IMG_CONTEXT_TOKEN, InternVL, _load_conversation_module,
    )

    inst = InternVL(model_path=path)
    inst.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    inst.conversation = _load_conversation_module(path)

    pp = json.loads((path / "preprocessor_config.json").read_text(encoding="utf-8"))
    inst._norm = (torch.tensor(pp["image_mean"], dtype=torch.float32),
                  torch.tensor(pp["image_std"], dtype=torch.float32))
    inst._rescale = float(pp.get("rescale_factor", 1.0 / 255.0))
    size = pp.get("size") or {}
    inst._tile_size = int(size.get("height", 448))
    inst._max_patches = int(pp.get("max_patches") or 12)

    cfg = _read_json(path)
    proc_cfg = json.loads((path / "processor_config.json").read_text(encoding="utf-8"))
    # 256 = (448/14)**2 * downsample_ratio(0.5)**2, the value the model exposes as
    # num_image_token; read from processor_config instead of from the weights.
    inst.num_image_token = int(proc_cfg.get("image_seq_length")
                               or (inst._tile_size // 14) ** 2 * int(cfg.get("downsample_ratio", 0.5) * 2) ** 2)
    inst.template_name = cfg.get("template", "internvl2_5")
    inst.img_context_token_id = inst.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    # _build_query reads self.model.config.system_message; a lightweight stand-in
    # avoids loading weights while keeping the real code path.
    inst.model = types.SimpleNamespace(config=types.SimpleNamespace(
        system_message=cfg.get("system_message")))

    def single(img):
        inst._prompts = [PROMPT]
        inputs, per_image = inst._prepare([img])
        n = int(per_image[0])
        pv = inputs["pixel_values"]
        toks = _count_ids(inputs["input_ids"][0], inst.img_context_token_id)
        return {"effective_px": n * inst._tile_size * inst._tile_size,
                "effective_shape": [n * inst._tile_size, inst._tile_size],
                "img_tokens": toks if toks >= 0 else n * inst.num_image_token,
                "n_patches": n,
                "num_image_token": inst.num_image_token,
                "tile_size": inst._tile_size,
                "pixel_values_shape": list(pv.shape)}

    def batch(imgs):
        inst._prompts = [PROMPT] * len(imgs)
        inputs, per_image = inst._prepare(list(imgs))
        return {"input_len": int(inputs["input_ids"].shape[1]),
                "pad_tokens": _pad_tokens(inputs.get("attention_mask")),
                "n_patches": [int(x) for x in per_image]}

    meta = {"processor": inst.tokenizer, "pad_target": inst.tokenizer,
            "method": "InternVL._tile/_build_query/_prepare (unchanged) with config-derived attributes",
            "replicated": False,
            "reason": "load() is required only to fill norm/tile/num_image_token; those are read from "
                      "preprocessor_config.json / processor_config.json instead of the weights"}
    return single, batch, meta


def make_llava(model: str, path: Path):
    """Replica of llava._generate_one minus model.generate (image encoding + ids)."""
    from transformers import AutoProcessor

    IMG_TOKEN = "<image>"
    proc = AutoProcessor.from_pretrained(path)
    tok = proc.tokenizer
    ip = proc.image_processor
    if getattr(proc, "patch_size", None) is None:
        proc.patch_size = 14
    tok.add_special_tokens({"additional_special_tokens": [IMG_TOKEN]})
    image_token_id = int(tok.convert_tokens_to_ids(IMG_TOKEN))
    cfg = _read_json(path)
    n_tokens = int(cfg.get("image_seq_length") or cfg.get("num_image_tokens") or 576)

    def _ids():
        prefix = tok("USER: ", add_special_tokens=False)["input_ids"]
        suffix = tok(f"\n{PROMPT} ASSISTANT:", add_special_tokens=False)["input_ids"]
        return prefix + [image_token_id] * n_tokens + suffix

    def single(img):
        ids = _ids()
        px = ip(images=_pil(img), return_tensors="pt")["pixel_values"]
        return {"effective_px": int(px.shape[-2]) * int(px.shape[-1]),
                "effective_shape": [int(px.shape[-2]), int(px.shape[-1])],
                "img_tokens": n_tokens,
                "num_image_tokens": n_tokens,
                "pixel_values_shape": list(px.shape),
                "txt_len": len(ids)}

    def batch(imgs):
        # LLaVA-1.5 wrapper is serial (supports_batching=False). The prompt is fixed
        # and the image block is a constant 576 tokens, so two prompts are identical
        # in length; the left-padded width is reported for completeness.
        width, pads = _left_pad_stats([_ids() for _ in imgs])
        return {"input_len": int(width), "pad_tokens": int(pads),
                "note": "synthesised: wrapper runs one image per generate() call"}

    meta = {"processor": proc, "pad_target": proc,
            "method": "replica of llava._generate_one (id construction + image_processor)",
            "replicated": True,
            "reason": "llava bypasses LlavaProcessor.__call__ and builds ids inline; no separable method"}
    return single, batch, meta


MAKERS = {
    "qwen25_vl_7b_instruct": lambda m, p: make_qwen25(m, p),
    "qwen3_vl_8b_instruct": lambda m, p: make_generic(m, p, with_hints=True),
    "medgemma_15_4b_it": lambda m, p: make_generic(m, p, with_hints=False),
    "medgemma_4b_it": lambda m, p: make_medgemma4b(m, p),
    "chexagent_8b": lambda m, p: make_chexagent(m, p),
    "internvl3_5_8b": lambda m, p: make_internvl(m, p),
    "llava_med_v1_5_7b": lambda m, p: make_llava(m, p),
}


# --------------------------------------------------------------------------- #
def audit_model(model: str, layers: list[str], selections: dict) -> dict:
    path = resolve_path(model, PROGRAM)
    single, batch, meta = MAKERS[model](model, path)

    side = ensure_left_padding(meta["pad_target"])
    print(f"[padding] {model}: ensure_left_padding -> {side!r} (wrapper={MODEL_SPECS[model]['wrapper']})",
          flush=True)
    if side != "left":
        print(f"FATAL: {model} padding_side is {side!r}, expected 'left'", file=sys.stderr)
        sys.exit(1)

    out = {
        "wrapper": MODEL_SPECS[model]["wrapper"],
        "precision": MODEL_SPECS[model]["precision"],
        "padding_side": side,
        "method": meta["method"],
        "replicated": meta["replicated"],
        "reason": meta["reason"],
        "diagnostics": meta.get("diagnostics", {}),
        "layers": {},
    }

    for layer in layers:
        sel = selections[layer]
        picks = sel["picks"]
        layer_out = {
            "n_scanned": sel["n_scanned"],
            "selected": {q: {"src_px": p["src_px"], "src_w": p["src_w"], "src_h": p["src_h"],
                             "path": p["path"]} for q, p in picks.items()},
            "images": [],
        }
        for q in ("min", "median", "max"):
            p = picks[q]
            m = single(p["path"])
            m["quantile"] = q
            m["src_px"] = p["src_px"]
            m["src_w"] = p["src_w"]
            m["src_h"] = p["src_h"]
            layer_out["images"].append(m)

        bims = [picks["min"]["path"], picks["max"]["path"]]
        b = batch(bims)
        b["images"] = ["min", "max"]
        layer_out["batch"] = b
        out["layers"][layer] = layer_out
        print(f"[{model}/{layer}] scanned={sel['n_scanned']} "
              f"eff_px={[im['effective_px'] for im in layer_out['images']]} "
              f"tokens={[im['img_tokens'] for im in layer_out['images']]} "
              f"batch_len={b['input_len']} pads={b['pad_tokens']}", flush=True)

    return out


def write_merged(out_path: Path, new_models: dict, layers: list[str]) -> dict:
    data = {}
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data.setdefault("models", {})
    data["prompt"] = PROMPT
    data["layers"] = layers
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["models"].update(new_models)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def render_table(data: dict, models: list[str], layers: list[str]) -> str:
    cols = ["model", "layer", "sel", "src_px", "effective_px", "img_tokens",
            "batch_input_len", "pad_tokens", "padding_side"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for model in models:
        md = data["models"].get(model)
        if not md:
            continue
        side = md["padding_side"]
        for layer in layers:
            ld = md["layers"].get(layer)
            if not ld:
                continue
            b = ld["batch"]
            for im in ld["images"]:
                row = [model, layer, im["quantile"], str(im["src_px"]),
                       str(im["effective_px"]), str(im["img_tokens"]),
                       str(b["input_len"]), str(b["pad_tokens"]), side]
                lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ALL_MODELS))
    ap.add_argument("--layers", default=",".join(LAYERS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    layers = [l.strip() for l in args.layers.split(",") if l.strip()]
    for m in models:
        if m not in MAKERS:
            print(f"unknown model {m!r}; known: {sorted(MAKERS)}", file=sys.stderr)
            sys.exit(2)
    for l in layers:
        if l not in LAYERS:
            print(f"unknown layer {l!r}; known: {list(LAYERS)}", file=sys.stderr)
            sys.exit(2)

    print(f"[env] torch={torch.__version__} python={sys.version.split()[0]}", flush=True)
    try:
        import transformers
        print(f"[env] transformers={transformers.__version__}", flush=True)
    except Exception:
        pass

    print("[scan] sizing every image in each layer ...", flush=True)
    selections = {}
    for l in layers:
        s = select_layer_images(l)
        selections[l] = s
        p = s["picks"]
        print(f"  {l}: n={s['n_scanned']} "
              f"min={p['min']['src_w']}x{p['min']['src_h']}({p['min']['src_px']}) "
              f"median={p['median']['src_w']}x{p['median']['src_h']}({p['median']['src_px']}) "
              f"max={p['max']['src_w']}x{p['max']['src_h']}({p['max']['src_px']})", flush=True)

    new_models = {}
    for m in models:
        new_models[m] = audit_model(m, layers, selections)

    data = write_merged(Path(args.out), new_models, layers)
    print("\n[markdown]\n" + render_table(data, models, layers), flush=True)
    print(f"\n[written] {args.out}", flush=True)


if __name__ == "__main__":
    main()
