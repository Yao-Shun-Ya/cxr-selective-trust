"""Rewrite LLaVA-Med's legacy flat config into the modern nested layout.

Why this is needed
------------------
``llava-med-v1.5-mistral-7b`` ships a flat config with
``"model_type": "llava_mistral"`` and ``"architectures": ["LlavaMistralForCausalLM"]``.
Neither survives in current transformers:

* ``LlavaMistralForCausalLM`` **does not exist** in 4.51.3 or 5.18.0 (checked);
* ``llava_mistral`` is not a registered ``model_type``.

What *does* exist is ``LlavaForConditionalGeneration`` with
``model_type: "llava"``, which expects the modern nested layout: a
``vision_config`` block and a ``text_config`` block. The flat LLaVA-1.5 keys map
onto those directly, and the ``LlavaForConditionalGeneration`` implementation
already knows to build ``vision_tower`` and ``multi_modal_projector`` from them.

This is a **config rewrite only** — no weights are touched. The original file is
kept as ``config.json.orig`` so the change is auditable and reversible.

Safety
------
A config rewrite can produce a model that *loads* but is subtly wrong, so after
running this you must smoke-test with ``scripts/27_smoke_one.py`` and confirm the
output is a sensible chest-X-ray reading. A load success on its own proves
nothing.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = ROOT / "models" / "llava-med-v1.5-mistral-7b"
DEFAULT_VISION = ROOT / "models" / "openai" / "clip-vit-large-patch14-336"

# The reference conversion of the same weights (downloaded for provenance) is the
# authority for fields the *legacy* config never carried.
REFERENCE_CONFIG = ROOT / "outputs" / "benchmark" / "llavamed_ref" / "config.json"

# Fields that MUST be present in text_config even when the source does not have
# them, with the reference release's values.
#
# ``rope_theta`` is the one that actually bit us. The legacy flat config has no
# ``rope_theta`` at all, so the rewritten config did not either, and transformers
# silently fell back to its default 10000 -- while these weights were fine-tuned
# from ``Mistral-7B-Instruct-v0.2``, whose RoPE base is **1000000**. The model
# still loaded, still ran, and emitted ``</s>`` as its first token: 1,890 of 1,900
# empty replies on the pre-run (mean 0.22 s/cell). Every prompt-format hypothesis
# was wrong; the tokeniser was right, the image features were right, the BOS was
# a real but harmless deviation. This dict is why that cannot recur silently.
REFERENCE_TEXT_FALLBACK = {
    "hidden_act": "silu",
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "max_position_embeddings": 32768,
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "rope_theta": 1000000.0,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "sliding_window": None,
    "use_cache": True,
    # NOTE: deliberately NOT the reference's 32064. That number belongs to the
    # reference conversion's vocabulary, which ships <pad> at 32001 and padding up
    # to 32064. This checkpoint's embed table is 32000 rows (verified in the
    # safetensors header) and the wrapper appends <image> at 32000 itself.
    "vocab_size": 32000,
}

# Keys that belong in text_config (Mistral-7B language tower)
TEXT_KEYS = [
    "attention_dropout", "bos_token_id", "eos_token_id", "hidden_act",
    "hidden_size", "initializer_range", "intermediate_size",
    "max_position_embeddings", "num_attention_heads", "num_hidden_layers",
    "num_key_value_heads", "rms_norm_eps", "rope_theta", "sliding_window",
    "tie_word_embeddings", "torch_dtype", "use_cache", "vocab_size",
    "pad_token_id", "head_dim",
]

# Keys that belong in vision_config (CLIP ViT-L/14-336)
VISION_KEYS = [
    "hidden_size", "image_size", "intermediate_size", "num_attention_heads",
    "num_hidden_layers", "patch_size", "projection_dim", "attention_dropout",
    "layer_norm_eps", "vision_use_head", "hidden_act", "model_type",
]


def derive_vision_config(model_dir: Path) -> dict | None:
    """Read the real CLIP architecture straight out of the safetensors.

    Much safer than hardcoding: the previous hardcoded values claimed
    ``hidden_size 768`` while the checkpoint stores 1024, and the mismatch only
    surfaced as a confusing load error.
    """
    try:
        import json as _json
        from safetensors import safe_open
        idx_path = model_dir / "model.safetensors.index.json"
        if not idx_path.exists():
            return None
        wm = _json.loads(idx_path.read_text(encoding="utf-8"))["weight_map"]

        def sh(k):
            with safe_open(str(model_dir / wm[k]), framework="pt") as f:
                return tuple(f.get_tensor(k).shape)

        P = "vision_tower.vision_model."
        hidden = sh(P + "embeddings.class_embedding")[0]
        inter = sh(P + "encoder.layers.0.mlp.fc1.weight")[0]
        patch = sh(P + "embeddings.patch_embedding.weight")[2]
        pos = sh(P + "embeddings.position_embedding.weight")[0]     # 1 + n_patches
        n_layers = sum(1 for k in wm if k.startswith(P + "encoder.layers.")
                       and k.endswith("self_attn.q_proj.weight"))
        n_patches = pos - 1
        side = int(round(n_patches ** 0.5))
        return {
            "hidden_size": int(hidden),
            "intermediate_size": int(inter),
            "patch_size": int(patch),
            "num_hidden_layers": int(n_layers),
            "num_attention_heads": 16,          # not derivable from shapes
            "image_size": int(side * patch),
        }
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    ap.add_argument("--vision-tower", default=str(DEFAULT_VISION))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    vision_dir = Path(args.vision_tower)
    cfg_path = model_dir / "config.json"

    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found")
        return 1

    flat = json.loads(cfg_path.read_text(encoding="utf-8"))
    # The separate CLIP directory was never downloaded here -- the vision tower
    # weights live INSIDE this checkpoint (391 `vision_tower.*` tensors), which is
    # why the model works at all. So a missing vision dir is not fatal any more:
    # the architecture is derived from the tensor shapes, and the reference
    # config fills in what shapes cannot express (num_attention_heads, ...).
    vision_src = {}
    if (vision_dir / "config.json").exists():
        vision_src = json.loads((vision_dir / "config.json").read_text(encoding="utf-8"))
    else:
        print(f"[warn] no vision tower config at {vision_dir}; "
              f"deriving the architecture from the checkpoint instead")

    # ---- text_config: carry over the language-tower fields --------------
    # Idempotent by design. The first version only understood the legacy FLAT
    # layout; run against an already-nested config it would have produced an
    # empty text_config and made things worse. So the source of the language
    # fields is either the flat file or an existing text_config, and anything
    # still missing is filled from REFERENCE_TEXT_FALLBACK (rope_theta above all).
    source_text = flat.get("text_config") if isinstance(flat.get("text_config"), dict) else flat
    reference = dict(REFERENCE_TEXT_FALLBACK)
    if REFERENCE_CONFIG.exists():
        try:
            ref_text = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8")).get("text_config", {})
            # the reference's vocab size belongs to its own tokenizer; keep ours
            ref_text.pop("vocab_size", None)
            reference.update({k: v for k, v in ref_text.items() if not k.startswith("_")})
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[warn] reference config unreadable ({exc}); using built-in values")

    text_config = {"model_type": "mistral"}
    for k, v in reference.items():
        if k in TEXT_KEYS:
            text_config[k] = v
    for k in TEXT_KEYS:
        if k in source_text:
            text_config[k] = source_text[k]
    text_config.setdefault("vocab_size", flat.get("vocab_size", 32000))
    text_config["model_type"] = "mistral"

    print("=== text_config sources ===")
    for k in sorted(text_config):
        if k in source_text:
            src = "source"
        elif k in reference:
            src = "REFERENCE FALLBACK" if REFERENCE_TEXT_FALLBACK.get(k) == text_config[k] \
                else "reference file"
        else:
            src = "default"
        mark = "  <-- filled in, was missing" if src != "source" else ""
        print(f"   {k:26s} = {str(text_config[k]):>12s}   [{src}]{mark}")

    # ---- pad token: what THIS tokenizer actually has ------------------------
    pad_token_id = 2          # Mistral's </s> row, which the wrapper uses as pad
    try:
        tok_cfg = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
        for sid, entry in (tok_cfg.get("added_tokens_decoder") or {}).items():
            if entry.get("content") == "<pad>":
                pad_token_id = int(sid)
    except (json.JSONDecodeError, OSError, ValueError):
        pass
    print(f"=== pad token: {pad_token_id} "
          f"({'a real <pad> token exists' if pad_token_id != 2 else 'no <pad> token; using </s>'}) ===")

    # ---- vision_config: DERIVED from the checkpoint ------------------------
    # The checkpoint's CLIP is ViT-L/14-336: hidden 1024, patch 14, mlp 4096.
    # Deriving these from the actual tensor shapes removes any chance of writing a
    # config that disagrees with the weights — a hardcoded value here previously
    # produced hidden_size 768 and failed with
    # "size mismatch for class_embedding: checkpoint 1024 vs model 768".
    #
    # Falls back to the reference conversion's constants if the weights cannot be
    # inspected.
    derived = derive_vision_config(model_dir)
    vision_config = {
        "model_type": "clip_vision_model",
        "hidden_act": "quick_gelu",
        "attention_dropout": 0.0,
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-05,
        "projection_dim": 768,
        "num_channels": 3,
        "vision_use_head": False,
    }
    vision_config.update(derived or {
        "hidden_size": 1024, "intermediate_size": 4096, "image_size": 336,
        "patch_size": 14, "num_hidden_layers": 24, "num_attention_heads": 16,
    })
    vision_config["model_type"] = "clip_vision_model"

    new = {
        "architectures": ["LlavaForConditionalGeneration"],
        "model_type": "llava",
        # --- the three fields that actually made loading work ---------------
        # `image_seq_length` lets LlavaProcessor take the image-token count
        # directly instead of computing (h // patch_size) * (w // patch_size),
        # which is what raised "TypeError: int // NoneType" because this
        # checkpoint's processor has patch_size = None. Set from the CLIP grid:
        # (336 / 14) ** 2 = 576.
        "image_seq_length": 576,
        # These two are forwarded into get_image_features() by the model, whose
        # forward calls it with `vision_feature_layer` / `vision_feature_select_strategy`.
        # The mm_vision_select_* names below are the OLD LLaVA naming and are
        # ignored by the HF implementation — having only those was why the vision
        # path could not resolve its inputs.
        "vision_feature_layer": -2,
        "vision_feature_select_strategy": "default",
        "multimodal_projector_bias": True,
        "projector_hidden_act": "gelu",
        "ignore_index": -100,
        "image_token_index": 32000,
        # The reference conversion pads at 32001 (a distinct <pad> token); THIS
        # checkpoint's tokenizer has no <pad> at all, so padding falls back to
        # Mistral's EOS row. Writing 32001 here would name a token that does not
        # exist -- the wrapper masks the mistake at load time, which is exactly the
        # kind of "works by accident" a config file should not rely on.
        "pad_token_id": pad_token_id,
        "vocab_size": flat.get("vocab_size", 32000),
        "torch_dtype": flat.get("torch_dtype", "bfloat16"),
        # LLaVA-1.5 conventions (kept for provenance; the HF implementation reads
        # the vision_feature_* keys above)
        "mm_projector_type": flat.get("mm_projector_type", "mlp2x_gelu"),
        "mm_vision_select_layer": flat.get("mm_vision_select_layer", -2),
        "mm_vision_select_feature": flat.get("mm_vision_select_feature", "patch"),
        "mm_patch_merge_type": "flat",
        "mm_use_im_start_end": bool(flat.get("mm_use_im_start_end", False)),
        "mm_use_im_patch_token": bool(flat.get("mm_use_im_patch_token", True)),
        "mm_hidden_size": flat.get("mm_hidden_size", vision_config["hidden_size"]),
        # Point the tower at the LOCAL directory; the shipped value is
        # "openai/clip-vit-large-patch14-336", which cannot be fetched without
        # HF access.
        "vision_config": vision_config,
        "text_config": text_config,
    }
    # record where the tower actually lives, for provenance
    new["_vision_tower_path"] = str(vision_dir)

    print("=== flat keys that do NOT carry over ===")
    for k in sorted(flat):
        if k not in TEXT_KEYS and k not in new and not k.startswith("_"):
            print(f"   dropped: {k}")

    print("\n=== new config ===")
    print(json.dumps({k: v for k, v in new.items() if k != "vision_config"},
                     indent=2, ensure_ascii=False)[:1200])
    print("vision_config:", json.dumps(vision_config)[:300])

    if args.dry_run:
        print("\ndry run: nothing written")
        return 0

    backup = cfg_path.with_suffix(".json.orig")
    if not backup.exists():
        shutil.copy2(cfg_path, backup)
        print(f"\nbacked up original -> {backup.name}")
    cfg_path.write_text(json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {cfg_path}")

    # ---- preprocessor_config: match the reference conversion exactly --------
    # The shipped checkpoint has no preprocessor_config at all. The reference
    # conversion (chaoyinshe/llava-med-v1.5-mistral-7b-hf) uses
    # `size={"shortest_edge": 336}` with an explicit `crop_size`, NOT `size: 336`;
    # the difference changes the resize-then-crop behaviour and therefore what the
    # vision tower actually sees.
    pp_path = model_dir / "preprocessor_config.json"
    pp = {
        "crop_size": {"height": 336, "width": 336},
        "do_center_crop": True,
        "do_convert_rgb": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_processor_type": "CLIPImageProcessor",
        "image_std": [0.26862954, 0.26130258, 0.27577711],
        "processor_class": "LlavaProcessor",
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "size": {"shortest_edge": 336},
    }
    if pp_path.exists():
        shutil.copy2(pp_path, pp_path.with_suffix(".json.orig"))
    pp_path.write_text(json.dumps(pp, indent=2), encoding="utf-8")
    print(f"wrote {pp_path} (reference format: shortest_edge + crop_size)")

    print("\nNEXT: this only proves the config is well-formed. Run "
          "scripts/27_smoke_one.py and check the OUTPUT is a sensible reading — "
          "a model can load successfully from a wrong config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
