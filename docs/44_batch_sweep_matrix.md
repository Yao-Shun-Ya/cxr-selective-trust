# 44 — Batch-size Sweet Spots Across the Full Model Matrix

> Hardware: **RTX 4080 SUPER 32 GB** (32760 MiB) / 64 GB DDR5 / Windows WDDM.
> Tool: `scripts/26_batch_sweep.py`, results appended to
> `program/outputs/benchmark/batch_sweep.json`.
> Supersedes the partial sweeps in `docs/43`, which covered only three models.

---

## 1. Why this is a measured curve, not a parameter choice

On a fixed memory budget, throughput is **not monotone in batch size**. Two things
push in opposite directions:

* larger batches amortise kernel-launch and Python-side overhead → faster per image;
* larger batches need proportionally more KV cache → more VRAM, until the driver
  starts paging.

The second effect does not announce itself with an OOM. On Windows the WDDM
driver can **oversubscribe** dedicated VRAM, paging working sets to host memory;
every access to a paged page then crosses PCIe, whose bandwidth is roughly
25–30 GB/s on PCIe 4.0 x16 against 700–1000 GB/s of board memory. Throughput
therefore *falls* while memory readings look merely "full".

Measured on this machine, Qwen2.5-VL-7B at `bs=8` runs 1.50 s/img; at `bs=16` it
runs **1.89 s/img** — 26% slower — with no OOM raised.

**Correction to the earlier mechanism description.** An earlier write-up said the
CUDA allocator overflows into shared system memory. That is wrong: dedicated VRAM
is hard-isolated under WDDM, and `torch.empty` does not route to shared memory —
an over-budget allocation raises `CUDA out of memory`. What actually happens is
driver-level paging (oversubscription), and the `nvidia-smi` reading that exceeds
32 GB is the *dedicated+shared* total rather than physical VRAM alone. The
observation (throughput declines past a point) is right; the mechanism is
oversubscription, not allocator spill. This tool records both the allocator peak
(`torch.cuda.max_memory_allocated`) and the driver figure (NVML) precisely because
the gap between them is where oversubscription hides.

---

## 2. Method

* One model in memory at a time; the previous model is freed and the CUDA cache
  emptied before the next load (otherwise the VRAM numbers are meaningless).
* `max_new_tokens=128` fixed across every model, so runs are comparable.
* Real chest radiographs from `nih_shared_500.csv` and the project's own prompt,
  so image-token counts match what the experiments actually pay.
* A batch size is rejected when per-image time **rises** versus the previous
  step (`--slowdown-tolerance 0.10`), whether or not an OOM was raised.
* A host-RAM guard stops the sweep if free memory falls below a threshold —
  oversubscription pages VRAM into host RAM, and that is the condition which
  takes the machine down.
* The recommendation rule is deliberately not "the fastest measured point":
  it is **the smallest batch within 5% of the best throughput**, so VRAM is not
  spent for a negligible gain.

### Memory model to predict a ladder before spending time

With weights `W` and a per-sample cache `K`, peak ≈ `W + b·K`. Two measured
anchors are enough to solve for `K`:

```text
Qwen2.5-VL-7B, bf16:  W ≈ 14 GB (argued),  peak(bs=8) = 15.8 GB  -> K ≈ 0.23 GB/sample
```

The same arithmetic explains why 4-bit matters far more than batch tuning for the
larger models: quantising the weights frees ~10 GB, which is worth roughly 40
samples of headroom at `K ≈ 0.25`, whereas pushing a bf16 7B model from `bs=8` to
`bs=16` already crosses the physical limit.

---

## 3. Hypotheses tested (written before running)

Each row is a prediction, so the sweep can falsify it rather than merely confirm
whatever it finds.

| Model | Precision | Predicted sweet spot | Reasoning |
|---|---|---:|---|
| MedGemma-4B-IT | bf16 | 64 | measured in `docs/43` (22.3 GB, 0.50 s/img) |
| MedGemma-1.5-4B-IT | bf16 | 64 | same size class as above |
| MedGemma-4B-IT | 4bit | ~128 | weights shrink ~5.5 GB, so more room |
| CheXagent-8B | bf16 | ~16 | 16 GB weights leave little for cache |
| CheXagent-2-3B | bf16 | ~32 | small LLM + separate SigLIP encoder |
| Qwen2.5-VL-7B | bf16 | 8 | measured: 15.8 GB at bs=8, 16 already oversubscribes |
| Qwen2.5-VL-7B | 4bit | ~32 | 4× the headroom once weights are quantised |
| Qwen3-VL-8B | bf16 | 8 | measured 27.2 GB at bs=8 — already near the ceiling |
| Qwen3-VL-8B | 4bit | ~16 | |
| InternVL3.5-8B | bf16 | ~8 | tile-based preprocessing raises per-image cost |
| LLaVA-Med-7B | bf16 | ~16 | simpler vision tower than Qwen3 |
| Qwen3-VL-30B-A3B | **4bit only** | 1–2 | ~17 GB of weights before any cache |

`Qwen3-VL-30B-A3B` is deliberately 4-bit only: at bf16 its weights alone (~58 GB)
exceed the card.

---

## 4. Standing decisions

* **Do not run full experiments at a batch size whose measured VRAM exceeds
  ~90% of physical** (≈29 GB), even if it works in a timing test. Image size and
  generation length both vary across the real experiments, and a mid-run OOM
  wastes far more than the marginal speed was worth.
* **Baselines for the paper use the same `max_new_tokens` and the same manifest**
  as the sweep, so the recorded `s/img` figures translate directly into run
  budgets.
* Any change to image resolution or prompt length invalidates these numbers and
  requires re-measuring, because both change the image-token count and therefore
  the KV cache.
* The throughput-vs-batch curve itself (rise, plateau, decline) is a
  supplementary-material figure; the decline arm is the interesting part, since
  it is the empirical evidence for the "measure, don't maximise" claim.

### 4.1 A measurement trap: silent CPU offload

`BaseVLM._device_map_for()` returns `{"": 0}` ("put everything on GPU") **only
when a quantisation config is present**. With `load_in_4bit=False` it falls back
to `device_map="auto"`, and `auto` may decide to offload layers to CPU. The loader
says so explicitly:

```text
Some parameters are on the meta device because they were offloaded to the cpu.
```

Consequences, and how this document handles them:

* An offloaded model still runs, so the sweep produces a *number* — but a
  different number from a fully-resident model, and the two must not be mixed in
  one table.
* `torch.cuda.max_memory_allocated` **excludes** offloaded weights, so a small
  allocator peak can coexist with heavy host-RAM traffic. This is one reason the
  driver-side NVML reading is recorded separately.
* Any row measured with offloading is marked as such and re-measured with the
  model pinned to the GPU before being used for run budgeting.

The practical rule at this card size: **7–8B models at bf16 are the ones at
risk** (16 GB of weights leaves little room for activations at a large batch), so
those rows are the ones to re-verify.

---

## 5. Results

Measured on RTX 4080 SUPER 32 GB, 100 NIH images, `max_new_tokens=128`,
`--limit-per-batch 100`. `torch_peak` = allocator peak; `smi` = driver-side high
water mark (see caveat below).

### 5.1 Models with an established sweet spot

Measured on RTX 4080 SUPER 32 GB, 100 NIH images (64 for Qwen3-VL),
`max_new_tokens=128`. `torch_peak` = allocator peak.
Every row produced with CPU and disk offload refused (`--gpu-only`).

| Model | Precision | Sweet spot | s/img | torch_peak | Decline point |
|---|---|---:|---:|---:|---|
| MedGemma-4B-IT | 4bit | **64** | 0.41 | 14.2 GB | none seen to 64 |
| MedGemma-4B-IT | bf16 | **64** | 0.40 | 21.0 GB | bs=96 → 0.88 s/img (+120%) |
| MedGemma-1.5-4B-IT | 4bit | **64** | 0.41 | 14.2 GB | none seen to 64 |
| MedGemma-1.5-4B-IT | bf16 | **64** | 0.40 | 21.0 GB | bs=96 → 0.75 s/img (+86%) |
| Qwen2.5-VL-7B | 4bit | **8** | 1.28 | 13.5 GB | bs=16 → 1.45 s/img |
| Qwen2.5-VL-7B | bf16 | **8** | 1.31 | 18.1 GB | bs=20 → 3.17 s/img (+112%) |
| **CheXagent-8B** | bf16 | **4** | 7.14 | **29.7 GB** | **bs=6 → 117.1 s/img (+1540%)** |
| **InternVL3.5-8B** | bf16 | **n/a (serial)** | 4.40 | 17.3 GB | no batch gain by construction |

#### InternVL3.5-8B: why batch size does not apply

The sweep shows a flat curve (4.40 / 4.42 / 4.44 s/img at bs=1/2/4, then a
spurious 5.09 at bs=8), and identical memory at every step. That is not a
measurement artefact: **the wrapper runs this model serially.** Each image is
tiled independently into a variable number of 448×448 patches, and the model's
own `chat()` path is per-conversation, so there is no shared padded batch to
form. The "decline" at bs=8 is therefore meaningless — nothing about the model
changed between steps; only the number of *sequential* calls did.

This is honestly a limitation of the wrapper rather than of the model. Real
batching would require per-image patch grids and variable-length image-token runs
packed into one call, i.e. reimplementing InternVL's collation. It is left as
serial because 4.4 s/img is already usable at the planned sample sizes, and
because a batch path that silently mis-tiles would corrupt every downstream
metric.

**Reporting caveat:** for InternVL the "sweet spot" column is not a tuning
result, so the paper should present its throughput as a measured single-image
rate rather than as an optimum.

**Qwen3-VL-8B detail** (measured in `.venv-modern`, transformers 5.18): the curve
is still improving at bs=8, so the operating point is a *ceiling*, not a plateau:

| bs | s/img | torch_peak |
|---:|---:|---:|
| 1 | 5.25 | 17.21 GB |
| 2 | 3.63 | 18.07 GB |
| 4 | 2.68 | 19.79 GB |
| 8 | **2.18** | 23.23 GB |
| 12 | *thrash* | 32.2 GB (driver) |

The bs=12 run is instructive: `nvidia-smi` reported 100% GPU utilisation but only
**86 W** of a 320 W budget, with 32,202 of 32,760 MiB allocated. That combination —
full utilisation, negligible power, memory saturated — is the signature of paging
rather than compute, and it is the state to abort on rather than wait out.

Conclusions that generalise beyond these numbers:

1. **The 4-bit penalty to speed is nil.** MedGemma-4B runs 0.41 s/img at 4-bit
   versus 0.40 s/img at bf16 — the same throughput for 7 GB less VRAM. There is
   no throughput cost to quantisation here, only a memory benefit.
2. **Throughput rises, plateaus, then collapses — and the collapse is not an
   OOM.** Every 7–8B model above shows a sharp reversal while still executing.
   CheXagent-8B is the extreme case: 7.14 → 117.1 s/img, a 16× collapse with no
   exception raised.
3. **Model size does not predict the ceiling; weight footprint does.**
   CheXagent-8B tops out at bs=4 while Qwen3-VL-8B is still improving at bs=8,
   despite nominally similar parameter counts. CheXagent's vision tower and
   query-token stack carry a much larger fixed cost.

### 5.1.1 CheXagent-8B: the clearest oversubscription evidence in the matrix

| bs | s/img | torch_peak | SMI (driver) |
|---:|---:|---:|---:|
| 2 | 13.43 | 28.85 GB | 30.55 GB |
| 4 | 7.14 | 29.73 GB | 31.57 GB |
| **6** | **117.11** | 30.88 GB | 31.58 GB |

The reversal happens while the allocator peak is still below the 32 GB physical
limit, and the driver figure sits at ~99% of it. This is the pattern the
supplementary figure should show: the failure mode is driver paging, not an
allocator error.

### 5.1.2 Why CheXagent-8B batching was broken, and the fix

CheXagent-8B's processor cannot batch at all out of the box, for two independent
reasons discovered by instrumenting the pipeline:

1. **No pad token.** Its tokenizer ships without one, so `padding=True` raises
   *"Asking to pad but the tokenizer does not have a padding token"*. Fixed by
   assigning `pad_token = eos_token` at load time.
2. **`pixel_values` layout.** The model's own `forward` does
   `pixel_values.sum(dim=(2, 3, 4))` followed by `pixel_values[image_mask]`, so
   it **requires 5-D input** `(batch, n_images, C, H, W)` — the extra image-slot
   axis is deliberate (it is how one sample carries several images). Upstream
   `CheXagentProcessor.__call__` produces 5-D output but wraps it as
   `[torch.tensor(pv) for pv in ...]` when already batched, nesting one axis too
   many and failing above bs=1 with
   *"Sizes of tensors must match except in dimension 1"*.

   The wrapper therefore bypasses the processor and assembles
   `(n, 3, H, W) → unsqueeze(1) → (n, 1, 3, H, W)` itself. A first attempt that
   instead *removed* the extra axis failed with
   *"Dimension out of range (expected to be in range of [-4, 3], but got 4)"* —
   evidence that the 5-D contract is real and must be honoured, not normalised away.

Batching is accepted only after output equivalence: batch[0] is compared against
the serial result at every batch size, and all matched (`no_finding`).

#### Measured shapes (for the record)

| path | n=1 | n=2 |
|---|---|---|
| `image_processor` directly | `(1,3,448,448)` | `(2,3,448,448)` |
| `processor.__call__` (upstream, broken when batched) | `(1,1,3,448,448)` | `(1,2,3,448,448)` |
| wrapper output (required by `forward`) | `(1,1,3,448,448)` | `(2,1,3,448,448)` |

### 5.2 Caveat on the SMI column

`nvidia-smi`'s figure is a **high-water mark that does not fall back** within a
process: on the MedGemma-1.5 ladder it read 24.96 GB at bs=16, 24.96 at bs=32 and
24.96 at bs=64, i.e. a cached ceiling rather than a per-batch value. Treat it as a
"how close did this *run* get to the wall" indicator, and `torch_peak` as the
per-batch allocator figure. Oversubscription is flagged when SMI approaches or
exceeds 32 GB — which happened at MedGemma bf16 bs=96 (31.8 GB), Qwen bf16 bs=20
(31.7 GB) and Qwen 4bit bs=16 (20.2 GB allocator, i.e. the allocator was not the
limiting factor there).

### 5.3 Observation on Qwen2.5-VL bf16: the least squares fit of the ceiling

Fitting `peak ≈ W + b·K` on the clean bf16 ladder:

```text
bs=8  -> 18.09 GB
bs=16 -> 24.79 GB        K ≈ 0.84 GB/sample,  W ≈ 11.4 GB
bs=20 -> 29.66 GB
```

so bs=20 is the last point before the allocator alone exceeds the card. The
measured collapse at that exact step is consistent with the paging model rather
than with an allocator failure.

### 5.4 A measurement trap that invalidated one row

The first full-sweep run reported `qwen25_vl_7b_instruct [bf16] bs=20 @ 6.53
s/img`. That number is wrong — not because the model is slow, but because the
loader had silently offloaded part of it:

```text
Some parameters are on the meta device because they were offloaded to the cpu.
host RAM free: 49.6 GB -> 37.7 GB
```

Re-measured with offload refused (`--gpu-only`), the same configuration gives
**1.31 s/img at bs=8** and **3.17 s/img at bs=20** — roughly 5× faster, with the
same qualitative shape. Reasons the trap is easy to fall into, and the guard now
in place:

* `BaseVLM._device_map_for` returns `{"": 0}` only when a quantisation config is
  present. Without one it passes `device_map="auto"`, and `auto` may offload.
* An offloaded model still runs, so no exception is raised and the sweep happily
  reports a plausible-looking number.
* Offloaded weights are invisible to `torch.cuda.max_memory_allocated`, so the
  allocator peak understates the real footprint.
* `--gpu-only` now passes `max_memory={0: "<total-2>GiB", cpu: "0GiB"}`, which
  turns a silent offload into a loud failure.

**Rule adopted: no throughput number is used for run budgeting unless it was
produced with offload refused.** Rows measured before that guard are marked and
re-measured.

### 5.5 Loader / environment work: what was fixed, what is still blocked

Every item below was a genuine defect, not a tuning question.

#### FIXED

| Model | Defect | Fix |
|---|---|---|
| CheXagent-8B | tokenizer has no pad token → `padding=True` fails | set `pad_token = eos_token` at load |
| CheXagent-8B | `pixel_values` must be **5-D** `(B, n_images, C, H, W)`; upstream processor nests one axis too many when batched | assemble the layout in the wrapper; verified against serial output |
| InternVL3.5-8B | `'InternVLChatModel' object has no attribute 'all_tied_weights_keys'` under transformers 5.x | run in `.venv-vl` (4.51.3) |
| InternVL3.5-8B | bare `AssertionError` on generate: `self.img_context_token_id` is only set inside `chat()` | set it from the tokenizer's `<IMG_CONTEXT>` before generating |
| InternVL3.5-8B | `"processor_class": "InternVLProcessor"` declared but **the class is not shipped**; `AutoProcessor` silently returns `Qwen2TokenizerFast` | implement the 448×448 tiling and normalisation in the wrapper from `preprocessor_config.json` |
| InternVL3.5-8B | `RuntimeError: Input type (float) and bias type (c10::BFloat16)` | cast `pixel_values` to the model dtype |
| InternVL3.5-8B | 5-D reshape/permute tiling silently transposed tiles (only a shape error revealed it) | split tiles by explicit slicing instead |

Also fixed in shared infrastructure:

* **`base.py::_device_map_for` discarded `max_memory` whenever a quantisation
  config was present**, so it returned `{"": 0}` and forced every layer onto the
  GPU regardless of the caller's budget. On the 30B model that meant 882 shards
  pushed at a 32 GB card; the Windows driver then paged VRAM into host RAM and
  the load crawled at ~15 s/shard (~3.7 h projected) at 85 W and 100 %
  utilisation. An explicit `max_memory` now wins, which makes "refuse to offload"
  expressible. Behaviour with `max_memory=None` is unchanged, so previously
  published 4-bit results stay reproducible.

  Effect on the 30B: before, a multi-hour silent thrash; after,
  `ValueError: ... Make sure you have enough GPU RAM to fit the quantized model`
  in **2.3 s**, with the card back to 495 MiB.

#### STILL BLOCKED

| Model | Blocker | Assessment |
|---|---|---|
| Qwen3-VL-30B-A3B | 4-bit weights ≈ 30 GB; does not fit a 32 GB card with any batch | **not feasible on this hardware** — retain as a "needs larger GPU" exclusion |
| CheXagent-2-3B | `'DynamicCache' object has no attribute 'get_max_length'` — remote code uses a **removed transformers API** | needs a `transformers==4.40.0` env; the CXR-specialist role is already covered by CheXagent-8B |
| LLaVA-Med-7B | `LlavaMistralForCausalLM` no longer exists in transformers; config is the legacy flat `llava_mistral` format needing nested `vision_config`/`text_config`; **and the vision tower `openai/clip-vit-large-patch14-336` is not on disk nor cached** (the 686-key index contains no `vision_tower` weights) | needs a config rewrite **plus** a ~1.7 GB extra download; deprioritised |

For the paper the honest statement is that the matrix covers the models that
load reproducibly, with these exclusions listed explicitly. **Do not write "9
models"** — write the number actually measured and name the exclusions.

#### CheXagent-2-3B detail (dropped)

Three defects were fixed before hitting the wall: no processor file (a path-based
wrapper was written); a spurious `tensorflow` requirement (its
`tokenization_chexagent.py` imports TF under `TYPE_CHECKING` **for a type hint
only** — `tf.*` is never called — but transformers' remote-code loader scans the
AST with `check_imports()` and **does not honour `TYPE_CHECKING`**); and a hard
`assert transformers.__version__ == "4.40.0"`, relaxed to a warning.

It then fails at generation with a removed-cache-API error, which is precisely
what that assertion protected against. Not fixable without a 4.40 environment.

### 5.6 Standing consequences

* **Cleared for full runs** at these operating points, with the listed headroom:

  | Model | Precision | bs | s/img | peak | env |
  |---|---|---:|---:|---:|---|
  | MedGemma-4B-IT | 4bit | 64 | 0.41 | 14.2 GB | `.venv-vl` |
  | MedGemma-4B-IT | bf16 | 64 | 0.40 | 21.0 GB | `.venv-vl` |
  | MedGemma-1.5-4B-IT | 4bit | 64 | 0.41 | 14.2 GB | `.venv-vl` |
  | MedGemma-1.5-4B-IT | bf16 | 64 | 0.40 | 21.0 GB | `.venv-vl` |
  | Qwen2.5-VL-7B | 4bit | 8 | 1.28 | 13.5 GB | `.venv-vl` |
  | Qwen2.5-VL-7B | bf16 | 8 | 1.31 | 18.1 GB | `.venv-vl` |
  | CheXagent-8B | bf16 | 4 | 7.14 | 29.7 GB | `.venv-vl` |
  | Qwen3-VL-8B | bf16 | 8 | 2.18 | 23.2 GB | `.venv-modern` |
  | InternVL3.5-8B | bf16 | serial | 4.40 | 17.3 GB | `.venv-vl` |

  **9 model × precision configurations are now measurable.** Excluded with
  stated reasons: Qwen3-VL-30B-A3B (does not fit), CheXagent-2-3B (removed
  transformers API), LLaVA-Med-7B (missing vision tower + removed model class).

* **Environment split is real and must be respected.** Qwen3-VL requires
  `.venv-modern` (transformers 5.18); everything else runs in `.venv-vl`
  (4.51.3). Never upgrade `.venv-vl` in place — that would invalidate the
  already-published 4-bit baselines.
* **CheXagent-8B is not a batch-scaling candidate.** 29.7 GB at bs=4 leaves
  ~2 GB of headroom, and bs=6 already collapses. Treat 4 as final.
* Re-measure if image resolution or prompt length changes: both alter the image
  token count, hence the KV term, hence where the ceiling sits.
* `--gpu-only` is mandatory for any number used in budgeting. See §5.4.
