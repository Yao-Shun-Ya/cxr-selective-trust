# Supplementary: Engineering Insights

> Draft framework for the supplementary-material section on throughput and
> deployment. Every number below is measured on **RTX 4080 SUPER 32 GB**,
> Windows 11, WDDM, `max_new_tokens=128`, on real chest radiographs from
> `nih_shared_500.csv`. Pending numbers are marked `TBD` rather than guessed.

---

## S1. Why this section exists

Two things a reviewer of a multi-model VLM study is entitled to ask, that a table
of AUROC and ECE cannot answer:

1. **Was the evaluation actually feasible?** A matrix of 8 models × several
   thousand images only means something if the compute story holds up.
2. **Are the batch sizes defensible, or were they tuned to whatever looked good?**

This section answers both with measured throughput curves, and reports one
finding that is methodological rather than merely practical: **throughput is not
monotone in batch size, and the failure mode is silent.**

---

## S2. Method

* Each model measured with **offload refused** (`max_memory={0: budget, cpu: 0}`),
  so a model that does not fit fails loudly instead of silently partly running on
  the CPU. This matters: an earlier measurement of Qwen2.5-VL-7B, taken without
  that guard, reported **6.53 s/img** where the clean figure is **1.31 s/img** — a
  **5× error** caused by silent CPU offload, with no warning in the output.
* Allocator peak reported from `torch.cuda.max_memory_allocated`; driver-side use
  from NVML. The two disagree systematically and both are reported.
* A batch size is rejected when per-image time *rises* versus the previous step,
  whether or not an out-of-memory error was raised.
* The operating point chosen is **the smallest batch within 5% of the best
  measured throughput**, not the fastest point — VRAM is not spent for a
  negligible gain.

---

## S3. The central finding: rise, plateau, then silent collapse

**Table S1 — throughput vs batch size (selected models).**

| Model | Precision | bs | s/img | VRAM peak | Regime |
|---|---|---:|---:|---:|---|
| MedGemma-4B | bf16 | 16 | 0.53 | 12.6 GB | improving |
| MedGemma-4B | bf16 | 32 | 0.47 | 15.4 GB | improving |
| MedGemma-4B | bf16 | **64** | **0.40** | 21.0 GB | best |
| MedGemma-4B | bf16 | 96 | 0.88 | 26.6 GB | **collapsed (+120%)** |
| Qwen2.5-VL-7B | bf16 | 4 | 1.59 | 16.7 GB | improving |
| Qwen2.5-VL-7B | bf16 | **8** | **1.31** | 18.1 GB | best |
| Qwen2.5-VL-7B | bf16 | 16 | 1.49 | 24.8 GB | declining |
| Qwen2.5-VL-7B | bf16 | 20 | 3.17 | 29.7 GB | **collapsed (+112%)** |
| CheXagent-8B | bf16 | 2 | 13.43 | 28.9 GB | improving |
| CheXagent-8B | bf16 | **4** | **7.14** | 29.7 GB | best |
| CheXagent-8B | bf16 | 6 | 117.11 | 30.9 GB | **collapsed (+1540%)** |

**The collapse is not an out-of-memory error.** Every row above completed without
raising an exception; the model simply became 2–16× slower.

**Mechanism.** Dedicated VRAM is hard-isolated under WDDM, so an over-budget
allocation normally raises `CUDA out of memory`. What instead happens near the
limit is **driver-level oversubscription**: parts of the working set are paged to
host memory and every access crosses PCIe, whose ~25–30 GB/s on PCIe 4.0 x16 is
roughly 30× slower than the board's 700–1000 GB/s. The diagnostic signature is
unmistakable and worth stating explicitly, because it is what tells you to abort
rather than wait:

```text
GPU utilisation ~100%,  power draw far below the board limit (~86 W of 320 W),
VRAM at ~99% of physical.
```

Full utilisation with negligible power is *paging*, not computing.

**This is why the operating point is measured, not maximised.** A practitioner
who sets the largest batch that fails to raise an error will silently get a
16× slower pipeline, and will attribute it to the model or the framework.

---

## S4. Quantisation: no throughput cost, only a memory benefit

**Table S2 — 4-bit vs bf16 at each model's operating point.**

| Model | Precision | bs | s/img | VRAM peak |
|---|---|---:|---:|---:|
| MedGemma-4B | 4bit | 64 | 0.41 | 14.2 GB |
| MedGemma-4B | bf16 | 64 | 0.40 | 21.0 GB |
| MedGemma-1.5-4B | 4bit | 64 | 0.41 | 14.2 GB |
| MedGemma-1.5-4B | bf16 | 64 | 0.40 | 21.0 GB |

At equal batch size, 4-bit is within 2.5% of bf16 while requiring **6.8 GB less
VRAM**. On this hardware 4-bit is therefore not an accuracy-versus-speed
trade-off at all; the only question it raises is whether it changes the *findings*,
which is what the 4-bit/fp16 comparison in the main text addresses.

---

## S5. Per-model operating points used for the main experiments

**Table S3 — recommended configurations.** Every row produced with offload refused.

| Model | Params | Precision | bs | s/img | VRAM peak | Headroom | Env |
|---|---:|---|---:|---:|---:|---:|---|
| MedGemma-4B-IT | 4B | 4bit | 64 | 0.41 | 14.2 GB | 55% | `.venv-vl` |
| MedGemma-4B-IT | 4B | bf16 | 64 | 0.40 | 21.0 GB | 34% | `.venv-vl` |
| MedGemma-1.5-4B-IT | 4B | 4bit | 64 | 0.41 | 14.2 GB | 55% | `.venv-vl` |
| MedGemma-1.5-4B-IT | 4B | bf16 | 64 | 0.40 | 21.0 GB | 34% | `.venv-vl` |
| Qwen2.5-VL-7B | 7B | 4bit | 8 | 1.28 | 13.5 GB | 58% | `.venv-vl` |
| Qwen2.5-VL-7B | 7B | bf16 | 8 | 1.31 | 18.1 GB | 43% | `.venv-vl` |
| Qwen3-VL-8B | 8B | bf16 | 8 | 2.18 | 23.2 GB | 27% | `.venv-modern` |
| InternVL3.5-8B | 8B | bf16 | 1 (serial) | 4.40 | 17.3 GB | 46% | `.venv-vl` |
| LLaVA-Med-7B | 7B | bf16 | 1 (serial) | 0.90 | 18.1 GB | 43% | `.venv-vl` |
| CheXagent-8B | 8B | bf16 | 4 | 7.14 | 29.7 GB | 7% | `.venv-vl` |
| Qwen3-VL-30B-A3B | 30B (MoE, 3B active) | prebuilt 4bit | — | — | — | — | excluded, see S5.1 |

**Headroom is not uniform, and that has an operational consequence.** CheXagent-8B
runs at 7% headroom and collapses at the very next batch size, so it is the one
model that cannot tolerate a change in image resolution or prompt length without
re-measuring. Two models (InternVL3.5, LLaVA-Med) are measured serially because
their architectures tile or expand each image independently; for those, batch
size is not a tuning parameter and the table reports a single-image rate.

---

## S5.1 Why the 30B MoE is excluded, despite fitting in memory

This is worth recording because the failure is *not* the one the memory
arithmetic predicts, and the same class of problem will recur with other
pre-quantized releases.

**The memory arithmetic is sound.** The pre-quantized checkpoint is 17.86 GB
(4 shards, SHA256-verified), which is comfortably within 32 GB. Measured
behaviour on the card:

| Quantity | Measured |
|---|---:|
| weight files | 17.86 GB |
| allocator peak at load | **27.7 GB** (runtime overhead ≈ +9.8 GB) |
| driver-side VRAM | **28.1–29.2 GB** of 32 GB |
| load time (torch 2.14.1+cu126) | 75.9 s |

**Two independent blockers prevent generation:**

1. **Expert layout mismatch.** The checkpoint stores MoE experts *per expert* in a
   packed format — 55,296 tensors as
   `model.language_model.layers.{0..47}.mlp.experts.{0..127}.gate_proj.weight_packed`
   (48 layers × 128 experts, with `weight_packed` / `weight_scale` /
   `weight_shape` triples, i.e. a compressed-tensors / llm-compressor layout).
   The transformers implementation in use expects a **fused** parameter,
   `layers.{i}.mlp.experts.gate_up_proj`, and finds **zero** matching tensors.
   The experts are therefore reported as MISSING and newly initialised. Converting
   would mean renaming and fusing 55k tensors — a re-quantisation job, not a
   configuration change.
2. **Disk offload cannot work on Windows.** The tail of the model does not fit in
   29 GB, so `compressed_tensors` attempts disk offloading and fails because it
   creates its cache entries with `os.symlink`
   (`compressed_tensors/offload/cache/disk.py:190`). Symlink creation on Windows
   requires Developer Mode or elevation, so the path raises
   `WinError 1314: A required privilege is not held by the client`. This is a
   POSIX assumption in the library, not a misconfiguration on our side.

Additionally, `compressed-tensors 0.19.0` declares `torch>=2.10.0` and calls seven
`torch.accelerator.*` APIs. The environment was therefore upgraded from
torch 2.5.1+cu121 to **2.14.1+cu126**, which was necessary and did help (load time
fell from 440.8 s to 75.9 s), but cannot address either blocker above.

Two notes on that upgrade, both verified rather than assumed:

* The default PyPI wheel on the mirror installs `torch 2.14.1+cpu` — **CUDA
  unavailable**. A CUDA build has to come from the pytorch wheel index
  (`--index-url https://download.pytorch.org/whl/cu126`, wheel
  `torch-2.14.1+cu126`). Checking `torch.version.cuda` after installing is
  therefore not optional: `cuda.is_available() == False` is easy to miss when the
  version string looks right.
* The other model in that environment, Qwen3-VL-8B, was re-verified after the
  upgrade and still runs correctly (1.4 s for one image, 17.8 GB), confirming the
  upgrade did not disturb the working configuration.

**Decision: excluded from the matrix.** Reporting throughput for a model whose
expert weights never loaded would be meaningless. If scale evidence from a 30B
model is required later, the realistic routes are to serve the checkpoint through
an engine that natively supports its per-expert layout (e.g. vLLM), or to obtain a
checkpoint in the fused layout this transformers version expects.

**Transferable lesson.** "The weights fit in VRAM" is necessary but not sufficient.
For pre-quantized checkpoints, three further questions decide feasibility, and
each must be checked *before* download: (a) does the runtime support the
quantisation format, (b) does it match the model implementation's parameter
naming, and (c) does any required fallback path (offload) work on this OS.

---

## S6. What does *not* predict the ceilingParameter count does not determine where the ceiling lies. CheXagent-8B tops out
at **bs=4** while Qwen3-VL-8B is still improving at **bs=8**, at nominally the same
size. The difference is the fixed cost each architecture carries — vision tower,
tiling, projector, per-sample image-token count — not the parameter count. Two
consequences:

* a per-sample KV budget cannot be read off the model size; it must be measured;
* any change to image resolution or prompt length changes the image-token count,
  hence the KV term, hence the ceiling. **These numbers are only valid for the
  exact preprocessing used in the study.**

---

## S7. Engineering notes that changed the results

Reported because each one silently corrupts measurements rather than failing:

1. **Silent CPU offload.** The loader's `device_map="auto"` may offload layers when
   no explicit memory budget is given. The only hint is a log line. Measured
   impact: **5×** apparent slowdown. Guard: pass
   `max_memory={0: budget, cpu: 0}` so it fails instead.
2. **Quantisation-aware layer placement was ignored.** Our loader used to
   discard the caller's memory budget whenever a quantisation config was present,
   forcing `{"": 0}`. Loading a 30B model then pushed all 882 shards at the GPU and
   thrashed for hours instead of failing; with the budget honoured it fails in
   **2.3 s**. Detection cannot substitute for prevention here, because the thrash
   happens during *load*, before any per-batch guard can run.
3. **Single-image smoke tests are mandatory.** A 30B model's failure mode is not an
   error but hours of near-zero progress. Verifying "loads AND generates one image
   within budget" before any sweep converts that into a 2-second check.
4. **`max_new_tokens` must be fixed across models.** Generation length drives the
   decode phase and therefore the measured rate; differing values make the table
   incomparable.

---

## S8. Reporting caveats to state in the paper

* Throughput figures are hardware-specific (RTX 4080 SUPER 32 GB, WDDM). The
  *shape* of the curve transfers; the numbers do not.
* The 30B model uses a **different quantisation scheme** (pre-quantized
  pack-quantized 4-bit / compressed-tensors) from the bitsandbytes NF4 used
  elsewhere. Its throughput is reported for feasibility, and the comparability
  caveat is stated with it.
* Two models are serial-only; reporting a "sweet spot" for them would be
  meaningless, so a single-image rate is given instead.
* The batch sizes in Table S3 are the ones used for the main-text experiments, so
  the reported run times are reproducible from this table alone.
