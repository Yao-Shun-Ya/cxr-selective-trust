# 48 — The Three-Layer 6K Evaluation: Authoritative Protocol

> Date: 2026-10-05. Machine: **RTX 4080 SUPER 32 GB** (32760 MiB) / 64 GB DDR5 /
> Windows 11 WDDM.
> Supersedes the per-dataset execution order in `docs/41_6k_curated_benchmark_plan.md`
> (which named no per-layer design and had no runner for PadChest or Open-i).
> Companion documents: `docs/49_data_manifest_contract.md` (manifest contract),
> `docs/44_batch_sweep_matrix.md` (batch sweet spots), `docs/46_engineering_insights_supplementary.md`
> (throughput).
>
> Everything numeric below is either read out of the code or printed by a command
> that was actually run; each figure names its source. Nothing here is estimated
> by hand.

---

## 1. The three layers

Each layer is one dataset, one question, one runner, one output directory. The
one-button driver is `program/scripts/run_all_layers.ps1`.

| | **NIH** | **PadChest** | **Open-i** |
|---|---|---|---|
| Question | in-domain: how much synthetic degradation before the selective signal breaks | cross-hospital: does the framework transfer to a different country and scanner population | free text: does it survive report generation, where the failure mode is hallucination rather than miscalibration |
| Manifest | `data/manifests/nih_eval_1732.csv` | `data/manifests/padchest_shift.csv` | `data/manifests/openi_rrg.csv` |
| Images | 1,732 | 1,500 | 947 |
| Splits run | `internal_test,source_val` | `internal_test` | `external_report` |
| Images root | `data/nih/images` | `data/padchest` | `data/open-i` |
| Cells / image | shift **19**, reliance **4**, precision **4**, ablation **10** | **2** | **2** |
| Runner(s) | `scripts/08_shift_benchmark.py`, `scripts/06_image_reliance.py`, `scripts/21_prompt_ablation.py` | `scripts/43_run_padchest_shift.py` | `scripts/44_run_openi_report.py` |
| Output dir | `outputs/matrix/nih_eval/<model>/` | `outputs/matrix/padchest/<model>/` | `outputs/matrix/openi/<model>/` |
| Calls (5 models) | 237,284 | 15,000 | 9,470 |
| Est. wall clock (5 models) | ≈ 133.0 h | ≈ 6.7 h | ≈ 4.3 h |

Layer totals above are the sum of the per-stage `est` column printed by
`run_all_layers.ps1 -DryRun`; the grand total it prints is **261,754 model calls,
143.9 h (6.0 days)** — re-derived 2026-10-08 from measured per-`(model, layer,
stage)` rates after the chexagent greedy-decoding and medgemma_15 2048-token
fixes moved it down from 173.6 h; see §7 «2026-10-06» and docs/50 §2.11.

### 1.1 NIH — in-domain synthetic shift (1,732 images)

* `shift` — **19 cells/image**: clean + 6 degradations (blur, noise, JPEG,
  brightness, contrast, resolution) × 3 severities. Runner `08_shift_benchmark.py`.
  Each model: `1,732 × 19 = 32,908` calls.
* `reliance` — **4 cells/image**: `original` / `blank` / `blur` / `shuffle`.
  Runner `06_image_reliance.py`. Each model: `1,732 × 4 = 6,928` calls.
* `precision` — the bf16 counterpart of `reliance` (**4 cells/image**), run only
  for the models whose primary configuration is 4-bit
  (`medgemma_4b_it`, `medgemma_15_4b_it`, `qwen25_vl_7b_instruct`, see
  `$FourBitModels` in `run_all_layers.ps1`). A bf16 8B model at batch 4 already
  sits at 29.7 GB on a 32 GB card, so the control is not run for a model that is
  already bf16 — that would compare bf16 with itself.
* `ablation` — prompt ablation (**5 variants × 2 conditions = 10 cells/image**),
  `MedGemma-4B` only (`$AblationModels`). Runner `21_prompt_ablation.py`:
  `1,732 × 10 = 17,320` calls.

The NIH evaluation manifest is **evaluation-only**: 1,054 `internal_test` +
452 `source_val` + 226 `internal_train` donor rows. See
`docs/49_data_manifest_contract.md` §5 for why the 226 rows exist and why the
analysis never reads them.

### 1.2 PadChest — cross-hospital native shift (1,500 images)

* **2 cells/image**: `original` + `blank` (flat grey).
* Runner: `scripts/43_run_padchest_shift.py`, invoked with
  `--conditions original blank --split internal_test`.
* No synthetic degradation grid. PadChest's scientific value is that it *is*
  a different hospital in a different country (different scanners, acquisition
  parameters, population and label vocabulary); that domain gap is already in
  the pixels. The `original − blank` gap is the out-of-domain image-reliance
  signal.
* Output schema is deliberately byte-compatible with `06_image_reliance.py`
  (`image_reliance_predictions.jsonl`, `image_reliance_per_image.csv`,
  `image_reliance_summary.json`, plus `blank_images/<image_id>` kept for audit),
  so `18`, `19` and `42` consume it unchanged.
* Known gap, must be stated and never computed: **PadChest's 120-label vocabulary
  contains no pneumonia label**, so a PadChest pneumonia AUROC does not exist.
  The debug sweep reports this as WARN 1 (§6).

### 1.3 Open-i — external report generation (947 images)

* **2 cells/image**: `original` + `blank`.
* Runner: `scripts/44_run_openi_report.py`, invoked with
  `--conditions original blank --split external_report --max-new-tokens 192`.
* The primary output is the generated **prose report**, not a label
  distribution. Open-i has a reference report (`reference_text`) and **no
  per-finding labels** — all 8 label columns carry `unknown`, which is the
  honest value because nothing was annotated. The 8-label AUROC machinery does
  not apply, and scoring it against a column of `unknown` would produce a
  metric-shaped number with no meaning.
* Metrics come from comparing the generation to the reference: **omission**
  (findings the reference asserts and the generation does not) and
  **hallucination** (findings the generation asserts and the reference does
  not), both produced by one labeller applied to *both* texts so a systematic
  labelling bias cancels instead of inflating one side.
* The `blank` control on a generation task is the sharpest image-reliance probe
  available: a model that writes a fluent report from a grey rectangle is
  generating from its language prior, and the real-vs-blank overlap quantifies
  what was ever image-driven.

---

## 2. Why each layer runs a different design

This is the central scientific decision in the 6K design, so it is written out
rather than implied.

**The decision.** The three layers do not share one experimental grid. NIH runs
the 19-cell synthetic-shift matrix plus reliance, precision control and prompt
ablation; PadChest runs native shift with a single blank control (2 cells);
Open-i runs report generation with a single blank control (2 cells).

**The reason in one sentence: compute moved off experiments that do not answer a
question.** A synthetic degradation grid answers *"how much artificial image
degradation does this model tolerate"*. That question is real, and NIH answers it
— with 19 cells per image, on the in-domain set where the degradation axis is
unconfounded. Re-applying blur, noise and JPEG on top of PadChest does not answer
a second question: it answers the same synthetic-robustness question again, on a
cohort whose scientific value is precisely that it did **not** receive any
artificial perturbation. The pixels already carry the shift. Likewise, a
classification grid on Open-i would score against `unknown`, because Open-i has
reference prose and no per-finding annotation.

So the budget was re-allocated rather than reduced:

| Design | Cost | Verdict |
|---|---|---|
| 19-cell synthetic grid on all three layers, 5–7 models | **≈ 355.3 h (14.8 days)** for 7 models, ≈ 175.0 h for 5 | answers the synthetic question three times, and on Open-i against `unknown` labels |
| Per-layer design as implemented, 5 models | **143.9 h (6.0 days)**, 261,754 calls | NIH answers synthetic shift, PadChest answers transfer, Open-i answers hallucination |

Both figures are reported together on purpose. The per-layer design is **not a
shortcut**: the NIH layer still runs the full 19-cell grid on the full 1,732-image
evaluation set for all five models (5 × 32,908 calls), which is where the
synthetic-robustness evidence lives. What was dropped is 17 additional cells on
each of the 1,500 PadChest images and each of the 947 Open-i images — i.e.
`17 × 1,500 = 25,500` and `17 × 947 = 16,099` cells per model, or **127,500 and
80,495 model calls across the five-model matrix** — that would have re-measured an
already-measured axis and, on Open-i, measured it against something that is not
ground truth.

For scale, two further configurations, both now reproducible:

* **All 7 models × all 3 layers with the 19-cell synthetic grid: 355.3 h (14.8 days).**
* **NIH layer only, all 7 models: 223.5 h.**

Reproducibility, and one bug this section used to hide
------------------------------------------------------
Every hours figure here is now produced by **two independent computations that
agree exactly**:

* `powershell -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1 -DryRun`
  — the delivered driver, which computes 143.9 h / 261,754 calls for the default
  five-model three-layer run and 245.7 h / 351,214 calls for all seven models.
* `.venv-vl\Scripts\python.exe scripts\47_cost_table.py` — a standalone cost table
  using the same cost model, which reproduces both figures line by line and adds
  the configurations the driver is not run in.

This matters because the driver's estimate was wrong until it was checked. Its
formula divided by a hard-coded 1.5 "batching speedup", but the measured rates it
consumes are *already* sweet-spot (batched) rates, so the division counted the
batching benefit twice and reported **125.4 h for a run that was then costed at
187.9 h** — an estimate 1.5x too optimistic. That is precisely the failure mode that
made an earlier run print `est 7m` and take 1.6 h (see `program\第一次跑.md`). The
same 1.5x bug would today turn 143.9 h into ~96 h. The earlier quoted alternatives
were wrong by the same factor: the all-layers counterfactual was 524 h, not 621 h,
and the seven-model NIH-only figure was 270.4 h, not 309.9 h. **Both of those
figures have since been re-derived** (355.3 h and 223.5 h) after the rate tables were
split per `(model, layer, stage)` on 2026-10-06 and again after the chexagent and
medgemma_15 rate fixes on 2026-10-08; the numbers in this paragraph are
kept as the record of that correction, not as current quotes.
All of them were corrected across `docs/41`, `docs/48`, `docs/49`, `README.md`,
`CONTINUE_HERE.md` and `HANDOVER.md`.

The lesson is worth keeping: an ETA is a measurement, not a forecast. If it is
not produced by a command you can re-run, it will drift.

The two slowest measured models — `llava_med_v1_5_7b` (1.66 s/call) and
`internvl3_5_8b` (6.03 s/call) — are therefore **left out of the default matrix**
and are opt-in via `-Models` (§4.6). They remain fully runnable.

**What each layer buys, stated as a claim it can support**

1. NIH (in-domain, 19 cells): the degradation response curve, and whether
   selective metrics degrade before accuracy does. NIH is also the only layer
   with enough per-label positives for AUROC and selective metrics.
2. PadChest (cross-hospital, 2 cells): threshold transfer — calibrate on
   `source_val` in the source domain, freeze, apply unchanged to a different
   country. A framework that only looks calibrated in-house is not a framework.
3. Open-i (free text, 2 cells): omission and hallucination rates under
   generation, plus the reference-vs-blank overlap.

---

## 3. Crash safety

The full run is **>5 days of continuous GPU time** (and >10 days if the opt-in
models are added). A power cut, a driver reset or a Ctrl-C during that window is
a realistic event, not an edge case, so resumability is a design requirement.

### 3.1 How `DurableJsonl` works

Module: `program/src/cxr_trust/inference/durable.py` (class `DurableJsonl`,
427 lines). API:

```python
from src.cxr_trust.inference.durable import DurableJsonl

with DurableJsonl(out_path, key=lambda r: (r["image_id"], r["condition"])) as out:
    pending = [j for j in jobs if out.needs(j.image_id, j.condition)]
    for batch in chunks(pending, batch_size):
        rows = run(batch)        # may raise; nothing is written
        out.write_many(rows)     # append -> fsync -> atomic progress marker
    print(out.stats())
```

Written path, per batch:

1. **Append** every new row to the JSONL.
2. **`os.fsync`** the append handle (`_fsync_file` flushes Python buffers *and*
   forces the write through the OS cache). A bare `f.flush()` only hands bytes to
   the operating system, so on Windows they can still sit in the file cache and a
   power loss loses them.
3. **Atomically rewrite the sidecar** `<name>.jsonl.progress.json`
   (mkstemp → fsync → `os.replace`) recording `bytes` = the durable byte offset,
   plus `records`, `appended_this_run` and `truncated_bytes_on_open`.

On open, before any work is scheduled:

* **Torn tail truncation.** If the file is longer than the durable offset, the
  file is `truncate()`d back to it. Everything past the last durable marker is
  suspect (either a complete row that was never fsynced, or half a line); a torn
  row would corrupt every downstream aggregate, so it is dropped and the work is
  simply redone.
* **Missing or absurd sidecar.** No sidecar (first run after the upgrade, or it
  was deleted) → the file is trusted as far as its last complete line
  (`_usable_size()`). A sidecar pointing past the end of the file, or a
  non-integer one, is ignored rather than believed. Neither case crashes.
* **Dedupe.** Rows are collapsed to exactly one row per key `(image_id,
  condition[, severity])`, preferring the successful attempt over a failed one
  and otherwise the newer attempt. Order is preserved. Rows whose key cannot be
  computed are kept rather than silently dropped.
* **Retry semantics.** A row whose `error` field is non-empty is treated as *not
  done* and re-attempted (`retry_failed=True` default), because a transient CUDA
  error must not become a permanent hole in the data. The reported resume count
  is therefore "cells already durable", not "lines in the file".
* **Replace, not duplicate.** `write_many` replaces an existing row with the same
  key (one atomic rewrite) instead of appending beside it, and repeats inside one
  batch are collapsed last-one-wins. Without that, retrying a failed cell inside
  a single run left the file holding both the failure and the success, and any
  reader that counted rows or took the first row would report the failure.

`stats()` returns `records_on_disk`, `already_done_this_run`,
`failed_rows_on_disk`, `appended_this_run`, `replaced_this_run`,
`truncated_bytes_on_open`, `rows_collapsed_on_open`, `durable_bytes`.

### 3.2 What every runner guarantees, and which runners do not

| Runner | Durable writer | Guarantee |
|---|---|---|
| `06_image_reliance.py` | **yes** | resumes from the durable offset |
| `08_shift_benchmark.py` | **yes** | resumes from the durable offset |
| `43_run_padchest_shift.py` | **yes** | resumes from the durable offset |
| `44_run_openi_report.py` | **yes** | resumes from the durable offset |
| `10_run_chexpert_rrg.py` | **yes** | adopted later. It used to open the file with `"a" if done else "w"`, so a file holding only failed rows looked empty and was wiped on the next run |
| `21_prompt_ablation.py` | **yes** | adopted later. `raw_outputs.jsonl` and the parsed file are two durable streams, so the audit trail can never disagree with the parsed records about how many calls happened |

Both adopted `DurableJsonl` later than the other four; the flush-only mode they
once had no longer exists — `45_debug_sweep.py` rejects any non-durable
prediction writer as a hard **error** (§6.1.4). One caveat remains: without
`--resume`, `21_prompt_ablation.py` opens the cell file in `"w"` mode and redoes
the cell from scratch — always let the driver pass the flag so a completed cell
is not overwritten.

Guarantees common to all six durable runners: a Ctrl-C or crash loses **at most
the in-flight batch**; the next invocation continues instead of redoing work; a
torn final line cannot survive a reopen; and a failed row is retried rather than
recorded as a permanent hole.

### 3.3 How to resume

Nothing special is required. **Re-run the same command.** Every stage is
independently resumable:

```powershell
cd D:\Paper\program
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1
```

* The orchestrator skips any stage already recorded complete in the state file
  (it prints `[skip] nih/medgemma_4b_it/shift (already complete)`).
* A stage that was *interrupted* is not marked complete, so it is re-entered and
  the runner itself resumes from the durable byte offset.
* `-Reset` deletes the state file so every stage is re-entered. It does **not**
  delete predictions; the durable writer still skips completed cells. To redo a
  stage from scratch, delete that stage's `*_predictions.jsonl` **and** its
  `.progress.json` sidecar.
* Interrupting mid-run is safe by design; this is stated in the driver's own
  output: *"Ctrl-C is safe at any point: every stage fsyncs each batch and
  records a durable byte offset, so a re-run continues instead of redoing work."*

### 3.4 The silent-failure guard in the driver

The failure shape this project has actually lost a run to is *"the script exited
0 and produced work that was wrong or empty."* `run_all_layers.ps1` therefore
does not trust the exit code alone. After each stage that exits 0 it calls
`Test-StageOutput`, which samples the **first 200 lines** of that layer's
predictions file and counts rows carrying an `"error": "..."` field. Tolerance is
`max(2, 1% of sampled rows)`:

* within tolerance → `[done]` and the stage is recorded in the state file;
* above tolerance → `[BAD ]`, the stage is **not** recorded complete, the top
  error message is printed, and the poisoned predictions file is **deleted** so a
  re-run redoes the stage from scratch rather than resuming on top of bad data;
* a missing or empty output file is also `[BAD ]`.

The same failure mode is defended at the batch level: the batch fallback in
`06`, `08`, `43` and `44` catches only

```python
_BATCH_REJECTION_ERRORS = (TypeError, ValueError, AttributeError, AssertionError,
                           NotImplementedError, KeyError, IndexError)
```

— i.e. only errors that mean *"this wrapper cannot batch"*. Anything else (CUDA,
driver, OOM) propagates, so the stage exits non-zero and is never recorded as
complete. Catching broad `Exception` there would convert a driver or OOM failure
into a file full of per-row error strings with exit code 0, which is the second
of the two silent-failure classes this project has already been bitten by (the
first was InternVL's wrapper assuming `Image.open(path)` while callers passed PIL
objects).

---

## 4. How to run it

All commands are from `D:\Paper\program`.

### 4.1 Preflight (CPU only, ~20 s)

```powershell
.\.venv-vl\Scripts\python.exe scripts\39_preflight.py
.\.venv-vl\Scripts\python.exe scripts\39_preflight.py --gpu-only
```

`scripts/39_preflight.py` checks, per model: that the interpreter the orchestrator
will use exists; that its torch / transformers versions match what the wrapper
needs; that the checkpoint directory is complete (weights + `config.json` +
tokenizer); the current GPU state (`nvidia-smi --query-gpu=memory.used,memory.total`)
and whether stray `python` processes are running. **CPU only — no weights are
loaded, no VRAM is used.** The first full run lost ~1 h to two failures a
20-second check would have caught (Qwen3-VL ran under the wrong interpreter and
did not recognise the `qwen3_vl` architecture; InternVL and LLaVA-Med loaded fine
but generated an error string for every row).

### 4.2 Debug sweep (read-only)

```powershell
.\.venv-vl\Scripts\python.exe scripts\45_debug_sweep.py
.\.venv-vl\Scripts\python.exe scripts\45_debug_sweep.py --strict   # warnings also exit 1
```

Checks: every Python file parses **and has no UTF-8 BOM**; every argparse
`choices=` list matches the model registry; every manifest an orchestrator
references exists and carries the columns the stage scripts read; every image the
evaluation manifests reference is on disk; label columns are the strings
`positive`/`negative`; no label column is constant; local imports resolve; every
script the orchestrators invoke exists. It writes its findings to
`outputs/state/debug_sweep.json` and returns 1 on errors (and on warnings with
`--strict`). Current state: **0 errors, 2 warnings** (see §6 for the two
warnings and their resolutions).

### 4.3 Crash-safety tests

```powershell
.\.venv-vl\Scripts\python.exe scripts\41_test_durable.py
.\.venv-vl\Scripts\python.exe scripts\41_test_runner_resume.py
.venv-vl\Scripts\python.exe scripts\49_test_legacy_durable.py
```

Both are CPU-only and take seconds; no model is loaded.

* `41_test_durable.py` — **37 assertions, all passing** (`durable jsonl: 37
  passed, 0 failed`). Covers the torn tail, an unflushed tail, retry-vs-done
  semantics, a missing sidecar, an absurd (past-EOF) sidecar, an exception raised
  mid-stage, a retry that must not accumulate duplicates, and a no-op dedupe.
* `49_test_legacy_durable.py` — **38 assertions, all passing**: the four older runners (06, 08, 10, 21) through their real CLIs, each clean, killed mid-run, resumed, and re-resumed to a byte-identical no-op.
* `41_test_runner_resume.py` — **40 assertions, all passing** (`runner
  crash/resume: 40 passed, 0 failed`). It really starts the 43 and 44 CLIs as
  **subprocesses with a stubbed model** (`scripts/_stub/stub_model.py`, activated
  by `RUNNER_STUB=1`) and verifies end to end: a clean run; a **HARD failure
  mid-run must exit non-zero** and the resume must complete with no duplicates;
  a simulated power cut plus resume; torn-tail truncation; and a third resume
  being a **byte-identical no-op**.

### 4.4 Plan only, nothing runs

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1 -DryRun
```

Prints per-stage call counts and hour estimates, the data manifest of each layer
and the TOTAL, then exits 0. Current output: **`TOTAL: 261,754 model calls,
143.9h (6.0 days of GPU time)`**, with a `rate src` column marking each stage's
rate as `measured` / `proxy-nih` / `default`.

### 4.5 Smoke run (minutes, on the GPU)

```powershell
# 3 layers, 4 images each, one fast model, no ablation -- ISOLATED (see 4.5.1)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1 `
  -Models medgemma_4b_it -Layers openi,padchest -Limit 4 -NoAblation `
  -OutputRoot outputs\pretest -StateFile outputs\state\pretest_state.json

# or a single runner directly, one study, both conditions
.\.venv-vl\Scripts\python.exe scripts\44_run_openi_report.py --model medgemma_4b_it `
  --limit 1 --manifest data\manifests\openi_rrg.csv --images-root data\open-i `
  --output-dir outputs\test\openi_smoke
.\.venv-vl\Scripts\python.exe scripts\43_run_padchest_shift.py --model medgemma_4b_it `
  --limit 1 --manifest data\manifests\padchest_shift.csv --images-root data\padchest `
  --output-dir outputs\test\pc_smoke
```

#### 4.5.1 A partial run MUST be isolated, and the driver enforces it

A `-Limit N` run and the full run must not share **either** the predictions root
**or** the state file. The reason is not tidiness:

* The state key is `layer/model/stage` and does **not** encode the image limit. A
  `-Limit 100` run therefore marks every stage complete, the full run skips them,
  and the study is silently capped at 100 images — with nothing in the logs to
  say so.
* If the roots are shared, the partial run's rows sit inside the real run's
  directories, where the durable writer treats them as finished work.

`run_all_layers.ps1` refuses any partial run that shares **either** half with the
production run, with **exit 3**. The complete truth table (all four flag
combinations plus the two full-run cases):

| `-Limit` | `-OutputRoot` | `-StateFile` | verdict |
|---|---|---|---|
| no | default | default | **ALLOW** — the full run, what the script is for |
| no | isolated | default | **ALLOW** — full set into another folder |
| no | any | isolated | **ALLOW** — full set, separate bookkeeping |
| yes | isolated | isolated | **ALLOW** — a pre-run test, fully contained |
| yes | **default** | any | **REFUSE** — partial results land in the production root |
| yes | any | **default** | **REFUSE** — stage keys ignore `-Limit`, so the full run would skip everything this run completed |

```
-Limit 100                                             -> exit 3, refused
-Limit 100 -OutputRoot outputs\pretest                 -> exit 3, refused (state shared)
-Limit 100 -StateFile outputs\state\pretest_state.json -> exit 3, refused (root shared)
-Limit 100 -OutputRoot outputs\pretest `
          -StateFile outputs\state\pretest_state.json  -> allowed, fully isolated
```

The third refusal was **missing from the first version of this guard and from the
first version of its test**: a partial run with the production root but an
isolated state file passed and wrote its rows into `outputs\matrix\`, where the
durable writer then counted them as finished work for the full run. The guard had
been written by enumerating the illegitimate cases branch by branch, and one
quadrant of a four-quadrant space was not enumerated. It is now a single condition
over the whole space (`-Limit` AND (root default OR state default)), and
`45_debug_sweep.py` section 8 tests **all six** cases rather than four, because
enumerating the cases is exactly what let one go missing.

`45_debug_sweep.py` therefore cannot regress unnoticed.

```powershell
# pre-run test: 100 images, two models (fast + the invariant one), NIH layer
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1 `
  -Models medgemma_4b_it,qwen25_vl_7b_instruct -Layers nih `
  -Limit 100 -OutputRoot outputs\pretest -StateFile outputs\state\pretest_state.json

# the real run afterwards: both defaults, disjoint from the above
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1
```

Because the roots differ, the pre-run test is deliberately **not** a head start:
those 100 images are computed twice. Sharing the root to save that work would
reintroduce exactly the collision the guard exists to prevent.

### 4.6 The full run

```powershell
cd D:\Paper\program
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1
```

Useful variations (all flags are declared in the driver's `param` block):

| Flag | Effect |
|---|---|
| `-Models a,b,c` | any subset of the 7 registered models; unknown names exit 2 |
| `-Layers nih,padchest` | any subset of `nih`, `padchest`, `openi`; unknown names exit 2 |
| `-Limit N` | images per layer (0 = each layer's full evaluation set) |
| `-BatchSize N` | override every model's measured sweet spot |
| `-NoAblation` | drop the MedGemma-4B prompt-ablation stage |
| `-Reset` | delete the state file and re-enter every stage |
| `-DryRun` | plan and exit |

The default matrix is **5 models**: `medgemma_4b_it`, `medgemma_15_4b_it`,
`qwen25_vl_7b_instruct`, `qwen3_vl_8b_instruct`, `chexagent_8b`. Adding the two
opt-in models:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1 `
  -Models chexagent_8b,qwen3_vl_8b_instruct,qwen25_vl_7b_instruct,medgemma_4b_it,`
medgemma_15_4b_it,internvl3_5_8b,llava_med_v1_5_7b
```

`llava_med_v1_5_7b` and `internvl3_5_8b` are omitted from the default matrix
because they are the two slowest measured models (1.66 s/call and 6.03 s/call
respectively). Both remain runnable via `-Models`; a full 7-model NIH-only run is
recorded at 270.4 h.

Within every layer the driver runs the **slowest model first**, so the models
most likely to be interrupted finish early and any later interruption happens on
cheap work.

### 4.7 Interpreter routing and state

* `qwen3_vl_8b_instruct` runs under **`.venv-modern`** (transformers ≥ 4.57 is
  required for the `qwen3_vl` architecture); every other model runs under
  **`.venv-vl`**. The driver picks the interpreter per model.
* Quantisation: `--load-in-4bit` is passed only for `medgemma_4b_it`,
  `medgemma_15_4b_it` and `qwen25_vl_7b_instruct` — only these have a *measured*
  4-bit number; the rest stay bf16 as benchmarked.
* Batch size defaults to the measured sweet spot (`medgemma_*` 64, `qwen*` 8,
  `chexagent_8b` 4, `internvl3_5_8b` 1, `llava_med_v1_5_7b` 1), capped at 8 on
  PadChest and Open-i because those are two-cell runs where the bf16 VRAM ceiling
  is the binding constraint.
* **State file: `D:\Paper\program\outputs\state\run_all_layers_state.json`**
  (override with `-StateFile`). It holds the list of completed
  `layer/model/stage` keys. Debug sweep output:
  `outputs\state\debug_sweep.json`. Environment report: `outputs\env_report.json`.

### 4.8 Measured throughput used by the estimate

Seconds per model **call** (one image + one prompt) at the 256-token budget:

| Model | s / call |
|---|---|
| `medgemma_4b_it` | 0.24 |
| `medgemma_15_4b_it` | 0.24 |
| `llava_med_v1_5_7b` | 1.66 |
| `qwen25_vl_7b_instruct` | 2.09 |
| `qwen3_vl_8b_instruct` | 3.55 |
| `chexagent_8b` | 5.97 |
| `internvl3_5_8b` | 6.03 |

The driver's estimate multiplies by a 1.5 batching gain and a 1.2 overhead factor
and adds 35 s stage startup plus a per-model load cost (`chexagent_8b` 75 s,
`internvl3_5_8b` 30 s, `llava_med_v1_5_7b` 220 s, `qwen3_vl_8b_instruct` 45 s).

---

## 5. What is analysed afterwards, and by which script

All analysis is **CPU only** — no GPU is needed once the predictions exist.

| Analysis | Script | Reads |
|---|---|---|
| Cross-hospital threshold transfer, degradation delta with a patient-cluster bootstrap interval, out-of-domain image reliance, and the Open-i report layer (omission / hallucination / reference-vs-blank overlap) | `scripts/42_geographic_shift_analysis.py` | `outputs/matrix/nih_eval/<model>`, `outputs/matrix/padchest/<model>`, `outputs/matrix/openi/<model>` |
| Selective metrics v2 (risk–coverage, patient/finding-aware) | `scripts/18_selective_metrics_v2.py` | any layer's predictions + its manifest |
| Accept / clarify / abstain three-way policy | `scripts/19_accept_clarify_abstain.py` | selective-metric outputs |
| NIH synthetic-degradation curve (single dataset) | `scripts/09_shift_analysis.py` | `08` output |
| Report-level judge (omission/hallucination labeller) | `scripts/20_rrg_labeler_eval.py` | reference + generated text |

The exact command the driver prints at the end of every run:

```powershell
.\.venv-vl\Scripts\python.exe scripts\42_geographic_shift_analysis.py `
    --source-dir outputs\matrix\nih_eval\medgemma_4b_it `
    --source-manifest data\manifests\nih_eval_1732.csv `
    --target-dir outputs\matrix\padchest\medgemma_4b_it `
    --target-manifest data\manifests\padchest_shift.csv `
    --report-dir outputs\matrix\openi\medgemma_4b_it `
    --output-dir outputs\geographic\medgemma_4b_it --n-boot 1000
```

Two honesty requirements `42` enforces in its own outputs:

1. **Threshold transfer** — the decision threshold is calibrated on `source_val`
   *inside the source domain*, frozen, then applied unchanged to PadChest; AUROC,
   sensitivity, specificity and realised coverage are reported in both domains.
2. **Every rate records its `evidence_method`** — finding labels are extracted
   from free text by a labeller, and the judged labeller (`--judge-model`) is
   preferred while the keyword+negation rule labeller is the fallback. Whichever
   was used is written into every output together with the judge's identity,
   because a rate computed by a rule that mis-reads negation is a different
   number from one computed by an adjudicator, and the paper must not silently
   swap them.

**The judged labeller is verified to work on the Open-i manifest**, not assumed
to be. The repository holds no text-only causal LLM, so the judge has to be one
of the multimodal checkpoints; `_load_judge` tolerates that by falling back from
`AutoModelForCausalLM` to the image-text classes. Run on this data:

```
.venv-vl\Scripts\python.exe scripts\20_rrg_labeler_eval.py --mode rule judge report \
    --reports data\manifests\openi_rrg.csv --limit 20 \
    --judge-model models\Qwen2.5-VL-7B-Instruct --judge-load-in-4bit \
    --output-dir outputs\rrg_evidence\openi_eval
```

Observed: the manifest's `reference_text` column is auto-detected, the judge
loaded via `AutoModelForImageTextToText` and labelled **140 rows with 0
unparsed**, and the agreement table reports `rule_vs_judge n=140
agreement=0.936 kappa=0.235`. That kappa is low *by design and must be reported
as such*: high raw agreement with a near-zero kappa means the two labellers agree
on the majority class (`absent`) and disagree where it matters, which is exactly
the argument for preferring the judge and for adding a human gold set. Do not
quote the 0.936 without the 0.235.

Two caveats recorded honestly: `--mode all` does **not** exist — the docstring
used to claim it did and argparse then rejected the command; the correct form is
`--mode rule judge report`. And the judge must not be the model under evaluation,
so a MedGemma-under-test run needs a judge from a different family (the example
uses Qwen2.5-VL).

Degradation deltas are reported **with a patient-cluster bootstrap interval**
(`--n-boot 1000`). A delta of −0.04 whose interval spans zero is not "the model
degrades across hospitals"; it is noise.

---

## 6. Known limitations and traps

### 6.1 Scientific limitations to state, never to paper over

1. **PadChest has no pneumonia label.** Its 120-label vocabulary contains no
   pneumonia entry, so a PadChest pneumonia AUROC **does not exist**. This is a
   data limitation, not a missing number to be filled in. It is the debug sweep's
   first warning.
2. **The 226 `internal_train` donor rows.** `nih_eval_1732.csv` contains 1,054
   `internal_test` + 452 `source_val` + **226 `internal_train`**. Those 226 exist
   *only* as donors for the `no_finding` repair (§6.4). The analysis scripts
   select `internal_test`/`source_val` and never read them; any script that
   reports an aggregate over the whole file without the `--split` filter is
   including training data it must not include.
3. **Open-i has no per-finding labels.** All 8 label columns are `unknown` by
   design. Any classification metric computed on Open-i is meaningless.
4. **RESOLVED — every runner is now crash-durable.** `21_prompt_ablation.py`
   (flush-only at the time this section was written) and
   `10_run_chexpert_rrg.py` have both been moved onto `DurableJsonl`, and
   `45_debug_sweep.py` now treats that as a hard check rather than a warning, so
   the regression cannot come back silently. See §3.2.

### 6.2 `--limit` is the first N rows, which is NOT a sample

This is the trap that produced wrong numbers once already. `--limit` takes the
**first N rows of the file**. `nih_shared_5000.csv` is ordered by `image_id` and
contains 3,268 `internal_train` rows, so a large `--limit` silently evaluates on
**training data** — with no warning. Measured consequence on the old workflow:
`--limit 300` yielded only 138 evaluation images and 12 pneumonia positives.

Mitigation in the code, all three scripts: `06_image_reliance.py`,
`08_shift_benchmark.py` and `21_prompt_ablation.py` now take a `--split` filter
and **shuffle before limiting** (`--limit` + `--split` together make a real
sample). `43_run_padchest_shift.py` and `44_run_openi_report.py` do the same. The
driver always passes `--split internal_test,source_val` for NIH,
`--split internal_test` for PadChest and `--split external_report` for Open-i.

### 6.3 Two import paths to the same registry file

`src/` has no `__init__.py`, so both of these import the *same file* as **two
separate module objects**:

```python
from src.cxr_trust.models.registry import MODEL_PATHS   # used by the runners
from cxr_trust.models.registry import MODEL_PATHS       # used by 39_preflight / 45_debug_sweep
```

They are distinct objects: **patching one does not affect the other.** This has
already hidden one bug. Consequence for tests: a stub must be installed **after**
the runner's own imports, which is why the resume test uses the `RUNNER_STUB=1`
seam (`scripts/_stub/sitecustomize.py`) rather than monkey-patching a registry
from the outside. This is the debug sweep's second warning.

### 6.4 The `no_finding` repair, and why it was needed

`nih_shared_5000.csv` has `label_no_finding` constant `"negative"` for **all
5,000** rows. Cause, traced: the sampler's tail-filling step orders candidates by
how many under-fed disease labels they cover, so plain normals sort last and the
budget runs out before any of them is selected. A constant label makes AUROC and
every selective metric for `no_finding` **undefined** while still printing as a
number in a table.

`scripts/40_build_eval_manifest.py` builds `nih_eval_1732.csv` and repairs this:
`no_finding` positives go from **0 → 347** (20% of the set), taken from the
extracted pool with preference for images already inside the evaluation splits,
and the least load-bearing rows are dropped to make room. The script now **exits
non-zero if the built manifest has any constant label column**. Full detail in
`docs/49_data_manifest_contract.md` §5.

### 6.5 The external-manifest adapter was rewritten

`scripts/25_format_external_manifests.py` is now negation-aware. The old version:

* wrote **0/1 labels** — and the label matrix tests `astype(str) == "positive"`,
  so every row silently became negative;
* omitted `patient_id` and `split` entirely;
* used bare substring matching, so `"no pleural effusion"` counted as a positive
  pleural effusion.

Recorded effect on the consolidation label: **false positives 62 → 82 corrected,
i.e. 20 bogus positives removed.** (This figure is the recorded measurement from
the rewrite; the script itself has no counter that recomputes it, so treat it as
a historical note rather than something `25_format_external_manifests.py` prints
today.) Any number produced from the pre-rewrite PadChest or Open-i manifests is
invalid. The pre-rewrite files are kept as `openi_rrg.csv.bak` (319,809 bytes)
and `padchest_shift.csv.bak` (106,676 bytes).

### 6.6 Engineering traps

1. **Never write Python files with PowerShell `Set-Content -Encoding UTF8`.**
   On Windows PowerShell 5.1 that writes a UTF-8 **BOM**, and Python then fails
   with `invalid non-printable character U+FEFF`. This actually happened in this
   repository; `45_debug_sweep.py` now checks every `scripts/*.py` and
   `src/**/*.py` for the `\xef\xbb\xbf` prefix and reports it as an ERROR.
2. **`os.replace` fails with WinError 5** if the destination file is still open by
   the same process. `DurableJsonl._rewrite` therefore closes the append handle
   before the swap and reopens it afterwards — which is also the correct order on
   POSIX, where a handle opened before the swap would keep pointing at the
   unlinked inode and silently append into nothing.
3. **The batch fallback must stay narrow.** Catch only
   `(TypeError, ValueError, AttributeError, AssertionError, NotImplementedError,
   KeyError, IndexError)` — see §3.4.
4. **Use the right interpreter per model.** `qwen3_vl_8b_instruct` → `.venv-modern`;
   everything else → `.venv-vl`. The wrong interpreter is not a warning, it is a
   silent 3-second failure per stage.
5. **Do not hand-edit `*.jsonl` or delete a `.progress.json`** expecting to lose
   only one row: deleting the sidecar makes the writer trust the file to its last
   complete line, and deleting the JSONL loses the resume record entirely.
6. **Memory budget.** A bf16 8B model at batch 4 on a 32 GB card already sits at
   29.7 GB, which is why the `precision` control is capped at batch 4 and is not
   run for models that are already bf16.

---

## 7. Changelog

| What | Where | Why |
|---|---|---|
| `run_all_layers.ps1` — one-button three-layer driver, resumable, per-stage call/time plan, `[BAD]` guard that deletes a poisoned predictions file | `scripts/run_all_layers.ps1` (new) | the three layers previously had to be driven by hand in the right order with the right flags |
| Per-layer design: NIH 19/4/4/10 cells, PadChest 2, Open-i 2 | `run_all_layers.ps1`; `docs/41` §"superseded" | compute moved off experiments that do not answer a question; ≈524 h → 187.9 h for 5 models (re-measured 2026-10-06: 173.6 h; 2026-10-08: 143.9 h) |
| `43_run_padchest_shift.py` — native-shift + blank-control runner, reliance-schema compatible | `scripts/43_run_padchest_shift.py` (new) | PadChest had no runner; the 19-cell grid was the wrong instrument for it |
| `44_run_openi_report.py` — prose-generation runner with blank control | `scripts/44_run_openi_report.py` (new) | Open-i has reference reports and no per-finding labels; AUROC does not apply |
| `42_geographic_shift_analysis.py` — threshold transfer, degradation delta with bootstrap CI, out-of-domain reliance, report layer | `scripts/42_geographic_shift_analysis.py` (new) | the transfer question needs two datasets at once and existed nowhere in the repo |
| `DurableJsonl` adopted by `06`, `08`, `43`, `44` | `src/cxr_trust/inference/durable.py` | >5-day runs; `flush()` alone loses the tail on Windows |
| `41_test_durable.py` — 37 assertions, passing | `scripts/41_test_durable.py` (new) | torn tail, unflushed tail, stale/missing sidecar, retry not duplicating |
| `41_test_runner_resume.py` — 40 assertions, passing, real subprocesses with a stub | `scripts/41_test_runner_resume.py`, `scripts/_stub/` (new) | "resumable" spans process boundaries and can only be tested by really killing and restarting a run |
| `39_preflight.py` — CPU-only environment/checkpoint/GPU check | `scripts/39_preflight.py` (new) | the first full run lost ~1 h to a wrong interpreter and unusable checkpoints |
| `45_debug_sweep.py` — repository debug sweep, 0 errors / 2 warnings | `scripts/45_debug_sweep.py` (new) | the failures that cost time all exit 0 and produce wrong or empty work |
| `25_format_external_manifests.py` rewritten negation-aware | `scripts/25_format_external_manifests.py` | old version wrote 0/1 labels, dropped `patient_id`/`split`, and read `"no pleural effusion"` as a positive |
| `40_build_eval_manifest.py` + `nih_eval_1732.csv` | `scripts/40_build_eval_manifest.py` | killed the `--limit`-on-training-data trap and repaired a constant `no_finding` column (0 → 347 positives) |
| `--split` filter + shuffle-before-limit in `06`, `08`, `21` | those scripts | `--limit` takes the first N rows, which is not a sample |

**Not yet done, explicitly:**

* ~~`10_run_chexpert_rrg.py` and `21_prompt_ablation.py` used to use flush-only
  writing — they have not adopted `DurableJsonl`.~~ **This bullet was wrong and is
  corrected here rather than annotated**: both scripts adopted `DurableJsonl`, all
  six prediction-writing runners are durable, and `45_debug_sweep.py` section 7 now
  asserts it on every run so the claim cannot go stale again. `49_test_legacy_durable.py`
  exercises the four older runners with real subprocesses (38 assertions).
* The three-layer run itself has **not** been executed end to end; all figures
  here are the driver's own plan plus the CPU-only test/sweep results.

A note on why that bullet survived so long: it had been produced by a
find-and-replace that rewrote its first clause ("still use flush-only" → "used to
use flush-only") and left the second clause behind, so the sentence contradicted
itself. Self-contradicting text is worse than stale text in an authoritative
document, because a reader cannot tell which half to believe. The guard now lives
in the sweep rather than in prose.

### 2026-10-05 — the ECE convention, the calibration universe, and three guards

| What | Where | Why |
|---|---|---|
| ECE bin convention corrected: `(0,0.1]` → `[0,0.1)` | `src/cxr_trust/evaluation/metrics.py`; the local copy in `18_selective_metrics_v2.py` deleted, that script now imports the canonical function | the bins had been built with `right=True`; the standard convention (Guo et al. 2017, sklearn) is left-closed, right-open. Not a measure-zero difference: **8.44%** of stored probabilities sit exactly on a bin edge, and **24.7%** in the RRG file. Every ECE number produced before today is on the old convention |
| Calibration sample universe made explicit | `04_evaluate.py` | probability metrics (AUROC / AUPRC / ECE / Brier) now run over every row that emitted a probability, `uncertain` rows included; classification and selective metrics (F1 / AURC / coverage) over definite calls only. The previous single `dropna` had silently removed **186 of 1,800** MedGemma `internal_test` decisions (**10.3%**) from the calibration metrics, with nothing in the code or the docs to say so |
| `n_prob` and `n_definite` columns added | `metrics_by_label.csv` | the two universes are now visible in the artefact instead of being an implicit property of the script that produced it |
| Paper numbers regenerated: MedGemma `internal_test` macro AUROC 0.612 → 0.622, ECE 0.232 → 0.220, Brier 0.198 → 0.188 | `papers/main_v2/sections/04_results.tex`; Table 2; `fig04_shift_ece_severity`; `main.pdf` rebuilt (19 pages, 0 warnings, 0 undefined references) | the corrected convention and the explicit universe both move the numbers, in the same direction; the abstract's "macro ECE 0.23" becomes 0.22 |
| Figure drift fixed: `13_make_figures.py` wrote to `figures/stage2/` while the manuscript embedded `papers/main_v2/figures/` | `13_make_figures.py` | the two directories had **no link at all**, so regenerating a figure produced a stale manuscript figure and nothing complained. `save()` now refreshes both |
| Calibration endpoints reconciled | `07_calibration_selective.py` | the `no_finding` temperature-scaling pair had its "before" endpoint from the corrected convention and its "after" endpoint from the old one. Re-running the script makes both endpoints come from one pass; `04` and `07` now agree to the digit (0.35689) |
| Three new behavioural guards: ECE convention (§9), stage-aware `[BAD]` guard loading the driver's real function (§10), calibration universe (§11) | `45_debug_sweep.py` | the sweep had been checking that files exist and parse; these three check that the *behaviour* is still the documented one. The `[BAD]` guard was previously asserted against a re-implementation rather than the driver's own function, which is the same defect one level up |
| Isolation guard asserted over **all six** flag combinations (§8) | `45_debug_sweep.py` | a `-Limit` run must separate **both** `-OutputRoot` and `-StateFile`; the driver refuses with exit 3 otherwise (see 4.5.1) |
| All six prediction-writing runners crash-safe | `41_test_durable.py` (37 assertions), `41_test_runner_resume.py` (40), `49_test_legacy_durable.py` (38) | **115 assertions, all passing**: append + fsync + atomic byte-offset sidecar, torn-tail truncation, replace-not-append on retry |
| `45_debug_sweep.py` now has 11 sections | `45_debug_sweep.py` | the three new guards above plus the six that existed |

### 2026-10-06 — the five pre-run fixes, and a cost model that measures instead of assumes

Full evidence, including the negative results worth keeping, is in
`docs/50_resume_prerun_findings.md` §2. Summary of what changed and why:

| What | Where | Why |
|---|---|---|
| `21_prompt_ablation.py` batches (`--batch-size`, `generate_batch`) | `scripts/21_prompt_ablation.py`; batch size measured by `scripts/60_measure_ablation_batch.py` | the stage was serial at 8.28 s/cell against 0.63 batched: ~38 h of the budget. Batch **32** is the measured choice for medgemma_4b (all five variants land at 0.77–0.97 s/cell, 10.1 GiB peak); anything unmeasured is capped at 8 |
| Ablation crash/resume case run at **two** batch sizes | `scripts/49_test_legacy_durable.py` | the old assertion ("the kill lands on the 2nd image") silently depended on there being one call per image. 38 source assertions, 49 executed |
| chexagent loads from an fp16 copy, with **no** `device_map`, then a verified move to the GPU | `src/cxr_trust/models/chexagent.py`; `scripts/62_convert_chexagent_fp16.py` | three stacked causes: 31.15 GiB of fp32 does not fit 30.6 GiB free; `torch_dtype` is ignored for this remote-code class (measured); and a `device_map` triggers accelerate's `caching_allocator_warmup(factor=2)`, which alone pages a 15.58 GiB model. Result: 15.61 GiB resident, 3.45 s/cell at bs=4 (that run was still beam search; greedy decoding then cut it to 0.477 s/cell, docs/50 §2.11); ~49 h saved |
| llava_med's `config.json` given the reference `text_config` (`rope_theta` 1e6) | `scripts/29_fix_llava_med_config.py` | the rewritten config had lost `rope_theta`, so transformers used 10000 while the weights were tuned at 1e6: the model loaded, ran, and answered `</s>` (1,890 of 1,900 replies empty). After the fix it answers — but still not in the contract (0/16), so the parser is *not* loosened and the model stays opt-in |
| Per-model token budget (`Get-MaxNewTokens`; medgemma_15 = 2048, was 1024) | `scripts/run_all_layers.ps1` | MedGemma-1.5 opens a reasoning channel on hard images — blank 100/100, resolution/severe 117/300 — and at bf16 on **every** image; 256 tokens truncate inside it. Measured: 256→0/8, 512→0/8, 768→6/8, 1024→8/8. The 8-image probe was too small for the precision stage: at 1024 the bf16 control still lost 3/64 replies to truncation, so the budget is **2048** (docs/50 §2.11). The other models keep 256 |
| Batches grouped by condition instead of by image | `06_image_reliance.py`, `08_shift_benchmark.py`, `43_run_padchest_shift.py`, `44_run_openi_report.py` | a batch costs what its **longest** member costs, so an image-major order made every batch contain a reasoning cell: 3.30 s/cell against 0.825 for the 256-token run. Homogeneous batches make only the hard conditions pay |
| Rate table split per `(model, layer, stage)` with a visible `rate src` | `scripts/run_all_layers.ps1`, `scripts/47_cost_table.py` | one constant per model was applied to 4-bit shift, bf16 precision *and* the serial ablation: the same run was 27.6x under on one stage and 0.33x over on another. PadChest/Open-i still have no measurement of their own and are labelled `proxy-nih` rather than presented as measured. Plan total: 187.9 h → 173.6 h → **143.9 h** |
| Coverage census travels with the metrics | `scripts/09_shift_analysis.py` → `shift_coverage_by_condition.csv`, `coverage_report.json`, COVERAGE WARNING, `--min-coverage` / `--fail-on-low-coverage`; the driver runs it after every run and folds the verdict into its RUN SUMMARY | the analysis silently dropped unparsed rows and non-binary labels, then reported `n_label_instances` over the survivors. On the pre-run, `medgemma_15_4b_it` `resolution/severe` keeps only **32%** of its label instances (68 of 100 replies did not parse) — a drop correlated with the condition, i.e. a biased subsample, not a smaller one. The gate was then exercised end to end: fail → `exit 3`, pass → `exit 0`, and a real 4-image run whose census ran on the file the stage had just written |
| Read/write key audit as a standing guard (sweep section 12) | `scripts/64_audit_row_keys.py`, called by `45_debug_sweep.py` | `59_probe_pretest_findings.py` read `probability_pos` where the schema writes `probability_positive`; `.get` returns `None` without raising, so that column was empty for every row and the comparison it fed used the status column instead, and nothing compared reads to writes. 190 written keys vs 391 read keys now: **0 near misses**. The checker runs its own positive control first — a blind checker reporting "clean" is worse than no checker — and keys a file writes itself are excluded so it does not cry wolf |
| Confidence provenance in the artifacts | `04_evaluate.py` (`n_conf_from_model` / `n_conf_imputed` / `conf_source`), `14_make_tables.py` (table08 gains `conf_source`) | no model in the matrix emits its own `confidence` (null in 26,462 of 26,462 rows), so the risk score behind `macro_aurc` is derived from the probability, with 0.5 imputed only where a definite status carries no probability (2.38% on the new-format files, **0.00%** on 04's pilot/overnight inputs). The convention is defensible; leaving it unstated was not. Six metrics artifacts were regenerated: four bit-identical, two found **stale** (produced before the 2026-10-05 calibration-universe change — they lacked `n_prob`/`n_definite`) and now current. Neither stale file feeds the paper's tables |
