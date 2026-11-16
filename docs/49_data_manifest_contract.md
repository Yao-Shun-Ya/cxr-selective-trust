# 49 — The Manifest Contract

> Date: 2026-10-05.
> Scope: the exact contract every file in `program/data/manifests/` must satisfy
> for the stage scripts to read it correctly, the three evaluation manifests with
> their verified row counts and split composition, the `no_finding` repair, the
> PadChest pneumonia gap, and an inventory of every manifest on disk.
> Companion: `docs/48_three_layer_protocol.md` (what is run on top of these files).
>
> Every row count and byte size below was read off disk, not recalled.

---

## 1. Why a contract exists

The stage scripts (`06_image_reliance.py`, `08_shift_benchmark.py`,
`21_prompt_ablation.py`, `43_run_padchest_shift.py`, `44_run_openi_report.py`,
`18_selective_metrics_v2.py`, `19_accept_clarify_abstain.py`) never read a raw
dataset CSV. They read a **manifest**: one flat table that points at local images
and carries the ground truth. The raw NIH/PadChest/Open-i files have different
column names, different label vocabularies and different label encodings, and
`25_format_external_manifests.py` / `40_build_eval_manifest.py` are the adapters
that produce a manifest from each.

A manifest that violates the contract does not raise. It produces a metric. That
is why the contract is written down and why `45_debug_sweep.py` enforces part of
it automatically.

---

## 2. Required columns

### 2.1 Columns every manifest must have

| Column | Type | Meaning | Consumer |
|---|---|---|---|
| `image_id` | string, unique | study identifier; also the resume key of every runner | all runners, `DurableJsonl` key, join key to `patient_id` |
| `image_relpath` | string | path to the image **relative to the layer's `--images-root`** | image loader |
| `dataset` | string | `nih` / `padchest` / `openi` / `chexpert_rrg` | provenance recorded into every prediction row |
| `patient_id` | string | patient / study cluster key | `42` patient-cluster bootstrap, `18`/`19` clustering |
| `split` | string | which split the row belongs to; the `--split` filter selects on it | every runner |

### 2.2 Columns a classification manifest must have

Eight columns named `label_<finding>` for the eight reported findings:

```text
label_atelectasis      label_cardiomegaly    label_consolidation
label_edema            label_pleural_effusion label_pneumonia
label_pneumothorax     label_no_finding
```

`nih_eval_1732.csv` additionally carries the six ChestX-ray14 labels that are not
part of the eight (`label_lung_opacity`, `label_mass_nodule`,
`label_emphysema`, `label_fibrosis`, `label_pleural_thickening`,
`label_hernia`); they are carried for provenance and are not reported.

### 2.3 Columns a report-generation manifest must have

| Column | Meaning |
|---|---|
| `reference_text` | the reference report (`findings_section`), used as the omission/hallucination reference |
| `default_prompt` | the prompt the runner sends when `--prompt` is not overridden |

`openi_rrg.csv` carries both. `chexpert_rrg_all.csv` is the older
report-generation manifest and carries both as well.

### 2.4 Purely informational columns

`view` is carried by the NIH, PadChest and Open-i manifests and is not part of
the contract — no stage script branches on it.

---

## 3. Labels must be the strings `"positive"` / `"negative"` — and here is why

**The requirement.** A label cell is either the exact string `positive` or the
exact string `negative`. Not `1`/`0`, not `True`/`False`, not `TRUE`, not `pos`,
not empty.

**Why.** Two independent code paths break silently otherwise.

1. **The label matrix is built by string comparison.**

   ```python
   # scripts/23_build_stratified_manifest.py:72
   return np.column_stack([(df[c].astype(str) == "positive").to_numpy() for c in cols])
   ```

   `astype(str)` means an integer column `0/1` becomes `"0"`/`"1"`, which never
   equals `"positive"`. **Every row silently becomes negative** — the manifest
   still has the right shape, the sampler still runs, and the resulting label
   counts are all zero positives with no error anywhere.

2. **The metric loaders silently drop anything else.**

   ```python
   # scripts/18_selective_metrics_v2.py:130-132
   gt_val = gt.get(f"label_{label}")
   if gt_val not in {"positive", "negative"}:
       continue
   ```

   `19_accept_clarify_abstain.py:121` has the same guard. A cell holding `1`, or
   `unknown`, or an empty string, does not raise — the finding is simply **absent
   from the metric**. A label that is absent and a label with zero positives look
   identical in most summary tables.

**What `unknown` means.** `unknown` is *not* a violation of the contract, it is
the honest value for a dataset where nothing was annotated:
`openi_rrg.csv` has all 8 label columns set to `unknown` because Open-i ships
reference reports and no per-finding labels. The consequence is that no
classification metric may be computed on Open-i — the correct reaction is to run
the report layer, not to fill the labels in.

**No label column may be constant.** A constant column makes AUROC and every
selective metric for that finding **undefined** while still printing a number.
`40_build_eval_manifest.py` now exits non-zero if the manifest it just built has
any constant label column, and `45_debug_sweep.py` reports constant columns it
finds in any manifest an orchestrator references.

---

## 4. The three evaluation manifests

### 4.1 `nih_eval_1732.csv` — NIH, in-domain (1,732 rows, 20 columns, 336,065 bytes)

| Split | Rows | Role |
|---|---|---|
| `internal_test` | 1,054 | the in-domain test set |
| `source_val` | 452 | threshold / temperature calibration inside the source domain — **never** used to fit and report on the same rows |
| `internal_train` | 226 | **donors for the `no_finding` repair only** (see §5) |
| **total** | **1,732** | unique patients: 951 |

**Scored set: 1,054 + 452 = 1,506.** The 226 `internal_train` rows are donors and
are excluded by `--split internal_test,source_val`; only the 1,506 are evaluated.

Positives per label (from `nih_eval_1732_report.json`):

| Label | Positives | Label | Positives |
|---|---|---|---|
| atelectasis | 764 | pneumonia | 206 |
| cardiomegaly | 436 | pneumothorax | 397 |
| consolidation | 542 | no_finding | 347 |
| edema | 263 | | |
| pleural_effusion | 998 | | |

**The 226 `internal_train` rows exist for exactly one reason** and must not be
reported on. `scripts/40_build_eval_manifest.py` needs `no_finding`-positive
images that are not already exclusively-negative evaluation rows; the extracted
pool supplied 183 inside the evaluation splits, so the remainder were taken as
training-pool donors and dropped in exchange for the least load-bearing
evaluation rows. The analysis scripts select `internal_test` / `source_val`
(explicitly, via `--split internal_test,source_val`) and **never read** the 226.
Any script that aggregates the whole file without a `--split` filter is including
226 training images in a reported metric.

`images_missing_on_disk: 0` in the report, and `45_debug_sweep.py` independently
confirms **1732/1732 present**.

### 4.2 `padchest_shift.csv` — PadChest, cross-hospital (1,500 rows, 14 columns, 272,787 bytes)

| Split | Rows |
|---|---|
| `internal_test` | 1,500 |

* **765 unique patients** — this is the cluster key for the bootstrap interval in
  `42_geographic_shift_analysis.py`. 1,500 images over 765 patients means images
  are **not** independent observations and a per-image bootstrap would understate
  the interval.
* Column order is `image_id, image_relpath, dataset, patient_id, view, split,
  label_<8 findings>`.
* Labels are positive/negative strings, produced by the negation-aware
  `25_format_external_manifests.py`.

**Known data limitation: there is no pneumonia label.** PadChest's 120-label
vocabulary contains no pneumonia entry, so **a PadChest pneumonia AUROC does not
exist**. This must be stated as a limitation, never computed, never filled in from
another cohort, and never silently dropped from a table without a footnote. The
debug sweep reports it as a WARN (see `docs/48` §6).

### 4.3 `openi_rrg.csv` — Open-i, report generation (947 rows, 16 columns, 417,524 bytes)

| Split | Rows |
|---|---|
| `external_report` | 947 |

* `reference_text` present for every row; `default_prompt` present for every row.
* **All 8 label columns are `unknown`**, which is the honest value: Open-i has
  reference reports and no per-finding annotation. The report layer is the only
  valid analysis on this file.
* `images_missing_on_disk: 0`; the sweep confirms **947/947 present**.

### 4.4 Image availability — all three verified present

| Manifest | Present on disk |
|---|---|
| `nih_eval_1732.csv` | 1732 / 1732 |
| `padchest_shift.csv` | 1500 / 1500 |
| `openi_rrg.csv` | 947 / 947 |

A missing image does not stop a run; it becomes a **silently skipped job**. That
is why the sweep checks this rather than trusting the download step.

---

## 5. The `no_finding` repair

**The defect.** `nih_shared_5000.csv` carries `label_no_finding` = `"negative"`
for **all 5,000 rows**. Cause, traced through the sampler:

* `stratified_select` in `23_build_stratified_manifest.py` balances the **seven
  disease labels** and treats `no_finding` as "handled separately", so it is never
  given a positive target;
* the tail-filling step then orders candidates by *novelty* — how many
  still-under-fed disease labels each candidate covers — so images positive for
  several diseases sort first and plain normals sort **last**, exactly where the
  budget runs out.

Result: **0 of 5,000** rows with `no_finding` positive, while the full pool
(`nih_full.csv`) has 60,361 of 112,120. A constant label makes AUROC and every
selective metric for `no_finding` undefined — it prints as a number and means
nothing.

**The repair.** `scripts/40_build_eval_manifest.py` enforces a floor
(`--no-finding-min`, default 0.20) by swapping in `no_finding`-positive images
from the extracted pool, preferring images already inside the evaluation splits,
and dropping the least load-bearing rows to keep the total at 1,732. Recorded in
`nih_eval_1732_report.json`:

```json
"no_finding_repair": {
  "target": 347, "before": 0, "after": 347, "swapped": 347,
  "drop_strategy": "least-load-bearing",
  "donor_scope": "whole extracted pool (183 available inside the evaluation splits)",
  "dropped_label_counts": {"atelectasis": 328, "consolidation": 56,
                           "pleural_effusion": 323, "cardiomegaly": 0,
                           "edema": 0, "pneumonia": 0, "pneumothorax": 0},
  "disabled": false
}
```

So `no_finding` goes **0 → 347 positives (20% of the 1,732-row manifest)**, paid for
mostly out of `atelectasis` (−328) and `pleural_effusion` (−323), the two most
abundant labels, which is what "least load-bearing" means in practice.
`cardiomegaly`, `edema`, `pneumonia` and `pneumothorax` lost nothing — the rare,
high-value labels were deliberately not touched.

**The guard.** `40_build_eval_manifest.py` now **exits non-zero if the built
manifest has any constant label column**, so this class of defect cannot be
produced silently again.

**Consequence for the literature.** `nih_shared_5000.csv` has never been usable
for a `no_finding` metric and is not used for evaluation. This is a defect of the
upstream sampling, not of the data; `docs/48` and `45_debug_sweep.py` both label
it as such.

---

## 6. Inventory of `program/data/manifests/`

| File | Rows | Bytes | Kind | Purpose / status |
|---|---|---|---|---|
| `nih_eval_1732.csv` | 1,732 | 336,065 | classification | **the NIH evaluation set — the only NIH manifest used for reported metrics**: 1,054 `internal_test` + 452 `source_val` + 226 donors (→ **1,506 scored**) |
| `nih_eval_1732_report.json` | — | 1,281 | provenance | split composition, positives per label, `no_finding_repair` record, `images_missing_on_disk` |
| `padchest_shift.csv` | 1,500 | 272,787 | classification | **the PadChest evaluation set** (765 patients, `internal_test`); no pneumonia label exists |
| `openi_rrg.csv` | 947 | 417,524 | report | **the Open-i evaluation set** (`external_report`); `reference_text` present, all 8 labels `unknown` |
| `chexpert_rrg_all.csv` | 264 | 131,395 | report | the earlier external RRG set (`reference_text` + `default_prompt`, no label columns); kept for historical continuity |
| `nih_full.csv` | 112,120 | 21,855,117 | classification | the extracted NIH pool (60,361 `no_finding` positives of 112,120); the donor pool for the `no_finding` repair, **not** an evaluation manifest |
| `nih_shared_5000.csv` | 5,000 | 974,037 | classification | the 6K-plan sampling manifest; three splits incl. 3,268 `internal_train`. **`label_no_finding` is constant `negative` and it is not used for evaluation** — superseded by `nih_eval_1732.csv` |
| `nih_shared_5000_report.json` | — | 1,426 | provenance | sampling report for the above |
| `nih_shared_2500.csv` | 2,500 | 487,094 | classification | earlier mid-scale sampling manifest |
| `nih_shared_2500_report.json` | — | 1,421 | provenance | sampling report |
| `nih_shared_500.csv` | 500 | 97,709 | classification | the smallest sampling manifest (used by the throughput benchmarks in `docs/46`) |
| `nih_pilot_1000.csv` | 1,000 | 195,123 | classification | the pilot manifest behind the Stage 2 results; pilot only |
| `openi_rrg.csv.bak` | — | 319,809 | report | pre-rewrite backup — **invalid labels, 0/1 encoding, no `patient_id`/`split`** |
| `padchest_shift.csv.bak` | — | 106,676 | classification | pre-rewrite backup — **invalid labels, 0/1 encoding, no `patient_id`/`split`** |

Two caveats about the table:

* `.bak` files are kept only as evidence of the rewrite described in §7. Nothing
  should read them.
* `nih_full.csv` and the `nih_shared_*` family are **inputs to manifest
  construction**, not evaluation manifests. Only `nih_eval_1732.csv`,
  `padchest_shift.csv` and `openi_rrg.csv` are evaluated, and only those three are
  checked for image presence by the debug sweep.

---

## 7. How each evaluation manifest is produced

| Manifest | Producer | Notes |
|---|---|---|
| `nih_eval_1732.csv` (+ report) | `scripts/40_build_eval_manifest.py` | selects `internal_test` + `source_val` from `nih_shared_5000.csv`, swaps in the `no_finding` donors, drops the least load-bearing rows, refuses to write a manifest with a constant label column |
| `padchest_shift.csv` | `scripts/25_format_external_manifests.py` | negation-aware phrase matching over PadChest's free-text labels; writes positive/negative strings plus `patient_id` and `split` |
| `openi_rrg.csv` | `scripts/25_format_external_manifests.py` | report-generation adapter: `reference_text` + `default_prompt`, `split = external_report`, all 8 labels honestly `unknown` |

```powershell
cd D:\Paper\program
.\.venv-vl\Scripts\python.exe scripts\40_build_eval_manifest.py
.\.venv-vl\Scripts\python.exe scripts\40_build_eval_manifest.py --no-finding-min 0.20
.\.venv-vl\Scripts\python.exe scripts\25_format_external_manifests.py
.\.venv-vl\Scripts\python.exe scripts\25_format_external_manifests.py --report
```

### 7.1 What the old `25_format_external_manifests.py` did wrong

Three contract violations at once, all of them silent:

1. it wrote **0/1 labels**, and the label matrix tests
   `astype(str) == "positive"`, so **every row became negative**;
2. it **omitted `patient_id` and `split`** entirely — which removed the cluster
   key the bootstrap needs and made the `--split` filter impossible;
3. it used **bare substring matching**, so `"no pleural effusion"` counted as a
   positive pleural effusion. (It also mapped PadChest's `Normal` by substring
   against a lowercased label string, so anything containing `normal` lit up
   `no_finding`.)

The rewrite matches phrases with a leading-negation window
(`no`, `no_`, `without`, `sin`, `absence of`, `negative for`) and derived the
PadChest vocabulary explicitly per finding. **Recorded effect on consolidation:
false positives 62 → 82 corrected, i.e. 20 bogus positives removed.** Numbers
computed from the pre-rewrite manifests are invalid; the `.bak` files preserve
them only as evidence.

---

## 8. Traps

1. **`--limit` takes the first N rows of the file, which is NOT a sample.**
   `nih_shared_5000.csv` is ordered by `image_id` and holds 3,268
   `internal_train` rows, so a large `--limit` silently evaluates on **training
   data**. Measured: `--limit 300` yielded only 138 evaluation images and 12
   pneumonia positives. `06`, `08` and `21` now take a `--split` filter and
   shuffle before limiting; `43` and `44` do the same. Always pass `--split`.
2. **Never write a manifest (or any `.py`) with PowerShell
   `Set-Content -Encoding UTF8`** on Windows PowerShell 5.1: it writes a UTF-8
   BOM, and Python fails with `invalid non-printable character U+FEFF`. This
   happened in this repository; `45_debug_sweep.py` checks for the BOM.
3. **A constant label column is a bug, not a finding.** Check with
   `45_debug_sweep.py` before analysing anything.
4. **The 226 `internal_train` rows in `nih_eval_1732.csv` are donors.** Filter
   with `--split internal_test,source_val`, or the metric silently includes
   training data.
5. **`patient_id` is the cluster key.** Both `nih_eval_1732.csv` (951 patients)
   and `padchest_shift.csv` (765 patients) contain repeated patients; bootstrapping
   over images instead of patients understates every interval reported.
6. **Do not "fix" `unknown` in `openi_rrg.csv`.** The absence of labels is the
   reason the Open-i layer is a report-generation layer; filling the labels in
   would manufacture ground truth.
