# cxr-trust-audit

CPU-only reproducibility audit for medical vision-language model (VLM) prediction
files. It is the reusable companion to:

- *Responsiveness Before Reliance: Four Reversals in Measuring Medical
  Vision-Language Models* (`papers/main_v9/`);
- *Auditing the Denominator: Coverage Census and Fault-Tested Durability for
  Medical VLM Benchmarks* (`papers/bench_v3/`).

No model, image or network access is required.

## Install

```bash
pip install -e audit
# or, from GitHub:
pip install "git+https://github.com/Yao-Shun-Ya/cxr-selective-trust.git#subdirectory=audit"
```

## Quick start (no private data)

```bash
cxr-audit demo
# equivalent, if the console script is not on PATH:
python -m cxr_audit demo
```

The demo writes a synthetic JSONL/CSV in a temporary directory and prints a
census, responsiveness and selective-prediction report.

## Commands

### Denominator census

```bash
cxr-audit census path/to/shift_predictions.jsonl
cxr-audit census path/to/predictions.jsonl --key-fields image_id,condition,severity
```

Reports rows, unique keys, duplicate rows/groups, parse rate and condition
counts, recomputed from the file itself.

### Responsiveness

```bash
cxr-audit responsiveness path/to/shift_predictions.jsonl \
  --labels atelectasis,cardiomegaly,consolidation,edema,pleural_effusion,pneumonia,pneumothorax,no_finding \
  --clean clean --shift contrast --severity severe
```

Reports distinct status vectors on clean inputs, modal share, parse rate and the
P1 change rate against clean. A modal share near 1 means reliance is
*unidentifiable*, not zero.

### Selective baselines

```bash
cxr-audit selective path/to/risk_items_finding.csv
```

Compares status-aware confidence with Bernoulli entropy and probability margin:
error-detection AUROC and coverage at 5%/10% risk budgets.

## Input formats

- Prediction JSONL: one row per `(image_id, condition, severity)` with a
  `parsed` object containing `parse_success` and `findings`.
- Selective CSV: one row per finding with `error`, `p_calibrated` (or `p`), and
  `conf` (or `pred`), plus `patient_id` when available.

## Verified replay

The tool is a regression check on the manuscript, not only a template. With the
frozen outputs present, these invocations reproduce the published values:

```bash
# derm panel, MedGemma-1.5-4B: 2,940 rows, 0 duplicates, parse rate 0.95034
cxr-audit census program/outputs/crossmodal/derm/v0/medgemma_15_4b_it/shift_predictions.jsonl

# 95 distinct clean vectors, modal share 0.1105, P1 change 0.5493
cxr-audit responsiveness program/outputs/crossmodal/derm/v0/medgemma_15_4b_it/shift_predictions.jsonl \
  --labels actinic_keratosis,basal_cell_carcinoma,benign_keratosis,dermatofibroma,melanoma,melanocytic_nevus,vascular_lesion \
  --clean clean --shift contrast --severity severe

# confidence AUROC 0.7366, coverage@5% 0.0926; entropy/margin 0.6019, 0.0001
cxr-audit selective program/outputs/selective_v2/medgemma_15_4b_it/risk_items_finding.csv
```

The responsiveness output also reports `modal_vector` as seven `unparsed`
entries: 65 clean parse failures make "unparsed" the most frequent vector. That
is the intended reading — parse failure is a status, not a missing row.

## Tests

```bash
python -m pytest -q audit/tests      # 4 passed
```

CI runs the same command on Python 3.10 and 3.12, then `cxr-audit demo`,
`pip check` and the citation-metadata assertions.

## Scope

The tool audits *reported* numbers. It does not run inference, load model
weights, or redistribute images. Dataset licences remain those of the original
releases. Code, manifests and derived tables are MIT; see `LICENSE` and
`NOTICE`.

## Citation

If you use this tool, cite the manuscript and the software record:
[10.5281/zenodo.24483387](https://doi.org/10.5281/zenodo.24483387). See
`CITATION.cff` at the repository root; the archive and the DOI are versioned
together.
