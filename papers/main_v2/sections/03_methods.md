---

> **HISTORICAL SNAPSHOT — frozen Markdown mirror, last synced 2026-09-16.**
> It describes the Stage 2 pilot revision of the paper (1,000-study NIH pilot + 264-sample CheXpert-plus-RRG external set). It contains no pre-correction metrics, but it predates the explicit calibration-universe passage (n_prob / n_definite, 1,800 probability vs 1,614 definite decisions) now in the LaTeX section.
> Authoritative text: `papers/main_v2/sections/03_methods.tex` — the only copy to edit; it is what was rebuilt into `papers/main_v2/main.pdf` (19 pages).
> Do **not** hand-edit this mirror: two hand-maintained copies are how it diverged in the first place. See `docs/48_three_layer_protocol.md` §7 (2026-10-05).

# 3. Methods

## 3.1 Task and label space

We study chest X-ray (CXR) vision-language models on structured multi-label finding prediction. The primary label space contains eight findings:

```text
atelectasis
cardiomegaly
consolidation
edema
pleural_effusion
pneumonia
pneumothorax
no_finding
```

Each model outputs a structured JSON object containing:

- `findings[label] ∈ {positive, negative, uncertain, not_mentioned}`;
- `probability_positive[label] ∈ [0,1]`;
- an optional short rationale.

All models share one prompt and one parser. Missing labels are mapped to `not_mentioned`; invalid statuses are mapped to `uncertain`; invalid probabilities are set to null. Parse success and parse failure are recorded separately.

## 3.2 Models

We evaluate openly available vision-language models:

| Model | Role |
|---|---|
| MedGemma-4B-IT | medical generalist primary model |
| Qwen2.5-VL-7B-Instruct | open-weight general baseline |
| CheXagent-8B | CXR-specific pilot/sensitivity model |

Models are loaded locally with 4-bit quantization and a single-GPU placement when available. MedGemma and Qwen are used for the main Stage 2 experiments; CheXagent is used only for a small pilot because it is substantially slower under CPU offload.

## 3.3 Data

### NIH ChestX-ray14 pilot

We use a stratified pilot of 1,000 studies from NIH ChestX-ray14. The manifest contains patient-level splits:

```text
internal_train
source_val
internal_test
```

`source_val` is used only for calibration fitting and threshold selection. `internal_test` is used only for evaluation. Image-level ground-truth labels are derived from the official NIH labels via a fixed mapping into the primary label space.

### CheXpert-plus-RRG external report set

CheXpert-plus-RRG contains:

```text
findings_section:  62 samples
impression_section: 202 samples
```

Each sample provides a frontal CXR image and reference report text. It is used as an external report-grounded evaluation set for hallucination, omission, and image reliance, not as a large-scale classification benchmark.

## 3.4 Structured output and parsing

The structured prompt asks the model to return one JSON object with findings and probabilities. The parser:

1. extracts the first JSON object;
2. tolerates Markdown code fences;
3. normalises misspelled label keys with fuzzy matching;
4. maps missing labels to `not_mentioned`;
5. records parse failure explicitly.

## 3.5 Distribution shift benchmark

We evaluate six image-quality shifts at three severity levels:

| Shift | Mild | Moderate | Severe |
|---|---|---|---|
| Blur | σ=0.5 | σ=1.0 | σ=2.0 |
| Noise | σ=0.02 | σ=0.05 | σ=0.10 |
| JPEG | q=70 | q=40 | q=20 |
| Brightness | ×0.8 | ×1.2 | ×1.5 |
| Contrast | ×0.7 | ×1.3 | ×1.6 |
| Resolution | ×0.5 | ×0.25 | ×0.125 |

Each image is evaluated under clean, mild, moderate, and severe conditions. For every image–condition–severity combination we store the raw output, parsed output, parse success, probability estimates, elapsed time, and ground truth.

## 3.6 Image-reliance evaluation

For image reliance we compare the original image against:

- blank gray image;
- strong blur;
- shuffled image from the same manifest.

For each condition and label we measure:

- status flip count relative to original;
- image sensitivity (fraction of images with at least one finding flip);
- mean absolute change in `probability_positive`.

## 3.7 Calibration

For each finding we fit temperature scaling on `source_val` and apply it to `internal_test` without refitting. We report:

- AUROC;
- expected calibration error (ECE);
- Brier score.

Temperature scaling changes calibration but not ranking, so AUROC is invariant by construction.

## 3.8 Selective prediction

For each finding prediction we derive a confidence score:

```text
confidence = P(positive)        if predicted positive
confidence = 1 - P(positive)    if predicted negative
confidence = 0.5                if uncertain
risk = 1 - confidence
```

We compute risk–coverage curves by accepting the lowest-risk cases first, and report AURC, coverage at 5% and 10% risk, and selective accuracy at 80% coverage.

## 3.9 CheXpert-RRG report-level evaluation

RRG reference text is converted to finding-level evidence using a rule-based keyword + negation-window method (`rule_based_v0`). For each primary finding we define:

- disease hallucination: model positive while reference evidence is not present;
- negation failure: model positive while reference explicitly negates the finding;
- omission: model negative while reference evidence is present.

This evidence extractor is a pilot implementation; final analysis should use a validated report labeller or human adjudication.

## 3.10 Statistics

We use image-level cluster bootstrap to estimate 95% confidence intervals. For paired clean-vs-shift comparisons we resample images with replacement and compute paired bootstrap differences in macro AUROC, ECE, and Brier score. The current Stage 2 analysis uses 200 bootstrap resamples for pilot speed; the final paper should use at least 1,000.
