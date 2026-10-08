---

> **HISTORICAL SNAPSHOT — frozen Markdown mirror, last synced 2026-09-16.**
> Its MedGemma `internal_test` row and the sentence below it are the **pre-correction** numbers: macro ECE 0.232, AUROC 0.612, Brier 0.198. The corrected values (0.220 / 0.622 / 0.188) live in the LaTeX section and in Table 2.
> Authoritative text: `papers/main_v2/sections/04_results.tex` — the only copy to edit; it is what was rebuilt into `papers/main_v2/main.pdf` (19 pages).
> Do **not** fix numbers by hand in this file: two hand-maintained copies are how it diverged in the first place. See `docs/48_three_layer_protocol.md` §7 (2026-10-05).

# 4. Results

## 4.1 Structured outputs and parse stability

Both MedGemma-4B-IT and Qwen2.5-VL-7B produced structurally valid JSON for all 1,000 NIH pilot studies after the prompt and parser were finalised:

| Model | Samples | Parse failures |
|---|---:|---:|
| MedGemma-4B-IT | 1000 | 0 |
| Qwen2.5-VL-7B | 1000 | 0 |

However, parse stability did not imply reliable image understanding.

## 4.2 Overall NIH performance and calibration

Macro metrics across the eight findings are shown in Table 1.

| Model | Split | Macro AUROC | Macro ECE | Macro Brier | Macro AURC | Mean coverage @5% risk |
|---|---|---:|---:|---:|---:|---:|
| MedGemma | internal_train | 0.647 | 0.133 | 0.122 | 0.120 | 0.733 |
| MedGemma | source_val | 0.636 | 0.154 | 0.124 | 0.137 | 0.713 |
| MedGemma | internal_test | 0.612 | 0.232 | 0.198 | 0.228 | 0.357 |
| Qwen2.5-VL | internal_train | 0.500 | 0.088 | 0.087 | 0.071 | 0.662 |
| Qwen2.5-VL | source_val | 0.502 | 0.070 | 0.070 | 0.061 | 0.662 |
| Qwen2.5-VL | internal_test | 0.509 | 0.153 | 0.154 | 0.117 | 0.398 |

MedGemma showed moderate but non-trivial discrimination (macro AUROC 0.61–0.65) and clear calibration degradation on `internal_test` (macro ECE 0.232). Qwen remained close to chance discrimination (macro AUROC ≈ 0.50), despite acceptable-looking calibration and selective metrics.

## 4.3 Qwen is structurally valid but image-invariant

On 300 NIH images, MedGemma changed many predictions under counterfactual image conditions:

| Condition | MedGemma image sensitivity | Qwen image sensitivity |
|---|---:|---:|
| blank gray | 1.000 | 0.0033 |
| strong blur | 0.947 | 0.0033 |
| shuffled image | 0.877 | 0.0033 |

MedGemma’s mean absolute probability change was 0.138–0.210. Qwen’s was 0.0006, effectively zero.

Qwen also predicted `no_finding = positive` for 988/1000 studies and had macro AUROC ≈ 0.50 across all eight findings. Its apparently low selective risk and high selective accuracy therefore reflected a degenerate prediction strategy rather than reliable image interpretation.

**Finding:** Structured output validity, low selective risk, and high selective accuracy can coexist with image invariance.

## 4.4 Shift effects are heterogeneous and non-monotonic

MedGemma macro AUROC and ECE under shift are shown in Table 2.

| Condition | Severity | Macro AUROC | Macro ECE |
|---|---|---:|---:|
| clean | none | 0.620 | 0.163 |
| blur | mild | 0.612 | 0.166 |
| blur | moderate | 0.603 | 0.160 |
| blur | severe | 0.615 | 0.137 |
| noise | mild | 0.675 | 0.149 |
| noise | moderate | 0.704 | 0.151 |
| noise | severe | 0.520 | 0.142 |
| JPEG | mild | 0.591 | 0.167 |
| JPEG | moderate | 0.710 | 0.149 |
| JPEG | severe | 0.621 | 0.140 |
| brightness | mild | 0.614 | 0.175 |
| brightness | moderate | 0.600 | 0.158 |
| brightness | severe | 0.597 | 0.164 |
| contrast | mild | 0.670 | 0.165 |
| contrast | moderate | 0.620 | 0.163 |
| contrast | severe | 0.617 | 0.164 |
| resolution | mild | 0.633 | 0.148 |
| resolution | moderate | 0.705 | 0.134 |
| resolution | severe | 0.581 | 0.104 |

Image sensitivity generally increased with severity (e.g., noise 0.53 → 0.70 → 0.79; resolution 0.37 → 0.60 → 0.71), but AUROC did not follow a monotonic degradation pattern. Several shifts even improved calibration.

Paired macro bootstrap tests on MedGemma showed:

- no shift produced a significant AUROC decrease;
- JPEG moderate: ΔAUROC = +0.069, p = 0.05;
- resolution moderate: ΔAUROC = +0.068, p = 0.10;
- significant ECE improvements for blur severe, JPEG moderate/severe, noise mild/severe, and resolution mild/moderate/severe;
- only brightness mild showed a significant ECE increase (ΔECE = +0.010, p = 0.04).

**Finding:** Distribution shift does not produce a simple monotonic degradation of CXR VLM reliability. Shift response is condition-specific and must be evaluated jointly with image reliance and calibration.

## 4.5 Calibration

Per-finding temperature scaling fitted on `source_val` did not change discrimination. It produced mixed calibration effects:

- `no_finding` ECE improved substantially: MedGemma 0.363 → 0.093;
- most other findings showed little improvement or slight worsening;
- Qwen’s `no_finding` ECE improved from 0.583 to 0.258, but AUROC stayed at chance.

These results suggest that global or per-label temperature scaling is insufficient for multi-label CXR reliability.

## 4.6 Selective prediction

On pooled `internal_test` label-instances, MedGemma achieved:

```text
global temperature: 5.67
AURC: 0.201
coverage@5% risk: 0.002
coverage@10% risk: 0.006
selective accuracy@80% coverage: 0.680
```

Qwen achieved:

```text
global temperature: 5.34
AURC: 0.095
coverage@5% risk: 0.000
coverage@10% risk: 0.897
selective accuracy@80% coverage: 0.901
```

Qwen’s selective metrics appear strong but are invalid as clinical evidence because its predictions are image-invariant and dominated by `no_finding`. This demonstrates that selective risk must be conditioned on image reliance.

## 4.7 CheXpert-plus-RRG external report evaluation

On 264 RRG images, MedGemma produced:

```text
overall hallucination rate: 0.135
overall negation-failure rate: 0.006
overall omission rate: 0.122
overall uncertain rate: 0.090
```

Finding-level results:

| Finding | Hallucination | Omission |
|---|---:|---:|
| atelectasis | 0.443 | 0.057 |
| cardiomegaly | 0.449 | 0.003 |
| consolidation | 0.019 | 0.127 |
| edema | 0.051 | 0.166 |
| no_finding | 0.083 | 0.252 |
| pleural_effusion | 0.019 | 0.213 |
| pneumonia | 0.019 | 0.054 |
| pneumothorax | 0.000 | 0.105 |

The high atelectasis/cardiomegaly hallucination rates may partly reflect model over-calling and limitations of the rule-based evidence extractor. On the first 50 RRG images, original-vs-blank status flip rate was 0.4525, indicating that MedGemma still depends on the image in this external set.

## 4.8 Summary of empirical findings

1. MedGemma is image-dependent, but its calibration and selective reliability are weak.
2. Qwen is structurally valid but image-invariant and near chance on findings.
3. Image-quality shift impacts prediction stability but does not monotonically degrade macro AUROC or ECE.
4. Selective metrics can look favourable for an image-invariant model.
5. RRG external report evaluation exposes both hallucination and omission behaviour.

Together these results motivate a shift-aware selective trust framework that evaluates calibration, uncertainty, and visual evidence reliance jointly rather than relying on accuracy or selective risk alone.
