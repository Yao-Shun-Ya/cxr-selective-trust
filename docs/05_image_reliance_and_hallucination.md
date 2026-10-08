# 05 — Image Reliance & Hallucination｜视觉证据依赖与幻觉协议

## 1. 核心问题

RQ4：

> 模型认为自己“可靠”的时候，是否真的建立在视觉证据上？

如果模型在 blank image / severe corruption 下仍给出相同答案且保持高 confidence，说明其可能主要依赖语言先验、标签先验或数据集捷径，而不是医学图像证据。

## 2. 实验总览

| 实验 | 干预 | 目的 | 关键输出 |
|---|---|---|---|
| A. Blank image | 替换为黑图 / 均值图 / 纯噪声 | 测语言先验 | prediction invariance；confidence drop |
| B. Severe corruption | 严重 blur / 噪声 / 遮挡 | 测视觉依赖 | performance drop；confidence change |
| C. Region masking | mask lung / heart / pleural 区 | 测关键区域依赖 | finding-specific drop |
| D. Counterfactual / negation | 图像不变，问题表述改变 | 测语言敏感性 | answer flip rate |
| E. Attention / saliency（可选） | attention rollout / IG | 补充机制证据 | heatmap overlap |

## 3. Experiment A — Image Removal

### 条件

- `original`：原图 + 问题；
- `blank_black`：全黑图 + 同一问题；
- `blank_mean`：数据集均值图 + 同一问题；
- `noise_only`：随机噪声图 + 同一问题。

### 评估

对每个 finding：

```text
same_prediction_rate = P(pred_blank == pred_original)
confidence_drop = conf_original - conf_blank
accuracy_drop = acc_original - acc_blank
```

高 `same_prediction_rate` + 高 `conf_blank` = 语言先验风险。

### 解释边界

- 不能直接把“blank 仍答对”判定为作弊：某些 finding 可能有 prevalence prior；
- 必须结合 label prevalence、prompt、模型类型解释；
- 要同时报告 blank 上的 accuracy，而不只是 consistency。

## 4. Experiment B — Severe Corruption

### 方法

对同一图像施加：

```text
severe blur
severe noise
patch shuffle
large occlusion
low resolution
```

### 评估

- performance drop；
- confidence drop；
- image-reliance score：

```text
IR = (metric_clean - metric_corrupted) / metric_clean
```

注意避免除零，可用 absolute difference 或 rank-based 替代。

### 期望发现

- 真实依赖视觉证据的模型：corruption 后 performance 和 confidence 均应下降；
- 不可靠模型：performance 下降，但 confidence 不降，甚至上升；
- 结合 selective prediction：高风险样本是否恰好是 image-reliance 低的样本。

## 5. Experiment C — Region Masking

### 区域定义

优先使用可复现的解剖区域：

- left lung；
- right lung；
- bilateral lung fields；
- heart / cardiac silhouette；
- pleural space；
- lower / upper zones。

### 获取 mask 的路径

1. 使用公开 segmentation model（需在 protocol 中固定版本）；
2. 使用数据集自带的 bounding box / segmentation（若有）；
3. 手工标注小子集用于验证自动 mask 质量；
4. 不允许用 test 结果反推最佳 mask。

### 评估

对 finding-specific 假设：

| Finding | 关键区域 |
|---|---|
| Cardiomegaly | heart |
| Pleural Effusion | pleural / lower lung |
| Pneumothorax | lung field |
| Atelectasis | lung field |
| Fracture | rib / bone region |

期望：

- mask 关键区域后，对应 finding 的预测应显著变化；
- 若不变，说明模型没有依赖该区域，或依赖了其他捷径。

## 6. Experiment D — Counterfactual / Negation Prompt

保持图像不变，改变问题：

```text
"Is there evidence of X?"
"Is there no evidence of X?"
"Does this image rule out X?"
"Which finding is most likely?"
```

评估：

- answer flip rate；
- confidence sensitivity；
- negation consistency；
- 是否出现 yes/no 偏差。

该实验用于补充区分“视觉依赖”与“语言先验”，不能单独作为 image reliance 结论。

## 7. Image Reliance Score

建议构建 study-level 与 finding-level 两个版本：

```text
IR_finding = normalized_drop_in_correctness
           + normalized_drop_in_confidence
           + agreement_original_corrupted
```

更稳健的版本：

```text
IR = z(-Δaccuracy) + z(-Δconfidence) + z(1 - agreement)
```

要求：

- 所有归一化只在 source_val / 内部集合上拟合；
- 在 external / shifted 上做 transfer；
- 明确说明 IR 是行为学证据，不是机制性证明；
- 与 attention/saliency 结果互相佐证，rather than 取代。

## 8. Hallucination 定义

幻觉不能由 Agent 或 LLM 自由判断。必须基于：

```text
prediction
  ↓
ground truth / report-derived evidence
  ↓
factual consistency
```

### Hallucination taxonomy

| 类型 | 定义 | 评估数据 |
|---|---|---|
| Disease hallucination | 报告不存在的异常 | label + report evidence |
| Negation failure | report 明确 absent，模型输出 present | report evidence |
| Localization error | 病灶存在但位置错误 | report / bbox / mask |
| Severity error | 严重程度描述错误 | report evidence |
| Laterality error | 左/右/双侧错误 | report evidence |
| Finding omission | 存在重要 finding 但模型未报告 | label + report |
| Hallucinated rationale | rationale 声称看到了不存在的视觉证据 | evidence subset + human audit |

### 指标

- hallucination rate（按 finding / study）；
- false positive rate；
- negation failure rate；
- laterality error rate；
- severity error rate；
- 幻觉率 vs confidence；
- 幻觉率 vs calibration；
- 幻觉率 vs image-reliance score。

## 9. 人工审核协议

### 目的

- 验证自动 report-derived evidence 的可靠性；
- 对少量样本做幻觉类型 adjudication；
- 避免 LLM-as-judge 自证。

### 抽样

- 100–300 例；
- 按 model、dataset、correct/incorrect、confidence 分层；
- 包含 high-confidence error 的 enrichment 样本；
- 必须从 test set 抽样，但只用于评估。

### 流程

```text
自动 evidence extraction
  ↓
两位标注者独立审核（若资源允许）
  ↓
不一致 → 第三方仲裁 / 讨论
  ↓
冻结 hallucination subset
```

### 记录

- 标注指南版本；
- 标注者背景（不记录身份）；
- 一致性指标（Cohen's κ / Fleiss' κ）；
- 仲裁结果；
- 任何 prompt/model 输出版本。

## 10. 结论表述边界

可以写：

> Under blank-image and severe-corruption controls, model X showed high prediction invariance and preserved confidence, suggesting reliance on non-visual priors for certain findings.

不能写：

> Model X is cheating / does not use images at all.

必须保留行为学证据的边界，并建议后续机制研究。

## 11. 与 selective trust 的结合

将 image-reliance 纳入风险分数：

```text
trust_score = f(calibrated_confidence,
                 uncertainty,
                 shift_score,
                 image_reliance)
```

如果 image-reliance 低的样本恰好对应高错误率，则说明：

> 拒答机制不仅应在模型“不确定”时触发，也应在模型“没有充分看图”时触发。
