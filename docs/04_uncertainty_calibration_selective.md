# 04 — Uncertainty, Calibration & Selective Prediction

> **现行度量约定（2026-10-05 起，务必遵守；本节由文档审计补入）**
> 1. **ECE 分箱是左闭右开 `[0,0.1)`**，不是 `(0,0.1]`。本项目约 **8.44%** 的存储概率正好落在分箱边界上（RRG 文件 **24.7%**），用错约定会系统性地改变每一个 ECE。
> 2. **校准样本宇宙必须显式**：概率类指标（AUROC / AUPRC / ECE / Brier）用**所有给出概率的行，含 `uncertain`**；分类与选择性指标（F1 / AURC / coverage）**只用 definite 判定**。`metrics_by_label.csv` 中的 `n_prob` / `n_definite` 两列就是这个约定的落点。
> 3. 两者分别由 `45_debug_sweep.py` §9 与 §11 的行为守卫保护。完整经过见 `docs/48_three_layer_protocol.md` §7（2026-10-05）与 `program/README.md` §10。
> 4. §2 标题写「三类 uncertainty」但实际列了 A–D 四节（D 为可选的 verbalized uncertainty）—— 记录于此，未改动正文。

## 1. 目标

回答 RQ2 和 RQ3：

1. 模型 confidence 在 shift 下是否还能预测 correctness；
2. 能否用多源 uncertainty 识别高风险病例；
3. 通过 abstention 能否显著降低自动接受病例的错误率。

## 2. 三类 uncertainty

### A. Self-consistency

同一图像、同一问题，使用 K 个语义等价 prompt / 多次采样：

```text
K = 5（pilot 可用 3，full 建议 5）
temperature 固定并记录
```

输出 agreement：

- 对每个 finding，计算 positive / negative 的多数票；
- 计算预测熵、方差、Fleiss' kappa；
- 计算模型自报 confidence 的均值和方差；
- 对 study-level 聚合：mean / max / entropy。

### B. Image-perturbation consistency

同一图像施加不改变诊断语义的轻微扰动：

```text
original
light blur
light brightness
light contrast
small crop / resize
```

计算：

- prediction agreement；
- confidence variance；
- per-finding stability；
- study-level semantic disagreement。

注意：这里使用的是轻扰动，不能与 Shift B 的严重 degradation 混为一谈。

### C. Semantic disagreement

若模型只能输出文本/标签：

- 对比多次 generation 的标签集合；
- 使用结构化解析后的 Jaccard / Hamming distance；
- 对 rationale 做 evidence coverage 检查（不单独作为 hallucination ground truth）；
- 若有 token logits，可补充 predictive entropy / mutual information。

### D. 可选：verbalized uncertainty

模型自报 confidence 单独作为一个 baseline，不要默认它可靠。必须与客观 correctness 做 calibration 分析。

## 3. Uncertainty 聚合

### Per-finding uncertainty

```text
u_label = w1 * self_consistency_entropy
        + w2 * perturbation_variance
        + w3 * (1 - verbalized_confidence)
        + w4 * parse_instability
```

### Study-level risk score

```text
risk_study = max over findings u_label
```

或使用加权聚合：

```text
risk_study = 1 - product(1 - u_label)
```

注意：最大值对“任意一个 finding 高风险”更敏感，适合临床安全；平均可能掩盖局部错误。两种都做敏感性分析。

### 权重确定

- 初期：等权 / 简单规则；
- 后期：在 source_val 上拟合轻量 logistic regression / gradient boosting；
- 必须 nested CV 或独立 calibration/validation split；
- 禁止在 test/external 上调权重；
- 报告不调权重 vs 学习权重两组结果。

## 4. Calibration

### 指标

- Expected Calibration Error（ECE）；
- Adaptive ECE / class-conditional ECE；
- Brier score；
- Negative log-likelihood；
- Reliability diagram；
- confidence–accuracy gap；
- high-confidence error rate（如 confidence ≥ 0.9 时的错误率）。

### 方法

| 方法 | 说明 | 何时使用 |
|---|---|---|
| Temperature scaling | 单参数，多分类常用 | 基础 baseline |
| Platt scaling | 二分类/逐 finding | 逐 label 校准 |
| Isotonic regression | 非参数 | 有足够 calibration 样本 |
| Beta calibration | 概率校准 | 概率分布偏斜时 |
| Dirichlet calibration | 多标签/多分类 | 可选 |
| Shift-aware calibration | 按 shift 类型 / 严重度分组或加权 | 本项目核心比较 |

### 正确协议

```text
source_val → fit calibrator
internal_test → evaluate
external_test / shifted_test → evaluate transfer
```

禁止：

- 在 test/external 上 fit temperature；
- 用 external 结果反推 calibration 参数；
- 只报告 ECE 提升而忽略 AUROC / selective risk。

## 5. Selective Prediction

### 定义

对每个样本得到 risk score `r(x)`，选择阈值 `τ`：

```text
r(x) < τ  → accept
r(x) ≥ τ  → abstain
```

目标是让 accepted subset 的错误率低于临床可接受阈值。

### Risk–Coverage

按 risk score 从低到高排序，逐步扩大 coverage：

- Coverage = 自动接受比例；
- Risk = 自动接受样本中的错误率（或 1 - accuracy / 1 - AUROC 等其他定义要写清）。

理想曲线：

```text
100% coverage → error 15%
 90% coverage → error 9%
 80% coverage → error 4%
 70% coverage → error 2%
```

### 指标

- AURC（Area Under Risk–Coverage Curve）；
- Coverage@5%Risk；
- Coverage@10%Risk；
- Selective accuracy；
- Selective AUROC / AUPRC；
- Error detection AUROC：用 risk score 区分 correct vs incorrect；
- Abstention rate；
- 每类 shift / dataset / finding 的 coverage@risk。

### 阈值选择

- 只在 source_val 上选择 τ；
- 可使用固定风险约束（coverage@5%）或固定 coverage（risk@80%）；
- 同时报告 oracle threshold 作为上界，明确标注 oracle 不可部署；
- 对每个模型、每类 shift 单独报告，不只报告 pooled。

## 6. Shift-aware Selective Trust Framework

```text
                      Medical Image
                           ↓
                          VLM
                           ↓
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
   Prediction       Self-consistency    Image perturbation
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
                 Uncertainty Aggregation
                           ↓
                  Shift-aware Calibration
                           ↓
                   Trust Score / Risk
                           ↓
                 ┌─────────┴─────────┐
                 ↓                   ↓
              Accept               Abstain
                 ↓                   ↓
            AI decision         Human review
```

### 输入特征

- predictive confidence；
- per-finding / study-level uncertainty；
- calibration score；
- image quality / OOD / shift score；
- image-reliance score；
- dataset / view metadata（若可用，注意不要把 dataset identity 当成捷径）。

### 输出

- calibrated correctness probability；
- risk score；
- accept / abstain 决策；
- 可解释拒答理由（高风险 finding / 低一致性 / 图像依赖弱 / 明显 OOD）。

### 关键消融

| 配置 | 目的 |
|---|---|
| VLM only | baseline |
| + uncertainty | 验证不确定性 |
| + calibration | 验证校准 |
| + shift-aware | 验证 shift 条件化 |
| + image-reliance | 验证视觉依赖 |
| Full | 完整方法 |

## 7. 常见陷阱

1. 只用 confidence 做 selective prediction，忽略 calibration transfer；
2. 用 test 选 threshold 或 calibrator；
3. 只报告 pooled risk–coverage，掩盖 external failure；
4. 把 abstention rate 当作越低越好，忽略 accepted error；
5. uncertainty 只验证 average correlation，不验证 top-risk 区域；
6. 忽略 parse failure，把解析失败当 abstain 或 drop；
7. 把模型自报 confidence 当作 ground truth uncertainty；
8. 在 severe shift 下强行追求 coverage，忽略安全约束。
