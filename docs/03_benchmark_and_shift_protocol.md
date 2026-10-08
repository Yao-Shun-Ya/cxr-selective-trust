---

> **历史快照（HISTORICAL SNAPSHOT）—— 本文是 Phase 0 的 shift taxonomy 与 benchmark 设计稿。**
> 现以 `docs/48_three_layer_protocol.md` 为准（NIH 19 格合成退化 + PadChest 原生偏移 + Open-i 报告层）；四轴分类与 severity 定义被沿用下来。
> 正文原样保留，作为 benchmark 设计的演进底稿。

# 03 — Benchmark & Shift Protocol｜分布偏移基准协议

## 1. 目标

建立一个可复现的 CXR VLM distribution-shift benchmark，使得：

- 不同模型在完全相同的图像、prompt、label space 下比较；
- 每类 shift 可单独分析，也可组合分析；
- shift 不仅影响 accuracy，还能被 calibration / selective prediction 指标刻画；
- 结果可回溯到 manifest + config + random seed。

## 2. Shift 四轴

```text
Shift A — Cross-dataset / Domain
Shift B — Image Quality / Acquisition
Shift C — View / Acquisition Geometry
Shift D — Case Complexity
```

### Shift A：Cross-dataset / Domain

定义：

- source：一个数据集（或同数据集内部 train/val 分布）；
- external：其他数据集，保持 label space 对齐后评估；
- 典型设置：CheXpert → PadChest / NIH；MIMIC-CXR → CheXpert / PadChest / NIH。

要求：

- 所有 external 评估使用同一 label mapping；
- 不进行 target-domain fine-tuning，除非作为额外实验；
- 报告数据集特征差异表：label prevalence、view、resolution、patient demographics（若可用且合规）。

### Shift B：Image Quality / Acquisition

必须建立医学有意义的 degradation taxonomy，而不是只加 Gaussian noise：

| Degradation | 参数建议 | 医学/采集对应 |
|---|---|---|
| Blur | Gaussian blur σ = 0.5 / 1.0 / 2.0 | 运动模糊、对焦不足 |
| Noise | Gaussian σ = 0.02 / 0.05 / 0.10 | 低剂量、探测器噪声 |
| JPEG compression | quality = 70 / 40 / 20 | PACS 压缩、传输损失 |
| Brightness | factor = 0.8 / 1.2 / 1.5 | 曝光变化 |
| Contrast | factor = 0.7 / 1.3 / 1.6 | 窗宽窗位差异 |
| Resolution reduction | downscale 1/2 / 1/4 / 1/8 | 低分辨率、远距离采集 |
| Gamma / windowing | γ = 0.7 / 1.4 | 显示/后处理差异 |

### Shift C：View / Acquisition Geometry

- PA vs AP；
- portable vs standard；
- lateral 是否可用（很多 finding 在 lateral 上定义不同，需谨慎）；
- 若 metadata 不可靠，只报告可见子集。

### Shift D：Case Complexity

- single finding；
- multiple findings；
- No Finding；
- conflicting / uncertain labels；
- low-prevalence finding；
- report unclear / uncertain（若有人工或 report evidence）。

## 3. Severity 定义

每个 shift 需要有两套 severity：

### 物理参数 severity

用于可复现生成：

```yaml
blur:    {mild: 0.5, moderate: 1.0, severe: 2.0}
noise:   {mild: 0.02, moderate: 0.05, severe: 0.10}
jpeg:    {mild: 70, moderate: 40, severe: 20}
```

### 医学可解释 severity

用于论文叙事和临床意义：

- mild：肉眼可见但不影响主要诊断；
- moderate：可能影响边界/细节判断；
- severe：明显损害诊断信息。

要求：

- 先用 pilot 检查 severity 是否真的造成单调性能变化；
- 保留物理参数与医学标签的 mapping 表；
- 不根据 clean accuracy 反向挑选 severity；先固定参数，再看结果。

## 4. Paired benchmark 设计

所有 shift 必须与 clean 成对：

```text
clean image I
  ├── blur(I)
  ├── noise(I)
  ├── jpeg(I)
  ├── brightness(I)
  ├── contrast(I)
  ├── downsample(I)
  └── masked(I)
```

这样可以使用 paired test / bootstrap，并消除病例组成差异。

每个图像-模型-条件组合保存一条长表记录：

```text
dataset, image_id, model, prompt_id, shift_type, severity,
label, prediction, confidence, uncertainty, correct,
raw_output_hash, runtime, cost
```

## 5. Benchmark 对照条件

| 条件 | 目的 |
|---|---|
| clean source | 上界参考 |
| clean external | cross-domain 主体 |
| quality shift | 图像质量鲁棒性 |
| view shift | 采集视角鲁棒性 |
| complexity shift | 多标签/困难病例 |
| blank image | language prior / image reliance |
| severe corruption | 是否仍盲目自信 |
| region masking | 关键区域依赖 |

## 6. Prompt 协议

所有模型使用同一套 prompt template，但允许最小适配：

- 固定 system / instruction；
- 固定 finding list 与 JSON schema；
- 只改变问题的表述，不改变答案定义；
- self-consistency 实验中，使用 5 个语义等价 prompt；
- prompt 原文、version、hash 必须写入 config；
- 不针对某个模型 tuning prompt；若必须适配，单独记录并做敏感性分析。

## 7. 输出协议

统一输出：

```json
{
  "findings": {
    "cardiomegaly": "positive",
    "edema": "negative",
    "pleural_effusion": "positive"
  },
  "confidence": {
    "cardiomegaly": 0.87,
    "edema": 0.93,
    "pleural_effusion": 0.81
  },
  "rationale": "short evidence-based rationale",
  "image_reliance_claim": "visual evidence mentioned or not"
}
```

解析时必须记录：

- JSON 是否合法；
- 是否缺 label；
- confidence 是否在 [0,1]；
- rationale 长度；
- parse failure 是否与 shift 相关；
- 若模型不能输出概率，使用 semantic / self-reported confidence 替代并明确标注。

## 8. Shift 分析维度

每个结果至少按以下维度分组：

- model；
- dataset / domain；
- shift type；
- severity；
- finding；
- No Finding vs abnormal；
- single vs multiple finding；
- view；
- prompt。

## 9. 数据与结果命名

```text
experiments/<phase>/<experiment_id>/
  config.yaml
  manifest.csv
  predictions.jsonl
  metrics.json
  summary.md
  figures/
  logs/
```

实验 ID 建议：

```text
<phase>_<model>_<dataset>_<shift>_<method>_<seed>
```

例如：

```text
02_baseline_gpt4o_chexpert_clean_conf_seed0
03_shift_chexpert_blur_moderate_unc_seed0
```

## 10. Benchmark 版本管理

- `benchmark_version`：数据、label mapping、prompt、shift 参数一起版本化；
- 任何修改必须 bump version；
- 最终论文报告 `v1.0` 固定版本结果；
- 旧版本结果保留在 artifacts，不覆盖，不删除。

## 11. 失败模式先验清单

在设计 shift 时记录并观察：

- 模型忽略图像，只输出高频 finding；
- domain shift 导致 prevalence prior 主导；
- severe degradation 下 confidence 不降反升；
- parse failure 增加；
- 模型对某些 finding 系统性过度预测；
- multi-finding 病例中“只答一个”；
- negation 被忽略；
- laterality / view 信息错误。
