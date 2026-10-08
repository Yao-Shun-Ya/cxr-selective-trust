---

> **历史快照（HISTORICAL SNAPSHOT）—— 本文是 Phase 0 的数据与标签协议，开头的 Resource lock 指针指向的是**已作废的 v1 锁定**（CheXpert 做 source）。**
> 实际落地的数据面见 `docs/49_data_manifest_contract.md`（三个评测集清单）与 `docs/48`；label space、split 硬约束、预处理与合规条款仍然有效。
> 正文原样保留。

# 02 — Data Protocol｜数据与标签协议

> Resource lock: `docs/14_resource_locks.md` 已锁定 CheXpert (source)、PadChest + NIH (external)、MIMIC (deferred Stage B)。

## 1. 数据策略总原则

1. **公开数据先跑通，credentialed data 后加入。**
2. **Pilot-first**：每个数据集先抽 500–1000 studies，不下载全量后再发现 pipeline 不兼容。
3. **Patient-/study-level split**：同一患者、同一 study 不得跨 train/val/test。
4. **标签协调优先于模型选择**：先定义统一 finding set，再写 loader。
5. **报告证据用于幻觉评估，但不能在推理时泄露。**
6. **任何 PHI、报告文本、访问凭证不得进入 git。**

## 2. 数据集与访问优先级

| 数据集 | 访问难度 | 阶段 | 主要用途 | 备注 |
|---|---|---|---|---|
| CheXpert | 公开申请 | 第一批 | internal / external | 标签体系成熟，适合快速启动 |
| NIH ChestX-ray14 | 公开 | 第一批 | external / cross-dataset | 规模大，标签来自 NLP，需注意噪声 |
| PadChest | 公开申请 | 第一批 | external / cross-dataset | 多标签丰富，适合外部验证 |
| MIMIC-CXR | Credentialed | 第二批 | 主 external / report evidence | 需 DUA；正式实验再加入 |
| Open-i | 公开 | 可选 | external | 规模小，可用于压力测试 |
| 合作医院数据 | 需 IRB/DUA | 升级包 | clinical external validation | 只有合作到位才纳入 |

### 访问顺序

```text
Stage A: CheXpert / PadChest / NIH
  ↓
跑通 loader、标签映射、pilot、metrics、API、缓存
  ↓
Stage B: MIMIC-CXR
  ↓
补充 report-derived evidence、negation、hallucination
  ↓
Stage C: optional additional external
```

## 3. 统一 label space

建议第一阶段锁定以下 CXR findings（可按数据集实际可用性调整）：

```text
Atelectasis
Cardiomegaly
Consolidation
Edema
Enlarged Cardiomediastinum
Fracture
Lung Lesion
Lung Opacity
Pleural Effusion
Pleural Other
Pneumonia
Pneumothorax
Support Devices
No Finding
```

### 标签协调规则

- 最终主分析使用 **8–14 个核心 finding**，避免标签过多导致 API 输出失稳；
- 每个 finding 映射为 `positive / negative / uncertain / not-mentioned`；
- 主指标默认计算 `positive`；对 uncertain 做敏感性分析；
- 明确 `not-mentioned` 不等于 `negative`；
- 每个数据集的原始标签映射必须写入 `configs/datasets.yaml` 和 `docs` 附录；
- 对 CheXpert 的 uncertain label 单独报告两种处理：uncertain-as-positive、uncertain-as-negative/excluded。

## 4. Pilot 采样方案

### 目标

用最小成本验证：

- loader 与标签映射正确；
- 模型输出能被结构化解析；
- shift 下确实有性能 / calibration 变化；
- metrics 脚本、缓存、日志、可视化可用。

### 规模

| 集合 | 规模 | 抽样策略 |
|---|---:|---|
| Pilot source | 500–1000 studies | 按 finding 分层，保证每个 finding 有足够正例 |
| Pilot external | 500–1000 studies | 与 source 同 label space |
| Quality shift subset | Pilot source 内 200–300 studies | 固定 paired subset，生成 mild/moderate/severe |
| Report evidence subset | 100–300 studies | 有人工可核查的 report / label |

### 分层抽样

- 按 `No Finding` vs abnormal；
- 按 finding 共现数量分层：single / multiple；
- 按 view：PA / AP / lateral（若 metadata 可用）；
- 按数据集常见 label 分布校正，但不要过度匹配导致不真实。

### 扩量条件

只有 Gate 1、Gate 2 通过后，才从 pilot 扩到 full evaluation。Full set 建议每个数据集 2,000–5,000 studies；若 API 成本高，可保留分层子集 + bootstrap CI。

## 5. Split 设计

### 评测导向 split

VLM 多为零样本 / few-shot 评估，因此本项目 split 主要是：

- `source_val`：用于 temperature scaling / threshold selection；
- `internal_test`：同分布测试；
- `external_test`：不同数据集；
- `shifted_test`：在固定 test subset 上施加 shift；
- `reliance_test`：用于 blank / corruption / masking。

### 硬约束

- 同一 patient 的图像只能出现在一个 split；
- 若数据集没有 patient ID，使用 study ID 或图像级 ID 并在限制中说明；
- 所有 shift 基于同一个 clean test subset 成对生成；
- calibration fitting 只能使用 source_val，禁止使用 test/external；
- 若使用少量 few-shot，必须固定 seed，并确保 few-shot 样本不来自 test。

## 6. 图像预处理协议

### 基础预处理

- DICOM/原始格式 → 8-bit PNG / RGB；
- 保持原始宽高比，短边 resize 到模型支持范围；
- 默认不做激进直方图均衡，以保留域特征；
- 保存 `preprocess_version`，所有实验引用同一版本；
- 模型接口若要求特定尺寸，使用统一 resize policy（如 fit + pad）。

### 元数据

尽量保留并写入 manifest：

- dataset, patient_id_hash, study_id, image_id
- view_position（PA/AP/lateral）
- laterality（若可用）
- original_width/height
- finding labels
- split
- shift_type, severity
- source_file_hash

禁止在仓库中保存原始 patient ID；使用 hash 或数据集内部匿名 ID。

## 7. Report-derived evidence 与幻觉评估

仅在允许访问 report 的数据集（如 MIMIC-CXR）上建立 evidence subset：

```text
report text
  ↓
section parsing
  ↓
finding / negation / severity / laterality extraction
  ↓
structured evidence table
  ↓
prediction vs evidence comparison
```

### 证据字段

| 字段 | 示例 |
|---|---|
| finding | pleural effusion |
| presence | present / absent / uncertain |
| negation | yes / no |
| severity | mild / moderate / severe / unknown |
| laterality | left / right / bilateral / unknown |
| evidence_sentence | 原文句子 |
| annotator | report-derived / manual |
| adjudication | 是否人工复核 |

### 禁止事项

- 不用 LLM 单独判定“是否幻觉”作为最终标签；
- 不把 report text 作为模型输入，除非实验明确研究 report-grounded setting；
- 不把 report 数据上传到未获批的外部 API。

## 8. 数据版本与 manifest

每次数据准备必须产出：

```text
data/manifests/
  manifest_<dataset>_<split>_<preprocess_version>.csv
  label_mapping_<dataset>.csv
  dataset_card_<dataset>.md
  checksums.txt
```

manifest 至少包含：

- image path / relative path
- dataset
- split
- patient_id_hash / study_id_hash
- labels
- view
- original size
- shift fields（若适用）

## 9. 数据质量检查清单

- [ ] 图像可读，无损坏文件
- [ ] 标签映射覆盖所有原始标签
- [ ] 每个 finding 正负样本数量与分布
- [ ] patient/study 无跨 split 泄漏
- [ ] 图像方向、laterality、view 是否正常
- [ ] 重复图像 / 近重复图像检查
- [ ] 数据集间 label distribution 差异表
- [ ] 数据使用许可与 DUA 状态记录
- [ ] 所有 API 使用前完成合规确认
