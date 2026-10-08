# 17 — Unified Prompt & JSON Schema v1

> 目标：让所有被评测的 VLM 使用同一套结构化接口，便于跨模型比较和 parse failure 统计。
>
> **现行模型名单（2026-10-05）**：CheXagent-8b + MedGemma-4B-IT + Qwen2.5-VL-7B-Instruct（三层默认矩阵另含 MedGemma-15B-IT 与 Qwen3-VL-8B）。**GPT-4o 已放弃** —— §5/§6 里出现的 `gpt4o`、CheXpert manifest 路径属历史记录；**schema 本身（§2）未变**，仍是今天所有 runner 使用的接口。
> 三个评测集与 manifest 契约见 `docs/49_data_manifest_contract.md`；协议见 `docs/48`。

## 1. 文件位置

| 文件 | 作用 |
|---|---|
| `configs/prompts/prompt_v1.yaml` | prompt 版本、labels、output schema 定义 |
| `src/medvlm_trust/inference/prompts.py` | system prompt 与 user prompt builder |
| `src/medvlm_trust/inference/schema.py` | JSON extraction、schema validation、parse failure 标记 |
| `tests/test_schema_parser.py` | parser 单元测试（需安装 pytest；当前也可手动运行） |
| `scripts/run_smoke_test.py` | 用 manifest + 模型 wrapper 跑 20–100 张图 |

## 2. 统一输出 schema

```json
{
  "findings": {
    "atelectasis": "negative",
    "cardiomegaly": "positive",
    "consolidation": "uncertain",
    "edema": "negative",
    "pleural_effusion": "positive",
    "pneumonia": "negative",
    "pneumothorax": "negative",
    "no_finding": "negative"
  },
  "probability_positive": {
    "atelectasis": 0.12,
    "cardiomegaly": 0.88,
    "consolidation": 0.51,
    "edema": 0.20,
    "pleural_effusion": 0.82,
    "pneumonia": 0.15,
    "pneumothorax": 0.04,
    "no_finding": 0.18
  },
  "confidence": {
    "atelectasis": 0.88,
    "cardiomegaly": 0.84,
    "consolidation": 0.49,
    "edema": 0.80,
    "pleural_effusion": 0.81,
    "pneumonia": 0.85,
    "pneumothorax": 0.96,
    "no_finding": 0.82
  },
  "rationale": "Cardiac silhouette is enlarged; no clear pleural effusion.",
  "image_reliance_claim": "visual evidence used for cardiomegaly; uncertain for consolidation"
}
```

字段定义：

| 字段 | 定义 |
|---|---|
| `findings[label]` | `positive` / `negative` / `uncertain` / `not_mentioned` |
| `probability_positive[label]` | P(finding present \| image)，范围 [0,1]，用于 AUROC/AUPRC |
| `confidence[label]` | 对 `findings[label]` 判断的置信度，范围 [0,1]，用于 selective prediction |
| `rationale` | 简短图像证据说明；不用于 hallucination ground truth |
| `image_reliance_claim` | 模型是否声称使用了视觉证据 |
| Parse failure | JSON 非法、缺少 findings、类型错误、无法提取对象 |

## 3. 严格 parse 规则

- 优先提取第一个 JSON object；
- 允许 Markdown code fence；
- `findings` 缺失 → parse failure；
- 缺少 label → `not_mentioned`；
- 非法 status → `uncertain`；
- probability / confidence 超出 [0,1] 或非数值 → `null`，但不算整体 parse failure；
- parse failure 必须单独统计，不能直接丢弃样本；
- 同一图的 raw output、parsed output、parse error 都要缓存。

## 4. Prompt 设计原则

1. **图像优先**：明确要求只依据图像，不使用人口学或疾病流行率先验；
2. **固定 label set**：所有模型必须输出 primary 8 labels；
3. **区分 status 与 probability**：避免模型只给一个模糊 confidence；
4. **允许 uncertain**：不强迫模型在证据不足时二选一；
5. **禁止 Markdown**：减少 parser 负担；
6. **不暴露 ground truth**：prompt 中不包含标签或数据集身份；
7. **不针对单模型调 prompt**：若必须适配，单独记录为 model-specific prompt，并做敏感性分析。

## 5. 三模型 smoke test 协议

### 样本

- 从 CheXpert pilot manifest 中抽 20–50 张；
- 尽量覆盖 normal、single finding、multi-finding、uncertain；
- 固定 manifest 和图像顺序；
- 不重复使用 calibration/test 样本。

### 运行

```bash
python scripts/run_smoke_test.py \
  --manifest data/chexpert/manifests/pilot_100.csv \
  --images-root data/chexpert/raw/CheXpert-v1.0-small \
  --model dummy \
  --limit 100 \
  --output-dir artifacts/smoke/chexpert_dummy
```

真实模型需要先在 `src/medvlm_trust/models/` 实现对应 wrapper：

- `chexagent`
- `gpt4o`
- `qwen25_vl`

### 记录

每个模型必须输出：

```text
artifacts/smoke/<model>/
├── raw_outputs.jsonl
├── predictions.jsonl
└── parse_failure_summary.json
```

### Gate 判定

- parse failure ≤ 5%：继续；
- 5% < parse failure ≤ 20%：修 prompt / parser；
- parse failure > 20%：停止该模型，按 `docs/14_resource_locks.md` 替换为 MedGemma 1.5 或 Qwen2.5-VL-72B。

## 6. 当前局限

- 尚未接入真实模型；
- `probability_positive` 和 `confidence` 的语义可能被模型混用；
- GPT-4o 可能不能稳定提供 token-level confidence，因此只使用 verbalized confidence；
- Qwen2.5-VL-7B 的 logprob / confidence 可用性需要实测；
- CheXagent 的输出格式偏文本，parser 可能需要 model-specific fallback。
