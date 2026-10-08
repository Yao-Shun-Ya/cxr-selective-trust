---

> **历史快照（HISTORICAL SNAPSHOT）—— 本文描述的是早期包 `src/medvlm_trust`。**
> 现行流水线全部使用 `program/src/cxr_trust`；`medvlm_trust` 只被 `scripts/` 下三个早期脚本（`build_manifest` / `run_smoke_test` / `run_minimal_external_eval`）与 `tests/` 引用。
> 正文原样保留。

# medvlm_trust

核心 Python 包。模块职责：

| Module | Responsibility |
|---|---|
| `data/` | dataset loader、label harmonization、manifest、split、preprocessing |
| `models/` | VLM wrappers、API adapters、model registry |
| `inference/` | batch inference、structured prompt、JSON parser、cache |
| `shifts/` | blur、noise、compression、brightness、contrast、resolution、cross-domain |
| `uncertainty/` | self-consistency、perturbation、semantic disagreement、aggregation |
| `calibration/` | ECE、Brier、NLL、temperature、isotonic、shift-aware calibration |
| `selective_prediction/` | risk score、threshold、risk-coverage、AURC |
| `grounding/` | blank/corruption/masking、image-reliance score |
| `evaluation/` | metrics、bootstrap、paired tests、tables |
| `utils/` | IO、logging、config、seeding |

原则：

1. 不在此包中硬编码数据路径；
2. 不把 API key 写入代码；
3. 所有模型输出统一转为 `PredictionRecord`；
4. 每个指标函数可单元测试；
5. 正式结果必须来自 config-driven script。
