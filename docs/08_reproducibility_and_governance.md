---

> **历史快照（HISTORICAL SNAPSHOT）—— 本文的治理条款仍然有效，但其中的执行示例（`gpt4o` / `chexpert` 的 experiment_id 与 CLI）属于已被放弃的模型与数据集。**
> 现行执行入口是 `program/scripts/run_all_layers.ps1`，数据契约见 `docs/49`，协议见 `docs/48`。
> 正文原样保留：实验追踪与「禁止手改数据点」的规则没有变。

# 08 — Reproducibility & Governance｜可复现性与工程治理

## 1. 目标

保证任何一张主图、一个主表、一条结论都能追溯：

```text
论文 claim
  → metrics.json
  → predictions.jsonl
  → experiment config
  → data manifest
  → model version / prompt version
  → git commit
```

## 2. 目录职责

| 目录 | 用途 | 是否提交 git |
|---|---|---|
| `docs/` | 研究设计、协议、决策 | 是 |
| `configs/` | 数据/模型/shift/实验配置 | 是 |
| `src/medvlm_trust/` | 可复用包 | 是 |
| `scripts/` | CLI 脚本 | 是 |
| `experiments/` | 按阶段编号的实验入口和 summary | 是（结果轻量摘要可提交） |
| `artifacts/` | 预测、缓存、中间结果 | 否 |
| `data/` | 原始数据和 manifest | 否（manifest 可提交匿名版） |
| `figures/` | 最终图表 | 是（图源需可重跑） |
| `tables/` | 最终表格 | 是 |
| `papers/` | 论文草稿 | 是 |
| `references/` | BibTeX、矩阵导出 | 是（PDF 视版权） |
| `logs/` | 运行日志 | 否 |

## 3. 命名规范

### 实验

```text
<phase>_<model>_<dataset>_<shift>_<method>_<seed>
```

示例：

```text
03_gpt4o_chexpert_blur_moderate_unc_seed0
06_chexagent_padchest_clean_selective_seed0
```

### 文件

- Python 模块：`snake_case.py`
- config：`<dataset>.yaml` / `<model>.yaml` / `<shift>.yaml`
- 输出：`predictions.jsonl`、`metrics.json`、`summary.md`
- 图：`fig0X_<short_name>.pdf/png`

## 4. Config 驱动

所有实验必须通过 config 运行，禁止 notebook 手工改路径产出正式结果：

```yaml
experiment_id: 03_gpt4o_chexpert_blur_moderate_unc_seed0
phase: 03_shift_benchmark
model: gpt4o
dataset: chexpert
split: pilot_external
shift:
  type: blur
  severity: moderate
method:
  name: uncertainty
  k: 5
seed: 0
metrics:
  - auroc
  - auprc
  - f1
  - ece
  - brier
  - aurc
  - coverage_at_risk
```

每个 config 必须记录：

- model version / API snapshot / checkpoint hash；
- prompt version；
- dataset version / manifest hash；
- shift 参数；
- random seed；
- 代码 git commit。

## 5. 环境管理

- Python 3.10+；
- 优先 `uv` + `pyproject.toml` 或 conda；
- 固定依赖版本，记录 `uv.lock` / `requirements.lock`；
- 模型推理环境与数据分析环境可分离；
- 记录 CUDA / driver / PyTorch 版本；
- 所有脚本提供 `--config` 入口。

## 6. 缓存策略

缓存 key 必须包含：

```text
model_version + image_hash + prompt_hash + preprocess_version + shift_params
```

规则：

- 同一 key 不重复调用 API / GPU；
- `predictions.jsonl` 是唯一事实来源；
- metric 重算不重新推理；
- 解析失败也保存原始输出和 error type；
- 缓存目录不进 git。

## 7. 实验追踪

建议使用：

- 本地 `experiments/**/summary.md` + `metrics.json` 作为最低要求；
- 可选 MLflow / Weights & Biases 做比较；
- 每个实验记录：
  - experiment_id
  - config hash
  - git commit
  - dataset version
  - model version
  - start/end time
  - cost
  - metrics
  - notes

## 8. Git 与分支

- `main`：稳定、可复现；
- `dev`：日常集成；
- `exp/<id>`：实验分支，可丢弃；
- 大文件、数据、权重、缓存绝不提交；
- 合并前检查 `.gitignore` 与 `git status`；
- 论文结果对应的 commit 必须打 tag：`paper-v1.0`。

## 9. Agent 分工

### Agent 负责 70–80%

- dataset loader
- preprocessing / manifest
- prompt generation
- batch inference
- API adapter
- JSON parser
- metric calculation
- calibration / selective prediction 实现
- bootstrap / statistics
- plots / tables
- ablation scripts
- experiment tracking
- 文档整理

### 人类负责 20–30%

- 研究假设是否值得做
- 实验设计是否科学
- 医学意义与临床解释
- label mapping 的临床合理性
- hallucination / image reliance 证据边界
- 论文故事与创新点
- Gate 决策：继续 / 转向 / 停止

## 10. 数据治理

- PHI 零容忍：不把 MIMIC 报告、DICOM header、patient ID 上传到外部 API；
- API 使用前确认数据使用条款；
- 数据集下载路径、许可、DUA 状态记录在 `data/README.md`；
- manifest 只保留 hash / 匿名 ID；
- 人工审核数据匿名化；
- 论文引用数据时遵守数据提供方要求。

## 11. API / Compute 预算

### 预算表模板

| 项目 | 单价 | Pilot 量 | Full 量 | 预估成本 | 备注 |
|---|---:|---:|---:|---:|---|
| API model A | — | — | — | — | — |
| API model B | — | — | — | — | — |
| GPU 租用 | — | — | — | — | — |
| 存储 | — | — | — | — | — |

### 控制规则

- Pilot 阶段总预算设硬上限；
- 每个实验前估算 token / 图片数 / 重试率；
- 先跑开源模型验证 pipeline；
- 失败重试上限 2 次；
- 超预算 20% 必须记录原因，超 2 倍触发停止条件。

## 12. 代码质量

- `ruff` + `black`；
- `pytest` 覆盖 loader、parser、metrics；
- 所有 metric 函数必须有单元测试或已知例子验证；
- parser 必须有 failure case 测试；
- 不把 notebook 当生产代码；
- 函数命名、类型注解、docstring 保持统一；
- 日志使用 `logging`，不要 `print` 正式实验关键信息。

## 13. 复现命令模板

```bash
# 1. 环境
uv sync

# 2. 数据准备
python scripts/build_manifest.py --config configs/datasets.yaml --dataset chexpert

# 3. 推理
python scripts/run_inference.py --config experiments/03_shift_benchmark/configs/blur_moderate.yaml

# 4. 指标
python scripts/compute_metrics.py --predictions artifacts/.../predictions.jsonl

# 5. 图表
python scripts/make_figures.py --experiment-id 03_shift_benchmark
```

## 14. 审计清单

- [ ] 数据版本与 manifest hash 已记录
- [ ] 模型版本/API snapshot 已记录
- [ ] prompt version 已记录
- [ ] random seed 已记录
- [ ] patient/study split 无泄漏
- [ ] calibration 未使用 test/external
- [ ] threshold 未使用 test/external
- [ ] 所有主结果有 bootstrap CI
- [ ] 所有图表可由脚本重跑
- [ ] PHI 不在 git / logs / artifacts 明文出现
