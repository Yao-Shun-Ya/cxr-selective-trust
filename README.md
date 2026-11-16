# 医学视觉语言模型的测量有效性审计

**Responsiveness Before Reliance: Four Reversals in Measuring Medical Vision–Language Models**

本仓库是一个可审计的医学 VLM 评测工程：两篇论文、一套冻结预测产物、一个跨模态扩展臂、
一个闭源大模型探针，以及把它们固定在可复现状态下的故障注入测试。

> 版本化的是源码、协议、配置与论文源文件。权重、数据、预测产物、虚拟环境与内部工作笔记
> 不入库（见 `.gitignore` 的分节说明与下文第 4 节）。

---

## 1. 当前状态

| 项目 | 状态 |
|---|---|
| 主论文 | **v7 已投 arXiv**（`papers/main_v7/`，39 页，44 条参考文献）；中文镜像 `papers/main_v7_zh/`（44 页） |
| 工程副论文 | **bench_v2 已投 arXiv**（`papers/bench_v2/`，16 页，31 条参考文献） |
| CXR 主矩阵 | **已完成**：6 模型 × 3 层，275,218 次调用，39 个冻结预测文件（0.49 GiB），0 丢失行 |
| 跨模态本地臂 | **未运行**：三模态清单与 22-cell 网格已冻结（314,688 次调用），作为 v9 增量 |
| 跨模态 API 臂 | **运行中**：`deepseek-v4.1-flash`，9,400 次调用（另有 2,700 次 lite 档） |
| 人评 | 未开始；v8 之后的外部增量 |

---

## 2. 两篇论文

### 2.1 主论文：四个反转

三个被常规报告的数字——退化鲁棒性、图像依赖度、跨院迁移——通常被当作模型的属性。
论文证明它们是**模型–测量系统**的属性，并由四个未声明的前提决定：

| 前提 | 检查 | 违反后的反转 |
|---|---|---|
| P1 响应性 | 真实图像上是否产生可变输出 | 六个模型里五个在 PadChest 上"图像盲"，1,500 张图只给出 1–20 个不同答案 |
| P2 零模型 | 反事实对比是否与同度量空间的 real–real 零模型比较 | Qwen3-VL 在 Open-i 上看似强依赖（Jaccard 0.09），实则空白回复是固定拒答串 |
| P3 操作点 | 召回之外是否报告阳性判读率 | 对比度退化看似提升召回 +4.69 pp，实则假阳性 1,646 → 1,905 |
| P4 契约覆盖 | 输出 schema 与解码模式是否声明 | 4-bit 看似把依赖压制 2.21×，实则模型切换了解码通道（0.13% → 99.07%） |

### 2.2 副论文：分母审计

benchmark 论文通常只报分数，不报分母。副论文把分母本身作为研究对象：

- 五种写入器在同一 2,000-cell 负载上的代价对照（裸追加 / 逐行 fsync / 整体原子重写 / SQLite-WAL / 本工程）。
- 十类故障注入：崩溃、尾部撕裂、双写并发、ENOSPC、fsync 失败、rename 失败、侧车损坏、侧车超前、只读文件、配置漂移。
- 分母普查：从产物本身重算每个文件的分母，259 个预测文件里 255 个分母均匀、1 个含重复键。

---

## 3. 工程地图

```text
Paper/
├── program/
│   ├── src/cxr_trust/
│   │   ├── inference/        # prompts, prompts_crossmodal, prompts_report,
│   │   │                     # schema（解析器）, shifts（唯一算子实现）, durable（单写者）
│   │   ├── models/           # registry + 各模型 wrapper
│   │   └── evaluation/       # AUROC / ECE / 选择性预测
│   ├── scripts/
│   │   ├── 08_shift_benchmark.py        # CXR 19-cell 网格
│   │   ├── 43 / 44                      # PadChest 原生偏移 / Open-i 报告生成
│   │   ├── 104 / 93 / 117               # 表格一致性 / 工件审计 / 中英数字比对
│   │   ├── 120–122                      # 跨模态：清单 / 推理 / P1–P4
│   │   ├── 123_harness_experiments.py   # 五写入器 + 十故障矩阵
│   │   ├── 124_artifact_census.py       # 分母审计
│   │   ├── 125_validate_crossmodal.py   # 跨模态预检
│   │   └── api_probe/                   # 闭源 API 臂（01–04 + common）
│   └── tests/                # 解析器 / 指标 / durable / 算子像素摘要
├── papers/
│   ├── main_v7/  main_v7_zh/            # 主论文（英/中）
│   └── bench_v2/                        # 副论文
├── docs/                     # 方法与协议文档（15 篇入库）
├── configs/                  # 模型锁文件与数据配置
└── data/                     # 数据集（不入库）
```

**三个固定点**（改动会破坏可复现性，须同时更新测试）：

1. `src/cxr_trust/inference/shifts.py` 是唯一算子实现；18 个退化 cell 的像素摘要被钉在
   `tests/test_shifts.py` 里，CXR 臂与跨模态臂不允许各写一份。
2. `prompts_crossmodal` 的契约段与 CXR 臂逐字相同，只换解剖名词与标签集。
3. 每份产物都携带 `provenance`（模型、manifest 哈希、prompt 哈希、图像几何、token 预算）；
   不一致时 runner 拒绝续跑，而不是把两次实验混进同一个文件。

---

## 4. 代码可用性

| 资产 | 是否入库 | 获取方式 |
|---|---|---|
| 代码、协议、配置、论文源文件 | ✅ | `git clone` |
| 模型权重（约 142 GB） | ❌ | `configs/models.lock.yaml` 固定 repo id + revision |
| 数据集（NIH / PadChest / Open-i / MRI / OCT / HAM10000） | ❌ | 各自官方渠道；清单契约见 `docs/49` |
| 预测产物与冻结分析表 | ❌ | 由脚本重跑生成；`outputs/` 不入库 |
| 第三方权重与数据的许可声明 | ✅ | `NOTICE` |

API 密钥只从环境变量读取，任何文档、日志与产物中都不得出现。

---

## 5. 复现入口

### 5.1 自检（CPU，无需权重）

```powershell
cd D:\Paper\program
.\.venv-vl\Scripts\python.exe -m pytest -q tests
```

覆盖解析器、指标、`DurableJsonl` 的续跑与撕裂尾部、以及**算子像素摘要**（10 项）。

### 5.2 CXR 主矩阵

```powershell
cd D:\Paper\program
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1 -DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1 `
  -Models medgemma_4b_it,medgemma_15_4b_it,qwen25_vl_7b_instruct,`
qwen3_vl_8b_instruct,chexagent_8b,internvl3_5_8b
```

中断后重跑同一条命令即可续跑；已完成 stage 跳过，未完成的行从侧车记录的字节偏移继续。

### 5.3 论文数字复核（全 CPU）

```powershell
python scripts\104_table_consistency_check.py   # 80/80
python scripts\93_artifact_audit.py             # 26/26
python scripts\117_paper_audit.py               # 39 张表跨语言零差异
```

### 5.4 跨模态本地臂（v9，尚未运行）

```powershell
python scripts\120_prepare_crossmodal.py --modality all
python scripts\125_validate_crossmodal.py        # 覆盖式预检：4 模型 × 3 模态
powershell -File scripts\run_crossmodal_queue.ps1
python scripts\122_analyze_crossmodal.py
```

### 5.5 闭源 API 臂

```powershell
cd D:\Paper\program\scripts\api_probe
$env:OPENAI_API_KEY = "..."          # 不入库
$env:OPENAI_BASE_URL = "https://amr-link.open-design.ai/v1"
$env:OPENAI_MODEL = "deepseek-v4.1-flash"

python 01_smoke_test.py               # 8 图像调用 + 1 纯文本基线
python 02_build_probe_manifest.py --profile full_multimodal   # 9,400 行
python 03_run_api_probe.py --dry-run
python 03_run_api_probe.py --profile full_multimodal
python 04_audit_api_results.py
```

设计与端点行为见 `program/scripts/api_probe/README.md`。

### 5.6 工程副论文的实验

```powershell
python scripts\123_harness_experiments.py     # 五写入器 + 十故障矩阵
python scripts\124_artifact_census.py --root outputs
```

---

## 6. 数据与模型

| 层 | 数据 | 规模 | 用途 |
|---|---|---|---|
| 域内 | NIH ChestX-ray14 | 1,506 计分例（manifest 1,732，含 226 供体行） | 19-cell 合成退化网格 |
| 跨院 | PadChest | 1,500 | 原生分辨率 + 空白对照 |
| 报告 | Open-i | 947 | 自由文本报告生成 + 空白对照 |
| 跨模态 | 脑 MRI（Figshare）/ 视网膜 OCT（Kermany）/ HAM10000 | 600 / 600 / 588 | 2×2 语料×物理设计 |

模型：CheXagent-8B、MedGemma-4B、MedGemma-1.5-4B、Qwen2.5-VL-7B、Qwen3-VL-8B（含 Thinking）、
InternVL3.5-8B；LLaVA-Med 作为被测量的排除项单独记录。

调用口径：6 模型矩阵 270,324 次；加 LLaVA-Med 的 5,294 次为 275,618 次；
41 阶段普查 284,695 次（含第二推理模型的 9,477 次与 LLaVA 的 4,894 次）。

---

## 7. 文档索引

入库文档只保留方法与协议类；项目宪章、决策日志、风险登记与状态快照不入库。

| 文档 | 内容 |
|---|---|
| `docs/02_data_protocol.md` | 数据、访问顺序、标签协调、切分 |
| `docs/03_benchmark_and_shift_protocol.md` | 退化分类与 severity |
| `docs/04_uncertainty_calibration_selective.md` | 不确定性、校准、选择性预测 |
| `docs/05_image_reliance_and_hallucination.md` | 图像依赖与幻觉分类 |
| `docs/08_reproducibility_and_governance.md` | 数据治理与实验追踪 |
| `docs/17_prompt_and_schema.md` | 统一 prompt 与 JSON schema |
| `docs/28_shift_benchmark_protocol.md` | 正式 shift 协议 |
| `docs/29_chexpert_rrg_and_statistics.md` | 报告层与统计协议 |
| `docs/42_academic_integrity_and_writing_standards.md` | 学术诚信红线 |
| `docs/43_hardware_profiling_and_wddm_oversubscription.md` | 显存超订实测 |
| `docs/44_batch_sweep_matrix.md` | batch 拐点矩阵 |
| `docs/45_weight_download_through_cn_mirrors.md` | 权重获取路径 |
| `docs/46_engineering_insights_supplementary.md` | 工程洞见 |
| `docs/48_three_layer_protocol.md` | 三层次评测协议 |
| `docs/49_data_manifest_contract.md` | manifest 契约 |

---

## 8. 十条铁律

1. 数据、PHI、报告文本、token 与密钥永不入库。
2. 正式结果只能由脚本产生，不能来自 notebook 里的手工路径。
3. 所有模型输出解析为统一 JSON；解析失败率与标签完整度单独报告。
4. 所有切分在患者或病灶层面不交叉。
5. 报告判别力时必须同时报告校准与操作点。
6. 关键比较给聚类 bootstrap 区间；配对比较用配对子集。
7. 先预检后全量；预检不通过不启动长跑。
8. 任何一个数字都必须能回到冻结产物，且能由第三方重算。
9. 图表由脚本从冻结 CSV 重生成，不允许手改数据点。
10. 负性结果与失败模式必须报告，不得只保留显著项。
