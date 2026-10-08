# Selective Trust for Medical Vision-Language Models under Distribution Shift

> **本仓库版本化的是代码、协议、配置与论文源文件。** 本地工作笔记（`HANDOVER.md`、`CONTINUE_HERE.md`、`docs/` 中的状态快照）不属于对外交付物，未纳入版本控制。

**分布偏移条件下医学视觉语言模型的选择性可信预测研究**

> 本目录是该论文的**唯一工程主目录（single source of truth）**：既承载研究设计、文献、协议、实验记录和论文材料，也承载后续全部代码、配置、结果与图表。

---

## Code availability（仓库范围与可复现获取路径）

本仓库按顶级期刊的 code availability 要求组织：**方法可审查，受限资产不重分发**。

> **审稿人请从 [`REPRODUCING.md`](REPRODUCING.md) 开始**：环境、权重获取、CPU-only 自检、
> 三层次评测入口与各项数字的真实状态都在那一份里。授权范围见 `LICENSE`（含明确的范围声明）
> 与 `NOTICE`（第三方权重与数据的许可免责声明）。

| 资产 | 是否版本化 | 获取方式 |
|---|---|---|
| 代码（`program/scripts`、`program/src`、`scripts`、`tests`） | ✅ 是 | `git clone` |
| 协议与数据契约（`docs/` 中 15 篇方法文档） | ✅ 是 | 见 `docs/` |
| 配置与提示词（`configs/`、`program/configs/`） | ✅ 是 | `git clone` |
| 论文源文件（`papers/`） | ✅ 是 | `git clone` |
| 模型权重（约 142 GB） | ❌ 否 | `configs/models.lock.yaml` 固定 HF repo id + revision；`python scripts/fetch_models.py --all` |
| 数据集（NIH / PadChest / Open-i / CheXpert-RRG） | ❌ 否 | 各数据集官方发布渠道与数据使用协议；本地清单契约见 `docs/49` |
| 虚拟环境 | ❌ 否 | `program/env/create_envs.ps1` + `requirements-*.txt` |
| 第三方论文 PDF（`references/pdf/`） | ❌ 否 | 版权归原文作者，请从出版方获取 |

权重不入库的三条理由：Git 单文件 100 MB 上限与 Git LFS 免费额度（10 GiB 存储 / 10 GiB 月带宽）无法承载；MedGemma 受 Gemma Terms of Use 与 HAI-DEF 约束禁止镜像权重、LLaVA-Med 为 research-use-only 不可再分发；期刊要求的是**可复现的获取路径**，不是权重副本。

**权威代码树**：`program/src/cxr_trust/` 与 `program/scripts/` 是当前执行路径，论文中的每一个公式都与这两处一一对应。顶层 `src/medvlm_trust/`、`scripts/`、`experiments/`、`tests/` 是立项期的早期骨架，保留用于追溯设计演进，**不作为结果的可复现依据**；`scripts/fetch_models.py` 是这一层里唯一仍在服役的脚本。

---

## 0. 当前状态

| 项目 | 状态 |
|---|---|
| 阶段 | **Phase 5：6K Curated Benchmark 代码架构就绪，等待 GPU** |
| 数据 | **已锁定 6,000 例精筛多中心数据**：NIH (评测集 1,732), PadChest (1500, 地理偏移), Open-i (947, 报告生成) |
| 三层次评测 | **一键驱动已就绪**：`program\scripts\run_all_layers.ps1`（默认 261,754 次调用 / 143.9 h）；崩溃安全与测试全通过；**7 模型 × 100 张 NIH 预跑已全部跑完**（发现见 `docs/50`/`docs/52`），**正式 6K 全跑尚未开跑**。协议见 `docs/48` |
| 模型 | **已锁定**：CheXagent-8b + MedGemma-4B-IT + Qwen2.5-VL-7B-Instruct；三层默认矩阵另含 `google/medgemma-1.5-4b-it` 与 Qwen3-VL-8B。注意不存在 "MedGemma-15B"：`1.5` 是 MedGemma 的修订版本号，两个 MedGemma 检查点均为 4B |
| 学术准则 | **严格遵循 `docs/42` 学术诚信红线**：拥抱 Negative Findings (零数据预支，无阈值捏造)，基准退回 `main_v2.pdf`。 |
| 核心现象 | shift 响应非单调；Qwen2.5-VL 存在极端的“无图盲猜（Visual Reliance ≈ 0）”语言先验幻觉；预跑深挖另见：图像依赖呈三梯队谱系、退化打击的是**阳性召回**而非宏观准确率（`docs/52`）。 |
| 工程架构 | `program/` 已重构三大主脚本支持自适应 Batch 并行与 OOM 防御回退。 |
| 决策原则 | **真实硬件上机前，不虚构任何统计数据，让物理实证说话。** |

### 上一版状态（Phase 4，2026-09 中旬；已被上方 Phase 5 状态取代，原样保留）

| 项目 | 状态 |
|---|---|
| 阶段 | **Phase 4 进行中：Main v2 已按评审完成排版与统计修订** |
| 数据 | **已锁定 v2**：NIH (source/internal) + CheXpert-plus-RRG (external report/reliance) |
| 模型 | **已锁定 v2**：CheXagent-8b + MedGemma-4B-IT + Qwen2.5-VL-7B-Instruct（全开源） |
| 文献 | **50 篇 matrix v1 + v1.1 更新已建立**；2025–2026 重点核验已形成初步 synthesis |
| 核心现象 | shift 响应非单调；image reliance 与 calibration 比 accuracy 更能暴露不可靠性 |
| 最近一个 Gate | **Gate 1/2 已通过 pilot 版本**：Qwen image-invariant；MedGemma image-dependent；shift 响应非单调；RRG 提供 external report/hallucination 信号 |
| 工程 | `program/` 已完成：Stage 2 实验、图表生成脚本、论文初稿骨架 |
| 决策原则 | 先 pilot，后 full benchmark；未过 Gate 不扩大实验规模 |

---

## 1. 一句话论文主张

> Existing medical VLMs can show promising diagnostic performance, but distribution shifts can induce severe and poorly calibrated failures. This project investigates whether model uncertainty can identify high-risk predictions and whether reliability estimates remain grounded in actual visual evidence, and develops a **shift-aware selective trust framework** that automatically accepts reliable predictions and defers uncertain cases.

中文：

> 医学 VLM 在分布偏移下会出现“性能下降但自信不降”的危险失败。本项目研究不确定性是否能识别高风险回答，以及这种可靠性判断是否真正依赖视觉证据，并据此构建 **shift-aware selective trust framework**，实现自动接受可靠预测、拒答高风险病例并转人工复核。

---

## 2. 研究问题（RQ）与核心假设

### RQ1 — VLM 在跨数据集 / 图像质量变化下究竟怎么崩？

- 哪一类 shift 最危险？
- 哪种疾病/标签最容易出错？
- 是否出现 **confidence 上升或保持，accuracy 下降**？

**H1**：跨数据集和图像质量 shift 会造成显著性能下降，且 calibration 的恶化可能比 accuracy 下降更严重。

### RQ2 — 模型的“自信”到底可信吗？

- confidence 是否能预测 correctness？
- shift 下 calibration 是否系统性失效？

**H2**：在 source domain 校准良好的 confidence，迁移到 external / degraded domain 后会显著 miscalibrated，形成 high-confidence error。

### RQ3 — 能不能自动识别“不应该信”的病例？

- 能否用 uncertainty / consistency / calibration 构造 risk score？
- Risk–Coverage 曲线能否支持“拒答换安全”？

**H3**：多源不确定性聚合可以显著优于 single confidence，并在固定风险下保持可用 coverage。

### RQ4 — 这个“风险判断”到底是不是看了图？

- image removal / corruption / region masking 后，预测和 confidence 是否发生合理变化？
- 模型高置信回答是否建立在视觉证据上？

**H4**：部分高置信预测主要依赖语言先验而非视觉证据；将 image-reliance 纳入 trust score 可提升 selective prediction 的安全性。

---

## 3. 创新定位

不要将论文定位成“又一个 calibration 方法”。本项目的研究对象是：

1. **Distribution shift**：跨数据集、图像质量、采集视角、病例复杂度。
2. **Selective trust**：不是只校准 confidence，而是决定何时接受、何时拒答。
3. **Image reliance**：风险判断是否建立在真实视觉证据上。
4. **External validation**：source → external 的完整评估链路。

一句话区分：

> CALM-VLM 类工作研究“如何校准 VLM”；本项目研究“模型在 domain shift 下还能不能判断自己什么时候应该被信任”。

> 注：2025–2026 相关文献结论需在 Phase 0 逐篇核验，尤其是 CALM-VLM、医学 VLM trustworthiness scoping review、跨数据集 CXR VLM evaluation 等工作。当前只能作为待验证情报线索，不能直接写入论文。

---

## 4. 最终交付物

### 科学交付物

- 1 篇 Q1 目标论文（先完成可投 Q2 的完整版本，再加强到 Q1 版本）。
- 1 个可复现的 **CXR distribution-shift + selective trust benchmark**。
- 1 个 **shift-aware selective trust framework**。
- 1 套 image-reliance / hallucination 评估协议。

### 工程交付物

- 数据加载、标签协调、分组切分、预处理 pipeline。
- 多模型统一推理接口（本地 VLM + API VLM）。
- shift 生成器、uncertainty 计算、calibration、selective prediction、image reliance、evaluation 模块。
- 实验配置、结果缓存、统计脚本、主图和主表。
- 论文 LaTeX / Markdown 草稿与 supplementary。

---

## 5. 阶段与 Gate 总览

| 阶段 | 时间 | 核心目标 | 关键产物 | Gate |
|---|---:|---|---|---|
| Phase 0 文献侦察 | 第 1 周 | 确认 gap，建立 30–50 篇矩阵 | Literature Matrix + gap statement | Gate 0 / Gate 1 前置 |
| Phase 1 最小 baseline | 第 2–3 周 | 跑通数据与 3 个 VLM | Pilot pipeline + structured outputs | Gate 1：shift 存在 |
| Phase 2 Shift benchmark | 第 4–6 周 | 建立跨数据集与质量 shift | 四轴 shift benchmark | Gate 2：可靠性退化 |
| Phase 3 核心现象 | 第 6 周 | 证明 confidence ≠ reliability | Figure 2/3/4 | Gate 2 |
| Phase 4 Uncertainty | 第 7–8 周 | 验证 uncertainty 预测错误 | Uncertainty score + calibration | Gate 3：uncertainty→error |
| Phase 5 Selective prediction | 第 8–9 周 | Risk–Coverage 曲线 | AURC / Coverage@Risk / selective accuracy | Gate 3 |
| Phase 6 Image reliance | 第 10 周 | 验证是否真正看图 | Blank/corrupt/mask 实验 | Gate 4：image reliance |
| Phase 7 Full method | 第 11 周 | 融合 shift-aware trust score | Full framework + ablation | — |
| Phase 8 统计与图表 | 第 12–13 周 | bootstrap、显著性、主图 | 7 figures + tables | — |
| Phase 9 写作投稿 | 第 14–16 周 | Methods/Results/Discussion/package | 投稿版本 | — |

更细的周计划、Gate 判据与止损规则是内部工作记录，不在本仓库；本仓库只发布方法与
协议类文档，完整清单见 §8。

---

## 6. 16 周执行表（压缩版）

| 周 | 阶段 | 必须完成 | 周末产物 |
|---:|---|---|---|
| 1 | Literature | 30–50 篇矩阵；核验 2025–2026 最新工作 | 文献矩阵与 gap 陈述 |
| 2 | Data + Pilot | 至少 2 个公开数据集 loader；每集 500–1000 例 pilot | 可复现 pilot split |
| 3 | Baseline | 3 个 VLM 统一接口；固定 JSON 输出 | Baseline metrics table |
| 4 | Internal evaluation | AUROC/AUPRC/F1/ECE/Brier | Internal results + 校准图 |
| 5 | Cross-domain | source → external 评测 | External degradation figure |
| 6 | Shift benchmark | blur/noise/compression/brightness/contrast/resolution | Shift taxonomy + severity curves |
| 7 | Uncertainty | self-consistency + perturbation + semantic disagreement | Uncertainty scores + error prediction |
| 8 | Calibration | 温度缩放/isotonic/shift-aware 对比 | Reliability diagrams + ECE/Brier |
| 9 | Selective prediction | risk–coverage + AURC + coverage@risk | 主图 Risk–Coverage |
| 10 | Image reliance | blank/corrupt/region mask | Image-reliance figure |
| 11 | Full method | trust score + ablation | Full method table |
| 12 | Statistics | bootstrap CI、paired tests、多重比较 | 统计附录 |
| 13 | Figures/tables | 7 张主图定稿 | camera-ready 图源 |
| 14 | Paper | Methods + Results | 论文初稿 |
| 15 | Discussion | 临床意义、限制、失败案例 | 完整 draft |
| 16 | Submission | cover letter + supplementary + 复现包 | 投稿包 |

---

## 7. 工程目录

```text
Paper/
├── README.md                     # 项目总入口
├── docs/                         # 研究设计、协议、Gate、论文大纲
├── configs/                      # 数据集/模型/shift/实验配置
├── data/                         # 原始数据（不提交、不泄露 PHI）
│   ├── chexpert/
│   ├── padchest/
│   ├── nih/
│   └── mimic/                    # credentialed access，后期加入
├── src/medvlm_trust/             # 核心 Python 包
│   ├── data/                     # loader、标签协调、split
│   ├── models/                   # VLM wrapper / API adapter
│   ├── inference/                # 批处理、结构化输出解析
│   ├── shifts/                   # blur/noise/compression/contrast/cross-domain
│   ├── uncertainty/              # self-consistency/perturbation/semantic
│   ├── calibration/              # ECE/Brier/temperature/isotonic
│   ├── selective_prediction/     # risk score / threshold / metrics
│   ├── grounding/                # blank/corruption/region masking/image reliance
│   └── evaluation/               # 指标、统计、bootstrap
├── experiments/                  # 按阶段编号的实验脚本与记录
├── scripts/                      # CLI 入口与一次性脚本
├── notebooks/                    # 探索性分析，不承载最终结果
├── figures/                      # 最终图
├── tables/                       # 最终表
├── papers/                       # 论文草稿
├── references/                   # 文献矩阵导出/BibTeX
├── artifacts/                    # 中间结果、缓存、预测文件
└── logs/                         # 运行日志
```

模块边界与命名规范见 `docs/08_reproducibility_and_governance.md`。

---

## 8. 文档索引

本仓库只发布**方法与协议类**文档（下表 15 篇）。项目宪章、决策日志、风险登记、
执行状态、pilot 结果分析、投稿策略等属于内部工作记录：它们含未定稿的中间数值、
对第三方数据集访问状态的断言，以及会被时间淘汰的计划条目，发布后极易与正文结论
分叉，因此**不在本仓库中**（`.gitignore` §6 默认排除 `/docs/**`，逐条白名单放行下表）。

| 文档 | 内容 |
|---|---|
| `docs/02_data_protocol.md` | 数据集、访问顺序、标签协调、切分、预处理 |
| `docs/03_benchmark_and_shift_protocol.md` | 四轴 shift taxonomy、severity、benchmark 设计 |
| `docs/04_uncertainty_calibration_selective.md` | 三类 uncertainty、校准、selective prediction |
| `docs/05_image_reliance_and_hallucination.md` | image removal/corruption/masking、幻觉 taxonomy |
| `docs/08_reproducibility_and_governance.md` | 数据治理、实验追踪、compute 预算、agent 分工 |
| `docs/17_prompt_and_schema.md` | 统一 prompt 与 JSON schema v1 |
| `docs/28_shift_benchmark_protocol.md` | 正式 shift benchmark 协议与变量清单 |
| `docs/29_chexpert_rrg_and_statistics.md` | RRG external、shift benchmark 与 bootstrap 统计协议 |
| `docs/42_academic_integrity_and_writing_standards.md` | 学术诚信与写作规范：禁预支数据、反数据挖掘、负性结果必须报告、公式须与 `src/` 一一对应 |
| `docs/43_hardware_profiling_and_wddm_oversubscription.md` | WDDM 显存超订实测与论文 Supplementary 表述 |
| `docs/44_batch_sweep_matrix.md` | RTX 4080 SUPER 32 GB 上 batch size 拐点实测矩阵 |
| `docs/45_weight_download_through_cn_mirrors.md` | 大陆网络下权重获取路径（`HF_ENDPOINT` 镜像）与 gated 仓库注意事项 |
| `docs/46_engineering_insights_supplementary.md` | 可进 Supplementary 的工程洞见框架（未定数值显式标 TBD） |
| `docs/48_three_layer_protocol.md` | **三层次 6K 评测权威协议**：三层设计、成本、崩溃安全、运行与分析命令、全部陷阱 |
| `docs/49_data_manifest_contract.md` | **manifest 契约**：必需列、positive/negative 字符串要求与原因、三个评测集清单、no_finding 修复 |

### 8.1 如何跑三层次 6K 评测（一键）

三层 = NIH（域内合成偏移）/ PadChest（跨医院原生偏移）/ Open-i（自由文本报告生成），
每层跑各自问题的设计，由一条命令驱动，**随时可断点续跑**。

```powershell
cd D:\Paper\program

# 0. 体检 + 全仓扫描 + 崩溃安全测试（都很快，都不加载模型）
.\.venv-vl\Scripts\python.exe scripts\39_preflight.py
.\.venv-vl\Scripts\python.exe scripts\45_debug_sweep.py
.\.venv-vl\Scripts\python.exe scripts\41_test_durable.py          # 37 passed, 0 failed
.\.venv-vl\Scripts\python.exe scripts\41_test_runner_resume.py    # 40 passed, 0 failed
.venv-vl\Scripts\python.exe scripts\49_test_legacy_durable.py   # 38 passed, 0 failed

# 1. 只看计划（不跑任何东西）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1 -DryRun

# 2. 小烟测（几分钟）
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1 `
  -Models medgemma_4b_it -Layers openi,padchest -Limit 4 -NoAblation

# 3. 正式全跑
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1
```

* 默认 5 模型 × 3 层 = **261,754 次模型调用 / 143.9 h（6.0 天）**（`-DryRun` 实测输出）。
* 常用参数：`-Models` / `-Layers nih,padchest,openi` / `-Limit` / `-BatchSize` /
  `-NoAblation` / `-Reset` / `-DryRun`。LLaVA-Med 与 InternVL 是最慢的两个模型，
  不进默认矩阵，需要时用 `-Models` 显式加入。
* **中断后重跑同一条命令即可续跑**：已完成 stage 会被 skip，未完成的由 runner 从
  durable 字节偏移继续（`DurableJsonl`：append → fsync → 原子字节偏移 sidecar）。
* 状态文件：`program\outputs\state\run_all_layers_state.json`；
  预测输出：`program\outputs\matrix\{nih_eval|padchest|openi}\<model>\`。
* 跑完之后（纯 CPU）：`scripts\42_geographic_shift_analysis.py`（阈值迁移 + bootstrap + 报告层）、
  `scripts\18_selective_metrics_v2.py`、`scripts\19_accept_clarify_abstain.py`。

> 完整的每层命令、成本对比、续跑语义与全部陷阱：`docs/48_three_layer_protocol.md`；
> 数据契约：`docs/49_data_manifest_contract.md`。

---

## 9. 未来 7 天立刻要做的三件事 (RTX 4080 到货后)

1. **执行 6K 高优精筛验证 (Curated Evaluation)**
   - **一键入口：`program\scripts\run_all_layers.ps1`（见 §8.1）**，先 `-DryRun` 看计划。
   - NIH (1,732 例): 使用 `data/manifests/nih_eval_1732.csv`（
     `nih_shared_5000.csv` 的 `label_no_finding` 全为 negative、且含 3,268 行训练数据，
     已被取代，**不要用于评测**），跑 `08_shift_benchmark.py` / `06_image_reliance.py` /
     `21_prompt_ablation.py`。
   - PadChest (1,500 例): `scripts/43_run_padchest_shift.py` —— 原生跨医院偏移 + 灰底对照（2 cell/图）。
     注意：PadChest 词表里**没有 pneumonia 标签**，其 pneumonia AUROC 不存在，只能写 limitation。
   - Open-i (947 例): `scripts/44_run_openi_report.py` —— 报告生成 + 灰底对照（2 cell/图），
     再由 `scripts/42_geographic_shift_analysis.py` 计算 omission / hallucination。
     Open-i 的 8 个标签列全部诚实地是 `unknown`，**不能做分类指标**。

2. **修正 4-bit / FP16 精度对比**
   - 通过 `22_benchmark_batch.py` 量化吞吐量，在本地 `.venv-vl-modern` 跑通新基线。
   - 利用分离环境解决 `Qwen3-VL` 需要 `transformers >= 4.57` 的依赖冲突。

3. **撰写最终论文图表与投稿准备**
   - 不再鼓吹 Coverage@5% 提高，将全文核心贡献收缩至 **"Safely rejecting hallucinations and unreliable priors under shift"**。
   - 利用 `19_accept_clarify_abstain.py` 产出的复合风险统计数据重绘核心结论图。

> **硬规则：在 4080 到位前，不启动大规模推理，所有数据抽样和协议均采用纯 CPU 准备。**

---

## 10. 十条工程与研究铁律

1. 数据永远不提交仓库；PHI、report 文本、下载链接和 access token 一律不进入 git。
2. 所有实验必须由 config 驱动，禁止 notebook 中手工改路径后直接产出“正式结果”。
3. 所有模型输出必须解析为统一 JSON；解析失败率单独报告。
4. 所有 split 必须 patient-/study-level group split，禁止同一病例跨 split。
5. 所有 calibration 结果必须同时报告 discrimination、calibration、selective risk。
6. 所有结论必须给 bootstrap 置信区间；关键比较使用 paired test。
7. 先 pilot 后 full；先 Gate 后扩规模；失败要止损，不追加沉没成本。
8. 幻觉不靠 LLM 自由裁量，必须有 report-derived evidence / ground truth 支撑。
9. 图表必须能从 artifact + config 一键重跑，不允许“手改数据点”。
10. 每个 Gate 结束时更新决策日志（ADR），写清楚继续/转向/放弃的理由。
