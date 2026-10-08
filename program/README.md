# CXR Selective Trust — Program Directory

> 论文：**Selective Trust for Medical Vision-Language Models under Distribution Shift**
> 本目录是实际运行目录：数据、模型、脚本、配置、输出都在这里。

> **只想知道怎么跑 6K 三层次评测？** 直接跳到 [§0](#0-6k-三层次评测一键运行)。
> 本节后的 §1–§9 描述的是早期 pilot 流程，仍然可用，但已不是主流程。

---

## 0. 6K 三层次评测（一键运行）

三份数据各回答一个问题，所以各跑各的设计 —— 不是同一套 19 格网格跑三遍。

| 层 | 数据 | 问题 | 设计 | 格/图 | runner |
|---|---|---|---|---|---|
| NIH 评测集 | 1,732 | 同域：退化到哪一步信号先崩 | 19 格合成退化 + 4 格图像依赖 + bf16 对照 + 提示词消融 | 19 + 4 | `08` / `06` / `21` |
| PadChest | 1,500 | 跨院：换个国家还成不成立 | **原生偏移** + blank 对照（不叠合成退化） | 2 | `43_run_padchest_shift.py` |
| Open-i | 947 | 自由文本：遗漏与幻觉 | 散文报告 + blank 对照 → 判官比对 | 2 | `44_run_openi_report.py` |

**为什么 PadChest 不套 19 格网格**：PadChest 的价值本身就是「不同国家、不同机器」，
域差异已经在像素里。再叠高斯模糊和 JPEG 压缩，回答的是 NIH 层已经回答过的合成鲁棒性
问题。把 19 格铺满三层需要 524 小时，各司其职只要 **173.6 小时**（2026-10-06 实测重标定；此前报价 187.9 小时偏低，见 §11）。

```powershell
# ① 体检（约 20 秒，不用显卡）
.\.venv-vl\Scripts\python.exe scripts\39_preflight.py

# ② 崩溃安全与续跑回归（约 2 分钟，不用显卡；共 115 项断言）
.\.venv-vl\Scripts\python.exe scripts\41_test_durable.py           # 37
.\.venv-vl\Scripts\python.exe scripts\41_test_runner_resume.py     # 40（43/44 真子进程）
.\.venv-vl\Scripts\python.exe scripts\49_test_legacy_durable.py    # 38（06/08/10/21 真子进程）

# ③ 仓库 debug 扫描 + 文档/代码一致性
.\.venv-vl\Scripts\python.exe scripts\45_debug_sweep.py
.\.venv-vl\Scripts\python.exe scripts\53_check_doc_code_consistency.py

# ④ 只看计划与预算，什么都不跑
powershell -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1 -DryRun

# ⑤ 短程试水（**必须**同时隔离输出根与状态文件，见下）
powershell -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1 `
    -Limit 100 -OutputRoot outputs\pretest -StateFile outputs\state\pretest_state.json

# ⑥ 全量：261,754 次模型调用，173.6 小时；随时 Ctrl-C，重跑同一命令即续
powershell -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1

# 也可以按层或按模型切（最慢的模型排在每层最前）
powershell -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1 -Layers nih
powershell -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1 -Models chexagent_8b
```

### 跑前测试与正式全量的隔离（**必须成对使用**）

`-Limit` 只跑前 N 张时，**输出根与状态文件必须一起换**，否则会静默毁掉正式跑：

| 只换了一个 | 会发生什么 |
|---|---|
| 都不换 | 100 张的结果写进正式目录；状态文件把每个 stage 记为完成 → 正式跑全部 `[skip]`，**研究被永久卡在 100 张** |
| 只换 `-StateFile` | 状态跳过了，但 100 张仍写进正式目录 `outputs\matrix\`，durable writer 会把它们当作已完成工作 |
| 只换 `-OutputRoot` | 数据隔离了，但状态键 `layer/model/stage` 不含图片数 → 正式跑仍会跳过所有已完成的 stage |
| 两个都换 | 完全隔离。正式跑看不到、也不会跳过任何东西（**正确做法**） |

前三种都会被驱动脚本**主动拒绝**（exit 3），不会让主人误踩。
`45_debug_sweep.py` 会实跑六种组合（四种 `-Limit` 情形 + 两种全量）验证这个守卫没被改坏。

```powershell
# 跑前测试（隔离）
powershell -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1 `
    -Models medgemma_4b_it,qwen25_vl_7b_instruct -Layers nih `
    -Limit 100 -OutputRoot outputs\pretest -StateFile outputs\state\pretest_state.json

# 之后正式跑：两个默认值，与上面的数据完全不重合
powershell -ExecutionPolicy Bypass -File scripts\run_all_layers.ps1
```

**预算从哪来**：`run_all_layers.ps1 -DryRun` 与独立的 `scripts\47_cost_table.py`
用同一套成本模型各算一遍，两者必须吻合（`53_check_doc_code_consistency.py` 会核对）。
数字对不上就是有 bug —— 这个检查上线前，估算公式曾把批加速除了两次，报价比实际低 1.5 倍。

跑完后的纯 CPU 分析：

```powershell
.\.venv-vl\Scripts\python.exe scripts\42_geographic_shift_analysis.py `
    --source-dir outputs\matrix\nih_eval\medgemma_4b_it `
    --source-manifest data\manifests\nih_eval_1732.csv `
    --target-dir outputs\matrix\padchest\medgemma_4b_it `
    --target-manifest data\manifests\padchest_shift.csv `
    --report-dir outputs\matrix\openi\medgemma_4b_it `
    --output-dir outputs\geographic\medgemma_4b_it --n-boot 1000
```

**断电了会怎样**：每个阶段的每一批都 `fsync`，并原子写入一个字节偏移侧车文件
（`*.progress.json`）。重开时先把偏移之后的字节全部截掉（撕裂的尾行不会被当成结果），
再折叠被取代的尝试（同一个 key 只保留一行，优先保留成功的那次），然后接着跑。
硬故障（驱动/OOM）会以非零码退出并把阶段标为未完成，不会假装成功。

- 阶段级跳过：`outputs\state\run_all_layers_state.json`
- 行级续跑：每个 jsonl 旁边的 `*.progress.json`
- 协议全文：`..\docs\48_three_layer_protocol.md`
- 清单契约：`..\docs\49_data_manifest_contract.md`

---

## 1. 当前资源

### 数据

```text
素材/ChestX-ray14/
  data/Data_Entry_2017_v2020.csv
  data/train_val_list.txt
  data/test_list.txt
  data/images/images_001.tar.gz ... images_012.tar.gz

素材/CheXpert-plus-RRG/
  findings section/valid-00000-of-00001.parquet   # 62 samples
  impression section/valid-00000-of-00001.parquet # 202 samples
```

### 模型

```text
models/chexagent8b/CheXagent-8b-main        # CheXagent-8b
models/medgemma/MedGemma-4B-IT              # MedGemma-4B-IT
models/qwen/Qwen2.5-VL-7B-Instruct          # Qwen2.5-VL-7B-Instruct
```

### 论文任务调整

- **NIH ChestX-ray14**：source / internal，多标签分类，主 calibration / selective / shift 实验。
- **CheXpert-plus-RRG**：external report / image-reliance 集；有参考 findings/impression，适合 hallucination、image reliance、report-grounded 评估。
- 注意：CheXpert-plus-RRG 没有直接的 14 类 classification labels；如果要做 external classification，需要先从 reference report 抽标签，并标记为 `report-derived`。

---

## 2. 环境准备（第一次运行）

> 当前全局 Python 环境不能直接跑模型：
> - `torch 2.1.0` 是 CPU-only；
> - `transformers 4.57.3` 与当前 torch 不兼容；
> - `accelerate`、`qwen-vl-utils`、`bitsandbytes` 缺失。
>
> 必须创建独立环境。

### Windows PowerShell

```powershell
cd D:\Paper\program
.\env\create_envs.ps1
```

会创建：

```text
.venv-vl           # Qwen2.5-VL + MedGemma
.venv-chexagent    # CheXagent-8b
```

### Git Bash

```bash
cd /d/Paper/program
bash env/create_envs.sh
```

### 检查环境

```powershell
./.venv-vl/Scripts/python.exe scripts/run_overnight.py `
.\.venv-vl\Scripts\python.exe scripts\00_check_env.py
```

---

## 3. 数据准备

### 3.1 生成 manifests

```powershell
./.venv-vl/Scripts/python.exe scripts/run_overnight.py `
.\.venv-vl\Scripts\python.exe scripts\01_build_manifests.py
```

输出：

```text
data/manifests/nih_full.csv
data/manifests/nih_pilot_1000.csv
data/manifests/chexpert_rrg_all.csv
data/chexpert_rrg/images/...
```

### 3.2 解压 NIH 图像

只解压 pilot 需要的 1000 张：

```powershell
./.venv-vl/Scripts/python.exe scripts/run_overnight.py `
.\.venv-vl\Scripts\python.exe scripts\02_extract_nih_images.py `
  --manifest data\manifests\nih_pilot_1000.csv
```

全部解压（约额外 45GB）：

```powershell
./.venv-vl/Scripts/python.exe scripts/run_overnight.py `
.\.venv-vl\Scripts\python.exe scripts\02_extract_nih_images.py --all
```

解压后路径：

```text
data/nih/images/images/00000001_000.png
```

---

## 4. 运行模型

> 8GB RTX 4060 建议一次只加载一个模型。
> 7B/8B 模型如果显存不足，使用 `--load-in-4bit`，或者依赖 `device_map=auto` 做 CPU offload。

### Qwen2.5-VL

```powershell
./.venv-vl/Scripts/python.exe scripts/run_overnight.py `
.\.venv-vl\Scripts\python.exe scripts\03_run_model.py `
  --model qwen25_vl_7b_instruct `
  --manifest data\manifests\nih_pilot_1000.csv `
  --images-root data\nih\images `
  --output-dir outputs\pilot\qwen_nih `
  --limit 50 `
  --load-in-4bit
```

### MedGemma

```powershell
./.venv-vl/Scripts/python.exe scripts/run_overnight.py `
.\.venv-vl\Scripts\python.exe scripts\03_run_model.py `
  --model medgemma_4b_it `
  --manifest data\manifests\nih_pilot_1000.csv `
  --images-root data\nih\images `
  --output-dir outputs\pilot\medgemma_nih `
  --limit 50 `
  --load-in-4bit
```

### CheXagent

```powershell
.\.venv-chexagent\Scripts\python.exe scripts\03_run_model.py `
  --model chexagent_8b `
  --manifest data\manifests\nih_pilot_1000.csv `
  --images-root data\nih\images `
  --output-dir outputs\pilot\chexagent_nih `
  --limit 50
```

---

## 5. 评估

### 预测分布诊断

```powershell
./.venv-vl/Scripts/python.exe scripts/run_overnight.py `
.\.venv-vl\Scripts\python.exe scripts\05_diagnose_predictions.py `
  --predictions outputs\pilot\medgemma_nih\predictions.jsonl `
  --output-dir outputs\pilot\medgemma_nih\diagnosis
```

```powershell
./.venv-vl/Scripts/python.exe scripts/run_overnight.py `
.\.venv-vl\Scripts\python.exe scripts\04_evaluate.py `
  --predictions outputs\pilot\qwen_nih\predictions.jsonl `
  --output-dir outputs\pilot\qwen_nih\metrics
```

指标：

- AUROC / AUPRC / F1；
- ECE / Brier / high-confidence error；
- AURC / coverage@5% risk / coverage@10% risk / selective accuracy；
- error detection AUROC。

---

## 6. 推荐执行顺序

```text
1. create_envs
2. 01_build_manifests
3. 02_extract_nih_images --manifest ... --limit 1000
4. 03_run_model --model qwen25_vl_7b_instruct --limit 50
5. 03_run_model --model medgemma_4b_it --limit 50
6. 03_run_model --model chexagent_8b --limit 50
7. 04_evaluate each run
8. 对比 parse failure / AUROC / ECE / AURC
```

---

## 7. 目录说明

```text
program/
├── configs/                   # 数据、模型、prompt、实验配置
├── src/cxr_trust/             # 核心 Python 包
│   ├── data/                  # NIH / CheXpert-RRG / manifest
│   ├── models/                # CheXagent / MedGemma / Qwen wrappers
│   ├── inference/             # schema, prompts, runner
│   └── evaluation/            # metrics
├── scripts/                   # 00-04 可执行脚本
├── env/                       # 环境创建脚本
├── data/                      # manifest 和导出图像（不提交）
├── outputs/                   # 推理输出和 metrics
└── 素材/                      # 原始数据
```

---

## 8. 重要限制

- `CheXpert-plus-RRG` 只有 62 + 202 个样本，适合 report-level / image-reliance / qualitative evaluation，不适合单独作为大规模 classification external benchmark。
- 如果论文需要大规模 external classification，需要：
  - 用 CheXpert labeler 从 RRG reference report 抽标签；
  - 或后续补充更小的 external 数据集（如 VinDr-CXR）。
- 8GB GPU 对 7B/8B 模型很紧张；建议先 `--load-in-4bit`，不行再 CPU offload。
- CheXagent 与 Qwen/MedGemma 可能需要不同 transformers 环境，已拆成两个 venv。


---

## 9. 无人值守夜间运行

```powershell
cd D:\Paper\program
./.venv-vl/Scripts/python.exe scripts/run_overnight.py `
  --limit 1000 `
  --models medgemma_4b_it qwen25_vl_7b_instruct `
  --max-retries 3 `
  --retry-sleep 300
```

特点：

- 自动解压、推理、评估、诊断；
- 每步失败自动重试；
- 模型推理使用 `--resume`，中断后可继续；
- 写 `outputs/logs/overnight_*.log`；
- 写 `outputs/overnight/status.json`；
- 每个模型目录内有 `progress.log` 和 `progress.json`；
- Windows 下会阻止系统自动睡眠。

监控：

```powershell
Get-Content outputs\overnight\medgemma_4b_it_1000\progress.log -Wait
Get-Content outputs\overnight\status.json -Wait
```

---

## 10. 最近变更（2026-10-05）

一句话：**这一批全是「数值是怎么算出来的」的修正** —— 约定错了、样本宇宙是隐式的，两者都不报错，只是安静地把数字算歪。

### 10.1 ECE 分箱约定修正

`(0,0.1]` → `[0,0.1)`。原实现用了 `right=True`，而标准约定（Guo et al. 2017、sklearn）是左闭右开。
**这不是「测度为零」的差异**：8.44% 的存储概率正好落在分箱边界上（RRG 文件里是 24.7%），所以每一个 ECE 都被它影响 —— 2026-10-05 之前产出的所有 ECE 都是旧约定。

代码层面还发现同一函数有两份实现：`metrics.py` 与 `18_selective_metrics_v2.py` 里的本地副本。**本地副本已删除**，该脚本改为 import 唯一实现。

### 10.2 校准样本宇宙显式化

`04_evaluate.py` 原来用一个 `dropna` 同时处理两类指标，于是把 `uncertain` 行**从两类里一起丢掉**。现在：

| 指标类别 | 样本宇宙 |
|---|---|
| 概率类（AUROC / AUPRC / ECE / Brier） | 所有给出概率的行，**含 `uncertain`** |
| 分类与选择性（F1 / AURC / coverage） | 只有 definite 判定 |

代价被量化了：旧口径静默移除了 MedGemma `internal_test` **1,800 条判定中的 186 条（10.3%）**，而代码与文档都没有提过这件事。`metrics_by_label.csv` 新增 `n_prob` 与 `n_definite` 两列，把两个宇宙写进产物本身。

### 10.3 论文数字重算与图漂移修复

| 项 | 旧 | 新 |
|---|---|---|
| MedGemma `internal_test` macro AUROC | 0.612 | **0.622** |
| 同上 macro ECE | 0.232 | **0.220** |
| 同上 macro Brier | 0.198 | **0.188** |

Table 2 与 `fig04_shift_ece_severity` 已重生成，`papers/main_v2/main.pdf` 已重建（19 页，0 warning、0 undefined reference）。

另修一个**静默的图漂移**：`13_make_figures.py` 写的是 `figures/stage2/`，而正文嵌入的是 `papers/main_v2/figures/`，两者之间**本来没有任何关联** —— 重新生成图，论文里那张还是旧的，且不会有任何报错。现在 `save()` 同时刷新两处。

### 10.4 校准端点对齐

`no_finding` 温度缩放那一对的「前」端点是修正后约定算的、「后」端点却是修正前算的 —— 一条曲线两端不同口径。重跑 `07_calibration_selective.py` 让两端出自同一遍，`04` 与 `07` 现在逐位一致（0.35689）。

### 10.5 新增守卫（防止再犯）

`45_debug_sweep.py` 增加三个**行为**守卫（不只是「文件在不在、能不能解析」）：

1. §9 —— ECE 分箱约定；
2. §10 —— 阶段感知 `[BAD]` 守卫，**加载驱动脚本里的真函数**（此前是对着一份重新实现的副本断言，等于把同一个缺陷抬高了一层）；
3. §11 —— 校准样本宇宙。

§8 则实跑 `-Limit` 隔离的**全部六种**旗标组合。加上六个写预测 runner 的崩溃安全测试（`41_test_durable.py` 37 项、`41_test_runner_resume.py` 40 项、`49_test_legacy_durable.py` 38 项，**共 115 项断言全通过**），扫描脚本现在有 11 个 section。

> 协议侧同批记录见 `..\docs\48_three_layer_protocol.md` §7「2026-10-05」条目。

---

## 11. 最近变更（2026-10-06）

一句话：**预跑暴露的五个问题全部收口，成本模型从「每模型一个常数」改成「每 (模型, 层, 阶段) 一个实测值」。** 逐条证据见 `..\docs\50_resume_prerun_findings.md` §2。

### 11.1 消融阶段终于会批处理了（省约 35 小时）

`21_prompt_ablation.py` 加了 `--batch-size` 与 `generate_batch`（沿用 06/08/43/44 那套 `_BATCH_REJECTION_ERRORS` 回退契约：**只有**"这个 wrapper 不会批处理"才回退逐图，驱动/显存错误照旧抛出）。批大小不是猜的：五个 prompt 变体输出长度差别很大，`60_measure_ablation_batch.py` 实测 bs=1 是 5.67–9.93 s/cell，bs=32 全部落到 0.77–0.97（峰值 10.1 GiB / 空闲 25.8 GiB）。驱动用 `Get-AblationBatch` 只给实测过的模型发 32，其余一律封顶 8。

**回归测试也跟着改了**：`49_test_legacy_durable.py` 的 21 号用例现在跑**两遍**（`--batch-size 1` 与 `2`）。旧用例断言"模拟崩溃落在第 2 次模型调用=第 2 张图"，批处理一上线这个巧合就没了，报出来的是一句假的 durability 失败。

### 11.2 chexagent 不是显存估算问题，是三个叠在一起的问题（计划里 65.8 → 32.1 小时）

按 docs/50 §2.2 的 A→B→C 顺序实测下来，A（钉 `device_map={"":0}`）会**填满 32 GiB 卡**：31.15 GiB 的 fp32 权重装不进 30.6 GiB 空闲，加载本身就开始走驱动分页（100% "utilisation" / 82 W，单个 shard 两分钟）。而 `torch_dtype=torch.float16` 对这个 remote-code 类**被完全忽略**——`_diag_chexagent_dtype.py` 实测 config 记着 fp16、参数量却是 31.15 GiB 的 float32（`low_cpu_mem_usage=True` 也一样）。

最终修法：`62_convert_chexagent_fp16.py` 把 checkpoint 在盘上转成 fp16（31.15 → 15.58 GiB，同时修正 config 的 `torch_dtype`），包装器**不再传 device_map**（因为 `_load_pretrained_model` 一见到 device_map 就会做 `caching_allocator_warmup(factor=2)`，光这一步就足以让 15.58 GiB 的模型分页），改为 CPU 载入 → 显式转换 → 搬到 GPU，并在搬完后**校验驻留量**，超了就抛错而不是慢慢分页。

实测：驻留 15.61 GiB、0 个 meta/cpu 参数、单图 2.93 s（原本 38.69→118.05 s）、生产批大小下 **均值 2.92 s/cell（76 格，70/76 可解析；中位数 3.45、最小 1.26、最大 5.04）**。计划表里用的是**均值**：估算式是「调用数 × 秒/格」，需要的是总墙钟，中位数只在分布单峰时才等价于它。

### 11.3 llava_med 的真因是 `rope_theta`，不是对话模板

预跑里 1900 行里 1890 行 raw 为空、均值 0.22 s/cell——那不是"拒绝回答"，是立刻吐 `</s>`。`_diag_llava.py` 逐 token 排除了分词器与模板，并发现**加 BOS 反而更糟**（模型照样立刻 EOS；`LlavaProcessor` 端到端也复现）。真因是 29 号脚本重写的 config 丢了 `rope_theta`：旧 flat config 没有这个字段，transformers 于是静默用了默认 **10000**，而这批权重来自 `Mistral-7B-Instruct-v0.2`（RoPE base **1e6**）。

29 号脚本已改成幂等，并从仓库里那份参考转换 `outputs/benchmark/llavamed_ref/config.json` 补齐缺失字段。修完后模型确实生成了（且随图像变化），但**仍然不吐 JSON**（reliance 0/16）——它是 VQA 风格的模型，按红线**不放宽解析器**，继续留在 opt-in 名单外，并由阶段守卫标成可见失败。

### 11.4 medgemma_15 的思考通道：由「看不清的图」触发，也由 bf16 触发

原来记的"失败率与 prompt 相关"是**错的**：`precision` 与 `reliance` 跑的是同一个脚本、同一个 prompt，只差 `--load-in-4bit`。按条件拆开看：

| 阶段 / 精度 | original | blank | blur | shuffle | resolution / jpeg |
|---|---|---|---|---|---|
| reliance, 4-bit | 0/100 | **100/100** | 46/100 | 0/100 | — |
| shift, 4-bit | 0/100 | — | 25/300 | — | **117/300** / 47/300 |
| reliance, **bf16** | 100/100 | 100/100 | 100/100 | 100/100 | — |

也就是说：**图越无法支撑答案，它越要去"想"**——而 blank 正是图像依赖度的负对照。旧预算 256 token 下，blank 对照格 **100% 没有可用数据**。

修法是给这个模型单独的 token 预算（`$MaxNewTokensOverride`，1024；实测 256→0/8、512→0/8、768→6/8、1024→8/8），并把批次按条件分组（批的耗时由最长成员决定，按图像混编会让每个批都付出 1024 token 的代价——同一阶段实测：混编 bs=64 均值 **3.61 s/cell**，分组后 bs=32 均值 **2.34 s/cell**）。生产配置下三阶段实测均值：shift 4-bit **2.344**、reliance 4-bit **2.479**、precision bf16 **11.154** s/cell，可解析率 98.8% / 100% / 100%。预填充一个「空且已闭合」的思考块**行不通**（模型照样重新开始思考，0/8），这条也记在文档里以免有人再试。

### 11.5 成本模型拆到 (模型, 层, 阶段)，并带上来源

`$SecondsPerCall` 不再是一个模型一个常数（它曾把同一个数用在 4-bit shift、bf16 precision 和串行 ablation 上，于是同一轮里既有 27.6x 低估、也有 0.33x 高估）。现在查表顺序是「model/layer/stage → model/stage → 阶段别名 → 按模型兜底」，计划表里直接打一列 **rate src**（`measured` / `proxy-nih` / `default`）——PadChest 与 Open-i 尚未有各自实测，就老实标 `proxy-nih`，不假装是测过的。表里全部用**均值**（估算式需要总墙钟；中位数只在分布单峰时才等价，而思考通道恰恰制造了重尾）。`47_cost_table.py` 镜像同一套模型，`53` 号脚本继续把两者锁在 1 小时以内。当前总计：**261,754 次调用 / 173.6 小时（7.2 天）**，分层 NIH 158.9 h、PadChest 9.1 h、Open-i 5.7 h。

### 11.6 覆盖率随指标一起产出（不再静默丢行）

`09_shift_analysis.py` 过去把不能用的行直接丢掉：未解析的行、ground truth 非二值的标签，全都不留痕迹，而 `shift_metrics_by_condition.csv` 只在幸存者上报告 `n_label_instances`——读起来像样本量，其实是**余数**。现在每格都带 `n_attempted` / `n_parse_ok` / `n_parse_fail` / `n_label_planned` / `n_label_instances` / `parse_coverage` / `label_coverage` / `n_gt_dropped`，低于 `--min-coverage`（默认 0.95）会打出 COVERAGE WARNING 并写进 `coverage_report.json`（`--fail-on-low-coverage` 则退出码 3）。

预跑里这件事有多大：`medgemma_15_4b_it` 的 `resolution/severe` 只有 **32%** 的标签实例存活（100 格里 68 格没解析），`jpeg/severe` 69%、`blur/severe` 76%。丢掉的数据与条件相关，偏差就不是随机的。协议侧说明见 `..\docs\28_shift_benchmark_protocol.md` §3/§5。

### 11.7 键名审计、来源可见性，与覆盖率门禁的端到端验证（2026-10-06 续）

**审计**。同一类 bug 的实例是 `59_probe_pretest_findings.py` 读 `probability_pos`（schema 写的是 `probability_positive`）——`.get` 取不到键不会报错，于是那一列**每行都是 None**，下游拿它比较时静默换成了 status 列。为了让这种「读到的键根本没人写」不可能再漏，新增 `scripts/64_audit_row_keys.py` 并接成 `45_debug_sweep.py` 的 **section 12**：

- 写入面：schema（**实际调用** `ParsedPrediction.to_dict()` 得到，不是 grep 字符串）+ `outputs/` 里 85 个预测文件的全部键 + 六个 runner 的行字典（含尚无产物的 43/44）= 190 个键；
- 读取面：`scripts/` 与 `src/` 里所有 `x["key"]` / `x.get("key")` 字面量 = 391 个键；
- **结论：0 命中**——全仓没有任何读取键是某个写入键的截断/延长/2 编辑距离内变体。

这个检查器本身被校准过两次，值得记下来：只做编辑距离时它**漏掉了真 bug**（截断 5 个字符，不是 2 个），于是加了 `--selftest` 阳性对照（断言必须能抓到 `probability_pos`），sweep 会先跑对照再跑扫描——「检查器失明了还报 clean」是比不检查更坏的情况；随后又冒出约 30 条误报（`auc`/`batch`/`Path`/`n_findings` 等），全部是**同文件自己写的局部结构**，现在按「本文件写过的键不算读取」过滤掉。

同一个脚本还会普查**取值**：哪些写入键在抽样里永远是空。`error` 永远是空是设计（无生成错误）；`parsed.image_reliance_claim` 永远是空且**没有任何消费者**（未使用字段，不是静默丢数）；`parsed.findings.*.confidence` 永远是空——追到消费者后发现 `04_evaluate.py` 本来就是**从概率派生** confidence（与 07 同口径），只在「有明确判定但没给概率」时用 0.5 兜底：新格式文件上是 **2.38%**，而 04 真正的输入（pilot/overnight 扁平文件）上是 **0.00%**。所以那些选择性指标并非建在常数上——**我第一次的判断是错的，这里按证据改回来**；真正缺的是产物没说这件事：`metrics_by_label.csv` 现在带 `n_conf_from_model` / `n_conf_imputed` / `conf_source`，`04` 每次运行都会打印补值占比，`table08` 也多了一列 `conf_source`。

**顺手纠正一处陈旧产物**。按新 schema 重生成六份 metrics 时，四份与已发布产物**逐位一致**（含论文 table08 依赖的两个 `*_1000`），两份（`overnight/medgemma_4b_it_2`、`pilot/medgemma_nih`）数值有变化——原因是它们的旧版本**缺 `n_prob`/`n_definite` 两列**，即停留在 2026-10-05「校准样本宇宙显式化」之前。它们不进论文表格，`docs/24` 也没有引用其点位数值，现已更新为当前口径。备份在 `..\backups\metrics_pre_provenance\`。

**覆盖率门禁端到端实跑**（此前只做了 `-DryRun` 与单独跑 09）：

| 路径 | 怎么触发 | 结果 |
|---|---|---|
| 失败 | 隔离根 `outputs\gatecheck` 放预跑的 medgemma_15 shift 文件，state 里把阶段全标完成（GPU 不参与） | `exit 3`，摘要 `1 of 1 model(s) have cells below 95%`，并明确「不是推理失败」 |
| 通过 | 同上换成 qwen25 的干净文件 | `exit 0`，`coverage_ok: true`，`min_label_coverage: 1.0` |
| 交接 | 真跑一次 4 张图的 NIH 流程 | 阶段写出 `shift_predictions.jsonl` → 驱动随即在**刚写的文件**上产出 `coverage_report.json`（19 格 / 76 cell / 100% 解析）→ `exit 0` |


