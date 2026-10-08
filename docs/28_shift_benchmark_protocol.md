---

> **历史快照（HISTORICAL SNAPSHOT）—— 本文是早期 shift benchmark 协议草案（当时按 NIH pilot 100–200 张设计）。**
> 正式的 shift 设计现以 `docs/48_three_layer_protocol.md` 为准（NIH 19 格合成退化 + PadChest 原生偏移）。
> 正文原样保留：四轴 shift taxonomy 与 severity 定义后来被沿用，是设计演进的底稿。

# 28 — Formal Shift Benchmark｜正式分布偏移实验协议

> 数据：NIH ChestX-ray14 pilot 100–200 张
> 模型：MedGemma-4B-IT（primary），Qwen2.5-VL-7B（grounding control）
> 目的：观察 shift 下 accuracy ↓、ECE ↑、image sensitivity ↓、selective risk ↑

---

## 1. Shift 条件

| Condition | Severity | Parameter |
|---|---|---|
| clean | none | original |
| blur | mild / moderate / severe | Gaussian σ = 0.5 / 1.0 / 2.0 |
| noise | mild / moderate / severe | Gaussian σ = 0.02 / 0.05 / 0.10 |
| jpeg | mild / moderate / severe | quality = 70 / 40 / 20 |
| brightness | mild / moderate / severe | factor = 0.8 / 1.2 / 1.5 |
| contrast | mild / moderate / severe | factor = 0.7 / 1.3 / 1.6 |
| resolution | mild / moderate / severe | downscale = 0.5 / 0.25 / 0.125 |

---

## 2. 每个 job 记录的变量

每个 `(image_id, condition, severity)` 保存：

```text
image_id
dataset
split
condition
severity
param
raw_output
error
elapsed_sec
parsed.parse_success
parsed.parse_error
parsed.findings[label].status
parsed.findings[label].probability_positive
ground_truth.label_*
```

目标：后续可以直接画：

- performance vs severity；
- ECE vs severity；
- image sensitivity vs severity；
- selective risk vs severity；
- per-finding degradation heatmap。

---

## 3. 自动分析变量

`09_shift_analysis.py` 会输出：

```text
shift_metrics_by_condition.csv
shift_metrics_by_finding.csv
shift_image_flips.csv
shift_image_sensitivity.csv
shift_coverage_by_condition.csv     ← 2026-10-06 新增
coverage_report.json                ← 2026-10-06 新增
shift_summary.json
```

> **覆盖率是随指标一起产出的，不是另一个没人打开的文件。** 2026-10-06 之前，
> `09` 的 `load_rows` 会静默丢弃两类行——`parse_success` 为假的行，以及 ground truth
> 既非 `positive` 也非 `negative` 的标签——而 `shift_metrics_by_condition.csv` 只报告
> 幸存者上的 `n_label_instances`，读起来像样本量，实际是**余数**。
> 现在每个 condition/severity 都带 `n_attempted` / `n_parse_ok` / `n_parse_fail` /
> `n_label_planned` / `n_label_instances` / `parse_coverage` / `label_coverage` /
> `n_gt_dropped`，低于 `--min-coverage`（默认 0.95）的格子会在终端打出 COVERAGE
> WARNING，并写入 `coverage_report.json`（`--fail-on-low-coverage` 可让它以退出码 3
> 结束）。
>
> 这不是形式主义：预跑里 `medgemma_15_4b_it` 的 `resolution/severe` 只有 **32%**
> 的标签实例存活（100 格中 68 格没解析出来），即在那个格子上计算出来的任何 AUROC
> 都是在一个**被条件筛选过的子集**上算的。丢弃与条件相关，偏差就不随机。

每个 condition/severity 记录：

```text
n
auroc
ece
brier
f1
high_conf_error
error_rate
aurc
coverage_at_5pct_risk
selective_accuracy_at_80pct_coverage
image_sensitivity
mean_flip_count
mean_abs_prob_diff
```

---

## 4. 运行

### MedGemma 100 张（推荐先跑）

```powershell
cd D:\Paper\program

.\.venv-vl\Scripts\python.exe scripts\run_shift_overnight.py `
  --models medgemma_4b_it `
  --limit 100 `
  --max-retries 3 `
  --retry-sleep 300
```

预计 job 数：

```text
100 × (1 clean + 6 shifts × 3 severities) = 1900 jobs
MedGemma 约 8–10 小时
```

### Qwen 100 张

```powershell
.\.venv-vl\Scripts\python.exe scripts\run_shift_overnight.py `
  --models qwen25_vl_7b_instruct `
  --limit 100 `
  --max-retries 3 `
  --retry-sleep 300
```

---

## 5. 输出

```text
outputs/shift_benchmark/medgemma_4b_it_100/
├── shift_predictions.jsonl
└── analysis/
    ├── shift_metrics_by_condition.csv
    ├── shift_metrics_by_finding.csv
    ├── shift_image_flips.csv
    ├── shift_image_sensitivity.csv
    ├── shift_coverage_by_condition.csv
    ├── coverage_report.json
    └── shift_summary.json
```

---

## 6. 下一步主图

1. `Figure shift-performance`: macro AUROC vs severity；
2. `Figure shift-calibration`: macro ECE vs severity；
3. `Figure shift-reliance`: image sensitivity vs severity；
4. `Figure shift-selective`: coverage@5% risk vs severity；
5. `Figure per-finding heatmap`: finding × severity AUROC/ECE 变化。

---

## 7. 统计接口

后续 `statistics` 脚本会对：

```text
clean vs mild / moderate / severe
condition × severity
finding × severity
```

做：

- image-level cluster bootstrap CI；
- paired bootstrap difference；
- McNemar / paired test；
- FDR 多重比较校正。
