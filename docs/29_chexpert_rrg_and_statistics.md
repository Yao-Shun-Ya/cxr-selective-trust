---

> **历史快照（HISTORICAL SNAPSHOT）—— 本文是 CheXpert-plus-RRG（264 张）external 与 bootstrap 统计的早期协议。**
> 报告层现由 Open-i（947 张，`openi_rrg.csv`）承担，见 `docs/48`/`docs/49`；`chexpert_rrg_all.csv` 只作历史延续保留（`docs/49` §6）。
> 正文原样保留：其中「rule_based_v0 应在正式投稿前升级」至今仍是未结事项。

# 29 — CheXpert-RRG External + Statistics｜外部队列与统计协议

---

## 1. CheXpert-plus-RRG external

数据：`data/manifests/chexpert_rrg_all.csv`，264 张，参考 findings / impression 文本。

脚本：

```text
scripts/10_run_chexpert_rrg.py
scripts/11_rrg_report_analysis.py
scripts/run_rrg_overnight.py
```

运行：

```powershell
cd D:\Paper\program

.\.venv-vl\Scripts\python.exe scripts\run_rrg_overnight.py `
  --limit 264 `
  --blank-limit 50 `
  --max-retries 3 `
  --retry-sleep 300
```

输出：

```text
outputs/rrg/medgemma/
├── rrg_predictions.jsonl
└── analysis/
    ├── rrg_report_metrics_by_label.csv
    ├── rrg_hallucination_summary.json
    ├── rrg_image_reliance_per_pair.csv
    └── rrg_image_reliance_summary.json
```

当前 report evidence 方法：

```text
rule_based_v0
```

它使用关键词 + 否定窗口，用于 pilot。  
正式论文前应升级为 CheXpert labeler 或人工审核子集。

---

## 2. Shift benchmark

脚本：

```text
scripts/08_shift_benchmark.py
scripts/09_shift_analysis.py
scripts/run_shift_overnight.py
```

运行 MedGemma 100 张：

```powershell
.\.venv-vl\Scripts\python.exe scripts\run_shift_overnight.py `
  --models medgemma_4b_it `
  --limit 100 `
  --max-retries 3 `
  --retry-sleep 300
```

运行 Qwen 100 张：

```powershell
.\.venv-vl\Scripts\python.exe scripts\run_shift_overnight.py `
  --models qwen25_vl_7b_instruct `
  --limit 100 `
  --max-retries 3 `
  --retry-sleep 300
```

输出：

```text
outputs/shift_benchmark/<model>_100/
├── shift_predictions.jsonl
└── analysis/
    ├── shift_metrics_by_condition.csv
    ├── shift_metrics_by_finding.csv
    ├── shift_image_flips.csv
    ├── shift_image_sensitivity.csv
    └── shift_summary.json
```

---

## 3. Bootstrap + paired statistics

脚本：

```text
scripts/12_bootstrap_stats.py
```

输入：`shift_flat_predictions.csv`

```powershell
.\.venv-vl\Scripts\python.exe scripts\12_bootstrap_stats.py `
  --flat-csv outputs\shift_benchmark\medgemma_4b_it_100\analysis\shift_flat_predictions.csv `
  --output-dir outputs\shift_benchmark\medgemma_4b_it_100\statistics `
  --n-boot 1000
```

输出：

```text
bootstrap_by_condition.csv
paired_clean_vs_shift.csv
```

内容：

- image-level cluster bootstrap 95% CI；
- clean vs mild/moderate/severe 的 paired bootstrap；
- AUROC / ECE / Brier delta 与 p-value。

---

## 4. 后续图

1. Performance vs severity；
2. ECE vs severity；
3. Image sensitivity vs severity；
4. Coverage@5% risk vs severity；
5. Per-finding heatmap；
6. Clean vs shift paired delta forest plot；
7. RRG report hallucination + image reliance。
