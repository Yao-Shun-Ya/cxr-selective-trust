# Figures

最终主图输出目录。命名：

```text
fig01_framework.{png,pdf}
fig02_image_reliance_main.{png,pdf}
fig03_shift_auroc_severity.{png,pdf}
fig04_shift_ece_severity.{png,pdf}
fig05_shift_image_sensitivity.{png,pdf}
fig06_risk_coverage.{png,pdf}
fig07_rrg_report_errors.{png,pdf}
fig08_shift_macro_auroc_heatmap.{png,pdf}
fig09_selective_trust_method.{png,pdf}
fig10_clinical_workflow.{png,pdf}
```

> 以上是**磁盘上真实存在**的文件名。此前本文件写的是一套从未存在的规划名（`fig02_dataset_performance` 等），已更正。

规则：

- 图必须由脚本生成；
- 图中数据点必须来自 artifact 中的 predictions/metrics；
- 禁止手工修改数据点；
- 每张图记录生成脚本、config、git commit；
- 同时保存 `.pdf` 和 `.png`（预览用）。


## Stage 2 已生成（共 10 张）

- `figures/stage2/fig01_framework.*`
- `figures/stage2/fig02_image_reliance_main.*`
- `figures/stage2/fig03_shift_auroc_severity.*`
- `figures/stage2/fig04_shift_ece_severity.*`
- `figures/stage2/fig05_shift_image_sensitivity.*`
- `figures/stage2/fig06_risk_coverage.*`
- `figures/stage2/fig07_rrg_report_errors.*`
- `figures/stage2/fig08_shift_macro_auroc_heatmap.*`
- `figures/stage2/fig09_selective_trust_method.*`
- `figures/stage2/fig10_clinical_workflow.*`

> **同名文件同时存在于 `papers/main_v2/figures/`** —— 正文实际嵌入的是那一份。两处此前**互不关联**，重新生成图会产生「论文里还是旧图」的静默漂移；`program/scripts/13_make_figures.py` 的 `save()` 现已同时刷新两处（`docs/48` §7，2026-10-05）。
> `fig04_shift_ece_severity` 已按修正后的 ECE 约定重新生成（2026-10-05）。
