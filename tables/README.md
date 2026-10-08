# Tables

最终主表和附表目录。命名：

```text
table01_dataset_summary.csv
table02_main_performance.csv
table03_selective_prediction.csv
table04_ablation.csv
table05_hallucination_taxonomy.csv
tableS01_per_finding.csv
```

规则：

- 表格由 metrics/artifacts 自动生成；
- 每列指标必须写明定义与单位；
- 主表必须报告 95% CI 或标准差；
- 统计显著性标记需说明检验方法。


## Stage 2 已生成

- `tables/stage2/table01_image_reliance.*`
- `tables/stage2/table02_shift_medgemma.*`
- `tables/stage2/table03_shift_qwen.*`
- `tables/stage2/table04_rrg_report_errors.*`
- `tables/stage2/table05_rrg_summary.*`
- `tables/stage2/table06_bootstrap_medgemma.*`
- `tables/stage2/table07_bootstrap_qwen.*`
- `tables/stage2/table08_main_performance_calibration.*`
