---

> **历史快照（HISTORICAL SNAPSHOT）—— 本目录的 `00_pilot/` … `08_ablation/` 脚手架**从未用于真实运行**（各目录内只有 `.gitkeep`）。**
> 实际实验由 `program/scripts/` 下的编号 runner（`00`–`57`）执行，产物在 `program/outputs/`；现行计划见 `docs/41`/`docs/48`。
> 正文原样保留。

# Experiments｜实验目录

每个实验一个子目录，结构固定：

```text
experiments/<phase>/<experiment_id>/
├── config.yaml
├── manifest.csv
├── predictions.jsonl
├── metrics.json
├── summary.md
├── figures/
└── logs/
```

阶段编号：

| 目录 | 阶段 |
|---|---|
| `00_pilot/` | 数据与 pipeline smoke test |
| `01_baseline/` | 模型 baseline |
| `02_cross_domain/` | source → external |
| `03_shift_benchmark/` | quality / view / complexity shift |
| `04_uncertainty/` | uncertainty 方法 |
| `05_calibration/` | calibration 方法 |
| `06_selective_prediction/` | risk-coverage |
| `07_image_reliance/` | blank / corruption / mask |
| `08_ablation/` | 消融 |

规则：

- 禁止覆盖历史结果；
- 每个 summary 必须记录结论和下一步；
- 正式结果必须关联 git commit 与 config hash。
