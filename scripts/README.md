# Scripts

CLI 入口目录。

> **现状（2026-10-05）**：本目录只有早期 bootstrap 阶段的四个脚本；**真正在跑的流水线在 program/scripts/**（`00`–`57`，入口是 `run_all_layers.ps1`）。下面列出的是实际存在的文件。

- `build_manifest.py` — 数据 manifest 与 split；
- `run_smoke_test.py` — 用 dummy 模型端到端跑通 smoke；
- `run_minimal_external_eval.py` — 最小 external 指标计算；
- `check_environment.py` — 环境体检。

规则：流水线由 `program/scripts/run_all_layers.ps1` 与 `program/configs/` 驱动；禁止硬编码实验参数，禁止手改结果文件。
