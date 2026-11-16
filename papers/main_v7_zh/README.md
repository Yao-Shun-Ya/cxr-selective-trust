# 响应性先于依赖性 —— v7.0（中文版）

44 页，7 图，39 表（正文 13、附录 26），44 条参考文献，0 overfull，0 未定义引用或文献。

## 相对 v6.0 的变化

v7 执行了 v6 留待处理的八项桶二任务，并把结果并入论证；没有新增长时训练。

| 项目 | 结果 | 位置 |
|---|---|---|
| 072 DenseNet-121，19 格 + PadChest | 对比度 severe 召回 0.567 → **0.588**、阳性率 0.360 → **0.378**、宏 AUROC 平稳 0.733 → 0.729；PadChest **0.515** | §5.2、§5.9、表 `tab:densenetgrid` |
| 077 第二家族 schema 扫描（Qwen3-VL） | v0/v1/v4 下空白病理召回均为 **0.000**；契约签名随 schema 改变（完整度 12.5% / 100% / 拒答） | §5.12.1、`tab:r2xfamily` |
| 083 Qwen3-VL 改写对照 | 真实图像一致率 **93.1% / 90.3%**，κ 0.723 / 0.701；空白答案不变（1.000），与 MedGemma p1 相反 | §5.12.3、`tab:r2para` |
| 080 真固定规则重跑（InternVL3.5） | 固定 0.5：对比度召回 0.510 → **0.575**、阳性率 0.404 → **0.449**；亮度 0.510 → **0.584**、0.404 → **0.457** | §5.2、`tab:fixedrerun` |
| 082/067 batch 确定性 + 种子 | batch 1/4/8：原文一致 0.88/0.87，状态 7.96/7.97 of 8，解析率 1.000（写入局限，不主张逐位可复现） | 局限性 |
| 081 LLaVA-Med NIH 子集 | 两种回答、众数 98.7%、结构化解析率 0；已加入审计表 | §5.1、`tab:audit` |
| 076 CheXbert 交叉核验 Open-i | 漏报排序反转（ρ **−0.49**），幻觉排序保持（0.71）；删除"one operating point"普适表述 | §5.10、`tab:chexbert` |
| 064 GPU smoke CI | self-hosted 预检 + 单图冒烟 + 断点续跑 | `.github/workflows/gpu-smoke.yml` |

核验阶段还修正了精度对照中 Qwen2.5-VL 的 VRS 旧值（0.008/0.025 → 0.003/0.015，由冻结预测重算），因此依赖范围为 0.87×–5.22×；结论"方向不普遍"不变。

## 相对 v5.0 的变化

v5 在自己的 limitations 中承认了三个缺口：没有监督基线（因此 VLM 的随机水平 AUROC 可能只是任务太难）、schema 扫描只覆盖一个模型、标记证据只来自一个模型。v6 用四项确证性实验在同 1,506 例研究上补齐（追加 31,626 个格子），并报告了原设计看不到的一个结果。

| 实验 | 结果 | 位置 |
|---|---|---|
| DenseNet-121 监督基线（3,871 张无患者重叠图像，固定 8 epoch） | 宏 AUROC **0.733**（95% 患者聚类 CI 0.718–0.749），六个 VLM 为 0.500–0.627 | §5.12.1 |
| MedGemma-1.5 五套 schema 扫描（15,060 次调用） | 空白图像病理召回全部 **0.000**；真实召回 0.493–0.614；三种空白失效形态 | §5.12.2、附录 F |
| 改写对照，MedGemma-4B（6,024 次调用） | 真实图像一致率 0.909/0.914，κ 0.62/0.64；空白图像只翻转 `no_finding` | §5.12.3、附录 F |
| 第二个推理模型，Qwen3-VL-8B-Thinking（10,542 格） | 5,120 token 下 **89.0%** 可用（并入 8,192 抽样 89.2%）；覆盖率 92.6% → 80.9%；配对召回 −13.2 pp（resolution）、−10.3 pp（noise） | §5.12.4、附录 F |

## 结论上的三处变化

1. "没有可用判别力" → "这六个 VLM 没有，而同一批标签上监督基线达到 0.733"。
2. "契约决定捷径形式" → "契约与模型共同决定；哪种 schema 产生哪种空白失效不可跨检查点迁移"。
3. "噪声是静默失效" → "轴对通用、象限不通用：对 MedGemma 的标记通道静默，对'是否终止'通道响亮；静默失效属于监测器"。

## 文件

- `main.tex` — 入口，`版本 6.0`
- `sections/05b_round2.tex` — 四项确证实验（新增）
- `sections/09b_round2_appendix.tex` — 对应附录表（新增）
- 摘要、引言、局限、结论均已同步

## 编译

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## 数字来源

全部来自 `program/outputs/analysis_round2/`：`round2_contract.csv`、`round2_label_ci.csv`、`deep_schema.csv`、`deep_paraphrase_agreement.csv`、`deep_thinking_cells.csv`、`deep_thinking_paired.csv`、`paired_positive_rate.csv`、`budget_sensitivity.csv`、`round2_audit.json`。一致性由 `program/scripts/104_table_consistency_check.py` 自动核对。
