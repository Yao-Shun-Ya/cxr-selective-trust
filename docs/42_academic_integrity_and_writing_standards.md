# 学术诚信与顶级期刊撰写标准 (Academic Integrity & Writing Standards)

本规范基于 2026-10-02 确立的学术红线，作为后续所有论文撰写、数据分析及图表生成的绝对准则。目标期刊设定为 **TMI, MedIA, Nature Medicine** 等极度注重实证与科学诚信的顶级刊物。

## 1. 绝对的数据真实性 (Data Integrity First)
* **禁止预支数据**：任何未在本地硬件（如 4080 Super）上实际运行跑通的实验数据，绝不能提前写进论文作为“预期结果”或“伪造结果”。
* **反数据挖掘 (Anti Data Dredging)**：严禁事后倒推阈值（如为了好看的 3.4% 错误率去反向挑剔 $\alpha, \beta, \gamma$ 参数）。所有的阈值选择逻辑必须在 hold-out 验证集上产生，并透明地报告在测试集上的真实表现。

## 2. 拥抱“负性结果”的巨大价值 (Embrace Negative Findings)
* **如实报告困境**：如果复合风险信号在当前 VLM 架构下导致 Coverage 极低（例如风险分数甚至无法接手一个病例），必须**理直气壮且诚实地**作为核心发现写出。
* **拒绝强行圆场**：在当前多模态大模型的狂热泡沫期，用硬核数据证明“大模型依靠语言先验盲猜，一旦遇到分布偏移且缺乏视觉依赖就会灾难性失效”，这本身就是最具震撼力的顶刊级贡献，无需为了故事圆满而强行拼接一个“完美解决”的方法论。

## 3. 克制且深度的叙事基调 (Restrained & Empirical Tone)
* **摒弃营销词汇**：禁用浮夸、煽动性词汇（如 Illusion of Competence, Catastrophically fail）。采用客观、中立、数据驱动的描述（如 Model over-reliance on linguistic priors, Calibration degradation under distribution shift）。
* **保留复杂与异质性**：如果实验发现了“非单调、异质性”的现象（例如分布偏移某些情况下反而改善了 ECE），必须深入探讨背后的科学机制，绝不允许为了推销线性逻辑而抹除或简化这些真实发现。

## 4. 框架声明的诚实边界 (Honesty in Methodological Claims)
* **概念蓝图 vs. 成熟系统**：对于提出的“感知分布偏移的选择性信任框架（Shift-aware selective trust framework）”，在代码完全补齐并完成彻底的消融实验（Ablation Study）前，必须诚实地定性为**“由实证结果推导出的概念性蓝图 (Conceptual Blueprint)”**。
* **杜绝公式虚构**：绝不凭空捏造未落地的公式体系，每一行数学表达式必须与本地 `src/` 代码库中的计算逻辑严格一一对应。

**总结**：后续一切推进，恪守 `main_v2.pdf` 奠定的高度自洽与实事求是底线。让最真实的硬核数据去撕开 VLM 的华丽伪装。
