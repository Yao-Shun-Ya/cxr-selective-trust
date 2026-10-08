# Papers

存放论文草稿、LaTeX source、supplementary、cover letter。

建议结构：

```text
papers/
├── main/
│   ├── main.tex
│   ├── sections/
│   └── figures/
├── supplementary/
└── submission/
```

写作原则：

- 所有 claim 必须对应 `experiments/**/metrics.json`；
- 图表版本与实验 ID 绑定；
- 主文只放经过统计验证的结果；
- negative results 与 limitations 必须如实报告。


## Current main version（现行）

- `main_v2/main.pdf` —— **19 页**（2026-10-05 按修正后的数值重建）
- `main_v2/main.tex`、`main_v2/sections/*.tex` —— **正文的权威版本，只改这里**
- `main_v2/overleaf_package.zip`
- `main_v2/figures/` —— 正文实际嵌入的图

## 早期版本（保留，勿当现行）

- `main/` —— v1 草稿（2026-09-15，18 页）；其 LaTeX 由 `program/scripts/17_md_to_tex.py` 从 `main/sections/*.md` 生成

> **注意 `sections/*.md`**：`main/sections/*.md` 与 `main_v2/sections/*.md` 都是**已冻结的 Markdown 镜像**（分别冻结于 2026-09-15 / 2026-09-16），含修正前的旧数值。
> **不存在** `.tex` → `.md` 的生成脚本（`17_md_to_tex.py` 方向恰好相反，且会覆盖 Main v2 的手工修订）。
> 请只编辑 `.tex`；不要手改 md 里的数字 —— 两份手工维护的副本正是它们当初分叉的原因。
