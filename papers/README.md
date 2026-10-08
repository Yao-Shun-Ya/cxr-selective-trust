# Papers

本目录只发布**现行正文**：Main v2 的 LaTeX 源、正文实际嵌入的图、编译产物 `main.pdf`。

```text
papers/
└── main_v2/
    ├── main.tex            # 权威正文入口
    ├── main.bbl / refs.bib
    ├── Makefile            # `make` 编译，需 TeX Live / MiKTeX
    ├── sections/*.tex      # 00_abstract … 07_figures
    ├── figures/            # 正文嵌入的 fig01–fig10（pdf + png）
    └── main.pdf            # 19 页，2026-10-05 按修正后的数值重建
```

写作原则（本项目自设的硬约束，详见 `docs/42_academic_integrity_and_writing_standards.md`）：

- 所有 claim 必须对应可重跑的 artifact（`experiments/**/metrics.json` 一类）；
- 图表版本与实验 ID 绑定；
- 主文只放经过统计验证的结果；
- negative results 与 limitations 必须如实报告。

## 只改 `.tex`

`main_v2/main.tex` 与 `main_v2/sections/*.tex` 是**正文的唯一权威版本**。

## 未发布的历史副本（留在本地工作树，刻意不进仓库）

本地磁盘上另有 `papers/main/`（2026-09-15 的 v1 草稿，18 页）、
`main_v2/sections/*.md`（2026-09-15 / 2026-09-16 冻结的 Markdown 镜像，含**修正前的旧数值**）
以及 `main_v2/overleaf_package.zip`（2026-09-16 的快照，早于 2026-10-05 的数值重建）。
它们已在 `.gitignore` §11 中排除：同一仓库里并存互相矛盾的数值会违反 `docs/42` 的诚信红线，
也不利于审稿人核对。需要历史版本请直接向作者索取。

> 顺带说明：不存在 `.tex` → `.md` 的生成脚本。`program/scripts/17_md_to_tex.py` 方向恰好相反，
> 且会覆盖 Main v2 的手工修订，**不要运行它**。
