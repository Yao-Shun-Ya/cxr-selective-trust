---

> **历史快照（HISTORICAL SNAPSHOT）—— 本文描述的是 `papers/main/` 这一版：Methods + Results 初稿，基于 2026-09-15 的 Stage 2 pilot 结果。**
> 现行论文是 `papers/main_v2/`（`main.pdf` 19 页，2026-10-05 重建）；`papers/main/` 只作早期版本保留。
> 其 `sections/*.tex` 由 `program/scripts/17_md_to_tex.py` 从 `sections/*.md` 生成 —— 正文顶部另有「已冻结镜像」横幅，不要手改。
> 正文原样保留。

# Main Paper Draft

Working title:

> Selective Trust for Medical Vision-Language Models under Distribution Shift

Current draft sections:

```text
sections/00_abstract.md
sections/01_introduction.md
sections/02_related_work.md
sections/03_methods.md
sections/04_results.md
sections/05_discussion.md
sections/06_conclusion.md
```

Key assets:

- `figures/stage2/fig01_framework.*`
- `figures/stage2/fig09_selective_trust_method.*`
- `figures/stage2/fig10_clinical_workflow.*`
- `figures/stage2/fig02_image_reliance_main.*`
- `tables/stage2/table01_image_reliance.*`
- `tables/stage2/table08_main_performance_calibration.*`

Status: Methods + Results first draft based on real Stage 2 pilot results.


## LaTeX draft

```text
main.tex
refs.bib
Makefile
sections/*.tex
```

Compile:

```bash
cd papers/main
make
```

A TeX distribution (TeX Live/MiKTeX) is required.
