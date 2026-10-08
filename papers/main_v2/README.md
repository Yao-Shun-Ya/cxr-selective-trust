# Main v2 Paper Draft

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

- `figures/fig01_framework.*`
- `figures/fig09_selective_trust_method.*`
- `figures/fig10_clinical_workflow.*`
- `figures/fig02_image_reliance_main.*`
- `tables/stage2/table01_image_reliance.*`
- `tables/stage2/table08_main_performance_calibration.*`

Status: all seven sections drafted; `main.pdf` rebuilt 2026-10-05 on the corrected metrics.

> 图同时存在于 `papers/main_v2/figures/`（正文嵌入的这一份）与仓库根的 `figures/stage2/`；`13_make_figures.py` 的 `save()` 会同时刷新两处。
> `sections/*.md` 是与 `.tex` 并存的**已冻结镜像**（含修正前数值）—— 请只编辑 `.tex`。


## LaTeX draft

```text
main.tex
refs.bib
Makefile
sections/*.tex
```

Compile:

```bash
cd papers/main_v2
make
```

A TeX distribution (TeX Live/MiKTeX) is required.


## v2 revisions

- Fixed table numbering/captions and cross-references.
- Expanded bootstrap to 1,000 resamples with BH FDR correction.
- Recompiled `main.pdf` —— **19 pages** after the 2026-10-05 regeneration (ECE-bin + calibration-universe corrections; see `docs/48` §7).
- Removed the Chinese version at user request.
