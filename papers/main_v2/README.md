# Main v2 — the published manuscript

Working title:

> Selective Trust for Medical Vision-Language Models under Distribution Shift

Status: all sections drafted; `main.pdf` is **19 pages**, rebuilt 2026-10-05 on the corrected
metrics (ECE-bin + calibration-universe corrections; see `docs/48_three_layer_protocol.md` §7).
The manuscript is anonymous (`\author{Anonymous Authors}`) for double-blind review.

## Layout

```text
main.tex              # authoritative entry point
refs.bib / main.bbl
Makefile
sections/
  00_abstract.tex
  01_introduction.tex
  02_related_work.tex
  03_methods.tex
  04_results.tex
  05_discussion.tex
  06_conclusion.tex
  07_figures.tex
figures/              # fig01–fig10, the copies actually embedded by 07_figures.tex
main.pdf
```

Compile:

```bash
cd papers/main_v2
make
```

A TeX distribution (TeX Live / MiKTeX) is required. `make` regenerates `main.pdf` from
`main.tex` + `sections/*.tex` + `refs.bib`; nothing here depends on files outside this directory.

## Edit `.tex` only

`main.tex` and `sections/*.tex` are the only authoritative text. Frozen Markdown mirrors and the
superseded v1 draft exist on the author's disk but are deliberately excluded from this repository
(see `papers/README.md` and `.gitignore` §11) — they carry pre-correction numbers and would put
contradictory values in one repo.

## Figures

`figures/` here holds the copies the manuscript embeds. The generating scripts live in
`program/scripts/` (e.g. `13_make_figures.py`) and read experiment artifacts under
`program/outputs/`; the generated stage-2 tables and figures are not version-controlled, since
they are reproducible from code + config and are large.

## v2 revisions relative to v1

- Fixed table numbering/captions and cross-references.
- Expanded bootstrap to 1,000 resamples with BH FDR correction.
- ECE binning and calibration-universe corrections, then `main.pdf` recompiled (19 pages).
- Removed the Chinese version at author request.
