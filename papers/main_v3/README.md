# Main v3 — honest draft grounded in the 7-model × 100-image pre-run

Working title:

> Selective Trust for Medical Vision-Language Models under Distribution Shift

Status: **draft, not the published manuscript.** All sections drafted; `main.pdf` is
**22 pages**, compiled 2026-10-09 with **zero undefined citations/references** and no
LaTeX warnings. This v3 absorbs the three head-line assets from the pre-run deep mine
(`docs/52_pretest_deep_mine.md`) — the seven-model image-reliance spectrum, the
positive-recall dose-response, and the failure-mode taxonomy. It is distinct from an
earlier `main_v3` that fabricated data and was deleted; see `CONTINUE_HERE.md`
(Priority 0 warning).

The manuscript is anonymous (`\author{Anonymous Authors}`) for double-blind review.

## Honesty boundaries this draft must keep (docs/42)

- Every pre-run number (spectrum, recall, no-finding drift, slot usage) is a **point
  estimate on 100 images per model** and is labelled so in prose/table captions.
- The shift-aware selective trust framework is described as a **conceptual blueprint**,
  not a validated method: its predictive components await ablation in the full run.
- No number in `main_v3` comes from the full three-layer run (NIH 1,732 / PadChest 1,500
  / Open-i 947), which has not yet been run on GPU.
- Figures are the v2 main-model figures (two-primary-model 300-study / RRG pilot); the
  7-model tables are presented inline. Regeneration against full-run artifacts is a
  post-run task.

## Layout

```text
main.tex             # entry point; \input{}s sections 00–07
refs.bib
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
figures/             # fig01–fig10 (v2 copies, embedded by 07_figures.tex)
main.pdf
```

Compile (requires TeX Live / MiKTeX):

```bash
cd papers/main_v3
make
```

or, equivalently on Windows:

```powershell
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## v3 changes relative to v2

- Added the three-tier image-reliance spectrum (VRS gradient 0.36 → 0.00) across seven
  models, and the two within-family generational pairs.
- Added the accuracy-paradox analysis: degradation suppresses positive recall while
  macro accuracy stays flat; degradation-type-specific (resolution ≈ noise > JPEG >
  blur/brightness/contrast).
- Added the failure-mode taxonomy (default-normal / refusal-to-answer /
  prose-without-the-contract), none requiring a parse error.
- Expanded related work (Felizzi 2025, Buckley 2026, Zafar 2026, etc.) and softened
  "first" claims to "joint evaluation".
- All pre-run quantities are labelled n=100 and the framework is labelled conceptual.