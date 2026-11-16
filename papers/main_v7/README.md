# Responsiveness Before Reliance — v7.0

39 pages, 7 figures, 39 tables (main text 13, appendix 26), 44 references,
0 overfull boxes, 0 undefined references or citations.

## What changed from v6.0

v7 executes the eight bucket-2 items that v6 left open and folds their outcomes
into the argument; no new long-running training run is included.

| Item | Result | Where |
|---|---|---|
| 072 DenseNet-121, 19-cell grid + PadChest | contrast-severe recall 0.567 -> **0.588**, positive rate 0.360 -> **0.378**, macro AUROC flat 0.733 -> 0.729; PadChest **0.515** | §5.2, §5.9, Table `tab:densenetgrid` |
| 077 second-family schema sweep (Qwen3-VL) | blank-image pathology recall **0.000** under v0/v1/v4; contract signature is schema-specific (completeness 12.5% / 100% / refusal) | §5.12.1, `tab:r2xfamily` |
| 083 paraphrase on Qwen3-VL | real-image agreement **93.1% / 90.3%**, kappa 0.723 / 0.701; blank answers unchanged (1.000), unlike MedGemma p1 | §5.12.3, `tab:r2para` |
| 080 true fixed-rule rerun (InternVL3.5) | fixed 0.5 rule: contrast 0.510 -> **0.575** recall, 0.404 -> **0.449** positive rate; brightness 0.510 -> **0.584**, 0.404 -> **0.457** | §5.2, `tab:fixedrerun` |
| 082/067 batch determinism + seed | batch 1/4/8: exact reply 0.88/0.87, status 7.96/7.97 of 8, parse 1.000 (limitation, no bitwise claim) | Limitations |
| 081 LLaVA-Med NIH subset | two replies, 98.7% modal, 0% structured parse; added to the audit | §5.1, `tab:audit` |
| 076 CheXbert cross-check of Open-i | omission ordering reverses (rho **-0.49**), hallucination ordering holds (0.71); "one operating point" removed | §5.10, `tab:chexbert` |
| 064 GPU smoke CI | self-hosted preflight + single-image smoke + resume | `.github/workflows/gpu-smoke.yml` |

A verification pass also corrected the Qwen2.5-VL VRS values in the precision
control (0.008/0.025 -> 0.003/0.015, computed from the frozen predictions); the
reported range is therefore 0.87x--5.22x, and the conclusion ("not directionally
universal") is unchanged.

## What changed from v5.0

v5 left three gaps that a reviewer could call fatal, stated in its own
limitations: no supervised baseline (so chance-level VLM AUROC could be task
difficulty), a schema sweep on a single model, and marker evidence from a single
model. v6 closes all three with four confirmatory experiments on the same 1,506
studies and 31,626 additional calls, and reports one outcome the earlier design
could not see.

| Experiment | Result | Where |
|---|---|---|
| DenseNet-121 supervised baseline (3,871 disjoint-patient images, fixed 8 epochs) | macro pathology AUROC **0.733** vs 0.500--0.627 for the six VLMs | §5.12.1 |
| Five-schema sweep on MedGemma-1.5 (15,060 calls) | blank-image pathology recall **0.000** for all five schemas; real recall 0.493--0.614; three distinct blank failure forms | §5.12.2, Appendix F |
| Paraphrase control, MedGemma-4B (6,024 calls) | real-image agreement 0.909/0.914, $\kappa$ 0.62/0.64; blank-image answer flips on `no_finding` only | §5.12.3, Appendix F |
| Second reasoning model, Qwen3-VL-8B-Thinking (10,542 cells) | usable answers on **89.0%** at 5,120 tokens (89.2% after an 8,192-token sample); coverage 92.6% -> 80.9% clean -> resolution; paired recall -13.2 pp (resolution), -10.3 pp (noise) | §5.12.4, Appendix F |

## The claim that changed

v5 said the silent-failure quadrant (noise damages evidence without firing the
monitor) is a MedGemma-1.5 result. v6 keeps the two-axis structure but relocates
the quadrant: with a channel that watches whether reasoning *terminates*, noise
is loud (11.5% coverage loss, third-highest) instead of silent. **Silent failure
belongs to the monitor, not to the corruption.** The axis pair is cross-model
(noticeability--damage Spearman 0.486 in both); the quadrant assignment is not.

Two secondary claims were narrowed rather than strengthened:

- "The contract decides the form of the shortcut" -> the contract and the model
  *jointly* decide it; which schema yields which blank-image failure form does
  not transfer across checkpoints.
- "No usable discrimination anywhere" -> "no usable discrimination in these six
  VLMs, while a supervised baseline on the same labels reaches 0.733".

## Files

- `main.tex` — entry point, `Version 6.0`
- `sections/05b_round2.tex` — the four confirmatory experiments (new)
- `sections/09b_round2_appendix.tex` — their four appendix tables (new)
- `sections/00_abstract.tex`, `01_introduction.tex`, `07_limitations.tex`,
  `08_conclusion.tex` — updated for the new evidence
- `refs.bib` — 41 entries, unchanged

## Build

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Provenance of the new numbers

All round-2 numbers come from `program/outputs/analysis_round2/`:
`round2_contract.csv`, `round2_label_ci.csv`, `round2_marker.csv`,
`deep_schema.csv`, `deep_paraphrase_agreement.csv`, `deep_baseline_vs_vlm.csv`,
`deep_thinking_cells.csv`, `deep_thinking_paired.csv`, `budget_sensitivity.csv`,
`round2_audit.json`. The reasoning grid keeps only strictly valid replies
(closing tag plus JSON after it); the 1,141 non-terminating cells are archived in
`thinking_invalid_archive.jsonl` and drive the budget-sensitivity table. Scripts:
`96`, `97`, `98`, `100`, `101`.
