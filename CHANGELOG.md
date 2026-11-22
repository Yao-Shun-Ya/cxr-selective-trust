# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
manuscript generations (`v7`, `v8`, `v9`) as its release axis; the audit CLI has
its own semantic version.

## [Unreleased]

## [9] - 2026-11-22

### Added

- `audit/` — `cxr-trust-audit` 0.1.0, a CPU-only, installable CLI (`cxr-audit`)
  with three subcommands: `census` (denominator, duplicate keys, parse rate),
  `responsiveness` (distinct/modal status vectors, P1 change) and `selective`
  (confidence/entropy/margin AUROC and coverage at 5%/10%), plus a synthetic
  `demo`. Four tests run without weights, data or network.
- Root `CITATION.cff` and `.zenodo.json` so GitHub, Zenodo and reference managers
  can generate a software citation; `preferred-citation` points at the manuscript.
- CI job for the audit CLI (install, tests, demo, `pip check`) and Makefile
  targets `audit-install`, `audit-cli-test`, `audit-demo`.
- Evaluation manifests are now versioned (`program/data/manifests/*.csv` and
  `MANIFEST_SHA256.json`).

### Changed

- Main manuscript v9 (50 pages): four-model cross-modal panel (32,184 calls),
  frozen-encoder readout probe, input-geometry sweep, selective-prediction
  baselines and a request-level call ledger. Bench manuscript v3 (18 pages) adds
  the remote-API failure section.
- `README.md` and `REPRODUCING.md` now lead with the CPU-only audit path and the
  current v9/bench_v3 state.
- Citation corrections: Khanmohammadi et al. updated to arXiv:2606.15910v3;
  Akhtar et al. (2609.21763), Hao & Floresca (2610.28000) and Kale & Bajwa
  (2611.14975) added to the concurrent-work discussion.

### Fixed

- Multiplicity correction across the twelve null contrasts, a single call
  accounting in the main text, and the data-source table for the three
  cross-modal corpora.

## [8] - 2026-11

### Added

- `papers/main_v8/` and the `deepseek-v4.1-flash` relay arm: 9,400 calls through
  an OpenAI-compatible relay, single provenance, no transport errors.
- Bench v3 remote-API section recording 9,400 calls and four failure classes.

## [7] - 2026-10

### Added

- `papers/main_v7/` (39 pages) and the Chinese mirror `papers/main_v7_zh/`.
- v7 controls: clean AUROC intervals, PadChest label hierarchy, fixed-rule rerun,
  CheXbert cross-check, determinism, DenseNet grid, cross-family schema,
  labeller agreement, LLaVA-Med NIH subset and paraphrase agreement.
- `papers/bench_v2/` (16 pages) on harness durability and denominator auditing.
