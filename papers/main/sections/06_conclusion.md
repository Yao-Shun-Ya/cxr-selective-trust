---

> **HISTORICAL SNAPSHOT — frozen Markdown mirror of the earlier `papers/main/` revision, last synced 2026-09-15.**
> It describes the Stage 2 pilot revision of the paper (1,000-study NIH pilot + 264-sample CheXpert-plus-RRG external set). `program/scripts/17_md_to_tex.py` generated `papers/main/sections/*.tex` from these `.md` files.
> Superseded by `papers/main_v2/sections/06_conclusion.tex` — `main_v2` is the current manuscript (19 pages); `main/` is kept only as the earlier revision.
> Do **not** hand-edit this mirror. See `docs/48_three_layer_protocol.md` §7 (2026-10-05).

# 6. Conclusion

We investigated whether open-source CXR VLMs can be selectively trusted under distribution shift and whether apparent reliability is grounded in visual evidence. Using a 1,000-study NIH pilot, a six-axis image-quality shift benchmark, counterfactual image-reliance tests, and a 264-sample CheXpert-plus-RRG external set, we found a sharp dissociation between structural validity and clinical reliability. MedGemma was image-dependent but only moderately discriminative and poorly calibrated. Qwen2.5-VL was effectively image-invariant yet produced valid structured outputs and apparently favourable selective metrics. Shift effects were heterogeneous and non-monotonic, and report-level hallucination and omission were present on external data.

These findings show that accuracy, calibration, or selective risk alone can mischaracterise CXR VLM trustworthiness. Reliable selective trust requires joint evaluation of calibration, uncertainty, shift response, and visual-evidence reliance. The shift-aware selective trust framework proposed here provides a step toward this goal and motivates future multicentre, reader-study, and prospective clinical validation.
