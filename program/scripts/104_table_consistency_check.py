#!/usr/bin/env python
"""104 - Table/number consistency check.

Main-text numbers are written in LaTeX by hand.  This script re-derives the
load-bearing ones from the frozen CSVs and asserts that the exact strings used in
`sections/*.tex` are present.  It is deliberately dumb: a missing string is a
failure, so a table edit that forgets one of the places a number appears (abstract,
results, appendix) fails the check.

Usage::

    .\\.venv-vl\\Scripts\\python.exe scripts\\104_table_consistency_check.py

Exit code 0 = all checks pass, 1 = at least one mismatch.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AN = ROOT / "outputs/analysis_round2"
SECT = ROOT.parent / "papers/main_v7/sections"


def tex_blob() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(SECT.glob("*.tex")))


def main() -> int:
    blob = tex_blob()
    checks: list[tuple[str, bool, str]] = []

    def need(label: str, needle: str) -> None:
        checks.append((label, needle in blob, needle))

    # --- round-2 grid -------------------------------------------------------
    grid = [json.loads(l) for l in
            (ROOT / "outputs/matrix/nih_eval/qwen3_vl_8b_thinking/shift"
             / "shift_predictions.jsonl").open(encoding="utf-8") if l.strip()]
    checks.append(("grid rows", len(grid) == 9401, f"{len(grid)} rows (expected 9,401)"))
    need("coverage 89.0", "89.0\\%")
    need("coverage 89.2", "89.2\\%")

    # --- budget table -------------------------------------------------------
    b = pd.read_csv(AN / "budget_sensitivity.csv")
    post = int(b[b["pass"] == "post_5120"]["n_unusable"].sum())
    checks.append(("post_5120 residual", post == 1161, f"{post} (expected 1,161)"))
    arc = [json.loads(l) for l in (AN / "thinking_invalid_archive.jsonl").open(encoding="utf-8") if l.strip()]
    checks.append(("archive rows", len(arc) == 4283, f"{len(arc)} (expected 4,283)"))
    need("archive 4,283", "4,283")

    # --- paired recall ------------------------------------------------------
    pair = pd.read_csv(AN / "deep_thinking_paired.csv").set_index("cell")
    checks.append(("resolution delta", round(pair.loc["resolution/severe", "delta_paired"], 3) == -0.132,
                   str(pair.loc["resolution/severe", "delta_paired"])))
    need("resolution -13.2", "-13.2")
    need("noise -10.3", "-10.3")

    # --- per-label CI -------------------------------------------------------
    ci = pd.read_csv(AN / "round2_label_ci.csv")
    ci = ci[ci.experiment == "second_model"].set_index("label")
    checks.append(("pneumonia q", float(ci.loc["pneumonia", "q_bh"]) < 0.001, str(ci.loc["pneumonia", "q_bh"])))
    need("pneumonia -11.0", "-11.0")

    # --- DenseNet -----------------------------------------------------------
    m = json.loads((ROOT / "outputs/densenet_baseline/metrics.json").read_text())
    checks.append(("densenet macro", round(m["eval_1506"]["macro_pathology_auroc"], 4) == 0.7329,
                   str(m["eval_1506"]["macro_pathology_auroc"])))
    need("densenet 0.733", "0.733")
    need("internal_test 1,352", "1,352")

    # --- schema / paraphrase ------------------------------------------------
    s = pd.read_csv(AN / "deep_schema.csv")
    blank = s[s.condition == "blank"].groupby("variant")["recall"].mean()
    checks.append(("blank recall 0", bool((blank < 1e-9).all()), str(blank.round(4).to_dict())))
    need("paraphrase kappa", "0.62--0.64")

    # --- DenseNet grid / PadChest -------------------------------------------
    dc = pd.read_csv(ROOT / "outputs/densenet_baseline/condition_metrics.csv")
    cs = dc[(dc.condition == "contrast") & (dc.severity == "severe")].iloc[0]
    checks.append(("densenet contrast recall", round(float(cs["recall"]), 4) == 0.5878,
                   str(cs["recall"])))
    need("densenet 0.588", "0.588")
    need("densenet 0.378", "0.378")
    pcm = pd.read_csv(ROOT / "outputs/densenet_baseline/padchest_metrics.csv").iloc[0]
    checks.append(("densenet padchest", round(float(pcm["macro_auroc"]), 4) == 0.5153,
                   str(pcm["macro_auroc"])))
    need("densenet 0.515", "0.515")

    # --- CheXbert labeller cross-check --------------------------------------
    lc = json.loads((ROOT / "outputs/analysis_chexbert/labeller_comparison.json")
                    .read_text(encoding="utf-8"))
    rho_om = round(float(lc["spearman_omissions"]["rho"]), 2)
    rho_ha = round(float(lc["spearman_hallucinations"]["rho"]), 2)
    checks.append(("chexbert omission rho", rho_om == -0.49, str(rho_om)))
    checks.append(("chexbert hallucination rho", rho_ha == 0.71, str(rho_ha)))
    need("chexbert -0.49", "-0.49")
    need("chexbert 0.71", "0.71")

    # --- cross-family schema / paraphrase -----------------------------------
    xf = pd.read_csv(AN / "xfamily_schema.csv")
    v0b = xf[(xf.variant == "v0_simple") & (xf.condition == "blank")].iloc[0]
    checks.append(("xfamily v0 blank recall", abs(float(v0b["pathology_recall"])) < 1e-9,
                   str(v0b["pathology_recall"])))
    v1b = xf[(xf.variant == "v1_per_label_binary") & (xf.condition == "blank")].iloc[0]
    checks.append(("xfamily v1 completeness", round(float(v1b["probability_completeness"]), 4) == 1.0,
                   str(v1b["probability_completeness"])))
    need("xfamily 12.5", "12.5\\%")
    xp = pd.read_csv(AN / "xfamily_paraphrase.csv")
    p1 = xp[(xp.variant == "p1_paraphrase_short") & (xp.condition == "original")].iloc[0]
    p2 = xp[(xp.variant == "p2_paraphrase_second_reader") & (xp.condition == "original")].iloc[0]
    checks.append(("xfamily p1 kappa", round(float(p1["mean_kappa"]), 3) == 0.723, str(p1["mean_kappa"])))
    checks.append(("xfamily p2 kappa", round(float(p2["mean_kappa"]), 3) == 0.701, str(p2["mean_kappa"])))
    need("xfamily 93.1", "93.1\\%")
    need("xfamily 0.723", "0.723")

    # --- precision control (094) -------------------------------------------
    q = pd.read_csv(ROOT / "outputs/analysis_nih_mine/table_s6_quantization_delta.csv")
    qb = q[(q.model == "qwen25_vl_7b_instruct") & (q.condition == "blank")].iloc[0]
    checks.append(("qwen25 VRS 4bit", round(float(qb["vrs_4bit"]), 3) == 0.003,
                   str(qb["vrs_4bit"])))
    checks.append(("qwen25 VRS bf16", round(float(qb["vrs_bf16"]), 3) == 0.015,
                   str(qb["vrs_bf16"])))
    ratios = {}
    for m in ("medgemma_15_4b_it", "medgemma_4b_it", "qwen25_vl_7b_instruct"):
        row = q[(q.model == m) & (q.condition == "blank")].iloc[0]
        ratios[m] = round(float(row["vrs_bf16"]) / float(row["vrs_4bit"]), 2)
    checks.append(("precision ratios", (ratios["medgemma_15_4b_it"], ratios["medgemma_4b_it"],
                                        ratios["qwen25_vl_7b_instruct"]) == (2.21, 0.87, 5.22),
                   str(ratios)))
    need("precision 5.22", "5.22$\\times$")

    # --- batch determinism (082/096) ---------------------------------------
    dr = pd.read_csv(ROOT / "outputs/determinism/medgemma_4b_it/determinism.csv")
    exact1 = float((dr["exact_vs_1"] == 1).mean())
    exact2 = float((dr["exact_vs_2"] == 1).mean())
    mean1 = float(dr["status8_vs_1"].mean())
    mean2 = float(dr["status8_vs_2"].mean())
    checks.append(("determinism exact", round(exact1, 2) == 0.88 and round(exact2, 2) == 0.87,
                   f"{exact1:.2f}/{exact2:.2f}"))
    checks.append(("determinism status", round(mean1, 2) == 7.96 and round(mean2, 2) == 7.97,
                   f"{mean1:.2f}/{mean2:.2f}"))
    need("determinism 7.96", "7.96--7.97")

    # --- LLaVA-Med NIH subset (081/095) -------------------------------------
    ll = pd.read_csv(ROOT / "outputs/analysis_responsiveness_llava/nih_subset.csv")
    lo = ll[ll.condition == "original"].iloc[0]
    checks.append(("llava nih n", int(lo["n"]) == 200, str(lo["n"])))
    checks.append(("llava nih distinct", int(lo["distinct"]) == 2, str(lo["distinct"])))
    checks.append(("llava nih top share", round(float(lo["top_share"]), 3) == 0.987,
                   str(lo["top_share"])))
    need("llava 98.7", "98.7\\%")
    need("calls 275,618", "275,618")
    checks.append(("stale calls 275,218 removed", "275,218" not in blob, "275,218"))
    need("six-model calls 270,324", "270,324")
    need("llava calls 5,294", "5,294")

    # --- fixed-rule rerun scope ---------------------------------------------
    fr = pd.read_csv(AN / "fixed_rule_rerun.csv")
    checks.append(("fixed-rule only internvl", set(fr["model"]) == {"internvl3_5_8b"},
                   str(sorted(set(fr["model"])))))
    need("fixed-rule degenerate", "degenerate and excluded")

    # --- clean-table Qwen2.5 row (raw predictions) --------------------------
    need("qwen25 clean 0.6578", "0.6578")
    need("qwen25 recall 0.0039", "0.0039")
    checks.append(("stale clean 0.6566 removed", "0.6566" not in blob, "0.6566"))

    # --- operating-point false-positive counts ------------------------------
    need("FP clean 1,646", "1,646")
    need("FP severe 1,905", "1,905")
    need("FP brightness 1,775", "1,775")
    need("FP qwen3 696", "696")
    checks.append(("stale FP removed", ("1,477" not in blob) and ("1,555" not in blob),
                   "1,477/1,555"))

    # --- schema v3 parse label ----------------------------------------------
    need("v3 unparsed", "22.3\\% unparsed")
    need("marker unparsed 1.20", "1.20\\%")
    checks.append(("stale marker 1.06 removed", "1.06\\%" not in blob, "1.06\\%"))
    need("chexagent definite 18.9", "only 18.9\\% of its slots")

    # --- report regime label ------------------------------------------------
    checks.append(("report regime label", "very high hallucination" in blob,
                   "very high hallucination"))

    # --- pooled damage in the abstract --------------------------------------
    dmg = pd.read_csv(ROOT / "outputs/analysis_sensitivity/damage_pooled.csv").set_index("condition")
    checks.append(("damage noise", round(float(dmg.loc["noise", "delta"]), 3) == -0.128,
                   str(dmg.loc["noise", "delta"])))
    checks.append(("damage resolution", round(float(dmg.loc["resolution", "delta"]), 3) == -0.169,
                   str(dmg.loc["resolution", "delta"])))
    need("damage 0.128", "-0.128")
    need("damage 0.169", "-0.169")
    checks.append(("stale damage removed", ("-0.099" not in blob) and ("-0.135" not in blob),
                   "-0.099/-0.135"))

    # --- sign convention and late-audit fixes ------------------------------
    need("sens delta_null", "$\\Delta_{\\text{null}}$ range")
    checks.append(("sens polarity", ("+0.643" in blob) and ("-0.648" not in blob),
                   "+0.643/-0.648"))
    need("fixedthr points to rerun", "Table~\\ref{tab:fixedrerun} reports")
    checks.append(("stale rerun sentence removed",
                   "A true rerun is still required" not in blob, "stale"))
    checks.append(("no version self-reference", "An earlier version" not in blob,
                   "An earlier version"))
    need("densenet split clarified", "own patient-disjoint 1,352-study")
    need("census reconciliation 9,077", "9,077")
    need("census thinking 9,477", "9,477")
    need("clinical FP composition", "64.5\\% of the 259")
    need("ordered preconditions", "ordered rather than parallel")
    fig = (ROOT.parent / "program/scripts/80_make_figures.py").read_text(encoding="utf-8")
    checks.append(("fig1 FP 16", "FP +16%" in fig and "FP +29%" not in fig, "fig1"))
    checks.append(("fig1 operating point", "operating-point shift" in fig, "fig1"))
    checks.append(("fig5 delta_null", "Delta_{{\\\\mathrm{{null}}}}" in fig and "gap {r['gap_text']" not in fig,
                   "fig5"))

    # --- report -------------------------------------------------------------
    failed = [c for c in checks if not c[1]]
    for label, ok, detail in checks:
        print(("PASS " if ok else "FAIL ") + f"{label}: {detail}")
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
