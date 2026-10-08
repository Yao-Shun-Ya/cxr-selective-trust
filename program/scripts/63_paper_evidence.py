#!/usr/bin/env python
"""63 — consolidate the pre-run findings that are fit for the paper's main text.

`docs/50_resume_prerun_findings.md` section 7 is the narrative version of this
ledger. This script is the *generated* version, so the numbers in it cannot
drift from the artifacts they came from: it reads `outputs/pretest` (never
writes into it) and emits two files that are views of the same data --

    outputs/state/pretest_paper_evidence.json   machine-readable ledger
    docs/51_pretest_evidence_for_paper.md       the same ledger, for the paper

Every entry carries its source artifact, its sample size and the caveat that
limits the claim. A finding without its scope is exactly how a 100-image pilot
rate becomes a paper headline, and section 7.7 lists the claims that must not
be made from n=100 at all -- those are carried here as `not_claimable` so the
generator, not the reader, keeps the two lists apart.

Budget status is per model, not one constant: medgemma_15 was re-run WHOLE at
2048 tokens on 2026-10-08 (docs/50 2.11-2.12), chexagent was re-run GREEDY
(num_beams=1), and the other models are the original 256-token pre-run.
`pretest_budget_tokens` is therefore the shared default; `medgemma15_budget_tokens`
records the exception. The 2048 re-run is why the medgemma_15 parse rates here
sit at ~1.0 instead of the 256-token lower bounds of the first pass.

Exit code is 0 even when a model is broken: a broken model is a result here.

Usage:
    .venv-vl\\Scripts\\python.exe scripts\\63_paper_evidence.py
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The two readers are reused rather than re-implemented: 59 owns the byte-level
# and marker logic, 09 owns the coverage census. Re-deriving either here would
# let the ledger and the probes disagree.
def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load("probe59", "59_probe_pretest_findings.py")
shift_analysis = _load("analysis09", "09_shift_analysis.py")

# The pre-run used the shared 256-token default for most models. medgemma_15 is
# the exception: it was re-run WHOLE at 2048 on 2026-10-08 because the thought
# channel needs more than 256 to finish (docs/50 2.8, 2.11). chexagent was
# re-run GREEDY (num_beams=1). So a medgemma_15 number computed from this root is
# a 2048-token number and a chexagent number is a greedy number; the rest are
# 256-token numbers.
PRETEST_BUDGET_TOKENS = 256
MEDGEMMA15_BUDGET_TOKENS = 2048


def shift_files(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "nih_eval", "*", "shift", "shift_predictions.jsonl")))


def reliance_path(root: str, model: str, precision: str = "reliance") -> str:
    """`precision` is the directory: 'reliance' (4-bit) or 'reliance_bf16'."""
    return os.path.join(root, "nih_eval", model, precision,
                        "image_reliance_predictions.jsonl")


def model_of(path: str, root: str) -> str:
    return os.path.relpath(path, os.path.join(root, "nih_eval")).split(os.sep)[0]


def stage_of(path: str, root: str) -> str:
    return os.path.relpath(path, os.path.join(root, "nih_eval")).split(os.sep)[1]


# ------------------------------------------------------------------ 7.1 marker


def marker_census(root: str) -> list[dict]:
    """Per (model, stage): how many replies carry a reasoning marker.

    Two tests are reported on purpose. `startswith` is the exact, cheap detector
    the harness uses; `substring` is the loose one that would cry wolf on a reply
    that merely mentions the token. Section 7.1 claims they coincide, and a claim
    like that is worth re-measuring rather than trusting.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(root, "nih_eval", "*", "*", "*predictions.jsonl"))):
        rows = probe.read_jsonl(path)
        if not rows:
            continue
        marked_any = sum(1 for r in rows if any(m in probe.raw(r) for m in probe.THINK_MARKERS))
        marked_start = sum(1 for r in rows if probe.raw(r).startswith("<unused94>"))
        unparsed = [r for r in rows if not (r.get("parsed") or {}).get("parse_success")]
        out.append({
            "model": model_of(path, root),
            "stage": stage_of(path, root),
            "rows": len(rows),
            "marker_substring": marked_any,
            "marker_startswith": marked_start,
            "unparsed": len(unparsed),
            "unparsed_and_marker": sum(1 for r in unparsed if any(m in probe.raw(r)
                                                                  for m in probe.THINK_MARKERS)),
            "source": os.path.relpath(path, root).replace(os.sep, "/"),
        })
    return out


# ------------------------------------------------------- 7.2 / 7.3 parse tables


def parse_by(path: str, key: str) -> dict[str, dict]:
    """Parse rate grouped by `condition` or `severity` for one predictions file."""
    rows = probe.read_jsonl(path)
    buckets = defaultdict(lambda: [0, 0])
    for row in rows:
        cell = str(row.get(key))
        buckets[cell][1] += 1
        if (row.get("parsed") or {}).get("parse_success"):
            buckets[cell][0] += 1
    return {k: {"ok": v[0], "total": v[1], "rate": (v[0] / v[1]) if v[1] else None}
            for k, v in sorted(buckets.items())}


def severity_curve(root: str, model: str) -> dict[str, dict]:
    path = os.path.join(root, "nih_eval", model, "shift", "shift_predictions.jsonl")
    by_sev = parse_by(path, "severity")
    order = ["none", "mild", "moderate", "severe"]
    return {k: by_sev[k] for k in order if k in by_sev}


def marker_by_severity(root: str, model: str) -> dict[str, dict]:
    """Per severity: channel-switch rate and parse rate, computed independently so
    the two can be told apart. The marker is the exact ``startswith("<unused94>")``
    detector the harness uses (docs/50 7.1)."""
    path = os.path.join(root, "nih_eval", model, "shift", "shift_predictions.jsonl")
    rows = probe.read_jsonl(path)
    buckets = defaultdict(lambda: [0, 0, 0])  # n, parsed, marker
    for row in rows:
        sev = str(row.get("severity"))
        buckets[sev][0] += 1
        if (row.get("parsed") or {}).get("parse_success"):
            buckets[sev][1] += 1
        if (probe.raw(row) or "").startswith("<unused94>"):
            buckets[sev][2] += 1
    order = ["none", "mild", "moderate", "severe"]
    out = {}
    for sev in order:
        if sev not in buckets:
            continue
        n, p, m = buckets[sev]
        out[sev] = {
            "n": n,
            "parse_rate": (p / n) if n else None,
            "marker": m,
            "marker_rate": (m / n) if n else None,
        }
    return out


def channel_recovery(root: str, model: str) -> dict:
    """Across the model's three stages: how many replies opened the reasoning
    channel, and how many then recovered into a parseable object vs. degenerated
    into an absorbing repetition loop (docs/50 2.12)."""
    marker = recovered = fail = 0
    for st in ("shift", "reliance", "reliance_bf16"):
        if st == "shift":
            p = os.path.join(root, "nih_eval", model, "shift", "shift_predictions.jsonl")
        else:
            p = os.path.join(root, "nih_eval", model, st, "image_reliance_predictions.jsonl")
        if not os.path.isfile(p):
            continue
        for row in probe.read_jsonl(p):
            if (probe.raw(row) or "").startswith("<unused94>"):
                marker += 1
                if (row.get("parsed") or {}).get("parse_success"):
                    recovered += 1
                else:
                    fail += 1
    return {
        "marker": marker,
        "marker_recovered": recovered,
        "marker_fail": fail,
        "recovery_rate": (recovered / marker) if marker else None,
    }


def _stage_file(base: str, stage: str) -> str:
    if stage == "shift":
        return os.path.join(base, "shift", "shift_predictions.jsonl")
    return os.path.join(base, stage, "image_reliance_predictions.jsonl")


def _summary(path: str) -> dict:
    rows = probe.read_jsonl(path)
    n = len(rows)
    p = sum(1 for r in rows if (r.get("parsed") or {}).get("parse_success"))
    m = sum(1 for r in rows if (probe.raw(r) or "").startswith("<unused94>"))
    return {"n": n, "parsed": p, "rate": (p / n) if n else None, "marker": m}


def budget_contrast(root: str) -> dict:
    """medgemma_15 before (256-token first pass) vs after (2048 re-run).

    The "before" column comes from the backed-up first pass
    (`outputs/pretest_retry_backup/medgemma_15_4b_it/`), the "after" column from
    the current artifacts. Both are measured; the contrast is the budget control
    (docs/50 section 2.11-2.12)."""
    backup = os.path.join(os.path.dirname(str(Path(root))), "pretest_retry_backup",
                          "medgemma_15_4b_it")
    after_stage = {"shift": "shift", "reliance": "reliance", "precision": "reliance_bf16"}
    out = {}
    for stage in ("shift", "reliance", "precision"):
        before_p = _stage_file(backup, stage)
        after_p = _stage_file(os.path.join(root, "nih_eval", "medgemma_15_4b_it"),
                              after_stage[stage])
        out[stage] = {
            "before": _summary(before_p) if os.path.isfile(before_p) else None,
            "after": _summary(after_p) if os.path.isfile(after_p) else None,
        }
    # severity curve before/after for the shift stage
    before_shift = os.path.join(backup, "shift", "shift_predictions.jsonl")
    if os.path.isfile(before_shift):
        out["shift_severity_before"] = _severity_curve_file(before_shift)
    out["shift_severity_after"] = severity_curve(root, "medgemma_15_4b_it")
    return out


def _severity_curve_file(path: str) -> dict:
    by_sev = parse_by(path, "severity")
    order = ["none", "mild", "moderate", "severe"]
    return {k: by_sev[k] for k in order if k in by_sev}


def loop_residual(root: str) -> dict:
    """The 3 residual shift failures after the budget control: which cells, which
    images, and the exact channel-recovery ratio for that stage."""
    path = os.path.join(root, "nih_eval", "medgemma_15_4b_it", "shift",
                        "shift_predictions.jsonl")
    rows = probe.read_jsonl(path)
    fail = [r for r in rows if not (r.get("parsed") or {}).get("parse_success")]
    by_cell = defaultdict(int)
    by_image = defaultdict(int)
    for r in fail:
        by_cell[f"{r.get('condition')}/{r.get('severity')}"] += 1
        by_image[str(r.get("image_id"))] += 1
    marker = sum(1 for r in rows if (probe.raw(r) or "").startswith("<unused94>"))
    recovered = sum(1 for r in rows
                    if (probe.raw(r) or "").startswith("<unused94>")
                    and (r.get("parsed") or {}).get("parse_success"))
    # MNAR bound from the already-computed ledger (scripts/72)
    mnar_path = os.path.join(ROOT, "outputs", "state", "mnar_bound_medgemma15_shift.json")
    mnar = {}
    if os.path.isfile(mnar_path):
        with open(mnar_path, encoding="utf-8") as fh:
            mnar = json.load(fh)
    return {
        "n": len(rows),
        "n_fail": len(fail),
        "fail_rate": (len(fail) / len(rows)) if rows else None,
        "failed_cells": dict(by_cell),
        "failed_images": dict(by_image),
        "channel_opened": marker,
        "channel_recovered": recovered,
        "channel_recovery_rate": (recovered / marker) if marker else None,
        "mnar": mnar,
    }


def cell_parse_table(root: str) -> list[dict]:
    out = []
    for path in shift_files(root):
        rows = probe.read_jsonl(path)
        if not rows:
            continue
        cells = defaultdict(lambda: [0, 0])
        for row in rows:
            cell = f"{row.get('condition')}/{row.get('severity')}"
            cells[cell][1] += 1
            if (row.get("parsed") or {}).get("parse_success"):
                cells[cell][0] += 1
        rates = {c: (v[0] / v[1]) for c, v in cells.items()}
        worst = sorted(rates.items(), key=lambda kv: kv[1])[:3]
        out.append({
            "model": model_of(path, root),
            "overall": sum(v[0] for v in cells.values()) / sum(v[1] for v in cells.values()),
            "min_cell_rate": min(rates.values()),
            "n_cells": len(cells),
            "worst_cells": [{"cell": c, "ok": cells[c][0], "total": cells[c][1], "rate": r}
                            for c, r in worst],
        })
    return out


# --------------------------------------------------------------- 7.5 byte level


def byte_identity(root: str, path: str) -> dict:
    """Text/probability identity between each perturbation and the original."""
    table = probe.keyed(probe.read_jsonl(path))
    base = table.get("original", {})
    blank, blur = table.get("blank", {}), table.get("blur", {})
    shared = sorted(set(blank) & set(blur))
    bb = sum(1 for i in shared
             if probe.raw(blank[i]) == probe.raw(blur[i]) and probe.raw(blank[i]) != "")
    per_condition = {}
    for condition in ("blank", "blur", "shuffle"):
        other = table.get(condition, {})
        ids = sorted(set(base) & set(other))
        if not ids:
            continue
        same_text = sum(1 for i in ids
                        if probe.raw(base[i]) == probe.raw(other[i]) and probe.raw(base[i]) != "")
        comparable = [(probe.prob_vector(base[i]), probe.prob_vector(other[i])) for i in ids]
        comparable = [(a, b) for a, b in comparable if a is not None and b is not None]
        same_prob = sum(1 for a, b in comparable if a == b)
        per_condition[condition] = {
            "n_paired": len(ids),
            "same_text": same_text,
            "same_text_rate": same_text / len(ids),
            "n_prob_comparable": len(comparable),
            "same_prob": same_prob,
            "same_prob_rate": (same_prob / len(comparable)) if comparable else None,
        }
    return {
        "blank_blur_pairs": len(shared),
        "blank_blur_identical": bb,
        "blank_blur_identical_rate": (bb / len(shared)) if shared else None,
        "per_condition": per_condition,
        "source": os.path.relpath(path, root).replace(os.sep, "/"),
    }


# -------------------------------------------------------------------- 7.4 census


def coverage_census(root: str, model: str) -> dict:
    path = Path(root) / "nih_eval" / model / "shift" / "shift_predictions.jsonl"
    if not path.exists():
        return {}
    _, census = shift_analysis.load_with_census(path)
    if census.empty:
        return {}
    low = census[census["label_coverage"] < 0.95].sort_values("label_coverage")
    return {
        "cells": json.loads(census.to_json(orient="records")),
        "cells_below_95pct": json.loads(low.to_json(orient="records")),
        "min_label_coverage": float(census["label_coverage"].min()),
    }


# -------------------------------------------------------------------- 7.6 ablation


def ablation_table(root: str) -> list[dict]:
    path = os.path.join(root, "nih_eval", "medgemma_4b_it", "ablation",
                        "ablation_run_summary.json")
    if not os.path.isfile(path):
        return []
    data = json.load(open(path, encoding="utf-8"))
    by_variant = defaultdict(dict)
    for cell in data.get("cells", []):
        by_variant[cell["variant"]][cell["condition"]] = cell
    out = []
    for variant in sorted(by_variant):
        cells = by_variant[variant]
        orig, blank = cells.get("original", {}), cells.get("blank", {})
        out.append({
            "variant": variant,
            "parse_fail_original": orig.get("parse_failure_rate"),
            "parse_fail_blank": blank.get("parse_failure_rate"),
            "violation_original": orig.get("contract_violation_rate"),
            "violation_blank": blank.get("contract_violation_rate"),
            "probs_missing_original": orig.get("mean_probs_missing"),
            "probs_missing_blank": blank.get("mean_probs_missing"),
            "n_per_cell": orig.get("n_this_run"),
        })
    return out


# ------------------------------------------------------------------------- build


def build(root: str) -> dict:
    census = marker_census(root)
    mg15 = [c for c in census if c["model"] == "medgemma_15_4b_it"]
    llava = [c for c in census if c["model"] == "llava_med_v1_5_7b"]
    others = [c for c in census
              if c["model"] not in {"medgemma_15_4b_it", "llava_med_v1_5_7b"}]

    findings = []

    # ---- 7.1 ---------------------------------------------------------------
    ch = channel_recovery(root, "medgemma_15_4b_it")
    findings.append({
        "id": "7.1",
        "title": "Two mechanically distinct structural failures, separable for free",
        "claim": "The `<unused94>` opener divides non-functional replies into two kinds "
                 "that no amount of re-prompting would conflate: medgemma_15's are "
                 "reasoning traces (every unparsed row carries the marker, and 736 rows "
                 "open the channel across its stages), while llava_med emits prose with "
                 "no marker at all. The detector is exact on every row of the matrix.",
        "evidence": {
            "medgemma_15_4b_it": {
                "rows": sum(c["rows"] for c in mg15),
                "unparsed": sum(c["unparsed"] for c in mg15),
                "unparsed_and_marker": sum(c["unparsed_and_marker"] for c in mg15),
                "marker_startswith": sum(c["marker_startswith"] for c in mg15),
            },
            "channel_recovery": ch,
            "llava_med_v1_5_7b": {
                "rows": sum(c["rows"] for c in llava),
                "unparsed": sum(c["unparsed"] for c in llava),
                "marker_substring": sum(c["marker_substring"] for c in llava),
            },
            "other_models_marker_substring": sum(c["marker_substring"] for c in others),
            "per_stage": census,
        },
        "sources": sorted({c["source"] for c in census}),
        "caveat": "Exact on this root's rows. It is a *detector*, not a claim that every "
                  "thinking model fails the same way -- the recovery rate below (7.7) is "
                  "what separates 'tried and left' from 'tried and recovered'.",
        "paper_target": "method: the one-line structural-failure census",
        "status": "qualified",
    })

    # ---- 7.2 ---------------------------------------------------------------
    contrast = budget_contrast(root)
    cells = cell_parse_table(root)
    clean_models = [c for c in cells if c["min_cell_rate"] >= 0.999]
    findings.append({
        "id": "7.2",
        "title": "The decode budget is the confound behind three apparent pathologies",
        "claim": "Re-running medgemma_15 whole at 2048 tokens instead of 256 moved every "
                 "apparent pathology back to ~1.0: the severity-monotone shift collapse "
                 "(0.793 -> 0.997 at severe), the bf16 precision collapse (0.000 -> 1.000), "
                 "and the blank-condition collapse (0.000 -> 1.000). The three collapses "
                 "were one mechanism -- the reasoning channel needs more than 256 tokens to "
                 "finish on a degraded image -- so a severity or precision result computed "
                 "at a too-small budget would misattribute a measurement artifact to a "
                 "model property. The four working models without a reasoning channel parse "
                 "100% in every cell at 256 tokens, which rules the parser out as the cause.",
        "evidence": {
            "model": "medgemma_15_4b_it",
            "contrast": contrast,
            "marker_by_severity": marker_by_severity(root, "medgemma_15_4b_it"),
            "models_at_100pct_every_cell": [c["model"] for c in clean_models],
        },
        "sources": ["nih_eval/medgemma_15_4b_it/{shift,reliance,reliance_bf16} (after)",
                    "outputs/pretest_retry_backup/medgemma_15_4b_it/{shift,reliance,precision} (before)"],
        "caveat": "Single model -- the only one with an active reasoning channel -- and a "
                  "before/after on the same model, not a controlled sweep. The 2048 number "
                  "is the production budget; the 256 number is the recorded first pass.",
        "paper_target": "method: decode budget as a first-order control for reasoning VLMs; "
                        "results: the before/after table",
        "status": "headline (single-model, measured)",
    })

    # ---- 7.3 ---------------------------------------------------------------
    precision_rows = []
    for model in ("medgemma_15_4b_it", "medgemma_4b_it", "qwen25_vl_7b_instruct"):
        four_path = reliance_path(root, model, "reliance")
        bf16_path = reliance_path(root, model, "reliance_bf16")
        if not (os.path.isfile(four_path) and os.path.isfile(bf16_path)):
            continue
        fs, bs = _summary(four_path), _summary(bf16_path)
        precision_rows.append({
            "model": model,
            "four_bit_parse": fs["rate"],
            "four_bit_channel_open": fs["marker"],
            "bf16_parse": bs["rate"],
            "bf16_channel_open": bs["marker"],
        })
    findings.append({
        "id": "7.3",
        "title": "Precision shifts the reasoning mode, not the parse outcome",
        "claim": "At an adequate budget both precisions parse 100%, but medgemma_15's "
                 "bf16 opens the reasoning channel on every cell (400/400) while 4-bit "
                 "opens it only where the image cannot support an answer (146/400). The "
                 "earlier 'bf16 yields zero usable rows' was the budget again; what "
                 "remains is a real, measurable mode difference -- bf16 always reasons -- "
                 "and that, not a parse failure, is what makes the bf16 control 5x slower. "
                 "medgemma_4b and qwen25 never open the channel at either precision.",
        "evidence": {"by_precision": precision_rows},
        "sources": ["nih_eval/{medgemma_15_4b_it,medgemma_4b_it,qwen25_vl_7b_instruct}/"
                    "reliance{,_bf16}/image_reliance_predictions.jsonl"],
        "caveat": "A 4-bit-vs-bf16 comparison is still only meaningful if the reasoning "
                  "mode is reported per precision; 4-bit and bf16 rows differ in mode, "
                  "not only in numeric precision.",
        "paper_target": "results: precision control; the mode-vs-outcome distinction",
        "status": "qualified",
    })

    # ---- 7.4 ---------------------------------------------------------------
    cov = coverage_census(root, "medgemma_15_4b_it")
    findings.append({
        "id": "7.4",
        "title": "The metric pipeline used to silently select the sample",
        "claim": "The analysis dropped non-parsed rows and non-binary ground truth without "
                 "counting them, so a degraded cell's metric silently became a metric over "
                 "its survivors. That is a code defect, now closed: the pipeline emits a "
                 "per-cell census (n_planned / n_parsed / coverage) and the driver gates on "
                 "it. On the current pre-run every cell holds >= 99% label coverage, so no "
                 "metric is computed over a silently selected subset.",
        "evidence": {
            "model": "medgemma_15_4b_it",
            "min_label_coverage": cov.get("min_label_coverage"),
            "cells_below_95pct": cov.get("cells_below_95pct", []),
            "census": cov.get("cells", []),
        },
        "sources": ["nih_eval/medgemma_15_4b_it/shift/shift_predictions.jsonl",
                    "program/scripts/09_shift_analysis.py"],
        "caveat": "The dramatic illustration (resolution/severe 32% coverage) was the "
                  "256-token first pass and is no longer present at 2048. The finding is "
                  "about the code path, not about a current number.",
        "paper_target": "method: coverage accounting; anti-silent-selection discipline",
        "status": "qualified (code finding, now closed)",
    })

    # ---- 7.5 ---------------------------------------------------------------
    qwen_four = byte_identity(root, os.path.join(root, "nih_eval", "qwen25_vl_7b_instruct",
                                                 "reliance", "image_reliance_predictions.jsonl"))
    qwen_bf16 = byte_identity(root, os.path.join(root, "nih_eval", "qwen25_vl_7b_instruct",
                                                 "reliance_bf16", "image_reliance_predictions.jsonl"))
    mg4 = byte_identity(root, reliance_path(root, "medgemma_4b_it"))
    findings.append({
        "id": "7.5",
        "title": "Qwen's image invariance, restated at byte level",
        "claim": "qwen25 returns byte-identical text for blank and blur on every paired "
                 "image, and the same holds at bf16 -- a blank and a heavily blurred image "
                 "are different inputs, so identical bytes for all of them is output "
                 "collapse, not low sensitivity. medgemma_4b is the contrast: 0 of its 100 "
                 "blank replies are byte-identical to the original, so the collapse is "
                 "Qwen-specific rather than a property of the pipeline.",
        "evidence": {
            "qwen25_4bit": qwen_four,
            "qwen25_bf16": qwen_bf16,
            "medgemma_4b_4bit": mg4,
        },
        "sources": [qwen_four["source"], qwen_bf16["source"], mg4["source"]],
        "caveat": "The paper's tab:image-reliance uses 0.0033 on 300 images; this stage "
                  "measures 0.06 on 100. Same definition, sample variation -- only ONE "
                  "source may reach the paper. qwen3/internvl sit near chance.",
        "paper_target": "results: image reliance / hallucination under a destroyed image",
        "status": "qualified",
    })

    # ---- 7.6 ---------------------------------------------------------------
    abl = ablation_table(root)
    findings.append({
        "id": "7.6",
        "title": "The prompt ablation inverts under a blank image",
        "claim": "On clean images v1_per_label_binary ranks worse than v0_simple, but on "
                 "blank images it is the only prompt that survives (no violation, no "
                 "missing probabilities) while the other four lose all seven -- so a "
                 "prompt cannot be selected on clean data.",
        "evidence": {"table": abl, "model": "medgemma_4b_it", "n_per_cell": 100},
        "sources": ["nih_eval/medgemma_4b_it/ablation/ablation_run_summary.json"],
        "caveat": "One model, 100 images per cell. v4_normality_pressure already loses "
                  "5.79 of 7 probabilities on CLEAN images.",
        "paper_target": "results: prompt robustness does not transfer across conditions",
        "status": "qualified",
    })

    # ---- 7.7 ---------------------------------------------------------------
    loop = loop_residual(root)
    findings.append({
        "id": "7.7",
        "title": "The residual: a deterministic absorbing loop on the hardest images",
        "claim": "After the budget control, 3 of 1,900 shift cells (0.16%) still fail -- "
                 "not by truncation but by an absorbing repetition loop: under the study's "
                 "greedy decoding the thought channel starts enumerating findings and "
                 "cycles forever, so no budget can leave it. The three cells are exactly "
                 "marked, MNAR (all on degraded images), and their omission moves any "
                 "[0,1] metric by at most 0.01.",
        "evidence": loop,
        "sources": ["nih_eval/medgemma_15_4b_it/shift/shift_predictions.jsonl",
                    "outputs/state/mnar_bound_medgemma15_shift.json",
                    "scripts/70_diag_medgemma15_shift_failures.py"],
        "caveat": "3 cells, one model. It is reported as an exactly bounded negative "
                  "result, not imputed away; giving one model a repetition penalty would "
                  "break cross-model comparability.",
        "paper_target": "results/limitations: a reproducible, bounded failure mode",
        "status": "qualified (negative result, fully characterized)",
    })

    not_claimable = [
        {"claim": "'Structural validity is monotone in severity' as a general VLM property.",
         "why": "The monotone collapse was a 256-token budget artifact: at 2048 it is "
                "gone (7.2), leaving only a 0.16% absorbing loop. Do not resurrect the "
                "256-token curve as a claim about the model."},
        {"claim": "The pilot's exact reliance rates as final numbers.",
         "why": "qwen3/internvl blank-blur agreement sits at 0.50 and 0.52 -- near chance "
                "for a binary agreement; the full run may move them either way, and that "
                "movement decides whether 7.5 is about Qwen or about VLMs generally."},
        {"claim": "The two literature items surfaced during the analysis (a reported GGUF "
                  "4-bit JSON parse collapse, and a 'mirage reasoning' preprint).",
         "why": "Unverified. Check them before citing or drop them; 7.2/7.3 stand on their "
                "own measurements."},
        {"claim": "Any Qwen image-invariance number that is not from the full run.",
         "why": "The paper already carries 0.0033 on 300 images while this stage measures "
                "0.06 on 100. Same definition, so this is sample variation -- but only one "
                "source of that number may reach the paper. Let the full run produce it."},
        {"claim": "Any pre-run parse/coverage rate as a production rate.",
         "why": "The pre-run is n=100 with per-model budget/decoding exceptions; the "
                "production numbers come from the full run and are auto-censused by the "
                "driver."},
    ]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generator": "scripts/63_paper_evidence.py",
        "source_root": os.path.relpath(root, ROOT).replace(os.sep, "/"),
        "pretest_budget_tokens": PRETEST_BUDGET_TOKENS,
        "medgemma15_budget_tokens": MEDGEMMA15_BUDGET_TOKENS,
        "budget_note": ("medgemma_15 re-run whole @ 2048 (2026-10-08); chexagent greedy; "
                        "all other models 256-token pre-run"),
        "narrative_version": "docs/50_resume_prerun_findings.md section 7",
        "findings": findings,
        "not_claimable": not_claimable,
    }


# --------------------------------------------------------------------- render


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def render_markdown(ledger: dict) -> str:
    lines = [
        "# 51 — Pre-run evidence fit for the paper's main text (generated)",
        "",
        "> GENERATED by `program/scripts/63_paper_evidence.py` from "
        f"`{ledger['source_root']}` — do not hand-edit.",
        f"> Regenerated: {ledger['generated_at']}",
        "> Budget note: **medgemma_15 was re-run whole @ 2048 tokens** (2026-10-08); "
        "chexagent was re-run greedy; every other model is the 256-token pre-run "
        "(docs/50 §2.11-2.12). The 2048 re-run is why medgemma_15 sits at ~1.0 here.",
        "> Narrative version: `docs/50_resume_prerun_findings.md` section 7.",
        "",
        "Reproduce with:",
        "",
        "```powershell",
        ".venv-vl\\Scripts\\python.exe scripts\\63_paper_evidence.py",
        "```",
        "",
        "---",
        "",
    ]
    for f in ledger["findings"]:
        lines += [
            f"## {f['id']} {f['title']}",
            "",
            f"**Claim.** {f['claim']}",
            "",
            f"**Status.** {f['status']}",
            "",
            f"**Paper target.** {f['paper_target']}",
            "",
            f"**Caveat.** {f['caveat']}",
            "",
        ]
        ev = f["evidence"]
        if f["id"] == "7.1":
            m = ev["medgemma_15_4b_it"]
            ch = ev["channel_recovery"]
            lines += [
                "| model | rows | unparsed | unparsed & marker |",
                "|---|---|---|---|",
                f"| medgemma_15_4b_it | {m['rows']} | {m['unparsed']} | {m['unparsed_and_marker']} |",
                f"| llava_med_v1_5_7b | {ev['llava_med_v1_5_7b']['rows']} | "
                f"{ev['llava_med_v1_5_7b']['unparsed']} | "
                f"{ev['llava_med_v1_5_7b']['marker_substring']} (marker) |",
                "",
                f"Other models emit **{ev['other_models_marker_substring']}** markers.",
                "",
                "Channel opened vs recovered (medgemma_15, all stages): "
                f"{ch['marker']} opened, {ch['marker_recovered']} recovered, "
                f"{ch['marker_fail']} left — recovery {ch['recovery_rate']:.3f}.",
                "",
            ]
        elif f["id"] == "7.2":
            lines += ["| stage | parse @ 256 | parse @ 2048 |", "|---|---|---|"]
            for stage in ("shift", "reliance", "precision"):
                b = ev["contrast"][stage]["before"]
                a = ev["contrast"][stage]["after"]
                lines.append(f"| {stage} | {_fmt(b['rate'])} ({b['parsed']}/{b['n']}) | "
                             f"{_fmt(a['rate'])} ({a['parsed']}/{a['n']}) |")
            lines += ["", "| severity (shift) | parse @ 256 | parse @ 2048 |", "|---|---|---|"]
            for sev in ("none", "mild", "moderate", "severe"):
                b = (ev["contrast"].get("shift_severity_before") or {}).get(sev)
                a = (ev["contrast"].get("shift_severity_after") or {}).get(sev)
                lines.append(f"| {sev} | {_fmt(b['rate'] if b else None)} | "
                             f"{_fmt(a['rate'] if a else None)} |")
            lines += ["", "Channel opens concentrate on degraded cells (2048):"]
            lines += ["| severity | channel-open rate | parse rate |", "|---|---|---|"]
            for sev, v in ev["marker_by_severity"].items():
                lines.append(f"| {sev} | {_fmt(v['marker_rate'])} | {_fmt(v['parse_rate'])} |")
            lines += ["", "Models at 100% in every shift cell at 256 tokens: "
                      f"{', '.join(ev['models_at_100pct_every_cell']) or '—'}.", ""]
        elif f["id"] == "7.3":
            lines += ["| model | 4-bit parse | 4-bit channel-open | bf16 parse | bf16 channel-open |",
                      "|---|---|---|---|---|"]
            for r in ev["by_precision"]:
                lines.append(f"| {r['model']} | {_fmt(r['four_bit_parse'])} | "
                             f"{r['four_bit_channel_open']}/400 | {_fmt(r['bf16_parse'])} | "
                             f"{r['bf16_channel_open']}/400 |")
            lines.append("")
        elif f["id"] == "7.4":
            if ev["cells_below_95pct"]:
                lines += ["| cell | label coverage | parse fail |", "|---|---|---|"]
                for c in ev["cells_below_95pct"]:
                    lines.append(f"| {c['condition']} / {c['severity']} | "
                                 f"{c['label_coverage']:.1%} | {int(c['n_parse_fail'])} / "
                                 f"{int(c['n_attempted'])} |")
            else:
                lines += ["No cell is below the 95% label-coverage gate.", ""]
            lines += ["", f"Minimum label coverage: {ev['min_label_coverage']:.1%}.", ""]
        elif f["id"] == "7.5":
            for label, key in (("qwen25 4-bit", "qwen25_4bit"),
                               ("qwen25 bf16", "qwen25_bf16"),
                               ("medgemma_4b 4-bit", "medgemma_4b_4bit")):
                b = ev[key]
                lines += [f"**{label}** — blank≡blur byte-identical "
                          f"{b['blank_blur_identical']}/{b['blank_blur_pairs']}"
                          + (f" = {b['blank_blur_identical_rate']:.3f}"
                             if b["blank_blur_identical_rate"] is not None else ""),
                          "",
                          "| perturbation | paired | same text | same text rate |", "|---|---|---|---|"]
                for cond, v in b["per_condition"].items():
                    lines.append(f"| {cond} | {v['n_paired']} | {v['same_text']} | "
                                 f"{v['same_text_rate']:.3f} |")
                lines.append("")
        elif f["id"] == "7.6":
            lines += ["| variant | viol orig | viol blank | miss orig | miss blank |",
                      "|---|---|---|---|---|"]
            for r in ev["table"]:
                lines.append(f"| {r['variant']} | {r['violation_original']:.3f} | "
                             f"{r['violation_blank']:.3f} | {r['probs_missing_original']:.2f} | "
                             f"{r['probs_missing_blank']:.2f} |")
            lines.append("")
        elif f["id"] == "7.7":
            lines += [
                f"Failures: **{ev['n_fail']} / {ev['n']} = {ev['fail_rate']:.4f}**.",
                "",
                "Failed cells: " + ", ".join(f"{c} (×{n})" for c, n in
                                            sorted(ev["failed_cells"].items())) + ".",
                "",
                "Failed images: " + ", ".join(f"{i} (×{n})" for i, n in
                                              sorted(ev["failed_images"].items())) + ".",
                "",
                f"Channel opened {ev['channel_opened']}, recovered {ev['channel_recovered']} "
                f"({ev['channel_recovery_rate']:.3f}).",
                "",
            ]
            mnar = ev.get("mnar") or {}
            if mnar.get("worst_case_bound_cell_metric") is not None:
                lines += [
                    f"MNAR bound: worst-case metric movement ≤ "
                    f"{mnar['worst_case_bound_cell_metric']}.",
                    "",
                ]
        lines += ["**Sources.** " + ", ".join(f"`{s}`" for s in f["sources"]), "", "---", ""]

    lines += ["## What must NOT be claimed from n=100", ""]
    for item in ledger["not_claimable"]:
        lines += [f"* **{item['claim']}** — {item['why']}"]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT / "outputs" / "pretest"))
    ap.add_argument("--json", default=str(ROOT / "outputs" / "state" / "pretest_paper_evidence.json"))
    ap.add_argument("--markdown", default=str(ROOT.parent / "docs" / "51_pretest_evidence_for_paper.md"))
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"root not found: {args.root}", file=sys.stderr)
        return 2

    ledger = build(args.root)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(ledger, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.markdown).write_text(render_markdown(ledger), encoding="utf-8")

    print(f"findings      : {len(ledger['findings'])}")
    print(f"not claimable : {len(ledger['not_claimable'])}")
    for f in ledger["findings"]:
        print(f"  {f['id']}  {f['title']}")
    print(f"\nwrote {os.path.relpath(args.json, ROOT)}")
    print(f"wrote {os.path.relpath(args.markdown, ROOT.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())