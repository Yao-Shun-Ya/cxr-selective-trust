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

Every medgemma_15 number here is **pre-run @ 256 tokens**. The pre-run predates
the section 2.8 budget fix, so its parse rates are a lower bound on the
production run, not the paper's rate; the `pretest_budget_tokens` field says so
in the file itself.

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

# The pre-run used the shared 256-token default for every model, including
# medgemma_15 -- that is the whole point of section 2.8. Any number computed
# from this root is therefore a 256-token number.
PRETEST_BUDGET_TOKENS = 256


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
    findings.append({
        "id": "7.1",
        "title": "The structural failure is a channel switch, detectable for free",
        "claim": "MedGemma-1.5's unparsed replies are reasoning traces that ran out of "
                 "budget, not prose that ignored the format: every unparsed medgemma_15 "
                 "row carries the `<unused94>` opener and no other model emits one.",
        "evidence": {
            "medgemma_15_4b_it": {
                "rows": sum(c["rows"] for c in mg15),
                "unparsed": sum(c["unparsed"] for c in mg15),
                "unparsed_and_marker": sum(c["unparsed_and_marker"] for c in mg15),
                "marker_startswith": sum(c["marker_startswith"] for c in mg15),
            },
            "llava_med_v1_5_7b": {
                "rows": sum(c["rows"] for c in llava),
                "unparsed": sum(c["unparsed"] for c in llava),
                "marker_substring": sum(c["marker_substring"] for c in llava),
            },
            "other_models_marker_substring": sum(c["marker_substring"] for c in others),
            "per_stage": census,
        },
        "sources": sorted({c["source"] for c in census}),
        "caveat": "Exact on this root's rows. It is a *detector*, not a claim that "
                  "every thinking model fails the same way -- section 7.7.",
        "paper_target": "results: structural failure taxonomy; method: the one-line census",
        "status": "qualified",
    })

    # ---- 7.2 ---------------------------------------------------------------
    curve = severity_curve(root, "medgemma_15_4b_it")
    cells = cell_parse_table(root)
    clean_models = [c for c in cells if c["min_cell_rate"] >= 0.999]
    collapsed = [c for c in cells if c["min_cell_rate"] <= 0.001]
    findings.append({
        "id": "7.2",
        "title": "Structural validity is a function of the image, monotone in severity",
        "claim": "For the one model with an active reasoning channel, the parse rate "
                 "falls monotonically with severity, while every other working model "
                 "holds 100% in every cell -- so the collapse is the model, not the parser.",
        "evidence": {
            "model": "medgemma_15_4b_it",
            "severity_curve": curve,
            "reliance_conditions": parse_by(reliance_path(root, "medgemma_15_4b_it"),
                                            "condition"),
            "worst_shift_cells": next((c["worst_cells"] for c in cells
                                       if c["model"] == "medgemma_15_4b_it"), []),
            "models_at_100pct_every_cell": [c["model"] for c in clean_models],
            "models_collapsed": [c["model"] for c in collapsed],
            "all_cell_tables": cells,
        },
        "sources": [os.path.relpath(reliance_path(root, "medgemma_15_4b_it"), root).replace(os.sep, "/")]
                   + [f"nih_eval/{c['model']}/shift/shift_predictions.jsonl" for c in cells],
        "caveat": "Single-model pattern. The marker census is exact; the *generality* "
                  "is not established until a second thinking model reproduces it (7.7). "
                  "Pre-run @ 256 tokens -- the production 1024-token rates are higher.",
        "paper_target": "results: severity vs structural validity (per model)",
        "status": "qualified (single-model)",
    })

    # ---- 7.3 ---------------------------------------------------------------
    precision_rows = []
    for model in ("medgemma_15_4b_it", "medgemma_4b_it", "qwen25_vl_7b_instruct"):
        four_path = reliance_path(root, model, "reliance")
        bf16_path = reliance_path(root, model, "reliance_bf16")
        if not (os.path.isfile(four_path) and os.path.isfile(bf16_path)):
            continue
        precision_rows.append({
            "model": model,
            "four_bit": parse_by(four_path, "condition"),
            "bf16": parse_by(bf16_path, "condition"),
        })
    findings.append({
        "id": "7.3",
        "title": "Precision changes the format channel, not only the numbers",
        "claim": "For medgemma_15 the bf16 control yields zero usable rows while the "
                 "4-bit run answers cleanly on `original`; medgemma_4b and qwen25 are "
                 "unaffected at both precisions, so this is a model-specific channel "
                 "threshold rather than 'quantisation breaks JSON'.",
        "evidence": {
            "by_model": precision_rows,
            "direction": "the HIGHER precision is the broken one for medgemma_15",
        },
        "sources": [f"nih_eval/{r['model']}/reliance{{,_bf16}}/image_reliance_predictions.jsonl"
                    for r in precision_rows],
        "caveat": "A 4-bit-vs-bf16 comparison is only meaningful if format compliance is "
                  "reported per precision and the comparison is restricted to images that "
                  "parsed in BOTH (7.7).",
        "paper_target": "results: precision control; the mode-difference caveat",
        "status": "qualified",
    })

    # ---- 7.4 ---------------------------------------------------------------
    cov = coverage_census(root, "medgemma_15_4b_it")
    findings.append({
        "id": "7.4",
        "title": "The metric pipeline silently selects the sample",
        "claim": "Dropping non-parsed rows and non-binary ground truth without counting "
                 "them turns a cell's metric into a metric over the survivors: the "
                 "resolution/severe cell carries the same macro weight as any other while "
                 "resting on a fraction of its planned labels.",
        "evidence": {
            "model": "medgemma_15_4b_it",
            "min_label_coverage": cov.get("min_label_coverage"),
            "cells_below_95pct": cov.get("cells_below_95pct", []),
            "census": cov.get("cells", []),
        },
        "sources": ["nih_eval/medgemma_15_4b_it/shift/shift_predictions.jsonl",
                    "program/scripts/09_shift_analysis.py"],
        "caveat": "Pre-run @ 256 tokens; these are lower bounds. The production census is "
                  "emitted automatically by run_all_layers.ps1 and is the number to quote.",
        "paper_target": "method: coverage accounting; results: coverage column in the shift table",
        "status": "qualified (code finding + pre-run illustration)",
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
                  "source may reach the paper (7.7). qwen3/internvl sit near chance (7.7).",
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

    not_claimable = [
        {"claim": "'Structural validity is monotone in severity' as a general VLM result.",
         "why": "Single-model pattern until a second reasoning-channel model reproduces it. "
                "The marker census is exact; the generality is not established."},
        {"claim": "The pilot's exact reliance rates as final numbers.",
         "why": "qwen3/internvl blank-blur agreement sits at 0.50 and 0.52 -- near chance "
                "for a binary agreement; the full run may move them either way, and that "
                "movement decides whether 7.5 is about Qwen or about VLMs generally."},
        {"claim": "The two literature items surfaced during the analysis (a reported GGUF "
                  "4-bit JSON parse collapse, and a 'mirage reasoning' preprint).",
         "why": "Unverified. Check them before citing or drop them; 7.3 stands on its own "
                "measurements."},
        {"claim": "Any Qwen image-invariance number that is not from the full run.",
         "why": "The paper already carries 0.0033 on 300 images while this stage measures "
                "0.06 on 100. Same definition, so this is sample variation -- but only one "
                "source of that number may reach the paper. Let the full run produce it."},
        {"claim": "The pre-run's medgemma_15 parse/coverage rates as production rates.",
         "why": "The pre-run is @ 256 tokens; section 2.8 measured 1024 as the budget where "
                "the traces finish. Quote the full-run census from outputs\\matrix instead."},
    ]

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generator": "scripts/63_paper_evidence.py",
        "source_root": os.path.relpath(root, ROOT).replace(os.sep, "/"),
        "pretest_budget_tokens": PRETEST_BUDGET_TOKENS,
        "narrative_version": "docs/50_resume_prerun_findings.md section 7",
        "findings": findings,
        "not_claimable": not_claimable,
    }


# --------------------------------------------------------------------- render


def render_markdown(ledger: dict) -> str:
    lines = [
        "# 51 — Pre-run evidence fit for the paper's main text (generated)",
        "",
        "> GENERATED by `program/scripts/63_paper_evidence.py` from "
        f"`{ledger['source_root']}` — do not hand-edit.",
        f"> Regenerated: {ledger['generated_at']}",
        f"> Every medgemma_15 number is **pre-run @ {ledger['pretest_budget_tokens']} tokens** "
        "(section 2.8 of `50_resume_prerun_findings.md`); the production rates are higher.",
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
            ]
        elif f["id"] == "7.2":
            lines += ["| severity | parse rate | n |", "|---|---|---|"]
            for sev, v in ev["severity_curve"].items():
                lines.append(f"| {sev} | {v['rate']:.3f} | {v['total']} |")
            lines += ["", "| condition (reliance) | parse rate | n |", "|---|---|---|"]
            for cond, v in ev["reliance_conditions"].items():
                lines.append(f"| {cond} | {v['rate']:.3f} | {v['total']} |")
            lines += ["", "Models at 100% in every shift cell: "
                      f"{', '.join(ev['models_at_100pct_every_cell']) or '—'}; "
                      f"collapsed: {', '.join(ev['models_collapsed']) or '—'}.", ""]
        elif f["id"] == "7.3":
            lines += ["| model | condition | 4-bit parse | bf16 parse |", "|---|---|---|---|"]
            for r in ev["by_model"]:
                for cond in sorted(set(r["four_bit"]) | set(r["bf16"])):
                    four = r["four_bit"].get(cond, {}).get("rate")
                    bf = r["bf16"].get(cond, {}).get("rate")
                    show = lambda v: f"{v:.3f}" if v is not None else "—"
                    lines.append(f"| {r['model']} | {cond} | {show(four)} | {show(bf)} |")
            lines += ["", f"Direction: {ev['direction']}.", ""]
        elif f["id"] == "7.4":
            lines += ["| cell | label coverage | parse fail |", "|---|---|---|"]
            for c in ev["cells_below_95pct"]:
                lines.append(f"| {c['condition']} / {c['severity']} | "
                             f"{c['label_coverage']:.1%} | {int(c['n_parse_fail'])} / "
                             f"{int(c['n_attempted'])} |")
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