"""Re-derive the medgemma_15 findings from the CLEAN 2048-token re-run data.

docs/50 section 7.2 and 7.3 are written from the FIRST pre-run, which ran
medgemma_15 at a 256-token budget, so the thought channel was truncated on every
degraded image and at bf16. The 2026-10-08 re-run (@2048, uniform) changes those
cells: shift is 1897/1900 parse (3 absorbing loops), reliance and reliance_bf16 are
both 400/400. This script re-computes the descriptive statistics from the artifacts
so the paper's claims track the clean data instead of the budget-truncated one.

It fits no thresholds and reads only counts/rates that are already in the data:
descriptive statistics, nothing optimised on hold-out. The coverage gate (0.95)
is fixed in run_all_layers.ps1 and is NOT revisited here.

Not part of the pipeline.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("outputs/pretest/nih_eval")
OUT = Path("outputs/state/channel_analysis_2048.json")


def load(path):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_marker(r):
    return (r.get("raw_output") or "").startswith("<unused94>")


def parse_ok(r):
    return (r.get("parsed") or {}).get("parse_success", False)


def sev_agg(path):
    """Per-severity: n, parse_ok, n_marker, marker_recovered (marker & parsed)."""
    rows = load(path)
    agg = defaultdict(lambda: {"n": 0, "parse_ok": 0, "marker": 0, "marker_recovered": 0,
                               "marker_fail": 0})
    for r in rows:
        s = r.get("severity")
        a = agg[s]
        a["n"] += 1
        if parse_ok(r):
            a["parse_ok"] += 1
        if is_marker(r):
            a["marker"] += 1
            if parse_ok(r):
                a["marker_recovered"] += 1
            else:
                a["marker_fail"] += 1
    out = {}
    for s in sorted(agg):
        a = agg[s]
        out[s] = {
            "n": a["n"],
            "parse_ok": a["parse_ok"],
            "parse_rate": round(a["parse_ok"] / a["n"], 4),
            "marker": a["marker"],
            "marker_rate": round(a["marker"] / a["n"], 4),
            "marker_recovered": a["marker_recovered"],
            "marker_recovered_rate": round(a["marker_recovered"] / a["marker"], 4)
            if a["marker"] else None,
            "marker_fail": a["marker_fail"],
        }
    return out


def condition_parse(path):
    """Per-condition parse rate (for the reliance stage)."""
    rows = load(path)
    agg = defaultdict(lambda: {"n": 0, "parse_ok": 0})
    for r in rows:
        c = r.get("condition")
        agg[c]["n"] += 1
        if parse_ok(r):
            agg[c]["parse_ok"] += 1
    return {c: {"n": a["n"], "parse_ok": a["parse_ok"],
                "parse_rate": round(a["parse_ok"] / a["n"], 4)}
            for c, a in sorted(agg.items())}


report = {}

# ---- 1. shift @2048: severity vs marker, severity vs parse ----
shift = ROOT / "medgemma_15_4b_it/shift/shift_predictions.jsonl"
report["shift_2048_by_severity"] = sev_agg(shift)

# independent marker counts from the census artifact, to cross-check
cov = ROOT / "medgemma_15_4b_it/shift/analysis/coverage_report.json"
if cov.exists():
    j = json.load(cov.open(encoding="utf-8"))
    report["shift_2048_coverage_summary"] = {
        "n_marker_rows": sum(c["n_marker"] for c in j["cells"]),
        "n_marker_unparsed": sum(c["n_marker_unparsed"] for c in j["cells"]),
        "n_parse_fail": sum(c["n_parse_fail"] for c in j["cells"]),
        "cells_below_threshold": j["cells_below_threshold"],
        "min_label_coverage": min(c["label_coverage"] for c in j["cells"]),
    }

# ---- 2. reliance @2048: per-condition parse ----
rel = ROOT / "medgemma_15_4b_it/reliance/image_reliance_predictions.jsonl"
report["reliance_2048_by_condition"] = condition_parse(rel)

# ---- 3. reliance_bf16 @2048: per-condition parse (does bf16 still yield nothing?) ----
rel_bf16 = ROOT / "medgemma_15_4b_it/reliance_bf16/image_reliance_predictions.jsonl"
report["reliance_bf16_2048_by_condition"] = condition_parse(rel_bf16)

# ---- 4. reliance @2048 image_sensitivity, if the runner emitted it ----
# the stage summary (image-reliance) prints mean_flip_count per condition; recompute
# the key numbers directly from pairs is out of scope here, so pull the already-
# written summary if present.
for name, p in [("reliance", rel), ("reliance_bf16", rel_bf16)]:
    summ = p.parent / (p.stem.replace("_predictions", "") + "_summary.json".replace("predictions_summary", "summary"))
    cands = list(p.parent.glob("*summary*.json")) + list(p.parent.glob("*.json"))
    # the reliance stage wrote a summary JSON printed to stdout; find it by name only if present
    report.setdefault(name + "_image_sensitivity", None)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print("\nwrote", OUT)