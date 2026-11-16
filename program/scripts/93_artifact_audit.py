#!/usr/bin/env python
"""93 - Final artifact audit for v5.0, v6.0, and the engineering paper.

CPU-only. Checks the invariants a reader would check first:

* split-pure manifest has 1,506 rows; SHA-256 matches the lock file;
* census has 41 stages (the 39-stage delivered matrix plus two round-2
  reasoning stages), 284,695 calls in total, zero error rows;
* required scripts and CSV artifacts exist;
* bibliography has at least 40 cited entries and no rendered "Anonymous";
* the manuscript source trees contain the expected version markers.

PDF page counts are checked separately with PyMuPDF.

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\93_artifact_audit.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT.parent / "papers"
OUT = ROOT / "outputs" / "analysis_sensitivity" / "artifact_audit.json"

checks = []


def check(name: str, ok: bool, detail: str) -> None:
    checks.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def main() -> int:
    man = ROOT / "data" / "manifests"
    lock = json.loads((man / "MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    name = "nih_eval_1506.csv"
    p = man / name
    rows = sum(1 for _ in csv.DictReader(open(p, encoding="utf-8-sig")))
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    check("manifest rows", rows == 1506, str(rows))
    check("manifest sha256", digest == lock[name]["sha256"], digest[:16])

    census = list(csv.DictReader(open(
        ROOT / "outputs" / "analysis_sensitivity" / "coverage_census.csv",
        encoding="utf-8-sig")))
    calls = sum(int(r["calls"]) for r in census)
    errors = sum(int(r["errors"]) for r in census)
    check("matrix files", len(census) == 41, str(len(census)))
    check("matrix calls", calls == 284695, str(calls))
    check("error rows", errors == 0, str(errors))

    required = ["78_responsiveness_audit.py", "79_export_selective_inputs.py",
                "82_image_side_baseline.py", "83_quality_language_audit.py",
                "84_padchest_abnormal_subset.py", "85_operating_point_ci.py",
                "87_label_hierarchy_prevalence.py", "88_per_label_ci_fdr.py",
                "89_fixed_threshold_posthoc.py", "90_coverage_census.py",
                "91_manifest_lock.py", "92_make_bench_figures.py",
                "103_densenet_bootstrap.py", "104_table_consistency_check.py",
                "105_outputs_manifest.py", "106_clean_auroc_ci.py",
                "107_padchest_label_hierarchy.py", "108_fixed_rule_rerun.py",
                "109_chexbert_openi.py", "110_compare_determinism.py",
                "111_densenet_grid.py", "112_analyze_xfamily_schema.py",
                "113_densenet_conditions.py", "114_report_labeller_agreement.py",
                "115_llava_nih.py"]
    missing = [s for s in required if not (ROOT / "scripts" / s).exists()]
    check("required scripts", not missing, "missing: " + ",".join(missing) if missing else "all present")

    csvs = ["label_hierarchy_prevalence.csv", "label_hierarchy_pairwise_tests.csv",
            "label_hierarchy_strata.csv", "per_label_bootstrap_fdr.csv",
            "fixed_threshold_posthoc.csv", "coverage_census.csv",
            "run_accounting.json"]
    missing = [s for s in csvs if not (ROOT / "outputs" / "analysis_sensitivity" / s).exists()]
    check("analysis artifacts", not missing, "missing: " + ",".join(missing) if missing else "all present")

    for paper in ["main_v5", "main_v5_zh", "main_v6", "main_v6_zh",
                  "main_v7", "main_v7_zh", "bench_v1"]:
        bbl = PAPERS / paper / "main.bbl"
        if not bbl.exists():
            check(f"{paper} bibliography", False, "main.bbl missing")
            continue
        text = bbl.read_text(encoding="utf-8", errors="ignore")
        n = text.count("\\bibitem")
        check(f"{paper} citations", n >= (40 if paper != "bench_v1" else 20), str(n))
        check(f"{paper} anonymous refs", "Anonymous" not in text, "no rendered Anonymous")

    markers = {"main_v5": "5.0", "main_v5_zh": "5.0", "main_v6": "6.0",
               "main_v6_zh": "6.0", "bench_v1": "1.0"}
    for paper, marker in markers.items():
        tex = (PAPERS / paper / "main.tex").read_text(encoding="utf-8")
        check(f"{paper} version marker", marker in tex, paper)

    OUT.write_text(json.dumps(checks, indent=2), encoding="utf-8")
    failed = [c for c in checks if not c["ok"]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
