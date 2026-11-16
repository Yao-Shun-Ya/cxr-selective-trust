#!/usr/bin/env python
"""90 - Coverage census and run accounting over the frozen matrix.

CPU-only, read-only.  Produces one row per (layer, model, stage/file) with:

* calls (JSONL rows), parse successes, error rows;
* parse rate and error rate;
* all-definite rate for structured layers (eight statuses, none uncertain /
  not_mentioned);
* summed ``elapsed_sec`` and bytes on disk.

Also writes a compact run-level JSON (total calls, GPU-hour sum, bytes,
per-model throughput) used by the benchmark-engineering paper.

Inputs:  outputs/matrix/{nih_eval,padchest,openi}/**/*predictions.jsonl
Outputs: outputs/analysis_sensitivity/coverage_census.csv
         outputs/analysis_sensitivity/run_accounting.json

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\90_coverage_census.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "outputs" / "matrix"
OUT = ROOT / "outputs" / "analysis_sensitivity"
LABELS = ["atelectasis", "cardiomegaly", "consolidation", "edema",
          "pleural_effusion", "pneumonia", "pneumothorax", "no_finding"]


def main() -> int:
    rows = []
    total_calls = total_elapsed = total_bytes = 0
    per_model: dict[str, dict[str, float]] = {}
    files = sorted(MATRIX.rglob("*predictions.jsonl"))
    for path in files:
        rel = path.relative_to(MATRIX)
        if any(part.startswith("_") for part in rel.parts):
            continue
        layer, model = rel.parts[0], rel.parts[1]
        stage = "/".join(rel.parts[2:-1]) or rel.parts[-1].replace("_predictions.jsonl", "")
        n = parse_ok = errors = all_def = structured = 0
        elapsed = 0.0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                if r.get("error"):
                    errors += 1
                p = r.get("parsed") or {}
                if p.get("parse_success"):
                    parse_ok += 1
                findings = p.get("findings") or {}
                if findings:
                    structured += 1
                    ok = all(str((findings.get(l) or {}).get("status") or "")
                             in {"positive", "negative"} for l in LABELS)
                    all_def += int(ok)
                e = r.get("elapsed_sec")
                if isinstance(e, (int, float)):
                    elapsed += float(e)
        size = path.stat().st_size
        total_calls += n
        total_elapsed += elapsed
        total_bytes += size
        m = per_model.setdefault(model, {"calls": 0, "elapsed_sec": 0.0})
        m["calls"] += n
        m["elapsed_sec"] += elapsed
        rows.append({
            "layer": layer, "model": model, "stage": stage,
            "calls": n, "parse_ok": parse_ok, "parse_rate": round(parse_ok / n, 6) if n else None,
            "errors": errors, "error_rate": round(errors / n, 6) if n else None,
            "structured_rows": structured,
            "all_definite": all_def,
            "all_definite_rate": round(all_def / structured, 6) if structured else None,
            "elapsed_sec": round(elapsed, 3),
            "bytes": size,
        })

    with open(OUT / "coverage_census.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    accounting = {
        "files": len(rows),
        "calls": total_calls,
        "elapsed_sec_sum": round(total_elapsed, 1),
        "gpu_hours_sum": round(total_elapsed / 3600, 2),
        "prediction_bytes": total_bytes,
        "prediction_gib": round(total_bytes / (1024 ** 3), 3),
        "per_model": {
            k: {"calls": v["calls"],
                "elapsed_sec": round(v["elapsed_sec"], 1),
                "gpu_hours": round(v["elapsed_sec"] / 3600, 2),
                "sec_per_call": round(v["elapsed_sec"] / v["calls"], 4) if v["calls"] else None}
            for k, v in sorted(per_model.items())
        },
    }
    (OUT / "run_accounting.json").write_text(
        json.dumps(accounting, indent=2), encoding="utf-8")
    print(json.dumps(accounting, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
