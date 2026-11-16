#!/usr/bin/env python
"""98 - Integrity audit of the round-2 outputs (CPU only, read-only).

Why this exists
---------------
The round-2 grid ran at 2,560 tokens and the runner reported a low parse rate
(65 % clean, 56 % severe). A parse failure is not a cosmetic detail here: the
truncated replies are the long reasoning traces, i.e. exactly the difficult
cells the damage axis cares about. Before any analysis is trusted, the audit
has to say *why* a reply failed:

  truncated     no </think>, reply hit the token cap (length near the cap)
  no_json       closed the reasoning block but never emitted an object
  malformed     emitted something that looks like JSON but the parser rejected it
  empty         no usable text

This script never calls a model. It writes:
  outputs/analysis_round2/round2_audit.json
  outputs/analysis_round2/round2_audit_cells.csv

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\98_round2_audit.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "analysis_round2"
GRID = (ROOT / "outputs" / "matrix" / "nih_eval" / "qwen3_vl_8b_thinking"
        / "shift" / "shift_predictions.jsonl")
ABL = ROOT / "outputs" / "ablation_mg15" / "medgemma_15_4b_it"
PAR = ROOT / "outputs" / "paraphrase_mg4" / "medgemma_4b_it"
DENSE = ROOT / "outputs" / "densenet_baseline"


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    out.append({"_torn": True})
    return out


def classify_failure(rec: dict, cap_chars: int) -> str:
    raw = rec.get("raw_output") or ""
    if not raw.strip():
        return "empty"
    closed = "</think>" in raw
    has_brace = "{" in raw
    if not closed and len(raw) >= cap_chars * 0.6:
        return "truncated"
    if not has_brace:
        return "no_json"
    return "malformed"


def cell_key(rec: dict) -> str:
    return f"{rec.get('condition')}/{rec.get('severity')}"


def audit_grid() -> dict:
    rs = [r for r in read_jsonl(GRID) if not r.get("_torn")]
    keys = defaultdict(int)
    cells = defaultdict(lambda: {"n": 0, "parse": 0, "close": 0, "chars": []})
    failures = Counter()
    fail_cells = Counter()
    for r in rs:
        keys[(str(r.get("image_id")), str(r.get("condition")), str(r.get("severity")))] += 1
        k = cell_key(r)
        c = cells[k]
        c["n"] += 1
        raw = r.get("raw_output") or ""
        c["chars"].append(len(raw))
        ok = bool((r.get("parsed") or {}).get("parse_success"))
        c["parse"] += int(ok)
        c["close"] += int("</think>" in raw)
        if not ok:
            # 2,560 tokens is roughly 10-11k characters of this model's prose;
            # the audit only needs an order-of-magnitude cap.
            why = classify_failure(r, cap_chars=10000)
            failures[why] += 1
            fail_cells[f"{k}|{why}"] += 1
    dups = sum(1 for v in keys.values() if v > 1)
    per_cell = []
    for k, c in sorted(cells.items()):
        chars = c["chars"]
        per_cell.append({
            "cell": k, "n": c["n"],
            "parse_rate": round(c["parse"] / c["n"], 4) if c["n"] else None,
            "close_rate": round(c["close"] / c["n"], 4) if c["n"] else None,
            "mean_chars": round(sum(chars) / len(chars), 1) if chars else None,
            "max_chars": max(chars) if chars else None,
        })
    return {
        "rows": len(rs),
        "unique_keys": len(keys),
        "duplicate_keys": dups,
        "parse_failures": sum(failures.values()),
        "parse_rate": round((len(rs) - sum(failures.values())) / len(rs), 4) if rs else None,
        "failure_reasons": dict(failures),
        "failure_by_cell_reason": dict(fail_cells),
        "cells": per_cell,
    }


def audit_ablation(root: Path, tag: str) -> dict:
    out = {}
    if not root.exists():
        return out
    for variant_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for cond_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
            path = cond_dir / "image_reliance_predictions.jsonl"
            if not path.exists():
                out[f"{variant_dir.name}/{cond_dir.name}"] = {"n": 0}
                continue
            rs = [r for r in read_jsonl(path) if not r.get("_torn")]
            keys = Counter(str(r.get("image_id")) for r in rs)
            ok = sum(1 for r in rs if (r.get("parsed") or {}).get("parse_success"))
            viol = sum(1 for r in rs if (r.get("contract") or {}).get("contract_violation"))
            out[f"{variant_dir.name}/{cond_dir.name}"] = {
                "n": len(rs),
                "unique": len(keys),
                "duplicates": sum(1 for v in keys.values() if v > 1),
                "parse_rate": round(ok / len(rs), 4) if rs else None,
                "contract_violation_rate": round(viol / len(rs), 4) if rs else None,
            }
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "grid": audit_grid(),
        "ablation": audit_ablation(ABL, "mg15"),
        "paraphrase": audit_ablation(PAR, "mg4"),
    }
    metrics = DENSE / "metrics.json"
    payload["densenet"] = (json.loads(metrics.read_text(encoding="utf-8"))
                           if metrics.exists() else {})
    preds = DENSE / "test_predictions.csv"
    payload["densenet"]["prediction_rows"] = (
        len(preds.read_text(encoding="utf-8").splitlines()) - 1 if preds.exists() else 0)

    (OUT / "round2_audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    g = payload["grid"]
    print(f"grid rows={g['rows']} unique={g['unique_keys']} dup={g['duplicate_keys']} "
          f"parse={g['parse_rate']}")
    print("failure reasons:", g["failure_reasons"])
    for row in g["cells"]:
        print(f"  {row['cell']:<18} n={row['n']:<5} parse={row['parse_rate']} "
              f"close={row['close_rate']} mean_chars={row['mean_chars']}")
    for tag in ("ablation", "paraphrase"):
        bad = {k: v for k, v in payload[tag].items()
               if v.get("n") and (v.get("duplicates") or v.get("n") != 1506)}
        print(f"{tag}: {len(payload[tag])} cells, anomalies: {bad if bad else 'none'}")
    print(f"densenet: eval_n={payload['densenet'].get('eval_n')} "
          f"macro={payload['densenet'].get('eval_1506', {}).get('macro_pathology_auroc')} "
          f"pred_rows={payload['densenet'].get('prediction_rows')}")
    print(f"wrote {OUT / 'round2_audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
