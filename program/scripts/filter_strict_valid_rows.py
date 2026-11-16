#!/usr/bin/env python
"""Keep only strictly valid Thinking replies; archive the rest for budget analysis.

Definition of "valid" for this model
-------------------------------------
Qwen3-VL-Thinking's chat template pre-fills `<think>` into the prompt, so the
decoded reply is reasoning text followed by `</think>` and then the answer.
A reply counts as a usable answer only when

  1. it contains a closing `</think>`, and
  2. a JSON object holding `findings` can be decoded from the text that follows
     the *last* close tag.

Anything else is not an answer: it may be a truncated trace (hit the token cap
before closing), a trace that closed but never emitted JSON, or a reply whose
JSON appears only inside the reasoning. Those rows are moved to an archive so
the termination/length analysis can still use them, and the grid file keeps
only rows the analyses may treat as predictions.

Usage:
  .\\.venv-vl\\Scripts\\python.exe scripts\\filter_strict_valid_rows.py \
      --path outputs/matrix/nih_eval/qwen3_vl_8b_thinking/shift/shift_predictions.jsonl \
      --archive outputs/analysis_round2/thinking_invalid_archive.jsonl \
      --pass-label cap3584
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tail_findings(raw: str) -> dict | None:
    """Decode the JSON answer that follows the last closing tag, if any."""
    t = raw.rfind("</think>")
    if t < 0:
        return None
    seg = raw[t + len("</think>"):]
    i = seg.find("{")
    if i < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(seg[i:])
    except Exception:
        return None
    f = obj.get("findings")
    return f if isinstance(f, dict) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--pass-label", default="unknown")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.is_absolute():
        path = ROOT / path
    archive = Path(args.archive)
    if not archive.is_absolute():
        archive = ROOT / archive
    if not path.exists():
        raise SystemExit(f"missing {path}")

    backup = path.with_name(path.name + time.strftime(".bak_%Y%m%d_%H%M%S"))
    shutil.copy2(path, backup)

    keep, drop = [], []
    with path.open(encoding="utf-8-sig") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            raw = rec.get("raw_output") or ""
            ok = bool((rec.get("parsed") or {}).get("parse_success"))
            if ok and tail_findings(raw) is not None:
                keep.append(line if line.endswith("\n") else line + "\n")
            else:
                drop.append(rec)

    if not drop:
        backup.unlink()
        print(f"[filter] nothing to drop; kept {len(keep)} rows")
        return 0

    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("a", encoding="utf-8") as fh:
        for rec in drop:
            raw = rec.get("raw_output") or ""
            fh.write(json.dumps({
                "pass": args.pass_label,
                "image_id": rec.get("image_id"),
                "condition": rec.get("condition"),
                "severity": rec.get("severity"),
                "chars": len(raw),
                "has_close": "</think>" in raw,
                "has_json_after_close": tail_findings(raw) is not None,
                "parse_success": bool((rec.get("parsed") or {}).get("parse_success")),
            }, ensure_ascii=False) + "\n")

    payload = "".join(keep).encode("utf-8")
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    sidecar = path.with_suffix(path.suffix + ".progress.json")
    if sidecar.exists():
        sidecar.unlink()

    print(f"[filter] kept {len(keep)} valid rows, archived {len(drop)} invalid rows")
    print(f"[filter] backup {backup.name}")
    print(f"[filter] archive {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
