"""Archive the real-model smoke evidence for the two new layers into outputs/smoke.

Kept as a script rather than a shell one-liner because the report layer's
deliverable is the generated *text*: it has to be captured verbatim somewhere a
human can read, not just summarised into numbers.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "smoke"


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: archive_smoke.py <padchest_dir> <openi_dir>")
        return 2
    pad_dir, oi_dir = Path(sys.argv[1]), Path(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Real-model smoke: the two NEW layers",
        "",
        "Purpose: the stubbed tests prove the I/O and resume contract. They cannot",
        "prove that a real wrapper follows the new prompts, produces prose for the",
        "report layer, or emits parseable JSON for the shift layer. These runs do.",
        "Tiny n on purpose: this is a wiring check, not a result.",
        "",
    ]

    for layer, src, cmd in (
        ("padchest", pad_dir,
         "43_run_padchest_shift.py --model medgemma_4b_it --limit 1 "
         "--conditions original blank --load-in-4bit"),
        ("openi", oi_dir,
         "44_run_openi_report.py --model medgemma_4b_it --limit 2 "
         "--conditions original blank --load-in-4bit"),
    ):
        lines.append(f"## {layer}")
        lines.append(f"command: `python scripts/{cmd}`")
        lines.append("")
        for f in sorted(src.glob("*.json")):
            d = json.loads(f.read_text(encoding="utf-8"))
            for k in ("layer", "design", "n_images", "n_studies", "cells_on_disk",
                      "contrast_source", "format_violations", "empty_outputs",
                      "blank_control", "by_condition",
                      "generation_failures_by_condition",
                      "structured_parse_ok_by_condition"):
                if k in d:
                    lines.append(f"* {k}: `{json.dumps(d[k], ensure_ascii=False)}`")
        lines.append("")

        for name in ("rrg_predictions.jsonl", "image_reliance_predictions.jsonl"):
            p = src / name
            if not p.exists():
                continue
            rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines()
                    if ln.strip()]
            lines.append(f"### {name} ({len(rows)} rows)")
            for r in rows:
                gen = (r.get("raw_output") or "").replace("\n", " ")[:320]
                ref = (r.get("reference_text") or "").replace("\n", " ")[:220]
                lines.append(f"* `{r.get('image_id')}` **[{r.get('condition')}]**")
                if gen:
                    lines.append(f"  * generated: {gen}")
                if ref:
                    lines.append(f"  * reference: {ref}")
            lines.append("")

        # same-condition report pairs: the whole point of the blank control
        pair_src = src / "rrg_predictions.jsonl"
        if pair_src.exists():
            by: dict[str, dict[str, str]] = {}
            for r in (json.loads(ln) for ln in
                      pair_src.read_text(encoding="utf-8").splitlines() if ln.strip()):
                by.setdefault(str(r["image_id"]), {})[str(r["condition"])] = \
                    r.get("raw_output") or ""
            lines.append("### original vs blank, per study")
            for iid, d in by.items():
                same = d.get("original", "") == d.get("blank", "")
                lines.append(f"* `{iid}` identical={same} "
                             f"(len original={len(d.get('original', ''))}, "
                             f"blank={len(d.get('blank', ''))})")
            lines.append("")
            lines.append("An identical pair means the model produced its report without "
                         "using the image at all -- the blank control's entire purpose.")
            lines.append("")

    shutil.copy(oi_dir / "rrg_predictions.jsonl", OUT / "openi_real_smoke.jsonl")
    shutil.copy(pad_dir / "image_reliance_predictions.jsonl",
                OUT / "padchest_real_smoke.jsonl")
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'README.md'} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
