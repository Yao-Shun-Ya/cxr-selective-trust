#!/usr/bin/env python
"""117 - Structural and numeric inventory for the v7 manuscripts.

This is an audit *instrument*, not a result generator.  It walks every section
file of the English and Chinese manuscripts and records:

  * every label and every \ref (duplicates, undefined references),
  * every table/figure block with its label and caption,
  * every numeric token inside and outside tables, with file:line context,
  * the numeric-token sequence of each table, so the English and Chinese
    versions of the same label can be compared cell by cell.

Outputs (under outputs/analysis_audit/):
  v7_inventory.json
  v7_inventory.txt
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT.parent / "papers"
OUT = ROOT / "outputs" / "analysis_audit"
LANGS = {"en": PAPERS / "main_v7" / "sections", "zh": PAPERS / "main_v7_zh" / "sections"}

NUM = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:\s*[eE][-+]?\d+)?%?")
LABEL = re.compile(r"\\label\{([^}]+)\}")
REF = re.compile(r"\\(?:ref|eqref|autoref)\{([^}]+)\}")
TABLE_START = re.compile(r"\\begin\{table\}")
FIG_START = re.compile(r"\\begin\{figure\}")
CAPTION = re.compile(r"\\caption\{(.+?)\}\s*(?:\\label\{[^}]+\})?", re.S)


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def blocks(text: str, env: str) -> list[tuple[int, int, str]]:
    out = []
    pat = re.compile(rf"\\begin\{{{env}\}}(.*?)\\end\{{{env}\}}", re.S)
    for m in pat.finditer(text):
        out.append((m.start(), m.end(), m.group(1)))
    return out


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def numeric_tokens(s: str) -> list[str]:
    s = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?", " ", s)
    s = s.replace("$", " ").replace("\\", " ")
    return [m.group(0).replace(" ", "") for m in NUM.finditer(s)]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    inv: dict = {"languages": {}, "cross_language": {}}
    for lang, sect in LANGS.items():
        labels: list[dict] = []
        refs: list[dict] = []
        tables: list[dict] = []
        figures: list[dict] = []
        numeric: list[dict] = []
        for path in sorted(sect.glob("*.tex")):
            raw = path.read_text(encoding="utf-8")
            text = strip_comments(raw)
            for m in LABEL.finditer(text):
                labels.append({"file": path.name, "line": line_of(text, m.start()),
                               "label": m.group(1)})
            for m in REF.finditer(text):
                refs.append({"file": path.name, "line": line_of(text, m.start()),
                             "ref": m.group(1)})
            for env, bucket in (("table", tables), ("figure", figures)):
                for start, _end, body in blocks(text, env):
                    lm = LABEL.search(body)
                    cm = CAPTION.search(body)
                    tm = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", body, re.S)
                    bucket.append({
                        "file": path.name,
                        "line": line_of(text, start),
                        "label": lm.group(1) if lm else None,
                        "caption": re.sub(r"\s+", " ", cm.group(1))[:400] if cm else "",
                        "numbers": numeric_tokens(tm.group(0)) if tm else [],
                        "caption_numbers": numeric_tokens(cm.group(1)) if cm else [],
                    })
            for i, line in enumerate(text.splitlines(), 1):
                if "\\begin{tabular}" in line or "\\end{tabular}" in line:
                    continue
                toks = numeric_tokens(line)
                if toks:
                    numeric.append({"file": path.name, "line": i,
                                    "tokens": toks, "text": line.strip()[:300]})
        label_names = [x["label"] for x in labels]
        dup = {k: v for k, v in Counter(label_names).items() if v > 1}
        ref_names = {x["ref"] for x in refs}
        undefined = sorted(ref_names - set(label_names))
        unreferenced = sorted(set(label_names) - ref_names)
        inv["languages"][lang] = {
            "files": len(list(sect.glob("*.tex"))),
            "labels": labels, "refs": refs,
            "tables": tables, "figures": figures,
            "numeric": numeric,
            "duplicate_labels": dup,
            "undefined_refs": undefined,
            "unreferenced_labels": unreferenced,
            "counts": {"labels": len(labels), "refs": len(refs),
                       "tables": len(tables), "figures": len(figures),
                       "numeric_lines": len(numeric),
                       "numeric_tokens": sum(len(x["tokens"]) for x in numeric)},
        }
    # cross-language table comparison by label
    en = {t["label"]: t for t in inv["languages"]["en"]["tables"] if t["label"]}
    zh = {t["label"]: t for t in inv["languages"]["zh"]["tables"] if t["label"]}
    for lab in sorted(set(en) & set(zh)):
        a, b = en[lab]["numbers"], zh[lab]["numbers"]
        if a != b:
            diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            inv["cross_language"][lab] = {
                "en_n": len(a), "zh_n": len(b),
                "first_diffs": [{"i": i, "en": a[i] if i < len(a) else None,
                                 "zh": b[i] if i < len(b) else None} for i in diff[:20]],
            }
    (OUT / "v7_inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    lines = []
    for lang, d in inv["languages"].items():
        lines.append(f"== {lang} ==")
        lines.append(json.dumps(d["counts"], ensure_ascii=False))
        lines.append(f"duplicate labels: {d['duplicate_labels']}")
        lines.append(f"undefined refs: {d['undefined_refs']}")
        lines.append(f"unreferenced labels: {d['unreferenced_labels']}")
        for t in d["tables"]:
            lines.append(f"TABLE {t['label']} [{t['file']}:{t['line']}] "
                         f"nums={len(t['numbers'])} :: {t['caption'][:120]}")
        for f in d["figures"]:
            lines.append(f"FIGURE {f['label']} [{f['file']}:{f['line']}] :: {f['caption'][:120]}")
    lines.append("== cross-language table numeric mismatches ==")
    for lab, d in inv["cross_language"].items():
        lines.append(f"{lab}: en={d['en_n']} zh={d['zh_n']} diffs={d['first_diffs']}")
    (OUT / "v7_inventory.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:80]))
    print(f"\nwrote {OUT/'v7_inventory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
