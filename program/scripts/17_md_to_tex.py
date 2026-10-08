#!/usr/bin/env python
"""Best-effort Markdown -> LaTeX converter for paper drafts."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "papers/main"
SRC = PAPER / "sections"

CITE_MAP = {
    "(Chen et al., 2024)": "chen2024chexagent",
    "(Sellergren et al., 2025)": "sellergren2025medgemma",
    "(Bai et al., 2025)": "bai2025qwen25vl",
    "(Cheng et al., 2025)": "cheng2025robustness",
    "(Gu et al., 2025)": "gu2025illusion",
    "(Buckley et al., 2026)": "buckley2026exploit",
    "(Felizzi et al., 2025)": "felizzi2025grounded",
    "(Zafar et al., 2026)": "zafar2026beyond",
    "(Dhinagar et al., 2026)": "dhinagar2026calm",
    "(Nguyen et al., 2026)": "nguyen2026answer",
    "(Li et al., 2025)": "li2025atcxr",
    "(Fisher, 2025)": "fisher2025negative",
    "(Rafferty et al., 2026)": "rafferty2026limitations",
    "(Heiman et al., 2025)": "heiman2025factchexcker",
}


def protect_citations(line: str) -> str:
    for raw, key in CITE_MAP.items():
        line = line.replace(raw, f"@@CIT:{key}@@")
    return line


def escape_latex(text: str) -> str:
    unicode_map = {
        "–": "--", "—": "---", "≈": "$\\approx$", "∈": "$\\in$",
        "×": "$\\times$", "≥": "$\\ge$", "≤": "$\\le$",
        "σ": "$\\sigma$", "Δ": "$\\Delta$", "→": "$\\rightarrow$",
        "±": "$\\pm$",
    }
    for a, b in unicode_map.items():
        text = text.replace(a, b)
    for a, b in [("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}")]:
        text = text.replace(a, b)
    return text

def inline(text: str) -> str:
    text = protect_citations(text)
    text = escape_latex(text)
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: r"\textbf{" + m.group(1) + "}", text)
    text = re.sub(r"\*(.+?)\*", lambda m: r"\emph{" + m.group(1) + "}", text)
    text = re.sub(r"`(.+?)`", lambda m: r"\texttt{" + m.group(1) + "}", text)
    text = re.sub(r"@@CIT:([^@]+)@@", lambda m: r"\citep{" + m.group(1) + "}", text)
    return text


def convert_table(lines: list[str]) -> list[str]:
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            continue
        rows.append([inline(c) for c in cells])
    if not rows:
        return []
    ncol = max(len(r) for r in rows)
    out = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{" + "l" * ncol + "}", r"\hline"]
    for r in rows:
        r = r + [""] * (ncol - len(r))
        out.append(" & ".join(r) + r" \\")
    out += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    return out


def convert(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            out.append(r"\begin{verbatim}")
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                out.append(lines[i]); i += 1
            out.append(r"\end{verbatim}")
            i += 1
            continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i]); i += 1
            out.extend(convert_table(block))
            continue
        if line.startswith("# "):
            title = re.sub(r"^\d+(\.\d+)*\.?\s+", "", line[2:].strip())
            out.append(r"\section{" + inline(title) + "}")
        elif line.startswith("## "):
            title = re.sub(r"^\d+(\.\d+)*\.?\s+", "", line[3:].strip())
            out.append(r"\subsection{" + inline(title) + "}")
        elif line.startswith("### "):
            title = re.sub(r"^\d+(\.\d+)*\.?\s+", "", line[4:].strip())
            out.append(r"\subsubsection{" + inline(title) + "}")
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:].strip()); i += 1
            out.append(r"\begin{itemize}")
            for item in items:
                out.append(r"\item " + inline(item))
            out.append(r"\end{itemize}")
            continue
        elif re.match(r"^\d+\. ", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                items.append(re.sub(r"^\d+\. ", "", lines[i]).strip()); i += 1
            out.append(r"\begin{enumerate}")
            for item in items:
                out.append(r"\item " + inline(item))
            out.append(r"\end{enumerate}")
            continue
        elif line.strip() == "":
            out.append("")
        else:
            out.append(inline(line))
        i += 1
    return "\n".join(out) + "\n"


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    for md in sorted(SRC.glob("*.md")):
        tex = SRC / (md.stem + ".tex")
        tex.write_text(convert(md), encoding="utf-8")
        print("wrote", tex)


if __name__ == "__main__":
    main()
