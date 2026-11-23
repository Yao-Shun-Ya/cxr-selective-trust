"""Build arXiv-ready source zips for the current manuscripts.

Each zip contains only files arXiv needs: the root ``main.tex``, the generated
``main.bbl``, ``refs.bib``, every section that is actually compiled, and the PDF
figures.  PNG previews, aux/log files, the compiled PDF and the unused short
abstracts are excluded.  Entries are POSIX paths with no wrapping directory, so
``main.tex`` sits at the archive root.

    python program/scripts/make_arxiv_packages.py
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "release" / "arxiv"
PAPERS = ("main_v9", "bench_v3")


def files_for(paper: str) -> list[Path]:
    base = ROOT / "papers" / paper
    files = [base / "main.tex", base / "main.bbl", base / "refs.bib"]
    files += sorted(p for p in (base / "sections").glob("*.tex")
                    if not p.name.endswith("_short.tex"))
    files += sorted((base / "figures").glob("*.pdf"))
    return files


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for paper in PAPERS:
        base = ROOT / "papers" / paper
        files = files_for(paper)
        missing = [str(p) for p in files if not p.is_file()]
        if missing:
            raise SystemExit(f"{paper}: missing {missing}")
        out = OUT / f"{paper}_arxiv.zip"
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9) as zf:
            for path in files:
                zf.write(path, arcname=path.relative_to(base).as_posix())
        names = zipfile.ZipFile(out).namelist()
        bad = [n for n in names if "\\" in n or n.startswith("/")]
        if bad:
            raise SystemExit(f"{paper}: non-POSIX entries: {bad}")
        if "main.tex" not in names:
            raise SystemExit(f"{paper}: main.tex is not at the archive root")
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f"{paper}: {len(files)} files, {out.stat().st_size / 2**20:.2f} MiB, "
              f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
