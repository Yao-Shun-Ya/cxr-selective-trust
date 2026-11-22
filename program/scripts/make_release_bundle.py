"""Build the v9 artifact bundle with POSIX paths and a SHA-256 manifest.

    python program/scripts/make_release_bundle.py

Writes ``release/cxr_trust_artifacts_v9.zip``. No images, no model weights.
Windows ``Compress-Archive`` is deliberately not used: it writes backslash
entry names, which some archive readers mis-handle (the arXiv upload failure in
this project was exactly that).
"""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "release"
OUT = REL / "cxr_trust_artifacts_v9.zip"

MANIFESTS = [
    "program/data/manifests/nih_eval_1506.csv",
    "program/data/manifests/nih_eval_1732.csv",
    "program/data/manifests/padchest_shift.csv",
    "program/data/manifests/openi_rrg.csv",
    "program/data/manifests/crossmodal_mri.csv",
    "program/data/manifests/crossmodal_oct.csv",
    "program/data/manifests/crossmodal_derm.csv",
    "program/data/manifests/MANIFEST_SHA256.json",
]

ANALYSIS_DIRS = [
    "program/outputs/analysis_crossmodal",
    "program/outputs/analysis_harness",
    "program/outputs/analysis_geometry",
    "program/outputs/analysis_selective_baselines",
    "program/outputs/probe",
]

PAPER = [
    "papers/main_v9/main.tex",
    "papers/main_v9/main.bbl",
    "papers/main_v9/main.pdf",
    "papers/main_v9/refs.bib",
    "papers/bench_v3/main.tex",
    "papers/bench_v3/main.bbl",
    "papers/bench_v3/main.pdf",
    "papers/bench_v3/refs.bib",
]

CODE_GLOBS = [
    "program/scripts/1[23][0-9]_*.py",
    "program/scripts/run_geometry_sweep.ps1",
    "program/scripts/run_crossmodal_6cell_queue.ps1",
    "program/scripts/api_probe/*.py",
    "program/scripts/api_probe/*.md",
    "program/scripts/api_probe/*.ps1",
    "program/src/cxr_trust/**/*.py",
    "program/env/lock-*.txt",
    "audit/cxr_audit/*.py",
    "audit/tests/*.py",
]

TOP = ["README.md", "REPRODUCING.md", "LICENSE", "NOTICE", "release/CITATION.cff",
       "release/README.md", "program/outputs/state/run_all_layers_state.json",
       "CITATION.cff", ".zenodo.json", "CHANGELOG.md", "CONTRIBUTING.md",
       "audit/README.md", "audit/pyproject.toml"]


def _add(files: set[Path], p: Path) -> None:
    if p.is_file():
        files.add(p)


def main() -> None:
    files: set[Path] = set()
    for rel in MANIFESTS + PAPER + TOP:
        _add(files, ROOT / rel)
    for rel in ANALYSIS_DIRS:
        for p in (ROOT / rel).rglob("*"):
            if p.is_file() and p.suffix.lower() in (".csv", ".json", ".jsonl", ".md"):
                files.add(p)
    for pattern in CODE_GLOBS:
        for p in ROOT.glob(pattern):
            _add(files, p)
    for p in (ROOT / "papers" / "main_v9" / "sections").rglob("*.tex"):
        files.add(p)
    for p in (ROOT / "papers" / "main_v9" / "figures").rglob("*.pdf"):
        files.add(p)
    for p in (ROOT / "papers" / "bench_v3" / "sections").rglob("*.tex"):
        files.add(p)
    for p in (ROOT / "papers" / "bench_v3" / "figures").rglob("*.pdf"):
        files.add(p)
    for p in (ROOT / "program" / "outputs" / "matrix").rglob("*_predictions.jsonl"):
        files.add(p)
    for p in (ROOT / "program" / "outputs" / "crossmodal").rglob("shift_predictions.jsonl"):
        files.add(p)
    for p in (ROOT / "program" / "outputs" / "crossmodal_retry").rglob("shift_predictions.jsonl"):
        files.add(p)
    for p in (ROOT / "program" / "outputs" / "crossmodal_retry4k").rglob("shift_predictions.jsonl"):
        files.add(p)
    for p in (ROOT / "program" / "outputs" / "geometry").rglob("shift_predictions.jsonl"):
        files.add(p)
    _add(files, ROOT / "program/outputs/api_probe/deepseek_v4_1_flash/predictions.jsonl")

    rels = sorted(f.relative_to(ROOT).as_posix() for f in files)
    lines = []
    for rel in rels:
        h = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        lines.append(f"{h}  {rel}")
    sums = "\n".join(lines) + "\n"
    (REL / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")

    REL.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        for rel in rels:
            zf.write(ROOT / rel, arcname=f"cxr_trust_artifacts_v9/{rel}")
        zf.writestr("cxr_trust_artifacts_v9/SHA256SUMS.txt", sums)
    print(f"{len(rels)} files -> {OUT} "
          f"({OUT.stat().st_size / 2**20:.1f} MiB)")
    bad = [n for n in zipfile.ZipFile(OUT).namelist() if "\\" in n]
    print("backslash entries:", len(bad))
    if bad:
        raise SystemExit("zip contains backslash paths")


if __name__ == "__main__":
    main()
