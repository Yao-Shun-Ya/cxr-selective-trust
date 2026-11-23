# Artifact release (v9)

This directory is the upload bundle for a persistent archive (Zenodo or an
institutional repository). It contains no images and no model weights.

## Contents

- `manifests/` — the four evaluation manifests, the three cross-modal manifests
  and `MANIFEST_SHA256.json`.
- `predictions/` — the frozen prediction JSONL files for the CXR matrix, the
  cross-modal panel and the 9,400-call relay arm.
- `analysis/` — the derived CSVs behind the tables (`analysis_crossmodal`,
  `analysis_harness`, `analysis_geometry`, `probe`).
- `paper/` — `main_v9.pdf` and the LaTeX source tree.
- `code/` — the scripts needed to rebuild every table, including
  `120–130`, `api_probe/`, and the installable `audit/` CLI (`cxr-audit`).
- `README.md`, `REPRODUCING.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE`,
  `NOTICE`, `CITATION.cff`, `.zenodo.json`.
- `SHA256SUMS.txt` — SHA-256 of every file in the bundle.

## Verify the CLI directly

```bash
pip install -e "audit[dev]"        # inside the unpacked bundle
python -m pytest -q audit/tests    # 4 passed
python -m cxr_audit demo
```

## Build

```powershell
cd D:\Paper
python program\scripts\make_release_bundle.py
```

The script writes `release/cxr_trust_artifacts_v9.zip` using POSIX paths (a
Windows `Compress-Archive` zip is not used because it writes backslashes and
breaks some archive readers).

## Upload

The deposit and its DOI are already reserved: `10.5281/zenodo.24483387`.
The DOI is written into both manuscript PDFs, `CITATION.cff`, `README.md`,
`audit/README.md` and this bundle's metadata. It becomes resolvable only when
the deposit is published.

1. Upload `cxr_trust_artifacts_v9.zip` to the existing deposit (the zip already
   contains the reserved DOI).
2. Verify the metadata imported from `.zenodo.json` (Software, MIT, ORCID
   0009-0006-1226-8145, publication date 2026-11-24).
3. Publish, then confirm `https://doi.org/10.5281/zenodo.24483387` resolves.
4. Do not rebuild the zip after publishing; if a file must change, create a new
   Zenodo version. The concept DOI above stays stable; the version DOI changes.
5. The only remaining identifier is the arXiv id, to be added to `CITATION.cff`
   and the README after announcement.

## Licence boundary

Code, manifests and derived tables: MIT. Predictions are derived from public
de-identified datasets and are released for research use; the original dataset
licences (NIH ChestX-ray14, PadChest, Open-i, Figshare MRI, Kermany OCT,
HAM10000) continue to apply. Model weights are not included.
