# Contributing

This repository is a measurement-validity study, not a model release. The two
things that matter most are that **every reported number can be recomputed from a
frozen artifact** and that **no patient data, image, credential or model weight
enters git**. Changes that weaken either property are rejected.

## Before you start

- Read `docs/42_academic_integrity_and_writing_standards.md` (reporting rules)
  and `docs/49_data_manifest_contract.md` (manifest contract).
- Do not open issues or pull requests containing patient data, report text,
  API keys, prediction rows or model outputs. Describe the problem instead.

## Fast path: CPU-only checks

No GPU, weights or datasets are needed for the checks below.

```bash
# audit CLI (Python >= 3.10)
python -m venv .venv-audit
.venv-audit/Scripts/python.exe -m pip install -e "audit[dev]"
.venv-audit/Scripts/python.exe -m pytest -q audit/tests      # 4 passed
.venv-audit/Scripts/python.exe -m cxr_audit demo

# legacy parser tests
python -m pytest -q tests                                     # 6 passed

# authoritative package
cd program && .venv-vl/Scripts/python.exe -m pytest -q tests  # 11 passed
```

`make test`, `make test-cxr`, `make audit-cli-test` and `make audit-demo` wrap
these commands. CI (`.github/workflows/ci.yml`) runs the same checks on Linux
and Windows.

## Editing rules

1. **One operator implementation.** Image shift operators live only in
   `program/src/cxr_trust/inference/shifts.py`; the pytest pixel digests pin
   them. Do not add a second implementation in a script.
2. **Schema first.** A new model or modality must go through
   `inference/schema.py`; parse-failure rates are reported, never dropped.
3. **No hand-edited results.** Tables and figures are regenerated from frozen
   CSVs by scripts; changing a data point by hand is a correctness bug.
4. **Deterministic writes.** Long runs use `DurableJsonl` (append, fsync, atomic
   offset sidecar) and resume from the sidecar; do not write prediction files
   with `open(..., "a")` directly.
5. **Run the fast checks before opening a pull request.** If you touched the
   audit CLI, also run `python -m pip check` inside `.venv-audit` and
   `python -m compileall -q audit/cxr_audit`.

## Commit messages

Use `type(scope): summary` with `feat`, `fix`, `docs`, `data`, `test`, `chore`
or `paper`. Keep the body focused on *why*; the diff already shows *what*. Do not
commit generated archives (`release/*.zip`), virtualenvs, caches, `tmp_*` files
or anything matched by `.gitignore`.

## Release checklist (maintainers)

1. Rebuild the manuscripts and confirm zero overfull boxes and zero undefined
   references.
2. Run `python program/scripts/104_table_consistency_check.py`,
   `93_artifact_audit.py` and `117_paper_audit.py`.
3. Run the audit CLI tests and `cxr-audit demo`.
4. Rebuild `release/cxr_trust_artifacts_v9.zip` with
   `python program/scripts/make_release_bundle.py`; upload the zip to Zenodo.
5. Put the new DOI in `README.md`, `release/README.md` and the manuscript's
   code-and-data-availability paragraph, then recompile.
6. Tag the commit (`v9`) and push `main` and the tag.

## Citation

Cite the manuscript, not this file. `CITATION.cff` at the repository root is the
authoritative record; the Zenodo DOI is added to it after each archive release.
