.PHONY: help dirs lint test test-cxr test-resume audit manifest-lock

PY ?= python
PROGRAM_PY ?= program/.venv-vl/Scripts/python.exe

help:
	@echo "Targets: dirs lint test test-cxr test-resume audit manifest-lock"
	@echo "  test          legacy parser tests (repository root)"
	@echo "  test-cxr      authoritative cxr_trust tests (run from program/)"
	@echo "  test-resume   CPU-only crash/resume test with the stubbed model"
	@echo "  audit         regenerate the coverage census and run accounting"
	@echo "  manifest-lock write nih_eval_1506.csv and MANIFEST_SHA256.json"

dirs:
	@mkdir -p artifacts logs figures tables

lint:
	ruff check program/src program/scripts

test:
	$(PY) -m pytest -q tests

test-cxr:
	cd program && $(PROGRAM_PY) -m pytest -q tests

test-resume:
	cd program && $(PROGRAM_PY) scripts/41_test_runner_resume.py

audit:
	cd program && $(PROGRAM_PY) scripts/90_coverage_census.py

manifest-lock:
	cd program && $(PROGRAM_PY) scripts/91_manifest_lock.py
