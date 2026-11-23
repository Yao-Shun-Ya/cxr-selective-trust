.PHONY: help dirs lint test test-cxr test-resume audit manifest-lock \
	audit-install audit-cli-test audit-demo arxiv-packages

PY ?= python
PROGRAM_PY ?= program/.venv-vl/Scripts/python.exe
AUDIT_PY ?= .venv-audit/Scripts/python.exe

help:
	@echo "Targets: dirs lint test test-cxr test-resume audit manifest-lock"
	@echo "         audit-install audit-cli-test audit-demo"
	@echo "  test          legacy parser tests (repository root)"
	@echo "  test-cxr      authoritative cxr_trust tests (run from program/)"
	@echo "  test-resume   CPU-only crash/resume test with the stubbed model"
	@echo "  audit         regenerate the coverage census and run accounting"
	@echo "  manifest-lock write nih_eval_1506.csv and MANIFEST_SHA256.json"
	@echo "  audit-install install the CPU-only cxr-audit CLI into .venv-audit"
	@echo "  audit-cli-test run the audit CLI unit tests (4 tests, no data)"
	@echo "  audit-demo    run the synthetic end-to-end demo (no data, no network)"
	@echo "  arxiv-packages build release/arxiv/{main_v9,bench_v3}_arxiv.zip"

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

audit-install:
	$(AUDIT_PY) -m pip install -e "audit[dev]"

audit-cli-test:
	$(AUDIT_PY) -m pytest -q audit/tests

audit-demo:
	$(AUDIT_PY) -m cxr_audit demo

arxiv-packages:
	$(PY) program/scripts/make_arxiv_packages.py
