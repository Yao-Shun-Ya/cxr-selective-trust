.PHONY: help dirs lint test

help:
	@echo "Targets: dirs lint test"

dirs:
	@mkdir -p artifacts logs figures tables

lint:
	ruff check src scripts

test:
	pytest -q
