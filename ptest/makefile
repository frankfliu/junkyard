SHELL := /bin/bash
PACKAGE_SLUG=ptest
PYTHON := python

.PHONY: install
install:
	$(PYTHON) -m pip install -e .[dev]

.PHONY: pre-commit
pre-commit:
	$(PYTHON) -m pip install pre-commit
	pre-commit install

#
# Formatting
#
.PHONY: fix
fix: ruff_fixes black_fixes dapperdata_fixes tomlsort_fixes

.PHONY: ruff_fixes
ruff_fixes:
	$(PYTHON) -m ruff check . --fix

.PHONY: black_fixes
black_fixes:
	$(PYTHON) -m ruff format .

.PHONY: dapperdata_fixes
dapperdata_fixes:
	$(PYTHON) -m dapperdata.cli pretty . --no-dry-run

.PHONY: tomlsort_fixes
tomlsort_fixes:
	$(PYTHON_ENV) toml-sort $$(find . -not -path "./.pytest_cache/*" -name "*.toml") -i

#
# Testing
#
.PHONY: tests
tests: pytest ruff_check black_check mypy_check dapperdata_check tomlsort_check

.PHONY: pytest
pytest:
	$(PYTHON) -m pytest --cov=./${PACKAGE_SLUG} --cov-report=term-missing tests

.PHONY: pytest_loud
pytest_loud:
	$(PYTHON) -m pytest --log-cli-level=DEBUG -log_cli=true --cov=./${PACKAGE_SLUG} --cov-report=term-missing tests

.PHONY: ruff_check
ruff_check:
	$(PYTHON) -m ruff check

.PHONY: black_check
black_check:
	$(PYTHON) -m ruff format . --check

.PHONY: mypy_check
mypy_check:
	$(PYTHON) -m mypy ${PACKAGE_SLUG}

.PHONY: dapperdata_check
dapperdata_check:
	$(PYTHON) -m dapperdata.cli pretty .

.PHONY: tomlsort_check
tomlsort_check:
	$(PYTHON_ENV) toml-sort $$(find . -not -path "./.pytest_cache/*" -name "*.toml") --check

#
# Dependencies
#
.PHONY: rebuild_dependencies
rebuild_dependencies:
	$(PYTHON) -m uv pip compile --upgrade --output-file=requirements.txt pyproject.toml
	$(PYTHON) -m uv pip compile --upgrade --output-file=requirements-dev.txt --extra=dev pyproject.toml

.PHONY: dependencies
dependencies: requirements.txt requirements-dev.txt

requirements.txt: pyproject.toml
	$(PYTHON) -m uv pip compile --upgrade --output-file=requirements.txt pyproject.toml

requirements-dev.txt: pyproject.toml
	$(PYTHON) -m uv pip compile --upgrade --output-file=requirements-dev.txt --extra=dev pyproject.toml

#
# Packaging
#
.PHONY: build
build:
	$(PYTHON) -m build
