# ======================================================================
# CC-Framework • Makefile (Tier A)
# Goals: one-command reproducibility, quality gates, and clean ergonomics
# ======================================================================

# -------- Config ----------
SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

PY        ?= python3
PIP       ?= pip
VENV_DIR  ?= .venv
ACT       := source $(VENV_DIR)/bin/activate

PKG_NAME  := cc
SRC_DIR   := src/cc
COV_MIN   ?= 80

# Experiments (Tier A only)
SESS_SMOKE ?= 200
SESS_MVP   ?= 5000

CONFIG_SMOKE ?= experiments/configs/toy.yaml
CONFIG_MVP   ?= experiments/configs/two_world.yaml

# -------- Phony ----------
.PHONY: help install setup lock deps fmt lint type test test-unit test-int cov bench \
        reproduce-smoke reproduce-mvp figures reports docs docs-serve \
        verify-invariants verify-statistics verify-audit \
        docker-build docker-run clean distclean \
        carto-install carto-smoke carto-mvp carto-verify-audit carto-verify-stats carto-suggest

# -------- Help -----------
help:
	@echo ""
	@echo "CC-Framework • Make targets"
	@echo "-------------------------------------------------------------"
	@echo "setup                Create venv, install deps, install pre-commit"
	@echo "install              Install package (editable) + [dev,docs,notebooks] extras"
	@echo "lock                 Freeze runtime deps to requirements.lock.txt"
	@echo "fmt                  Auto-format (ruff --fix, isort, black)"
	@echo "lint                 Ruff/Isort/Black checks (no changes)"
	@echo "type                 mypy type-checking"
	@echo "test                 Unit+integration tests with coverage (>= $(COV_MIN)%)"
	@echo "reproduce-smoke      $(SESS_SMOKE) sessions quick run (CPU)"
	@echo "reproduce-mvp        $(SESS_MVP) sessions main Tier A run (CPU)"
	@echo "figures              Generate plots (paper/figures, results/artifacts)"
	@echo "reports              Build evaluation reports (evaluation/reports)"
	@echo "docs                 Build MkDocs site to site/"
	@echo "docs-serve           Local preview at http://127.0.0.1:8000"
	@echo "verify-*             Stats, invariants, audit-chain checks"
	@echo "docker-build/run     CPU docker image for fully pinned env"
	@echo "carto-*              Cartographer agent CLI entrypoints"
	@echo "clean/distclean      Clean artifacts / wipe venv, locks"
	@echo "-------------------------------------------------------------"
	@echo ""

# -------- Environment ----
$(VENV_DIR)/bin/activate:
	$(PY) -m venv $(VENV_DIR)
	$(ACT); $(PIP) install --upgrade pip wheel setuptools

install: $(VENV_DIR)/bin/activate
	$(ACT); $(PIP) install -e .[dev,docs,notebooks]

setup: install
	$(ACT); pre-commit install

lock: install
	$(ACT); python -c "import pkgutil, sys; print('Python:', sys.version)"
	$(ACT); pip freeze --all | sed '/@ file:\/\//d' > requirements.lock.txt
	@echo "Wrote requirements.lock.txt"

deps: install ## alias
	@true

# -------- Code Quality ----
fmt: install
	$(ACT); ruff check --fix .
	$(ACT); isort .
	$(ACT); black .

lint: install
	$(ACT); ruff check .
	$(ACT); isort --check-only .
	$(ACT); black --check .

type: install
	$(ACT); mypy $(SRC_DIR)

# -------- Tests -----------
test: install
	$(ACT); pytest -q --maxfail=1 --disable-warnings \
	  --cov=$(SRC_DIR) --cov-report=term-missing --cov-fail-under=$(COV_MIN)

test-unit: install
	$(ACT); pytest tests/unit -q --disable-warnings

test-int: install
	$(ACT); pytest tests/integration -q --disable-warnings

cov: test ## alias
	@true

bench: install
	$(ACT); pytest benchmarks -q || true

# -------- Reproduce Runs (Tier A) -------
reproduce-smoke: install
	$(ACT); python -m experiments.run --config $(CONFIG_SMOKE) --n $(SESS_SMOKE) \
	  --results-dir results --seed 1337

reproduce-mvp: install
	$(ACT); python -m experiments.run --config $(CONFIG_MVP) --n $(SESS_MVP) \
	  --results-dir results --seed 1337

# -------- Analysis / Figures / Reports ---
figures: install
	$(ACT); python scripts/plot_cc_curves.py
	$(ACT); python scripts/summarize_results.py

reports: install
	$(ACT); python -c "from cc.analysis.reporting import build_all; build_all()"

# -------- Docs --------------
docs: install
	$(ACT); mkdocs build --strict

docs-serve: install
	$(ACT); mkdocs serve -a 127.0.0.1:8000

# -------- Verifications -----
verify-invariants: install
	$(ACT); python -c "from cc.utils.validation import run_invariant_suite; run_invariant_suite()"

verify-statistics: install
	$(ACT); python -c "from cc.analysis.cc_estimation import self_check; self_check()"

verify-audit: install
	$(ACT); python -c "from cc.io.storage import verify_hash_chain; verify_hash_chain('results/raw')"

# -------- Docker (CPU) -----
docker-build:
	docker build -f deployment/docker/Dockerfile -t cc-framework:cpu .

docker-run:
	docker run --rm -it \
	  -v $$PWD:/workspace -w /workspace \
	  cc-framework:cpu bash

# -------- Clean -------------
clean:
	rm -rf .pytest_cache .coverage htmlcov site dist build \
	  results/aggregates results/artifacts \
	  paper/figures paper/tables \
	  benchmarks/results || true
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf $(VENV_DIR) requirements.lock.txt

# -------- Cartographer Agent (CLI) ----------
carto-install:
	python -m pip install -U pip && pip install -e .[dev]

carto-smoke: carto-install
	python -m cc.cartographer.cli run \
		--config experiments/configs/smoke.yaml \
		--samples 200 \
		--fig figs/phase_smoke.png \
		--audit runs/audit.jsonl

carto-mvp: carto-install
	python -m cc.cartographer.cli run \
		--config experiments/configs/mvp.yaml \
		--samples 5000 \
		--fig figs/phase_e2_T10.png \
		--audit runs/audit.jsonl

carto-verify-audit: carto-install
	python -m cc.cartographer.cli verify-audit --audit runs/audit.jsonl

carto-verify-stats: carto-install
	python -m cc.cartographer.cli verify-stats \
		--config experiments/configs/mvp.yaml \
		--bootstrap 10000

carto-suggest: carto-install
	python -m cc.cartographer.cli suggest \
		--history runs/audit.jsonl \
		--out experiments/grids/next.json
