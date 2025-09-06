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

CONFIG_SMOKE ?= experiments/configs/smoke.yaml
CONFIG_MVP   ?= experiments/configs/mvp.yaml

# Output paths expected by tests
FIG_DIR    := paper/figures
SMOKE_CSV  := results/smoke/aggregates/summary.csv
AUDIT_LOG  := runs/audit.jsonl

# -------- Phony ----------
.PHONY: help install setup lock deps fmt lint type test test-unit test-int cov bench \
        reproduce-smoke reproduce-mvp reproduce-figures figures reports docs docs-serve \
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
	@echo "reproduce-smoke      $(SESS_SMOKE) sessions quick run (CPU) + CSV + 3 figs"
	@echo "reproduce-mvp        $(SESS_MVP) sessions main Tier A run (CPU)"
	@echo "reproduce-figures    Rebuild smoke CSV + 3 figs from audit history"
	@echo "figures              Project plots (paper/figures, results/artifacts)"
	@echo "reports              Build evaluation reports (evaluation/reports)"
	@echo "docs                 Build MkDocs site to site/"
	@echo "docs-serve           Local preview at http://127.0.0.1:8000"
	@echo "verify-*             Stats and audit-chain checks"
	@echo "docker-build/run     CPU docker image for fully pinned env"
	@echo "carto-*              Cartographer CLI entrypoints (aliases)"
	@echo "clean/distclean      Clean artifacts / wipe venv, locks"
	@echo "-------------------------------------------------------------"
	@echo ""

# -------- Environment ----
$(VENV_DIR)/bin/activate:
	$(PY) -m venv $(VENV_DIR)
	$(ACT); $(PIP) install --upgrade pip wheel setuptools

install: $(VENV_DIR)/bin/activate
	$(ACT); $(PIP) install -e .[dev,docs,notebooks] || $(PIP) install -e .[dev]

setup: install
	$(ACT); pre-commit install || true

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
# Runs the Cartographer CLI once (FH upper bound path), then emits the CSV and 3 figures
reproduce-smoke: install
	mkdir -p paper/figures results/smoke/aggregates runs
	$(ACT); python -m cc.cartographer.cli run \
		--config experiments/configs/smoke.yaml \
		--samples $(SESS_SMOKE) \
		--fig paper/figures/phase_diagram.pdf \
		--audit runs/audit.jsonl
	$(ACT); python -m cc.analysis.generate_figures \
		--history runs/audit.jsonl \
		--fig-dir paper/figures \
		--out-dir results/smoke/aggregates
	# compatibility for legacy tests:
	mkdir -p results/aggregates
	cp results/smoke/aggregates/summary.csv results/aggregates/summary.csv

# Larger local run (kept as alias; customize as needed)
reproduce-mvp: install
	mkdir -p $(FIG_DIR) results/mvp/aggregates runs
	$(ACT); python -m cc.cartographer.cli run \
		--config $(CONFIG_MVP) \
		--samples $(SESS_MVP) \
		--fig $(FIG_DIR)/phase_mvp.pdf \
		--audit $(AUDIT_LOG)
	$(ACT); python -m cc.analysis.generate_figures \
		--history $(AUDIT_LOG) \
		--fig-dir $(FIG_DIR) \
		--out-dir results/mvp/aggregates

# Rebuild smoke figures + CSV only (no new run)
reproduce-figures: install
	mkdir -p $(FIG_DIR) results/smoke/aggregates
	$(ACT); python -m cc.analysis.generate_figures \
		--history $(AUDIT_LOG) \
		--fig-dir $(FIG_DIR) \
		--out-dir results/smoke/aggregates

# -------- Analysis / Figures / Reports ---
figures: reproduce-figures

reports: install
	$(ACT); python -c "from cc.analysis.reporting import build_all; build_all()"

# -------- Docs --------------
docs: install
	$(ACT); mkdocs build --strict

docs-serve: install
	$(ACT); mkdocs serve -a 127.0.0.1:8000

# -------- Verifications -----
verify-statistics: install
	$(ACT); python -m cc.cartographer.cli verify-stats \
		--config $(CONFIG_SMOKE) --bootstrap 2000

verify-audit: install
	$(ACT); python -m cc.cartographer.cli verify-audit --audit $(AUDIT_LOG)

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
	  $(FIG_DIR) paper/tables \
	  benchmarks/results || true
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf $(VENV_DIR) requirements.lock.txt

# -------- Cartographer Agent (CLI) ----------
carto-install:
	$(PY) -m pip install -U pip && pip install -e .[dev]

carto-smoke: install
	$(ACT); python -m cc.cartographer.cli run \
		--config $(CONFIG_SMOKE) \
		--samples $(SESS_SMOKE) \
		--fig figs/phase_smoke.png \
		--audit $(AUDIT_LOG)

carto-mvp: install
	$(ACT); python -m cc.cartographer.cli run \
		--config $(CONFIG_MVP) \
		--samples $(SESS_MVP) \
		--fig figs/phase_e2_T10.png \
		--audit $(AUDIT_LOG)

carto-verify-audit: verify-audit
carto-verify-stats: verify-statistics

carto-suggest: install
	$(ACT); python -m cc.cartographer.suggest \
		--history $(AUDIT_LOG) \
		--out experiments/grids/next.json