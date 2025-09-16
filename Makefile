# ======================================================================
# CC-Framework • Makefile (Tier A)
# Goals: one-command reproducibility, quality gates, and clean ergonomics
# ======================================================================

# -------- Config ----------
SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PY        ?= python3
PIP       ?= pip
VENV_DIR  ?= .venv
ACT       := source $(VENV_DIR)/bin/activate
EXTRAS    := '.[dev,docs,notebooks]'

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

# -------- Week-3 fixed params (single θ) ----------
W3_D        ?= 0.55
W3_TPR_A    ?= 0.72
W3_TPR_B    ?= 0.65
W3_FPR_A    ?= 0.035
W3_FPR_B    ?= 0.050
W3_ALPHA    ?= 0.05        # hard cap; do not auto-relax
W3_DELTA    ?= 0.05        # 95% CI
W3_TARGET_T ?= 0.10        # planner half-width
W3_N1       ?= 200
W3_N0       ?= 200
W3_K1       ?= 124
W3_K0       ?= 18
W3_JSON     ?= runs/week3_methods.json
W3_FIG      ?= figs/fig_week3_roc_fh.png

# -------- Phony ----------
.PHONY: help dev install setup lock deps fmt lint type test test-week3 test-unit test-int cov bench \
        reproduce-smoke reproduce-mvp reproduce-figures figures reports docs docs-serve \
        verify-invariants verify-statistics verify-audit \
        docker-build docker-run clean distclean \
        carto-install carto-smoke carto-mvp carto-verify-audit carto-verify-stats carto-suggest \
        week3 week3-95 week3-90 week3-fig

# -------- Help -----------
help:
	@echo ""
	@echo "CC-Framework • Make targets"
	@echo "-------------------------------------------------------------"
	@echo "dev / install       Create venv, install package + extras (zsh-safe)"
	@echo "setup               Install + pre-commit hook (optional)"
	@echo "lock                Freeze deps -> requirements.lock.txt"
	@echo "fmt                 Auto-format (ruff --fix, isort, black)"
	@echo "lint                Ruff/Isort/Black checks (no changes)"
	@echo "type                mypy type-checking"
	@echo "test                Unit+integration + coverage >= $(COV_MIN)% (override: COV_MIN=20)"
	@echo "test-week3          Run Week-3 unit tests only"
	@echo "reproduce-smoke     $(SESS_SMOKE) sessions quick run + CSV + figs"
	@echo "reproduce-mvp       $(SESS_MVP) sessions main run"
	@echo "reproduce-figures   Rebuild smoke CSV + 3 figs from audit history"
	@echo "week3               FH–Bernstein+Wilson CIs at single θ (α=$(W3_ALPHA), δ=$(W3_DELTA))"
	@echo "week3-95 / week3-90 Variants for δ"
	@echo "docs / docs-serve   Build or serve MkDocs docs"
	@echo "docker-build/run    CPU docker image"
	@echo "clean/distclean     Clean artifacts / wipe venv & locks"
	@echo "-------------------------------------------------------------"
	@echo ""

# -------- Environment ----
$(VENV_DIR)/bin/activate:
	$(PY) -m venv $(VENV_DIR)
	$(ACT); $(PIP) install --upgrade pip wheel setuptools

dev: install

install: $(VENV_DIR)/bin/activate
	$(ACT); $(PIP) install -e $(EXTRAS) || $(PIP) install -e '.[dev]'

setup: install
	$(ACT); pre-commit install || true

lock: install
	$(ACT); python -c "import sys; print('Python:', sys.version)"
	$(ACT); $(PIP) freeze --all | sed '/@ file:\/\//d' > requirements.lock.txt
	@echo "Wrote requirements.lock.txt"

deps: install
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

# Week-3 focused tests (subset)
test-week3: install
	$(ACT); pytest -q --disable-warnings \
	  tests/unit/test_fh_intervals_alpha_cap.py \
	  tests/unit/test_variance_envelope.py \
	  tests/unit/test_bernstein_monotonicity.py \
	  tests/unit/test_methods_cli_smoke.py

cov: test
	@true

bench: install
	$(ACT); pytest benchmarks -q || true

# -------- Week-3 (single θ) -------
week3: install
	mkdir -p $(dir $(W3_JSON)) $(dir $(W3_FIG))
	$(ACT); python -m cc.cartographer.cli methods \
	  --D $(W3_D) \
	  --tpr-a $(W3_TPR_A) --tpr-b $(W3_TPR_B) --fpr-a $(W3_FPR_A) --fpr-b $(W3_FPR_B) \
	  --n1 $(W3_N1) --n0 $(W3_N0) \
	  --k1 $(W3_K1) --k0 $(W3_K0) \
	  --alpha-cap $(W3_ALPHA) --delta $(W3_DELTA) \
	  --target-t $(W3_TARGET_T) \
	  --figure-out $(W3_FIG) \
	  --json-out $(W3_JSON)
	@echo "✓ Week-3 JSON → $(W3_JSON)"
	@echo "✓ Week-3 FIG  → $(W3_FIG)"

week3-95: install
	$(MAKE) week3 W3_DELTA=0.05

week3-90: install
	$(MAKE) week3 W3_DELTA=0.10

week3-fig: install
	mkdir -p $(dir $(W3_FIG))
	$(ACT); python -m cc.cartographer.cli methods \
	  --D $(W3_D) \
	  --tpr-a $(W3_TPR_A) --tpr-b $(W3_TPR_B) --fpr-a $(W3_FPR_A) --fpr-b $(W3_FPR_B) \
	  --n1 $(W3_N1) --n0 $(W3_N0) \
	  --k1 $(W3_K1) --k0 $(W3_K0) \
	  --alpha-cap $(W3_ALPHA) --delta $(W3_DELTA) \
	  --target-t $(W3_TARGET_T) \
	  --figure-out $(W3_FIG)
	
week3-power: install
	$(ACT); python scripts/make_power_curve.py \
	  --I1 0.37,0.65 --I0 0.05,0.05 --D 0.55 --delta 0.05 \
	  --target-t 0.10 \
	  --out paper/figures/fig_week3_power_curve.png
	  
# -------- Reproduce Runs (Tier A) -------
reproduce-smoke: install
	mkdir -p paper/figures results/smoke/aggregates runs
	$(ACT); python -m cc.cartographer.cli run \
		--config $(CONFIG_SMOKE) \
		--samples $(SESS_SMOKE) \
		--fig paper/figures/phase_diagram.pdf \
		--audit runs/audit.jsonl
	$(ACT); python -m cc.analysis.generate_figures \
		--history runs/audit.jsonl \
		--fig-dir paper/figures \
		--out-dir results/smoke/aggregates
	mkdir -p results/aggregates
	cp results/smoke/aggregates/summary.csv results/aggregates/summary.csv

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

reproduce-figures: install
	mkdir -p $(FIG_DIR) results/smoke/aggregates
	$(ACT); python -m cc.analysis.generate_figures \
		--history $(AUDIT_LOG) \
		--fig-dir $(FIG_DIR) \
		--out-dir results/smoke/aggregates

# -------- Analysis / Figures / Reports ---
figures: reproduce-figures

reports: install
	$(ACT); python -m cc.cartographer.cli build-reports --mode all

ccc: install
	$(ACT); python scripts/generate_ccc.py

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
	$(PY) -m pip install -U pip && pip install -e '.[dev]'

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
