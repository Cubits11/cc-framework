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

# -------- Week-6 (α-cap ablation) ----------
W6_DIR      := results/week6
W6_AUDIT    := runs/audit_week6.jsonl
W6_FIG_DIR  := figures/week6
W6_RAILS    := keyword regex semantic and or

# -------- Phony ----------
.PHONY: help dev install setup lock deps fmt lint type test test-week3 test-unit test-int cov bench \
        reproduce-smoke reproduce-mvp reproduce-figures figures reports docs docs-serve \
        verify-invariants verify-statistics verify-audit \
        docker-build docker-run clean distclean \
        carto-install carto-smoke carto-mvp carto-verify-audit carto-verify-stats carto-suggest \
        week3 week3-95 week3-90 week3-fig week3-power \
        week5-pilot memo-week5 \
        $(addprefix week6-rail-,$(W6_RAILS)) week6-ablation memo-week6 week6-utility test-week6
.PHONY: init demo demo-rails
demo: demo-rails

init: $(VENV_DIR)/bin/activate
	$(ACT); $(PIP) install -r requirements.txt

demo-rails: init
	$(ACT); python scripts/rails_compare.py \
		--csv datasets/examples/rails_tiny.csv \
		--out results/baselines/rails_summary.csv \
		--fpr-min 0.04 --fpr-max 0.06

# -------- Help -----------
help:
	@echo ""
	@echo "CC-Framework • Make targets"
	@echo "-------------------------------------------------------------"
	@echo "dev / install       Create venv, install package + extras (zsh-safe)"
	@echo "setup               Install + pre-commit hook (optional)"
	@echo "init                Create venv + install lightweight demo requirements"
	@echo "demo                Run lightweight rails baseline demo"
	@echo "lock                Freeze deps -> requirements.lock.txt"
	@echo "fmt / lint / type   Code quality: ruff/isort/black/mypy"
	@echo "test                Unit+integration + coverage >= $(COV_MIN)%"
	@echo "test-week3          Run Week-3 unit tests only"
	@echo "reproduce-smoke     $(SESS_SMOKE) sessions quick run + CSV + figs"
	@echo "reproduce-mvp       $(SESS_MVP) sessions main run"
	@echo "reproduce-figures   Rebuild smoke CSV + 3 figs from audit history"
	@echo "week3               FH–Bernstein+Wilson CIs at single θ (α=$(W3_ALPHA), δ=$(W3_DELTA))"
	@echo "week3-95 / week3-90 Variants for δ"
	@echo "week5-pilot         Calibrate + run Week-5 pilot and figures"
	@echo "week6-ablation      Calibrate/write-back + run + figures for 5 rails (fail-fast)"
	@echo "memo-week6          Build Week-6 memo (and PDF if pandoc available)"
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

# -------- Week 5 pipeline ---------
week5-pilot: install
	mkdir -p results/week5_scan figures runs
	$(ACT); python scripts/calibrate_guardrail.py \
	        --config experiments/configs/week5_pilot.yaml \
	        --dataset datasets/benign \
	        --summary results/week5_scan/calibration_summary.json \
	        --audit runs/audit_week5.jsonl
	$(ACT); python -m cc.exp.run_two_world \
	        --config experiments/configs/week5_pilot.yaml \
	        --out-csv results/week5_scan/scan.csv \
	        --output results/week5_scan/analysis.json \
	        --audit runs/audit_week5.jsonl \
	        --calibration-summary results/week5_scan/calibration_summary.json
	$(ACT); python scripts/make_week5_figs.py \
	        --scan results/week5_scan/scan.csv \
	        --out-ci figures/week5_ci_comparison.png \
	        --out-roc figures/week5_roc_slice.png

memo-week5: install
	$(ACT); pandoc docs/week5_memo.md -o docs/week5_memo.pdf

# -------- Week-6 (α-cap ablation, fail-fast) ---------

# Helper: assert file exists
define _assert_file
	@if [ ! -f "$(1)" ]; then echo "FAIL: Missing artifact $(1)"; exit 1; fi
endef

# Helper: assert at least one PNG exists in dir
define _assert_any_png
	@if ! compgen -G "$(1)/*.png" > /dev/null; then echo "FAIL: No figures in $(1)"; exit 1; fi
endef

# Per-rail target with proper variable expansion
$(addprefix week6-rail-,$(W6_RAILS)): install
	@rail=$(@:week6-rail-%=%); \
	cfg="experiments/configs/week6_$${rail}.yaml"; \
	outdir="$(W6_DIR)/$${rail}"; \
	figdir="$(W6_FIG_DIR)/$${rail}"; \
	mkdir -p "$${outdir}" "$${figdir}" "$(dir $(W6_AUDIT))"; \
	echo "==> [Calibrate] $${rail}"; \
	$(ACT); python scripts/calibrate_guardrail.py \
		--config "$${cfg}" \
		--dataset datasets/benign \
		--summary "$${outdir}/calibration_summary.json" \
		--audit "$(W6_AUDIT)" \
		--write-config-out "$${outdir}/calibrated.yaml"; \
	echo "==> [Run+Checks] $${rail}"; \
	$(ACT); python scripts/run_with_checks.py \
		--config "$${outdir}/calibrated.yaml" \
		--out-json "$${outdir}/analysis.json" \
		--audit "$(W6_AUDIT)" \
		--seed 123 \
		----fpr-lo 0.00 --fpr-hi 0.08 \
		--calibration "$${outdir}/calibration_summary.json"; \
	echo "==> [Figures] $${rail}"; \
	$(ACT); python scripts/make_week6_figs.py \
		--inputs "$${outdir}/analysis.json" \
		--out-dir "$${figdir}"; \
	if [ ! -f "$${outdir}/analysis.json" ]; then echo "FAIL: Missing artifact $${outdir}/analysis.json"; exit 1; fi; \
	if [ ! -f "$${outdir}/calibration_summary.json" ]; then echo "FAIL: Missing artifact $${outdir}/calibration_summary.json"; exit 1; fi; \
	if ! compgen -G "$${figdir}/*.png" > /dev/null; then echo "FAIL: No figures in $${figdir}"; exit 1; fi; \
	tail -n 1 "$(W6_AUDIT)" >/dev/null || { echo "FAIL: audit file missing or empty"; exit 1; }; \
	echo "OK: week6-rail-$${rail}"

# Umbrella target: run all five rails
week6-ablation: $(addprefix week6-rail-,$(W6_RAILS))
	@echo "OK: week6-ablation complete."

# Optional: utility summaries per rail
# Usage: make week6-utility RAIL=keyword
week6-utility: install
	@if [ -z "$(RAIL)" ]; then echo "Usage: make week6-utility RAIL=keyword"; exit 2; fi
	$(ACT); python scripts/summarize_results.py --utility \
	  --final-results $(W6_DIR)/$(RAIL)/final_results.json \
	  --out-csv $(W6_DIR)/$(RAIL)/utility_summary.csv \
	  --out-fig $(W6_FIG_DIR)/$(RAIL)/utility_hist.png

# Memo (MD always; PDF if pandoc is installed)
memo-week6: install
	@echo "Memo ready at docs/week6_memo.md"
	@if command -v pandoc >/dev/null 2>&1; then \
		$(ACT); pandoc docs/week6_memo.md -o docs/week6_memo.pdf; \
		echo "Memo PDF → docs/week6_memo.pdf"; \
	else \
		echo "pandoc not found; skipping PDF build."; \
	fi

# Tests for Week-6 integration
test-week6: install
	$(ACT); pytest tests/integration/test_week6_writeback_and_alpha.py -q

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
