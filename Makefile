# ======================================================================
# CC-Framework • Makefile
# Goals: one-command reproducibility, quality gates, and claim governance proof gates
# ======================================================================

# ----------------------------------------------------------------------
# Shell
# ----------------------------------------------------------------------

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

PY        ?= python3
PIP       ?= python -m pip
VENV_DIR  ?= .venv
ACT       := source $(VENV_DIR)/bin/activate
EXTRAS    ?= .[dev,docs,notebooks]

PKG_NAME  := cc
SRC_DIR   := src/cc
COV_MIN   ?= 80

TYPE_TARGETS ?= \
	src/cc/adapters/base.py \
	src/cc/cartographer/audit.py \
	src/cc/cartographer/bounds.py \
	src/cc/cartographer/intervals.py \
	src/cc/io/storage.py \
	src/cc/kernel/sensitivity.py \
	src/cc/utils/artifacts.py \
	src/cc/utils/timing.py

# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------

SESS_SMOKE ?= 200
SESS_MVP   ?= 5000

CONFIG_SMOKE ?= experiments/configs/smoke.yaml
CONFIG_MVP   ?= experiments/configs/mvp.yaml

FIG_DIR    := paper/figures
SMOKE_CSV  := results/smoke/aggregates/summary.csv
AUDIT_LOG  := runs/audit.jsonl
PAPER_ARTIFACT_DIR := artifacts/paper

# ----------------------------------------------------------------------
# Week-3 fixed params
# ----------------------------------------------------------------------

W3_D        ?= 0.55
W3_TPR_A    ?= 0.72
W3_TPR_B    ?= 0.65
W3_FPR_A    ?= 0.035
W3_FPR_B    ?= 0.050
W3_ALPHA    ?= 0.05
W3_DELTA    ?= 0.05
W3_TARGET_T ?= 0.10
W3_N1       ?= 200
W3_N0       ?= 200
W3_K1       ?= 124
W3_K0       ?= 18
W3_JSON     ?= runs/week3_methods.json
W3_FIG      ?= figs/fig_week3_roc_fh.png

# ----------------------------------------------------------------------
# Week-6
# ----------------------------------------------------------------------

W6_DIR      := results/week6
W6_AUDIT    := runs/audit_week6.jsonl
W6_FIG_DIR  := figures/week6
W6_RAILS    := keyword regex semantic and or

# ----------------------------------------------------------------------
# Enterprise Reference
# ----------------------------------------------------------------------

DASHBOARD_DIR := apps/dashboard
ENTERPRISE_TESTS := tests/integration/test_enterprise_aws_emulation.py tests/e2e/test_enterprise_smoke.py

# ----------------------------------------------------------------------
# Claim Governance Control Panel
# ----------------------------------------------------------------------

GOV_CAPSULE_DIR      := examples/claim_governance_capsule
GOV_CAPSULE_OUT      := $(GOV_CAPSULE_DIR)/outputs
GOV_CAPSULE_EXPECTED := $(GOV_CAPSULE_DIR)/expected

GOV_REPORT           := $(GOV_CAPSULE_OUT)/cc_report.json
GOV_AUDIT            := $(GOV_CAPSULE_OUT)/claim_governance_audit.json
GOV_ENVELOPE         := $(GOV_CAPSULE_OUT)/claim_envelope.json
GOV_MANIFEST         := $(GOV_CAPSULE_OUT)/capsule_manifest.json

GOV_NOW              ?= 2026-01-02T00:00:00Z

GOV_UNIT_TESTS       := tests/unit/evidence/test_claim_governance.py
GOV_CAPSULE_TESTS    := tests/integration/test_claim_governance_capsule.py

# ----------------------------------------------------------------------
# Phony Targets
# ----------------------------------------------------------------------

.PHONY: \
	help \
	dev install setup init lock deps \
	fmt lint type security package-build \
	test test-unit test-int test-kernel test-release test-reporting test-week3 test-week6 cov bench \
	conformance differential acceptance test-compose canon-probe evidence-cards \
	check-artifact-boundary check-repro-clean \
	enterprise-smoke \
	reproduce-smoke reproduce-mvp reproduce-figures figures reports ccc \
	reproduce-paper verify-paper-artifacts paper-smoke \
	verify-invariants verify-statistics verify-audit \
	docs docs-serve \
	docker-build docker-run \
	clean distclean \
	carto-install carto-smoke carto-mvp carto-verify-audit carto-verify-stats carto-suggest \
	week3 week3-95 week3-90 week3-fig week3-power \
	week5-pilot memo-week5 \
	$(addprefix week6-rail-,$(W6_RAILS)) week6-ablation memo-week6 week6-utility \
	demo demo-rails \
	governance-help \
	governance-install \
	governance-cli \
	governance-imports \
	governance-quick \
	governance-tests \
	governance-full-tests \
	governance-capsule \
	governance-capsule-update \
	governance-capsule-verify \
	governance-capsule-clean \
	governance-capsule-inspect \
	governance-capsule-tamper-demo \
	governance-verify-report \
	governance-docs \
	governance-diff \
	governance-proof \
	governance-proof-fast \
	governance-warnings \
	governance-foundation-audit \
	verify-governance

# ======================================================================
# Help
# ======================================================================

help:
	@echo ""
	@echo "CC-Framework • Make targets"
	@echo "-------------------------------------------------------------"
	@echo "dev / install       Create venv, install package + extras"
	@echo "setup               Install package + pre-commit hook"
	@echo "init                Create venv + install lightweight demo requirements"
	@echo "demo                Run lightweight rails baseline demo"
	@echo "lock                Freeze deps -> requirements.lock.txt"
	@echo "fmt / lint / type   Code quality: ruff/isort/black/mypy"
	@echo "security            Run Bandit, detect-secrets, and pip-audit"
	@echo "package-build       Build sdist/wheel and check package metadata"
	@echo "test                Unit+integration + coverage >= $(COV_MIN)%"
	@echo "test-kernel         Kernel-only unit tests, strict mypy, and focused ruff"
	@echo "test-release        Kernel lane + paper reproduction integration checks"
	@echo "test-reporting      CC report/receipt unit tests and CLI fixture smoke"
	@echo "test-week3          Run Week-3 unit tests only"
	@echo "test-week6          Run Week-6 integration tests only"
	@echo "enterprise-smoke    Install enterprise deps, prep dashboard, run strict AWS+dashboard smoke"
	@echo "check-artifact-boundary Verify tracked/generated artifact boundary"
	@echo "check-repro-clean   Run short repro lane and fail on new generated diffs"
	@echo "reproduce-paper     Build deterministic paper artifacts under $(PAPER_ARTIFACT_DIR)"
	@echo "verify-paper-artifacts Verify hashes, schemas, metrics, and LP witnesses"
	@echo "paper-smoke         Static Paper-1 source checks; compile if latexmk exists"
	@echo "reproduce-smoke     $(SESS_SMOKE) sessions quick run + CSV + figs"
	@echo "reproduce-mvp       $(SESS_MVP) sessions main run"
	@echo "reproduce-figures   Rebuild smoke CSV + figures from audit history"
	@echo "week3               FH-Bernstein+Wilson CIs at single theta"
	@echo "week3-95 / week3-90 Variants for delta"
	@echo "week5-pilot         Calibrate + run Week-5 pilot and figures"
	@echo "week6-ablation      Calibrate/write-back + run + figures for all rails"
	@echo "memo-week6          Build Week-6 memo PDF if pandoc exists"
	@echo "governance-help     Show Claim Governance target help"
	@echo "governance-proof    Full local governance proof gate"
	@echo "docs / docs-serve   Build or serve MkDocs docs"
	@echo "docker-build/run    CPU docker image"
	@echo "clean/distclean     Clean artifacts / wipe venv & locks"
	@echo "-------------------------------------------------------------"
	@echo ""

# ======================================================================
# Environment
# ======================================================================

$(VENV_DIR)/bin/activate:
	$(PY) -m venv $(VENV_DIR)
	$(ACT); $(PIP) install --upgrade pip wheel setuptools

dev: install

install: $(VENV_DIR)/bin/activate
	$(ACT); $(PIP) install -e "$(EXTRAS)" || $(PIP) install -e ".[dev]" || $(PIP) install -e .

setup: install
	$(ACT); pre-commit install || true

init: $(VENV_DIR)/bin/activate
	@if [ -f requirements.txt ]; then \
		$(ACT); $(PIP) install -r requirements.txt; \
	else \
		$(ACT); $(PIP) install -e .; \
	fi

deps: install
	@true

lock: install
	$(ACT); python -c "import sys; print('Python:', sys.version)"
	$(ACT); $(PIP) freeze --all | sed '/@ file:\/\//d' > requirements.lock.txt
	@echo "Wrote requirements.lock.txt"

# ======================================================================
# Demo
# ======================================================================

demo: demo-rails

demo-rails: init
	mkdir -p results/baselines
	$(ACT); python scripts/rails_compare.py \
		--csv datasets/examples/rails_tiny.csv \
		--out results/baselines/rails_summary.csv \
		--fpr-min 0.04 \
		--fpr-max 0.06

# ======================================================================
# Code Quality
# ======================================================================

fmt: install
	$(ACT); ruff check --fix .
	$(ACT); isort .
	$(ACT); black .

lint: install
	$(ACT); ruff check .
	$(ACT); isort --check-only .
	$(ACT); black --check .

type: install
	$(ACT); mypy $(TYPE_TARGETS)

security: install
	$(ACT); $(PIP) install -e ".[security]" || $(PIP) install bandit pip-audit
	$(ACT); bandit -q -r $(SRC_DIR) -x src/cc/_legacy --severity-level medium
	$(ACT); detect-secrets scan \
		src scripts .github pyproject.toml Makefile README.md README_ENTERPRISE.md SECURITY.md docs \
		--exclude-files '(^|/)(artifacts/paper|docs/archive|docs/assets|site|node_modules|cdk\.out)(/|$$)|package-lock\.json$$' \
		> .detect-secrets.tmp.json
	$(ACT); python scripts/check_detect_secrets_results.py .detect-secrets.tmp.json
	rm -f .detect-secrets.tmp.json
	$(ACT); pip-audit --skip-editable

package-build: install
	$(ACT); $(PIP) install -e ".[dev]"
	$(ACT); python -m build
	$(ACT); python -m twine check dist/*

# ======================================================================
# Tests
# ======================================================================

test: install
	$(ACT); pytest -q --maxfail=1 --disable-warnings \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-fail-under=$(COV_MIN)

test-unit: install
	$(ACT); pytest tests/unit -q --disable-warnings

test-int: install
	$(ACT); pytest tests/integration -q --disable-warnings

test-kernel: install
	PYTHONPATH=src $(VENV_DIR)/bin/pytest tests/unit/kernel -q
	PYTHONPATH=src $(VENV_DIR)/bin/pytest tests/unit/kernel/test_sensitivity.py tests/unit/kernel/test_frechet_classes.py -q
	PYTHONPATH=src $(VENV_DIR)/bin/mypy src/cc/kernel --strict
	$(VENV_DIR)/bin/ruff check src/cc/kernel tests/unit/kernel

test-release: test-kernel
	PYTHONPATH=src $(VENV_DIR)/bin/python examples/minimal/run_bounds.py >/dev/null
	PYTHONPATH=src $(VENV_DIR)/bin/pytest \
		tests/integration/test_reproduce_paper.py \
		tests/integration/test_verify_paper_artifacts.py -q

test-reporting: install
	PYTHONPATH=src $(VENV_DIR)/bin/pytest tests/unit/reporting -q

# --- cross-implementation agreement (upgrade workstream W5) -----------------
# Agreement here establishes that two implementations of one specification
# compute the same values. It does NOT establish that either is correct: both
# were authored in this project and can share a misreading of the spec.

conformance: install
	PYTHONPATH=src $(VENV_DIR)/bin/python scripts/build_conformance_corpus.py --check
	node verifiers/node/cc_compose_verify.mjs
	PYTHONPATH=src $(VENV_DIR)/bin/pytest tests/conformance -q

differential: install
	PYTHONPATH=src $(VENV_DIR)/bin/python scripts/differential_compose.py --cases 4000 --seed 1

# The acceptance gate for consumability: an external consumer's PUBLISHED
# four-control result, reproduced from cc.compose. Not "the API exists" --
# their numbers, from this library.
acceptance: install
	PYTHONPATH=src $(VENV_DIR)/bin/pytest tests/acceptance -q

test-compose: install
	PYTHONPATH=src $(VENV_DIR)/bin/pytest tests/unit/compose tests/unit/cli -q
	$(VENV_DIR)/bin/ruff check src/cc/compose src/cc/cli/guard.py

# Reproduces the S1 canonicalization findings. Exits non-zero while any
# unintended-kernel class remains, so the findings stay falsifiable.
# Evidence cards for an external Atlas. The committed cards are the not-run
# scaffold: verdicts are host-specific and must not be published from a laptop.
evidence-cards: install
	PYTHONPATH=src $(VENV_DIR)/bin/python scripts/build_evidence_cards.py --check
	PYTHONPATH=src $(VENV_DIR)/bin/pytest tests/unit/evidence_card -q

canon-probe: install
	PYTHONPATH=src $(VENV_DIR)/bin/python scripts/canonicalization_probe.py
	PYTHONPATH=src $(VENV_DIR)/bin/pytest tests/unit/canonical -q

test-week3: install
	$(ACT); pytest -q --disable-warnings \
		tests/unit/test_fh_intervals_alpha_cap.py \
		tests/unit/test_variance_envelope.py \
		tests/unit/test_bernstein_monotonicity.py \
		tests/unit/test_methods_cli_smoke.py

test-week6: install
	$(ACT); pytest tests/integration/test_week6_writeback_and_alpha.py -q

cov: test
	@true

bench: install
	$(ACT); pytest benchmarks -q || true

# ======================================================================
# Enterprise Smoke
# ======================================================================

enterprise-smoke: $(VENV_DIR)/bin/activate
	$(ACT); $(PIP) install -e ".[enterprise,test]"
	@if [ -d "$(DASHBOARD_DIR)" ]; then \
		cd $(DASHBOARD_DIR) && npm ci; \
	else \
		echo "Skipping dashboard npm install; missing $(DASHBOARD_DIR)"; \
	fi
	$(ACT); python -c "import boto3, moto; print('enterprise Python dependencies available')"
	CC_ENTERPRISE_STRICT=1 PYTHONPATH=src $(VENV_DIR)/bin/pytest $(ENTERPRISE_TESTS) -q

# ======================================================================
# Week 3
# ======================================================================

week3: install
	mkdir -p $(dir $(W3_JSON)) $(dir $(W3_FIG))
	$(ACT); python -m cc.cartographer.cli methods \
		--D $(W3_D) \
		--tpr-a $(W3_TPR_A) \
		--tpr-b $(W3_TPR_B) \
		--fpr-a $(W3_FPR_A) \
		--fpr-b $(W3_FPR_B) \
		--n1 $(W3_N1) \
		--n0 $(W3_N0) \
		--k1 $(W3_K1) \
		--k0 $(W3_K0) \
		--alpha-cap $(W3_ALPHA) \
		--delta $(W3_DELTA) \
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
		--tpr-a $(W3_TPR_A) \
		--tpr-b $(W3_TPR_B) \
		--fpr-a $(W3_FPR_A) \
		--fpr-b $(W3_FPR_B) \
		--n1 $(W3_N1) \
		--n0 $(W3_N0) \
		--k1 $(W3_K1) \
		--k0 $(W3_K0) \
		--alpha-cap $(W3_ALPHA) \
		--delta $(W3_DELTA) \
		--target-t $(W3_TARGET_T) \
		--figure-out $(W3_FIG)

week3-power: install
	mkdir -p paper/figures
	$(ACT); python scripts/make_power_curve.py \
		--I1 0.37,0.65 \
		--I0 0.05,0.05 \
		--D 0.55 \
		--delta 0.05 \
		--target-t 0.10 \
		--out paper/figures/fig_week3_power_curve.png

# ======================================================================
# Reproduction Runs
# ======================================================================

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

figures: reproduce-figures

reproduce-paper: install
	mkdir -p $(PAPER_ARTIFACT_DIR)
	PYTHONPATH=src $(VENV_DIR)/bin/python scripts/reproduce_paper.py \
		--output-dir $(PAPER_ARTIFACT_DIR)

verify-paper-artifacts: install
	PYTHONPATH=src $(VENV_DIR)/bin/python scripts/verify_paper_artifacts.py \
		--artifact-dir $(PAPER_ARTIFACT_DIR)

paper-smoke: install
	PYTHONPATH=src $(VENV_DIR)/bin/python scripts/check_paper_source.py --latex

reports: install
	$(ACT); python -m cc.cartographer.cli build-reports --mode all

ccc: install
	$(ACT); python scripts/generate_ccc.py

# ======================================================================
# Artifact Boundary / Clean Repro
# ======================================================================

check-artifact-boundary: install
	$(ACT); python scripts/check_artifact_boundary.py --static

check-repro-clean: check-artifact-boundary
	baseline=$$(mktemp); \
	tmpdir=$$(mktemp -d); \
	trap 'rm -f "$$baseline"; rm -rf "$$tmpdir"' EXIT; \
	git status --porcelain=v1 --untracked-files=all > "$$baseline"; \
	PYTHONPATH=src $(VENV_DIR)/bin/python examples/minimal/run_bounds.py >/dev/null; \
	PYTHONPATH=src $(VENV_DIR)/bin/python scripts/reproduce_paper.py --output-dir "$$tmpdir/paper" >/dev/null; \
	PYTHONPATH=src $(VENV_DIR)/bin/python scripts/verify_paper_artifacts.py --artifact-dir "$$tmpdir/paper"; \
	PYTHONPATH=src $(VENV_DIR)/bin/python scripts/check_artifact_boundary.py --after-run --baseline "$$baseline"

# ======================================================================
# Docs
# ======================================================================

docs: install
	$(ACT); mkdocs build --strict

docs-serve: install
	$(ACT); mkdocs serve -a 127.0.0.1:8000

# ======================================================================
# Verifications
# ======================================================================

verify-invariants: install
	@if [ -f scripts/verify_invariants.py ]; then \
		$(ACT); python scripts/verify_invariants.py; \
	elif [ -d tests/unit/kernel ]; then \
		echo "No scripts/verify_invariants.py found; running kernel tests as invariant proxy."; \
		PYTHONPATH=src $(VENV_DIR)/bin/pytest tests/unit/kernel -q; \
	else \
		echo "No dedicated invariant verifier found."; \
	fi

verify-statistics: install
	$(ACT); python -m cc.cartographer.cli verify-stats \
		--config $(CONFIG_SMOKE) \
		--bootstrap 2000

verify-audit: install
	@if [ ! -f "$(AUDIT_LOG)" ]; then \
		echo "Missing $(AUDIT_LOG). Run make reproduce-smoke or make carto-smoke first."; \
		exit 1; \
	fi
	$(ACT); python -m cc.cartographer.cli verify-audit \
		--audit $(AUDIT_LOG)

# ======================================================================
# Docker
# ======================================================================

docker-build:
	docker build -f deployment/docker/Dockerfile -t cc-framework:cpu .

docker-run:
	docker run --rm -it \
		-v $$PWD:/workspace \
		-w /workspace \
		cc-framework:cpu bash

# ======================================================================
# Week 5
# ======================================================================

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
	@if command -v pandoc >/dev/null 2>&1; then \
		$(ACT); pandoc docs/week5_memo.md -o docs/week5_memo.pdf; \
		echo "Memo PDF → docs/week5_memo.pdf"; \
	else \
		echo "pandoc not found; skipping PDF build."; \
	fi

# ======================================================================
# Week 6
# ======================================================================

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
		--fpr-lo 0.00 \
		--fpr-hi 0.08 \
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

week6-ablation: $(addprefix week6-rail-,$(W6_RAILS))
	@echo "OK: week6-ablation complete."

week6-utility: install
	@if [ -z "$(RAIL)" ]; then \
		echo "Usage: make week6-utility RAIL=keyword"; \
		exit 2; \
	fi
	$(ACT); python scripts/summarize_results.py --utility \
		--final-results $(W6_DIR)/$(RAIL)/final_results.json \
		--out-csv $(W6_DIR)/$(RAIL)/utility_summary.csv \
		--out-fig $(W6_FIG_DIR)/$(RAIL)/utility_hist.png

memo-week6: install
	@echo "Memo ready at docs/week6_memo.md"
	@if command -v pandoc >/dev/null 2>&1; then \
		$(ACT); pandoc docs/week6_memo.md -o docs/week6_memo.pdf; \
		echo "Memo PDF → docs/week6_memo.pdf"; \
	else \
		echo "pandoc not found; skipping PDF build."; \
	fi

# ======================================================================
# Cartographer Agent
# ======================================================================

carto-install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

carto-smoke: install
	mkdir -p figs runs
	$(ACT); python -m cc.cartographer.cli run \
		--config $(CONFIG_SMOKE) \
		--samples $(SESS_SMOKE) \
		--fig figs/phase_smoke.png \
		--audit $(AUDIT_LOG)

carto-mvp: install
	mkdir -p figs runs
	$(ACT); python -m cc.cartographer.cli run \
		--config $(CONFIG_MVP) \
		--samples $(SESS_MVP) \
		--fig figs/phase_e2_T10.png \
		--audit $(AUDIT_LOG)

carto-verify-audit: verify-audit

carto-verify-stats: verify-statistics

carto-suggest: install
	@if [ ! -f "$(AUDIT_LOG)" ]; then \
		echo "Missing $(AUDIT_LOG). Run make carto-smoke first."; \
		exit 1; \
	fi
	mkdir -p experiments/grids
	$(ACT); python -m cc.cartographer.suggest \
		--history $(AUDIT_LOG) \
		--out experiments/grids/next.json

# ======================================================================
# Embedded Python snippets for governance targets
# Kept outside recipes to avoid Makefile heredoc/tab separator failures.
# ======================================================================

define GOV_IMPORTS_PY
import cc
from cc.evidence import ClaimGovernanceAudit, GovernanceVerdict, verify_claim_governance

print("cc import OK")
print("ClaimGovernanceAudit:", ClaimGovernanceAudit)
print("GovernanceVerdict:", GovernanceVerdict)
print("verify_claim_governance:", verify_claim_governance)
endef
export GOV_IMPORTS_PY

define GOV_INSPECT_PY
import json
from pathlib import Path

out = Path("$(GOV_CAPSULE_OUT)")
files = [
    "claim_governance_audit.json",
    "claim_envelope.json",
    "capsule_manifest.json",
    "cc_report.json",
    "extremal_upper.json",
    "extremal_lower.json",
    "confirmatory_protocol.json",
    "confirmatory_failure_matrix.json",
    "decay_policy.json",
    "bounds.json",
]

for name in files:
    path = out / name
    print("\n" + "=" * 92)
    print(name)
    print("=" * 92)

    if not path.exists():
        print("MISSING:", path)
        continue

    data = json.loads(path.read_text())

    for key in [
        "schema",
        "verdict",
        "report_id",
        "claim_id",
        "allowed_claim_level",
        "required_human_review",
        "state",
        "manifest_id",
        "package_id",
        "scenario_id",
        "kind",
        "status",
    ]:
        if key in data:
            print(f"{key}: {data[key]}")

    if "reasons" in data:
        print("reasons:", data["reasons"])
    if "non_claims" in data:
        print("non_claim_count:", len(data["non_claims"]))
    if "evidence_artifacts" in data:
        print("evidence_artifact_count:", len(data["evidence_artifacts"]))
    if "artifacts" in data:
        print("artifact_count:", len(data["artifacts"]))
    if "excluded_evidence_fields" in data:
        print("excluded_evidence_field_count:", len(data["excluded_evidence_fields"]))
endef
export GOV_INSPECT_PY

define GOV_FOUNDATION_AUDIT_PY
from pathlib import Path

path = Path("docs/memos/local_foundation_audit.md")
if path.exists():
    print(f"Foundation audit memo already exists: {path}")
    raise SystemExit(0)

path.parent.mkdir(parents=True, exist_ok=True)

path.write_text("""# Local Foundation Audit: Claim Governance Phases 1–5

## Scope

This audit reviews the implemented claim-governance foundation.

## Phase 1 — Claim Lifecycle

### Files inspected

### What the phase proves

### What it does not prove

### Forbidden transitions tested

### Remaining risks

### Local verdict

PASS / NEEDS_REVIEW / FAIL

## Phase 2 — ClaimEnvelope / BoundaryEnvelope

### Files inspected

### Does the envelope preserve boundaries?

### Does it preserve non-claims?

### Does it distinguish evidence from support?

### Remaining risks

### Local verdict

PASS / NEEDS_REVIEW / FAIL

## Phase 3 — Evidence Role Ontology

### Files inspected

### Are roles typed or decorative?

### Do roles define what they cannot support?

### Does unknown evidence stay conservative?

### Remaining risks

### Local verdict

PASS / NEEDS_REVIEW / FAIL

## Phase 4 — Confirmatory Evidence Protocol

### Files inspected

### Does exploratory evidence remain separated?

### Does protocol timing matter?

### Can post-hoc plans sneak through?

### Remaining risks

### Local verdict

PASS / NEEDS_REVIEW / FAIL

## Phase 5 — Deterministic Governance Capsule

### Files inspected

### Is the capsule reproducible?

### Are outputs generated rather than hand-authored?

### Does second-run determinism pass?

### Does tamper failure pass?

### Remaining risks

### Local verdict

PASS / NEEDS_REVIEW / FAIL

## Cross-phase invariants

- [ ] PASS does not mean deployment safety.
- [ ] Evidence hashes are checked.
- [ ] Decay is verification-time state, not signed live truth.
- [ ] Endpoint worlds are feasible/counterfactual, not likely-world claims.
- [ ] Exploratory evidence cannot silently become confirmatory.
- [ ] Unknown roles cannot strengthen a claim.
- [ ] Human review is not simulated unless artifact exists.
- [ ] Capsule can be rebuilt from fixed inputs.
- [ ] Expected artifacts are intentionally golden, not accidental.

## Final local foundation verdict

PASS / NEEDS_REVIEW / FAIL

## Changes required before next Codex prompt

1.
2.
3.
""", encoding="utf-8")

print(f"Foundation audit memo ready: {path}")
endef
export GOV_FOUNDATION_AUDIT_PY

# ======================================================================
# Claim Governance Help
# ======================================================================

governance-help:
	@echo ""
	@echo "CC-Framework • Claim Governance Targets"
	@echo "-------------------------------------------------------------"
	@echo "governance-install              Install editable package with governance/dev/docs deps"
	@echo "governance-cli                  Check cc-report CLI and verifier command"
	@echo "governance-imports              Check public governance imports"
	@echo "governance-quick                Ruff + focused governance tests"
	@echo "governance-tests                Governance verifier + capsule tests"
	@echo "governance-full-tests           Full pytest suite"
	@echo "governance-capsule              Rebuild deterministic governance capsule"
	@echo "governance-capsule-verify       Verify capsule against golden expected artifacts"
	@echo "governance-capsule-update       Regenerate expected capsule artifacts intentionally"
	@echo "governance-capsule-inspect      Print audit/envelope/manifest summary"
	@echo "governance-capsule-tamper-demo  Demonstrate hash/tamper failure behavior"
	@echo "governance-verify-report        Run cc-report verify-claim-governance on capsule report"
	@echo "governance-docs                 MkDocs strict build"
	@echo "governance-foundation-audit     Create local foundation audit memo if missing"
	@echo "governance-proof-fast           Fast local proof gate"
	@echo "governance-proof                Full local proof gate"
	@echo "verify-governance               Alias for governance-proof"
	@echo "-------------------------------------------------------------"
	@echo ""

# ======================================================================
# Claim Governance Installation / Checks
# ======================================================================

governance-install: $(VENV_DIR)/bin/activate
	$(ACT); $(PIP) install --upgrade pip setuptools wheel
	$(ACT); $(PIP) install -e "$(EXTRAS)" || $(PIP) install -e ".[dev]" || $(PIP) install -e .
	$(ACT); $(PIP) install pytest ruff mkdocs mkdocs-material pydantic numpy scipy pandas

governance-cli: governance-install
	$(ACT); cc-report --help
	$(ACT); cc-report verify-claim-governance --help

governance-imports: governance-install
	$(ACT); python -c "$$GOV_IMPORTS_PY"

governance-quick: governance-install
	$(ACT); ruff check src/cc/evidence src/cc/reporting tests/unit/evidence tests/unit/reporting tests/integration/test_claim_governance_capsule.py
	$(ACT); ruff format --check src/cc/evidence src/cc/reporting tests/unit/evidence tests/unit/reporting tests/integration/test_claim_governance_capsule.py
	$(ACT); pytest $(GOV_UNIT_TESTS) $(GOV_CAPSULE_TESTS) -q

governance-tests: governance-install
	$(ACT); pytest $(GOV_UNIT_TESTS) $(GOV_CAPSULE_TESTS) -q

governance-full-tests: governance-install
	$(ACT); pytest -q

# ======================================================================
# Claim Governance Capsule
# ======================================================================

governance-capsule: governance-install
	bash $(GOV_CAPSULE_DIR)/reproduce.sh

governance-capsule-verify: governance-install
	bash $(GOV_CAPSULE_DIR)/reproduce.sh --verify-only

governance-capsule-update: governance-install
	@echo "WARNING: This intentionally updates golden expected capsule artifacts."
	@echo "Only run this after a deliberate governance semantics change."
	bash $(GOV_CAPSULE_DIR)/reproduce.sh --update-expected

governance-capsule-clean:
	rm -rf $(GOV_CAPSULE_OUT)

governance-capsule-inspect: governance-capsule
	$(ACT); python -c "$$GOV_INSPECT_PY"

governance-verify-report: governance-capsule
	$(ACT); cc-report verify-claim-governance \
		$(GOV_REPORT) \
		--now $(GOV_NOW) \
		--out $(GOV_AUDIT)

governance-capsule-tamper-demo: governance-capsule
	@tmpdir=$$(mktemp -d); \
	echo "Copying capsule outputs to $$tmpdir"; \
	cp -R $(GOV_CAPSULE_OUT) "$$tmpdir/outputs"; \
	echo "Tampering with extremal_upper.json"; \
	printf "\n" >> "$$tmpdir/outputs/extremal_upper.json"; \
	echo ""; \
	echo "Now running verifier against tampered artifact set. Expected: nonzero / FAIL."; \
	set +e; \
	$(VENV_DIR)/bin/cc-report verify-claim-governance "$$tmpdir/outputs/cc_report.json" \
		--base-dir "$$tmpdir/outputs" \
		--now $(GOV_NOW) \
		--out "$$tmpdir/outputs/tampered_audit.json"; \
	status=$$?; \
	set -e; \
	echo "Verifier exit code: $$status"; \
	if [ "$$status" -eq 0 ]; then \
		echo "FAIL: tampered package unexpectedly passed"; \
		exit 1; \
	fi; \
	echo "OK: tampering was detected or forced review/failure."; \
	echo "Tampered audit: $$tmpdir/outputs/tampered_audit.json"

# ======================================================================
# Claim Governance Docs / Diff / Warnings / Audit
# ======================================================================

governance-docs: governance-install
	$(ACT); mkdocs build --strict

governance-diff:
	git diff --check
	git status --short

governance-warnings: governance-install
	@echo ""
	@echo "Known warning currently expected:"
	@echo "  Pydantic warning: field name 'schema' shadows BaseModel.schema"
	@echo ""
	@echo "Future hardening:"
	@echo "  Preserve public JSON field 'schema' while using an internal schema_ alias."
	@echo "  Do not patch casually because golden capsule artifacts may need updating."
	@echo ""

governance-foundation-audit: governance-install
	$(ACT); python -c "$$GOV_FOUNDATION_AUDIT_PY"

# ======================================================================
# Claim Governance Proof Gates
# ======================================================================

governance-proof-fast: governance-cli governance-imports governance-quick governance-capsule-verify governance-capsule-inspect governance-diff
	@echo ""
	@echo "OK: Fast governance proof passed."
	@echo ""

governance-proof: governance-cli governance-imports governance-quick governance-full-tests governance-capsule governance-capsule-verify governance-docs governance-diff
	@echo ""
	@echo "OK: Full governance proof passed."
	@echo ""

verify-governance: governance-proof

# ======================================================================
# Clean
# ======================================================================

clean:
	rm -rf .pytest_cache .coverage htmlcov site dist build \
		results/aggregates results/artifacts \
		$(FIG_DIR) paper/tables \
		benchmarks/results || true
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

distclean: clean
	rm -rf $(VENV_DIR) requirements.lock.txt
