"""
.PHONY: install test reproduce-mvp clean setup verify-invariants

install:
\tpython -m venv .venv && . .venv/bin/activate && pip install -e .[dev]

setup:
\tmake install
\tpre-commit install

test:
\t. .venv/bin/activate && pytest

reproduce-smoke:
\t. .venv/bin/activate && python src/cc/exp/run_two_world.py --n 200 --config src/cc/exp/configs/smoke_test.yaml

reproduce-mvp:
\t. .venv/bin/activate && python src/cc/exp/run_two_world.py --n 5000 --config src/cc/exp/configs/toy_A.yaml

reproduce-main:
\t. .venv/bin/activate && python src/cc/exp/run_two_world.py --n 20000 --config src/cc/exp/configs/main_study.yaml

reproduce-figures:
\t. .venv/bin/activate && python src/cc/analysis/generate_figures.py

reproduce-paper:
\tmake reproduce-figures && cd paper && pdflatex main.tex

verify-invariants:
\t. .venv/bin/activate && python src/cc/tests/test_invariants.py

verify-statistics:
\t. .venv/bin/activate && python src/cc/tests/test_bootstrap.py

verify-audit:
\t. .venv/bin/activate && python src/cc/tests/test_audit_chain.py

clean:
\trm -rf .venv .pytest_cache .coverage dist build logs
\tfind . -name __pycache__ -type d -exec rm -rf {} +
"""