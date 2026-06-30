# CC Framework Developer Manual

This manual is the current contributor guide for repository structure,
development commands, and validation lanes. For the shortest setup path, start
with `README.md`. For documentation navigation, use [docs/index.md](index.md).
For the active research lane, use [Paper Core](research/PAPER_CORE.md). For
kernel semantics and change discipline, use
[Strict Kernel Contract](architecture/STRICT_KERNEL_CONTRACT.md).

## 1. Current Package Map

| Area | Current role |
| --- | --- |
| `src/cc/kernel/` | Publication-facing finite-atom kernel, Frechet classes, metric diagnostics, and sample-complexity helpers. |
| `src/cc/core/` | Protocol models, evidence bundles, audit runners, and legacy workflow support. |
| `src/cc/exp/` | Two-world experiment runner and configs. |
| `src/cc/cartographer/` | Workflow, reporting, bounds utilities, and older atlas tooling. |
| `src/cc/evals/` | Benchmarks and Paper 1 example summaries. |
| `src/cc/reporting/` | Canonical report and receipt generation. |
| `src/cc/evidence/` | Assurance schemas and evidence-log helpers. |
| `src/cc/adapters/` | Vendor and guardrail adapter interfaces. |
| `src/cc/guardrails/` | Built-in guardrail implementations used by experiments. |
| `src/cc/io/` | Data loading, deterministic storage, seeds, and serialization helpers. |

New paper-facing math should usually land in `src/cc/kernel/` with focused
tests under `tests/unit/kernel/`. Use `experiments/`, `cc.cartographer`, and
`cc.core` for workflows or historical support unless the strict-kernel contract
explicitly says otherwise.

## 2. Setup

The package supports Python `>=3.10`. CI currently runs code and docs checks on
Python 3.10, 3.11, 3.12, and 3.13.

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip wheel setuptools
.venv/bin/pip install -e '.[dev,docs]'
```

Useful first checks:

```bash
PYTHONPATH=src .venv/bin/python examples/minimal/run_bounds.py
PYTHONPATH=src .venv/bin/pytest tests/unit/kernel -q
.venv/bin/mkdocs build --strict
```

The root `mkdocs.yml` is the active documentation configuration. The Makefile
wraps the docs build as `make docs`.

## 3. Entrypoints

Active project scripts are declared in `pyproject.toml`:

| Script | Target |
| --- | --- |
| `cc-bundle` | `cc.core.evidence_bundle:main` |
| `cc-cartographer` | `cc.cartographer.cli:main` |
| `cc-dependence-bench` | `cc.evals.dependence_benchmark:main` |
| `cc-report` | `cc.reporting.cli:main` |

Common module and Makefile entrypoints:

```bash
PYTHONPATH=src .venv/bin/python -m cc.evals.dependence_benchmark --help
PYTHONPATH=src .venv/bin/python -m cc.cartographer.cli --help
make test-kernel
make reproduce-paper
make verify-paper-artifacts
make paper-smoke
```

## 4. Development Workflows

For paper-facing kernel changes:

1. Update `src/cc/kernel/`.
2. Add or update focused tests under `tests/unit/kernel/`.
3. Update [Metric Taxonomy](theory/metric_taxonomy.md), [Paper Core](research/PAPER_CORE.md), or artifact verifiers if the change affects paper outputs.
4. Run `make test-kernel` and the relevant Paper Core validation lane.

For guardrail or experiment changes:

1. Put reusable guardrail implementations under `src/cc/guardrails/`.
2. Put adapter-specific behavior under `src/cc/adapters/`.
3. Put experiment orchestration under `src/cc/exp/` or `experiments/`.
4. Keep product-coupling, Frechet, and identified-set calculations labeled and routed through the kernel when they are paper-facing.

For documentation changes:

1. Update `README.md` for root-level contributor guidance.
2. Update [docs/index.md](index.md) for docs navigation.
3. Update [Paper Core](research/PAPER_CORE.md) for active research-scope changes.
4. Update [Strict Kernel Contract](architecture/STRICT_KERNEL_CONTRACT.md) when kernel semantics, serialization contracts, or metric meanings change.
5. Run `make docs`.

## 5. CI and Validation

GitHub Actions currently uses these lanes:

| Workflow | What it checks |
| --- | --- |
| `.github/workflows/ci.yml` | `ruff check .`, focused `mypy` on Python 3.12, and `pytest -q` across Python 3.10 through 3.13. |
| `.github/workflows/docs.yml` | `make docs` across Python 3.10 through 3.13. |
| `.github/workflows/security.yml` | Bandit and `pip-audit` on Python 3.12. |
| `.github/workflows/pre-commit.yml` | All configured pre-commit hooks on Python 3.12. |
| `.github/workflows/cartographer.yml` | Scheduled and manual cartographer smoke run. |

Release-facing validation lanes are summarized in
[Validation Matrix](validation_matrix.md):

| Lane | Primary commands |
| --- | --- |
| Paper Core v0.3 | `make test-kernel`, `make test-release`, `make reproduce-paper`, `make verify-paper-artifacts`, `make paper-smoke` |
| Full Python | `PYTHONPATH=src .venv/bin/pytest -q` |
| Enterprise Reference v0.1 | `make enterprise-smoke` |
| Docs | `make docs` |
| Security | `make security` |

Historical planning docs under `docs/design-specs/` may preserve older
proposals. Treat them as research history and prefer the current README, docs
index, Paper Core, and strict-kernel contract for contributor guidance.
