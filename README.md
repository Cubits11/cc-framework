# CC-Framework

**Sharp partial-identification bounds for composed AI guardrail failures under unknown dependence.**

[![CI](https://github.com/Cubits11/cc-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/Cubits11/cc-framework/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)

CC-Framework is research software for dependence-aware partial identification
of composed AI guardrail failures. It asks what a composed guardrail evaluation
actually identifies when singleton failure rates are observed but the joint
dependence structure is unknown or only partially constrained.

## Research Status

This repository is an active research prototype. Its first-paper core is a
finite binary atom kernel for sharp Frechet composition intervals, metric
diagnostics, and endpoint witness distributions. It is not a deployment safety
certificate, not a product platform, and not an AI alignment solution.

## Core Thesis

Composed AI safety should be evaluated as a dependence-aware
partial-identification problem, not as a product of independent guardrail
scores.

The central claim is narrow: many composition claims are underidentified unless
dependence among guardrail failures is measured, bounded, or explicitly
assumed. The kernel computes what follows from declared assumptions and
evidence; it does not validate the upstream data collection process.

## What This Is / What This Is Not

What this is:

- A Python research kernel for finite-atom guardrail failure composition.
- A partial-identification framework for unknown dependence.
- A metrics layer for identified-set and product-baseline diagnostics.
- A witness-oriented reproducibility scaffold for mathematical verification.
- A research program with explicit non-claims and future-work boundaries.

What this is not:

- Not a certification system for deployed models.
- Not proof of deployment safety.
- Not causal inference without causal assumptions.
- Not a guarantee of dataset representativeness or future performance.
- Not a replacement for red teaming or human governance.
- Not an enterprise, dashboard, AWS, or adapter-centered project.

## Mathematical Object

Let

```text
Z = (Z_1, ..., Z_m),  Z_i in {0, 1}
```

with the repository-wide convention:

```text
Z_i = 1 means guardrail failure / unsafe pass.
```

The unknown object is the joint law

```text
pi(z) = P(Z = z),  z in {0, 1}^m.
```

Evaluation evidence may include singleton failure marginals

```text
p_i = P(Z_i = 1)
```

and optional pairwise constraints

```text
q_ij = P(Z_i = 1, Z_j = 1).
```

For a declared Boolean composition event `phi(Z)`, define the feasible Frechet
class `F` as the set of atom distributions satisfying the supplied assumptions
and evidence. The kernel computes:

```text
L_phi = inf_{pi in F} E_pi[phi(Z)]
U_phi = sup_{pi in F} E_pi[phi(Z)]
```

These are sharp bounds conditional on supplied assumptions and evidence.

## Implemented Kernel Surface

The current publication-facing kernel surface is intentionally small:

- [src/cc/kernel/sensitivity.py](src/cc/kernel/sensitivity.py): finite binary
  atom LP, linear assumptions, sharp identified intervals, infeasibility
  detection, endpoint witness distributions.
- [src/cc/kernel/metrics.py](src/cc/kernel/metrics.py): formal estimand-layer
  diagnostics such as `fh_width`, `fh_position`,
  `independent_event_probability`, `independence_regret`, `cc_gain`, and
  `cc_shift`.
- [src/cc/kernel/frechet_classes.py](src/cc/kernel/frechet_classes.py):
  classical Frechet special cases and side-constrained finite Bernoulli bounds.
- [docs/theory/metric_taxonomy.md](docs/theory/metric_taxonomy.md): canonical
  metric domains and deprecated-name mapping.
- [docs/theory/theorem_ledger.md](docs/theory/theorem_ledger.md): mathematical
  claims, implementation witnesses, tests, and non-claims.

Other modules remain useful but should be described more carefully:

- `src/cc/core`, `src/cc/exp`, and `src/cc/cartographer` are protocol,
  workflow, or legacy surfaces.
- `experiments/`, `scripts/`, and `notebooks/` are experimental surfaces.
- `apps/dashboard`, vendor adapters, and enterprise references are application
  or demonstration surfaces, not the first-paper core.

## Minimal Example

This example uses the current kernel API to identify the probability that at
least one declared guardrail failure occurs, given exact singleton failure
marginals and no dependence assumption.

```python
from cc.kernel.metrics import (
    fh_width,
    independence_regret,
    independent_event_probability,
)
from cc.kernel.sensitivity import AssumptionSet, LinearQuery

labels = ("input_filter_failure", "policy_judge_failure")
marginals = {
    "input_filter_failure": 0.08,
    "policy_judge_failure": 0.05,
}

assumptions = AssumptionSet.empty(labels)
for label, value in marginals.items():
    assumptions = assumptions.with_marginal_interval(label, value, value)

query = LinearQuery.union(
    labels,
    labels,
    name="P(any guardrail failure)",
)

result = assumptions.identify(query)
width = fh_width(result.lower_bound, result.upper_bound)
product_baseline = independent_event_probability(marginals, query, labels=labels)
endpoint_regrets = (
    independence_regret(result.lower_bound, product_baseline),
    independence_regret(result.upper_bound, product_baseline),
)

print(result.lower_bound, result.upper_bound)
print(width, product_baseline, endpoint_regrets)
```

For a runnable reviewer-facing script with witness checks, see
[examples/minimal/run_bounds.py](examples/minimal/run_bounds.py).

## Metrics

The canonical metric taxonomy has four categories:

| Category | Metrics | Interpretation |
| --- | --- | --- |
| Identified-set diagnostics | `fh_width`, `fh_position` | Describe the sharp interval `[L_phi, U_phi]` and where a selected event risk lies inside it. |
| Assumption-comparison diagnostics | `independent_event_probability`, `independence_regret` | Compare a declared event risk with an explicit product-coupling baseline. |
| One-world normalization diagnostics | `cc_gain` | Normalize a composition failure risk by the largest singleton failure risk in the same setting. |
| Two-world movement diagnostics | `cc_shift` | Summarize movement in composition failure risk relative to singleton failure-rate movement across two settings. |

Older names such as `cc_max`, `cc_rel`, `delta_add`, and `delta_mult` are
legacy/deprecated compatibility surfaces. They should not be presented as the
front-door theory; see [docs/theory/metric_taxonomy.md](docs/theory/metric_taxonomy.md).

## Witnesses and Reproducibility

If the LP reports `[L_phi, U_phi]`, a reproducible result should expose endpoint
witness distributions `pi_L` and `pi_U` that satisfy the declared constraints
and achieve the lower and upper endpoints.

Witnesses verify mathematical feasibility relative to supplied constraints.
They do not verify dataset representativeness, causal validity, semantic
coverage, or deployment safety.

The repository already exposes endpoint solutions through
`IdentificationResult.lower_solution` and `IdentificationResult.upper_solution`.
Paper-core artifacts can be regenerated with `make reproduce-paper` and checked
with `make verify-paper-artifacts`. This pipeline verifies deterministic kernel
artifacts and endpoint witnesses; it is not a claim that the historical LaTeX
paper source is complete.

## Research Program Documents

- [Research Program](docs/research/RESEARCH_PROGRAM.md)
- [Paper Core](docs/research/PAPER_CORE.md)
- [Non-Claims](docs/research/NON_CLAIMS.md)
- [Roadmap](docs/research/ROADMAP.md)
- [Metric Taxonomy](docs/theory/metric_taxonomy.md)
- [Theorem Ledger](docs/theory/theorem_ledger.md)
- [Reproducibility Notes](docs/reproducibility.md)

## Repository Structure

```text
src/cc/kernel/              canonical finite-atom and metric kernel
src/cc/core/                protocol and legacy workflow support
src/cc/exp/                 two-setting experiment runners
src/cc/cartographer/        workflow, reporting, and older atlas utilities
examples/minimal/           smallest runnable atom-LP example
docs/theory/                metric taxonomy, theorem ledger, derivations
docs/research/              research spine, paper core, non-claims, roadmap
experiments/                experimental studies and demonstrations
apps/dashboard/             application surface, not first-paper core
```

## Installation and Validation

Create a local environment and install development dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip wheel setuptools
.venv/bin/pip install -e '.[dev,docs]'
```

Run the minimal example:

```bash
PYTHONPATH=src .venv/bin/python examples/minimal/run_bounds.py
```

Kernel validation commands:

```bash
PYTHONPATH=src .venv/bin/pytest tests/unit/kernel -q
PYTHONPATH=src .venv/bin/mypy src/cc/kernel --strict
.venv/bin/ruff check src/cc/kernel tests/unit/kernel
.venv/bin/mkdocs build --strict
```

## Experimental Surfaces

Experimental and historical material remains in the repository, but it is not
the README's central theory.

- Two-setting diagnostics and older Composition Coefficient ratios are protocol
  or legacy material. Use `cc_shift` for the canonical two-setting movement
  diagnostic.
- Correlation-cliff experiments are research demonstrations; see
  [experiments/correlation_cliff/README.md](experiments/correlation_cliff/README.md)
  and [docs/theory/correlation_cliffs.md](docs/theory/correlation_cliffs.md).
- Older audit packets, manifests, adapters, dashboards, and cloud references
  are supporting or application surfaces unless a specific document promotes
  them into a reviewed contract.

## Limitations and Non-Claims

The kernel operates on finite binary failure events and explicit linear
constraints. Its conclusions are conditional on those inputs.

Key limitations:

- Binary event reductions may hide score-level or semantic structure.
- Estimated marginals and pairwise constraints require their own statistical
  justification.
- Pairwise evidence generally does not identify the full joint law.
- Explicit atom enumeration has scaling limits as `m` grows.
- Static finite-atom bounds are not a sequential agent framework.

Non-claims:

- The project does not prove deployment safety.
- The project does not certify deployed models.
- The project does not infer causality without causal assumptions.
- The project does not guarantee dataset representativeness or future behavior.
- The project does not make cryptographic integrity equivalent to statistical
  validity.

See [docs/research/NON_CLAIMS.md](docs/research/NON_CLAIMS.md) for the stricter
claim boundary.

## Citation

No archival citation exists yet. Until a preprint or release artifact exists,
cite the repository URL and commit hash used for the analysis.

## License

MIT. See [LICENSE](LICENSE).
