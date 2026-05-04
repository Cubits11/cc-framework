# CC-Framework: Correlation Cliff Framework

**Dependence-aware evaluation of composed AI safety guardrails under uncertainty.**

[![Tests](https://github.com/Cubits11/cc-framework/actions/workflows/tests.yml/badge.svg)](https://github.com/Cubits11/cc-framework/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![Research Area](https://img.shields.io/badge/research-AI%20safety%20evaluation-purple)

> **Research status:** CC-Framework is an active research prototype. A formal preprint and archival release may be added later, but this README intentionally avoids placeholder arXiv, DOI, or publication badges until those records exist.

---

## Table of Contents

- [Research Statement](#research-statement)
- [Why This Matters](#why-this-matters)
- [Core Concept](#core-concept)
- [What CC-Framework Provides](#what-cc-framework-provides)
- [Repository Structure](#repository-structure)
- [Key Modules](#key-modules)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Example Workflow](#example-workflow)
- [Interpreting Results](#interpreting-results)
- [Reproducibility and Auditability](#reproducibility-and-auditability)
- [Research Provenance](#research-provenance)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Ethical Use](#ethical-use)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Research Statement

Modern AI safety systems increasingly rely on **composed guardrails**: multiple filters, classifiers, monitors, or policy checks layered together to reduce unsafe behavior. Standard evaluation often reports each guardrail's individual performance, but deployed systems can fail through **dependence**. Two strong guardrails may share the same blind spot, trigger on the same examples, fail under the same distribution shift, or interfere when combined.

**CC-Framework** studies this problem directly. It provides a Python research software framework for evaluating when composed guardrails exhibit **correlation cliffs**: regimes where changes in dependence structure cause large changes in the behavior of a composed safety system.

The framework combines:

- Fréchet-Hoeffding bounds for dependence-aware reasoning
- two-world evaluation for baseline-vs-shift comparison
- adversarial attack simulation
- uncertainty-aware statistical protocols
- alternative composition metrics
- audit-oriented reproducibility infrastructure

The goal is not to claim that composition is always good or always bad. The goal is to make composition behavior **measurable, bounded, inspectable, and reproducible**.

---

## Why This Matters

A composed safety system can look strong while providing little true redundancy.

Suppose two guardrails, `A` and `B`, are evaluated independently:

```text
P(A triggers) = known
P(B triggers) = known
```

A common but risky move is to infer the value of the composition from those marginal rates alone. But the composed behavior depends heavily on the unknown overlap:

```text
P(A triggers and B triggers)
```

If both guardrails fail on the same inputs, the composition may provide an illusion of safety. If they fail on different inputs, the composition may provide real coverage. Without modeling dependence, those cases can be confused.

CC-Framework asks:

> When multiple AI guardrails are composed, what can be honestly inferred about the composed system's safety when individual guardrail rates are known but the joint dependence structure is uncertain?

---

## Core Concept

CC-Framework uses a **two-world evaluation**:

- **World 0:** baseline, clean, or reference distribution
- **World 1:** shifted, adversarial, stressed, or deployment-like distribution

For each world, the framework tracks individual guardrail behavior and composed behavior.

The central quantity is the **Composition Coefficient (CC)**:

```text
CC = composition jump / best single-guardrail jump
```

where:

```text
composition jump =
|P(composition triggers in World 1) - P(composition triggers in World 0)|

best single-guardrail jump =
max(
  |P(A triggers in World 1) - P(A triggers in World 0)|,
  |P(B triggers in World 1) - P(B triggers in World 0)|
)
```

This reframes safety evaluation from:

```text
How good is each guardrail separately?
```

to:

```text
What does the composed system actually add under shift?
```

---

## What CC-Framework Provides

### 1. Dependence-aware composition analysis

CC-Framework uses Fréchet-Hoeffding bounds to reason about feasible joint behavior without assuming independence.

For two binary guardrails with marginal trigger probabilities `pA` and `pB`:

```text
max(0, pA + pB - 1) <= P(A = 1, B = 1) <= min(pA, pB)
```

These bounds allow the framework to compute feasible envelopes for composed behavior.

---

### 2. Two-world experimental protocol

The framework supports baseline-vs-shift comparisons for guardrail systems.

Examples of two-world setups:

| World 0 | World 1 |
|---|---|
| clean prompts | adversarial prompts |
| baseline user distribution | shifted deployment distribution |
| non-jailbreak examples | jailbreak examples |
| low-risk domain | high-risk domain |
| pre-mitigation system | post-mitigation system |

---

### 3. Adaptive statistical evaluation

The protocol layer includes research-grade statistical components such as:

- ICC-aware correction for clustered attack trials
- one-way random-effects ANOVA for ICC estimation
- ROPE-based Bayesian sequential testing
- Beta posterior modeling for proportions
- ATE estimation for world effects
- confidence intervals adjusted by design effect
- deterministic checkpoints for experiment recovery

These components are intended to make evaluation claims more statistically disciplined than raw pass/fail counts.

---

### 4. Adversarial attack simulation

CC-Framework includes controlled attack strategy abstractions for evaluating guardrail behavior under repeated adversarial pressure.

Implemented attacker families include:

- `RandomInjectionAttacker`
- `TemplatePromptAttacker`
- `GeneticAlgorithmAttacker`

The genetic attacker supports tournament selection, crossover, mutation, fitness caching, optional EMA smoothing, and diversity pressure.

These attackers are research utilities, not operational exploit tools.

---

### 5. Alternative composition metrics

The framework includes multiple complementary metrics for detecting constructive or destructive composition behavior:

- Euclidean distance to the ROC ideal point
- cost-weighted error
- delta Youden's J against an independence baseline
- Fréchet-Hoeffding envelope percentile

The purpose is to avoid relying on a single metric when composition behavior is ambiguous.

---

### 6. Audit-oriented reproducibility

CC-Framework includes infrastructure for evidence-oriented experiments:

- reproducible configuration
- manifest-style metadata
- stable JSON serialization
- tamper-evident JSONL audit chains
- SHA-256 linked records
- chain verification utilities
- provenance-aware experiment records

This makes the framework useful not only for running experiments, but for preserving the reasoning trail behind safety claims.

---

## Repository Structure

```text
cc-framework/
├── src/
│   └── cc/
│       ├── adapters/
│       │   └── base.py
│       │
│       ├── analysis/
│       │   └── alternative_metrics.py
│       │
│       ├── cartographer/
│       │   ├── audit.py
│       │   ├── bounds.py
│       │   └── intervals.py
│       │
│       ├── cli/
│       │   └── manifest.py
│       │
│       ├── core/
│       │   ├── attackers.py
│       │   ├── audit_runner.py
│       │   ├── evidence_bundle.py
│       │   ├── logging.py
│       │   ├── models.py
│       │   ├── protocol.py
│       │   └── stats.py
│       │
│       ├── exp/
│       │   └── run_two_world.py
│       │
│       └── guardrails/
│           └── ...
│
├── experiments/
│   ├── correlation_cliff/
│   │   ├── theory.py
│   │   ├── theory_core.py
│   │   └── simulate/
│   │       └── ...
│   │
│   └── fh_atlas/
│       └── ...
│
├── theory/
│   └── fh_bounds.py
│
├── docs/
│   └── ...
│
├── tests/
│   └── ...
│
└── README.md
```

---

## Key Modules

### `src/cc/core/protocol.py`

Adaptive two-world experiment engine.

Includes:

- ICC computation
- ROPE-based Bayesian sequential testing
- causal effect / ATE estimation
- guardrail factory layer
- experiment states and stopping reasons
- deterministic checkpointing
- audit-friendly summaries

Use this module when you want to run structured baseline-vs-shift evaluation.

---

### `src/cc/core/attackers.py`

Runtime attacker interface and concrete attack strategies.

Includes:

- `AttackStrategy`
- `RandomInjectionAttacker`
- `TemplatePromptAttacker`
- `GeneticAlgorithmAttacker`
- config dataclasses
- serialization hooks
- factory helpers

Use this module when you want repeatable adversarial pressure for guardrail evaluation.

---

### `src/cc/cartographer/bounds.py`

Fréchet-Hoeffding and ROC-envelope utilities.

Includes:

- ROC anchor handling
- AND/OR envelope logic
- n-way Fréchet-Hoeffding helpers
- Bernstein tail utilities
- confidence interval support
- sample size planning helpers

Use this module when you need dependence-aware bounds over composed ROC behavior.

---

### `theory/fh_bounds.py`

Expanded theoretical implementation layer.

Includes:

- intersection and union bounds
- composed J bounds
- independence baselines
- copula helpers
- composability interference metrics
- probability validation
- numerical stability helpers

Use this module for deeper mathematical experiments and theory-facing analysis.

---

### `src/cc/cartographer/audit.py`

Tamper-evident audit chain and FH-ceiling auditor.

Includes:

- stable JSON serialization
- SHA-256 record hashes
- previous-record hash links
- fsync-backed append
- chain verification
- chain rehashing discipline
- truncation to last valid record
- audit record construction
- FH ceiling checks

Use this module when experiment evidence needs to be inspectable after the fact.

---

### `src/cc/analysis/alternative_metrics.py`

Alternative metrics for composition analysis.

Includes:

- Euclidean distance to ROC perfection
- cost-weighted error
- delta Youden's J against independence
- Fréchet-Hoeffding envelope percentile

Use this module when one metric is insufficient to judge whether composition is constructive, neutral, or destructive.

---

### `experiments/correlation_cliff/`

Experimental research package for correlation cliff simulation and theory exploration.

Includes:

- public theory facade
- theory core
- simulation utilities
- path-based dependence experiments
- result summaries

Use this package for experimental exploration and research replication.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Cubits11/cc-framework.git
cd cc-framework
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the package:

```bash
pip install -e .
```

Install development dependencies if available:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

---

## Quick Start

### 1. Verify the repository

```bash
pytest
```

### 2. Import core theory utilities

```python
from cc.cartographer.bounds import fh_intervals

interval = fh_intervals(0.3, 0.6)
print(interval)
```

### 3. Use an attacker

```python
from cc.core.attackers import RandomInjectionAttacker

attacker = RandomInjectionAttacker(
    vocab_harmful=["bypass", "override", "exploit"],
    vocab_benign=["please", "help", "context"],
)

attack = attacker.generate_attack(history=[])
print(attack["prompt"])
```

### 4. Append an audit record

```python
from cc.cartographer.audit import append_jsonl, verify_chain

path = "results/audit.jsonl"

record = {
    "experiment": "demo",
    "metric": "composition_coefficient",
    "value": 0.72,
}

sha = append_jsonl(path, record)
verify_chain(path)

print("appended:", sha)
```

---

## Example Workflow

A typical CC-Framework workflow looks like this:

```text
1. Define two worlds
   - World 0: baseline distribution
   - World 1: shifted or adversarial distribution

2. Define guardrails
   - Guardrail A
   - Guardrail B
   - Composition rule: OR, AND, or conditional composition

3. Run repeated attack/evaluation trials

4. Estimate marginal and composed behavior

5. Compute composition jump and CC

6. Compare observed behavior against dependence-aware bounds

7. Report uncertainty and alternative metrics

8. Save evidence bundle, manifest, and audit-chain records
```

---

## Interpreting Results

### Low CC

```text
CC << 1
```

May indicate constructive composition. The composed system changes less under shift than the strongest individual guardrail.

### Near-neutral CC

```text
CC ≈ 1
```

May indicate that the composition is not adding much beyond the best individual guardrail.

### High CC

```text
CC > 1
```

May indicate destructive or unstable composition. The composed system may amplify shift, dependence, or failure interactions.

### Important caveat

CC should not be interpreted alone.

A serious interpretation should include:

- confidence intervals
- sample size
- failure mode inspection
- Fréchet-Hoeffding envelope position
- alternative metrics
- guardrail calibration details
- attack strategy assumptions
- world definition

---

## Reproducibility and Auditability

CC-Framework treats reproducibility as part of the research claim.

A strong experiment should record:

- git commit
- Python version
- dependency versions
- random seeds
- world definitions
- guardrail specifications
- attacker configuration
- composition rule
- raw outputs
- summary metrics
- confidence intervals
- generated figures
- audit record hashes

The audit layer supports chained JSONL records where each record points to the previous record's SHA-256 hash. This makes tampering detectable and helps preserve a transparent experiment trail.

---

## Research Provenance

This project was developed by **Pranav Bhave** as part of:

```text
Course: IST 496 Independent Research Study
Institution: Pennsylvania State University
Research focus: LLM safety guardrail composition
Primary artifact: CC-Framework
Faculty supervisor: Dr. Peng Liu
```

The research began from a practical concern:

> AI safety claims often rely on individual guardrail metrics, but composed systems can fail at the dependence layer.

CC-Framework is the resulting attempt to turn that concern into a rigorous, inspectable research software system.

---

## Limitations

CC-Framework is an active research prototype. Important limitations remain:

- Many experiments are synthetic or controlled rather than fully deployed production evaluations.
- Fréchet-Hoeffding bounds can be conservative and wide.
- Two-world evaluation simplifies continuous drift into a discrete comparison.
- Attack strategies are baseline research tools, not a complete red-team suite.
- Causal claims require assumptions beyond the framework's core dependence bounds.
- Real guardrail validation requires careful calibration, dataset design, and threat modeling.
- Some APIs and module boundaries may change as the framework matures.

These limitations are not hidden. They define the next stage of the research.

---

## Roadmap

### Near-term

- Freeze the stable public API boundary
- Add tutorial notebooks
- Add clearer end-to-end examples
- Improve documentation for audit-chain verification
- Separate stable modules from experimental modules
- Add a cleaned research log under `docs/`

### Medium-term

- Add real guardrail adapter demonstrations
- Add benchmark experiment packets
- Add stronger visualization utilities
- Add reproducibility bundles for key figures
- Expand alternative metrics and disagreement analysis
- Add more tests for edge cases and degeneracy policies

### Long-term

- Prepare a formal preprint
- Create an archival reproducibility release
- Add Zenodo DOI only after release stabilization
- Extend pairwise composition analysis to broader n-way systems
- Explore links to privacy auditing, subgroup vulnerability, and safety assurance

---

## Ethical Use

This repository is intended for defensive research, safety evaluation, and auditability.

Do not use CC-Framework to:

- attack real systems without permission
- bypass deployed safety mechanisms
- generate operational exploit instructions
- misrepresent prototype outputs as production certification
- claim safety without external validation

Use CC-Framework to:

- study guardrail composition
- document evaluation assumptions
- identify dependence-driven failure modes
- compare composed safety systems responsibly
- build reproducible research evidence

---

## Citation

A formal citation will be added if and when a preprint or archival software release is available.

For now, cite the repository as:

```bibtex
@software{bhave_cc_framework_2026,
  author = {Bhave, Pranav},
  title = {CC-Framework: Correlation Cliff Framework for Dependence-Aware AI Safety Evaluation},
  year = {2026},
  url = {https://github.com/Cubits11/cc-framework},
  note = {Research prototype developed during IST 496 Independent Research Study at Pennsylvania State University under the supervision of Dr. Peng Liu}
}
```

---

## Acknowledgments

CC-Framework was developed by **Pranav Bhave** at **Pennsylvania State University** as part of **IST 496 Independent Research Study: LLM Safety Guardrail Composition**.

Faculty supervision and research guidance were provided by **Dr. Peng Liu**.

The framework builds on classical probability, statistical inference, reproducible research software practices, and modern AI safety evaluation concerns.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
