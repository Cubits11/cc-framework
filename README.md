# CC-Framework

**Dependence-aware evaluation and audit receipts for composed AI guardrails.**

[![CI](https://github.com/Cubits11/cc-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/Cubits11/cc-framework/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![Research Area](https://img.shields.io/badge/research-AI%20safety%20evaluation-purple)

Measure guardrail composition without pretending independence.

> **Research status:** CC-Framework is an active research prototype. A formal preprint and archival release may be added later. This README intentionally avoids placeholder arXiv, DOI, publication, or certification badges until those records exist.

## What this is

- A research prototype for dependence-aware guardrail composition evaluation.
- A small Python kernel for Fréchet-Hoeffding bounds, composition metrics, and local audit evidence.
- A reproducibility scaffold for controlled experiments and evidence bundles.

## What this is not

- Not a safety certification system.
- Not an enterprise SaaS backend.
- Not an AWS-native platform.
- Not proof that a deployed AI system is safe.
- Not a complete red-team framework.

---

## Table of Contents

- [Research Statement](#research-statement)
- [What this is](#what-this-is)
- [Why This Matters](#why-this-matters)
- [Core Concept](#core-concept)
- [Kernel Contract](#kernel-contract)
- [Mathematical Invariants](#mathematical-invariants)
- [Assumptions Registry](#assumptions-registry)
- [Audit Packet v1](#audit-packet-v1)
- [Determinism and Reproducibility Contract](#determinism-and-reproducibility-contract)
- [Privacy-Auditing Extension Point](#privacy-auditing-extension-point)
- [What CC-Framework Provides Today](#what-cc-framework-provides-today)
- [Repository Structure](#repository-structure)
- [Key Modules](#key-modules)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Example Workflow](#example-workflow)
- [Interpreting Results](#interpreting-results)
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

**CC-Framework** studies this problem directly. It provides Python research utilities for evaluating dependence-sensitive regimes where changes in overlap or distribution shift can cause large changes in composed guardrail behavior.

The current stable candidates are:

- Fréchet-Hoeffding bounds for dependence-aware reasoning
- local tamper-evident audit chains
- minimal evidence bundles
- toy, keyword, and regex guardrails for reproducible examples

Experimental surfaces include:

- two-world evaluation for baseline-vs-shift comparison
- adversarial attack simulation utilities
- ICC-aware diagnostics and anytime-valid sequential stopping
- alternative composition metrics and plotting
- vendor guardrail adapters

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

## Kernel Contract

CC-Framework separates **stable research-kernel surfaces** from **experimental surfaces**.

This distinction matters because safety, privacy, or audit claims should depend only on explicitly documented contract surfaces, not on exploratory scripts.

### Contract levels

| Level | Meaning | Examples |
|---|---|---|
| **Kernel** | Intended stable research logic. Changes should preserve documented invariants. | Fréchet-Hoeffding bounds, probability validation, composition metrics, audit-chain verification |
| **Protocol** | Research workflow logic that may evolve but should preserve documented output semantics. | two-world protocol, ICC-aware evaluation, anytime-valid sequential testing, ATE summaries |
| **Experimental** | Exploration code, notebooks, simulations, and analysis scripts. APIs may change. | correlation-cliff experiments, atlas generation, path sensitivity studies |
| **Application** | Adapters, demos, toy guardrails, dashboards, and user-facing workflows. APIs may change fastest. | guardrail adapters, CLI demos, visualization scripts |

### Current kernel candidates

The following modules are the main public contract candidates:

```text
src/cc/cartographer/bounds.py
src/cc/cartographer/audit.py
src/cc/cartographer/intervals.py
src/cc/core/stats.py
src/cc/core/models.py
src/cc/core/evidence_bundle.py
theory/fh_bounds.py
```

The following modules are protocol or workflow surfaces:

```text
src/cc/core/protocol.py
src/cc/core/attackers.py
src/cc/core/audit_runner.py
src/cc/exp/run_two_world.py
experiments/correlation_cliff/
```

### Contract promise

A result should be described as **kernel-supported** only if it satisfies all of the following:

1. It is produced through a documented module or workflow.
2. It records its assumptions.
3. It passes probability and feasibility validation.
4. It has explicit uncertainty or a stated reason uncertainty is unavailable.
5. It can be reproduced or audited from saved artifacts.
6. It does not rely on undocumented notebook state or one-off manual edits.

This README defines the contract intent. A future release should promote this into a versioned machine-readable contract such as:

```text
docs/contracts/kernel_v1.yaml
```

Until that file exists, the README is the human-readable contract source.

---

## Mathematical Invariants

CC-Framework is built around invariants that should hold regardless of experiment framing.

### Probability invariants

| ID | Invariant |
|---|---|
| `INV-PROB-001` | All probabilities must be finite real numbers in `[0, 1]`. |
| `INV-PROB-002` | Marginal probabilities must be validated before bound computation. |
| `INV-PROB-003` | Degenerate cases must return explicit decisions, not silent misleading metrics. |

### Fréchet-Hoeffding invariants

For binary events `A` and `B`:

```text
max(0, pA + pB - 1) <= P(A and B) <= min(pA, pB)
```

| ID | Invariant |
|---|---|
| `INV-FH-001` | The lower FH bound must never exceed the upper FH bound. |
| `INV-FH-002` | Any observed joint probability must be inside the FH envelope or be flagged. |
| `INV-FH-003` | AND composition must be bounded by feasible intersection bounds. |
| `INV-FH-004` | OR composition must be bounded by feasible union bounds. |
| `INV-FH-005` | Bound calculations must not assume independence unless explicitly marked as an independence baseline. |

### Composition invariants

| ID | Invariant |
|---|---|
| `INV-COMP-001` | Composition rules must be declared explicitly: `AND`, `OR`, or documented custom rule. |
| `INV-COMP-002` | `CC` must be finite or explicitly marked degenerate when `J_best = 0`. |
| `INV-COMP-003` | Marginal behavior and joint behavior must be reported separately. |
| `INV-COMP-004` | Aggregate composition metrics must not be treated as subgroup guarantees. |

### Statistical invariants

| ID | Invariant |
|---|---|
| `INV-STAT-001` | Confidence intervals must state their method or be marked unavailable. |
| `INV-STAT-002` | Clustered or repeated attack trials should be corrected or explicitly marked as uncorrected. |
| `INV-STAT-003` | Sequential stopping must state its stopping rule. |
| `INV-STAT-004` | Effect sizes should be reported alongside binary significance claims where available. |

These invariants are intentionally conservative. If an experiment violates one, that does not automatically make the experiment useless, but it does downgrade the strength of the claim.

---

## Assumptions Registry

Every serious audit claim depends on assumptions. CC-Framework makes those assumptions explicit.

### Core assumptions

| ID | Assumption | Risk if false |
|---|---|---|
| `ASSUMP-001` | Guardrail outputs can be represented as binary events for the selected analysis. | Bounds may not reflect true continuous-score behavior. |
| `ASSUMP-002` | World 0 and World 1 are meaningfully comparable distributions. | Composition jump may reflect dataset mismatch rather than system behavior. |
| `ASSUMP-003` | The selected composition rule matches the deployed or simulated system. | AND/OR conclusions may not apply to the actual pipeline. |
| `ASSUMP-004` | Marginal rates are estimated from representative samples. | Bounds may be numerically valid but operationally misleading. |
| `ASSUMP-005` | Attack trials are either independent or corrected for dependence. | Uncertainty may be understated. |
| `ASSUMP-006` | Subgroup claims require subgroup-conditioned evaluation, not only aggregate metrics. | Concentrated failures may remain hidden. |
| `ASSUMP-007` | Audit-chain integrity preserves records after creation; it does not certify that the original experiment was correct. | Hashes can prove tampering, not truth. |
| `ASSUMP-008` | Fréchet-Hoeffding bounds are model-free but may be wide. | Conservative envelopes may be decision-inconclusive. |
| `ASSUMP-009` | Synthetic or toy guardrails are useful for validation but not evidence of production behavior. | Prototype demos may be overgeneralized. |

### Assumption-to-risk traceability

Audit reports should map each major conclusion to assumption IDs.

Example:

```yaml
claim: "OR composition remains inside the FH envelope under World 1 shift."
depends_on:
  - ASSUMP-001
  - ASSUMP-002
  - ASSUMP-003
  - ASSUMP-004
  - ASSUMP-008
risk_if_false:
  - "The binary reduction may hide score-level instability."
  - "The selected worlds may not represent deployment drift."
```

A future release should mirror this registry into a machine-readable file such as:

```text
docs/assumptions.yaml
```

---

## Audit Packet v1

An **Audit Packet** is the minimum evidence bundle needed to support a CC-Framework evaluation claim.

### Purpose

The audit packet answers:

```text
What was evaluated?
Under what assumptions?
With what configuration?
Using what code version?
With what random seeds?
What metrics were produced?
What uncertainty was reported?
What artifacts prove the run can be inspected?
```

### Minimum packet fields

```yaml
audit_packet_version: "v1"
run_id: string
created_at_utc: string

code:
  repository: string
  commit: string
  branch: string
  dirty_worktree: boolean

environment:
  python_version: string
  platform: string
  dependency_snapshot: string | null

experiment:
  worlds:
    world_0: object
    world_1: object
  guardrails: list
  composition_rule: string
  attacker: object | null
  sample_sizes: object
  seeds: object

assumptions:
  ids: list
  notes: object | null

metrics:
  marginals: object
  joint: object | null
  composition: object
  uncertainty: object | null
  alternative_metrics: object | null

validation:
  probability_checks: pass | fail | warning
  fh_envelope_checks: pass | fail | warning
  degeneracy_policy: string
  invariant_status: object

artifacts:
  raw_outputs: list
  summaries: list
  figures: list
  manifest: string | null
  audit_chain: string | null
  file_hashes: object
```

### Guarantees

Audit Packet v1 guarantees only the following:

1. The run configuration is inspectable.
2. The assumptions are stated.
3. The output artifacts are named.
4. File hashes can detect artifact modification after packet creation.
5. The audit-chain can detect tampering with appended JSONL records.

Audit Packet v1 does **not** guarantee:

1. The experiment design is correct.
2. The dataset is representative.
3. The guardrails are production-realistic.
4. The causal interpretation is valid.
5. The result certifies a deployed system as safe.

This distinction is critical. Auditability is not the same as correctness; it is the ability to inspect and challenge a claim.

---

## Determinism and Reproducibility Contract

CC-Framework treats reproducibility as part of the research claim.

### Determinism policy

A reproducible run should define:

| Field | Requirement |
|---|---|
| `seed` | All stochastic components must receive explicit seeds. |
| `seed_scope` | Seeds should specify whether they apply globally, per world, per attack strategy, or per cell. |
| `environment` | Python version and dependency versions should be recorded. |
| `config_hash` | Experiment configuration should be hashable or stored as an artifact. |
| `artifact_hashes` | Output files should be hashable and listed in the audit packet. |
| `git_commit` | The code commit should be recorded whenever available. |
| `dirty_worktree` | Runs from uncommitted local changes should be marked. |

### Reproducibility levels

| Level | Meaning |
|---|---|
| **R0: Narrative reproducibility** | The README or paper describes the method, but no run packet is available. |
| **R1: Script reproducibility** | Commands and scripts are available, but environment and artifacts are incomplete. |
| **R2: Artifact reproducibility** | Config, seeds, summaries, and output hashes are available. |
| **R3: Exact reproducibility** | Same commit, same environment, same config, same seeds, same outputs. |
| **R4: Independent reproducibility** | A separate environment or researcher reproduces the substantive result. |

Most current experiments should be treated as **R1-R2** unless an audit packet, environment snapshot, and artifact hashes are present.

### Golden artifact discipline

For publication-grade results, the repository should maintain:

```text
results/
├── golden/
│   ├── manifest.json
│   ├── audit_packet.yaml
│   ├── summary.csv
│   ├── figures/
│   └── hashes.json
```

A result should not be described as publication-grade unless the relevant golden artifact packet exists.

---

## Privacy-Auditing Extension Point

CC-Framework is not currently an ML privacy auditing framework. Its primary domain is composed AI safety guardrail evaluation.

However, its methodology transfers naturally to privacy-auditing research because both settings involve hidden risk beneath aggregate metrics.

### Methodological bridge

| CC-Framework concept | Privacy-auditing analogue |
|---|---|
| Guardrail failure under distribution shift | Privacy leakage under deployment or query shift |
| Marginal guardrail rates | Aggregate attack success rates |
| Unknown joint dependence | Hidden dependence between risk factors |
| Tail-dependence cliffs | Sudden leakage increases under access or subgroup changes |
| Two-world evaluation | Baseline model vs. defended model, or score-access vs. label-only access |
| Subpopulation-concentrated failures | Group-specific privacy vulnerability |
| Audit packet | Privacy report with assumptions, threat model, metrics, artifacts |

### Example privacy-audit adaptation

A future privacy-audit module could evaluate:

```text
World 0: baseline model access
World 1: changed access regime or defended model

Metric A: membership inference risk
Metric B: attribute inference risk
Subgroups: demographic or feature-defined slices
Composition question: where do aggregate metrics hide concentrated privacy exposure?
```

This is not implemented as a complete privacy framework in the current repository. It is a research extension path.

### Why this matters

In privacy auditing, aggregate attack success can appear manageable while specific groups face higher exposure. CC-Framework's central discipline — separating aggregate behavior from hidden concentrated failure modes — is directly relevant to that kind of audit design.

---

## What CC-Framework Provides Today

### 1. Dependence-aware composition analysis

CC-Framework uses Fréchet-Hoeffding bounds to reason about feasible joint behavior without assuming independence.

For two binary guardrails with marginal trigger probabilities `pA` and `pB`:

```text
max(0, pA + pB - 1) <= P(A = 1, B = 1) <= min(pA, pB)
```

These bounds allow the framework to compute feasible envelopes for composed behavior.

---

### 2. Experimental two-world protocol

The experimental protocol layer supports baseline-vs-shift comparisons for guardrail systems.

Examples of two-world setups:

| World 0 | World 1 |
|---|---|
| clean prompts | adversarial prompts |
| baseline user distribution | shifted deployment distribution |
| non-jailbreak examples | jailbreak examples |
| low-risk domain | high-risk domain |
| pre-mitigation system | post-mitigation system |

---

### 3. Experimental statistical evaluation

The protocol layer includes experimental statistical components such as:

- ICC-aware correction for clustered attack trials
- one-way random-effects ANOVA for ICC estimation
- anytime-valid e-process sequential testing
- deprecated Bayesian ROPE heuristic only behind an explicit legacy flag
- ATE estimation for world effects
- confidence intervals adjusted by design effect
- deterministic checkpoints for experiment recovery

These components are diagnostics unless their assumptions are stated and checked. The anytime-valid stopping rule controls Type-I error under the null stated in `docs/theory/anytime_valid.md`; causal claims still require their own identification assumptions.

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

CC-Framework includes prototype infrastructure for evidence-oriented experiments:

- reproducible configuration
- manifest-style metadata
- stable JSON serialization
- local tamper-evident JSONL audit chains
- SHA-256 linked records
- chain verification utilities
- provenance-aware experiment records

Unsigned local hash chains detect ordinary modification after recording, but they are not sufficient against an attacker who can rewrite and re-anchor an entire log. Use externally managed signing keys for stronger evidence bundles.

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
- anytime-valid e-process sequential testing
- deprecated Bayesian heuristic behind `--legacy-bayesian-heuristic`
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

### `src/cc/kernel/cliff.py`

Copula tail-dependence estimators and cliff certificates.

Includes:

- empirical lower and upper tail-dependence estimates
- Gaussian, Clayton, Gumbel, and Student-t copula model selection
- AIC/BIC candidate rankings
- bootstrap cliff certificates for sub-critical, critical, and super-critical co-failure regimes

Use this module when joint extreme guardrail co-failure is the object of inference.
See `docs/theory/correlation_cliffs.md` for the formal definition.

---

### `src/cc/kernel/sequential.py`

Anytime-valid Bernoulli e-process sequential testing.

Includes:

- explicit null `composed miss rate <= pre-registered p0`
- test-martingale/e-process wealth tracking
- Type-I error control for continuous monitoring via Ville's inequality
- null calibration and power simulation helpers with injectable RNGs

Use this module when early stopping must remain valid without sample-size pre-commitment.
See `docs/theory/anytime_valid.md` for the theorem and assumptions.

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

### 5. Generate a minimal evidence bundle

Create a small guardrail config:

```bash
cat > /tmp/cc-guardrails.json <<'JSON'
[
  {"name": "keyword_blocker", "params": {"keywords": ["secret", "bypass"]}},
  {"name": "regex_filter", "params": {"patterns": ["(?i)password"]}}
]
JSON
```

Run the bundle generator:

```bash
cc-bundle run \
  --prompt-source datasets/attack_prompts/basic.txt \
  --guardrails-config /tmp/cc-guardrails.json \
  --output-dir runs/evidence \
  --run-id demo \
  --unsigned \
  --disable-plots
```

Unsigned mode must be requested explicitly with `--unsigned`; otherwise pass `--private-key-path` with an externally managed Ed25519 key outside the output directory. The bundle generator never writes a private key into the output directory.

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

9. Map conclusions to assumptions and invariants
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
- subgroup or slice analysis where relevant
- assumption IDs
- invariant status

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
- The current README defines the public contract intent, but a full machine-readable contract registry is still a roadmap item.
- Privacy-audit applications are currently an extension direction, not a completed module.

These limitations are not hidden. They define the next stage of the research.

---

## Roadmap

### Near-term

- Freeze the stable public API boundary
- Add `docs/contracts/kernel_v1.yaml`
- Add `docs/assumptions.yaml`
- Add `docs/audit_packet_v1.schema.json`
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
- Add golden artifact packets for flagship experiments
- Add subgroup/slice analysis templates

### Long-term

- Prepare a formal preprint
- Create an archival reproducibility release
- Add Zenodo DOI only after release stabilization
- Extend pairwise composition analysis to broader n-way systems
- Explore formal links to privacy auditing, subgroup vulnerability, and safety assurance

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
- design more honest audit reports

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
