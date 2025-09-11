# CC Framework — Research & Development Guide

This document serves as a Codex-style internal manual for cc-framework developers. It maps the architecture, enumerates research directions, and codifies patterns for extending experiments, guardrails, and attackers.

## 1. Architectural Overview

- **Core Data Models**
  - `AttackResult`, `GuardrailSpec`, `WorldConfig`, and `ExperimentConfig` define experiment schema and outputs.
  - `CCResult` captures composability metrics and confidence intervals.

- **Attackers**
  - `AttackStrategy` establishes the contract for adaptive attackers with `generate_attack`, `update_strategy`, and `reset` methods.

- **Guardrails**
  - `Guardrail` interface mandates `blocks`, `score`, and `calibrate` for filtering implementations.

- **Two-World Protocol**
  - `TwoWorldProtocol` orchestrates adaptive attacks in parallel worlds and tracks timing, results, and metadata.

- **Composability Analysis**
  - Statistical routines compute Youden's J, bootstrap confidence intervals, and composability coefficients.

- **Cartographer Subsystem**
  - `cartographer.audit` provides hash-chained logging and report generation.
  - `cartographer.cli` exposes CLI entry points for running experiments and aggregating reports.

## 2. Development Environment & Workflow

1. **Setup**
   ```bash
   git clone https://github.com/<org>/cc-framework.git
   cd cc-framework
   make setup      # create virtual environment and install dependencies
   ```

2. **Quality checks**
   ```bash
   pre-commit run --files <changed_files>
   pytest -q tests/unit tests/integration
   ```

3. **Workflow tips**
   - Keep commits small and descriptive.
   - Record seeds and Git SHAs via the audit logger.
   - Update documentation alongside code changes.

## 3. Research Questions & Deep-Dive Areas

- How do composability coefficients scale beyond pairwise guardrail combinations?
- Which adaptive attack strategies (e.g., deep RL, genetic search) expose weaknesses in guardrail stacks?
- Can confidence interval estimation for J-statistics be tightened or replaced with Bayesian approaches?
- How do varying utility profiles affect CC outcomes across domains?
- What formal guarantees can be provided for tamper-evident audit chains?

## 4. Technical Improvements & Best Practices

- Populate placeholder YAML files in `src/cc/exp/configs/` with documented defaults.
- Introduce a registry pattern so new guardrails and attackers auto-register for CLI discovery.
- Expand unit and integration tests; keep coverage above the threshold defined in the `Makefile` (`COV_MIN` ≥ 80%).
- Extend the README and MkDocs documentation with diagrams illustrating world flow and guardrail stacking.
- Integrate `cc.utils.timing` to profile large-scale experiments.
- Version datasets and experiment outputs with metadata and seeds.

## 5. Implementation & Extension Guide

### 5.1 Environment Setup

```bash
make setup          # install deps and pre-commit hooks
make test           # run unit + integration tests
```

### 5.2 Running Experiments

- **Quick Smoke Test**
  ```bash
  make reproduce-smoke   # ~200 sessions, baseline figures & CSVs
  ```

- **Full MVP Run**
  ```bash
  make reproduce-mvp     # ~5,000 sessions, deeper analysis
  ```

- **Custom Run Workflow**
  1. **Plan** – choose guardrail stack, config, seed, and output directory.
  2. **Execute**
     ```bash
     python experiments/run.py \
       --config experiments/configs/toy.yaml \
       --seed 123 \
       --out results/demo/raw.jsonl
     ```
  3. **Compute metrics (policy FPR ≤ 5%)**
     ```bash
     python -m src.cc.analysis.cc_estimation \
       --raw results/demo/raw.jsonl \
       --policy.fpr_cap 0.05 --bootstrap.fixed 1000 \
       --out results/demo/metrics_fixed.json
     ```
  4. **Shuffle baseline & refit sensitivity (recommended)**
     ```bash
     python -m src.cc.analysis.cc_estimation \
       --raw results/demo/raw.jsonl --shuffle.k 1000 \
       --out results/demo/null_shuffle.json

     python -m src.cc.analysis.cc_estimation \
       --raw results/demo/raw.jsonl --bootstrap.refit 100 --grid 51 \
       --out results/demo/metrics_refit.json
     ```
  5. **Generate figures**
     ```bash
     python -m src.cc.analysis.generate_figures \
       --raw results/demo/raw.jsonl \
       --metrics results/demo/metrics_fixed.json \
       --out results/demo/figs
     ```
  6. **Audit** – append run metadata to `runs/audit.jsonl`.

### 5.3 Adding a New Guardrail

```python
# src/cc/guardrails/my_guardrail.py
from cc.guardrails.base import Guardrail

class MyGuardrail(Guardrail):
    def blocks(self, text: str) -> bool:
        ...
    def score(self, text: str) -> float:
        ...
    def calibrate(self, benign_texts: list, target_fpr: float = 0.05) -> None:
        ...
```

- Register via `GuardrailSpec(name="MyGuardrail", params={...})`.
- Provide a sample config and unit tests.

### 5.4 Creating a Custom Attacker

```python
from cc.core.attackers import AttackStrategy

class MyAttacker(AttackStrategy):
    def generate_attack(self, history):
        ...
    def update_strategy(self, attack, result):
        ...
    def reset(self, *, seed=None):
        ...
```

- Integrate into experiment YAML under `attacker.type` and `params`.

### 5.5 Extending the Protocol

- Use `TwoWorldProtocol.build_guardrail_stack` to inject additional guardrail layers or utility metrics.

### 5.6 Audit & Reporting

```bash
python -m cc.analysis.generate_figures \
  --history runs/audit.jsonl \
  --fig-dir paper/figures \
  --out-dir results/smoke/aggregates

python -m cc.cartographer.cli build-reports --mode all
```

## 6. Roadmap & Action Items

- [ ] Populate experiment configs with realistic defaults.
- [ ] Add registry-based discovery for guardrails and attackers.
- [ ] Expand statistical modules with Bayesian or sequential methods.
- [ ] Harden audit chain validation for large-scale deployments.
- [ ] Document best practices for tuning bootstrap parameters.

---

This guide should evolve with the project. Contributions and updates are welcome.

