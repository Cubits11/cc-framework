# CC Framework Research & Development Guide

This manual gives engineers a complete overview of the `cc-framework`, how its
components interact, and concrete steps for extending and validating the
system.  It is intended as a living document; feel free to submit pull requests
that refine or extend the material.

## 1. Architectural Overview

### 1.1 Core Packages

| Package | Responsibility |
|---------|----------------|
| `cc.core` | Dataclasses, metrics, protocol orchestration, and logging utilities. |
| `cc.guardrails` | Defensive components such as keyword filters and model wrappers. |
| `cc.analysis` | Statistical routines for compositional capabilities (CC) evaluation. |
| `cc.io` | Data loading, experiment persistence, and serialization helpers. |
| `cc.examples` | Small runnable scripts demonstrating framework features. |
| `cc._legacy` | Historical reference implementations kept for comparison. |

### 1.2 Data Flow

1. **Experiment configuration** is loaded or constructed with
   `ExperimentConfig`.
2. **Attack sessions** are executed, yielding `AttackResult` objects.
3. **Guardrail stacks** alter behaviour according to `GuardrailSpec` entries in
   each `WorldConfig`.
4. **Metrics** consume serialized results (`AttackResult.to_dict()`) and output
   `CCResult` summaries.
5. **Analyses** persist artefacts via `cc.io` for reproducibility.

### 1.3 Directory Structure

```
src/cc
├── core          # models, metrics, and protocol helpers
├── guardrails    # built‑in guardrail implementations
├── analysis      # statistical evaluation routines
├── io            # dataset and serialization utilities
└── _legacy       # reference implementations
```

## 2. Getting Started

### 2.1 Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

### 2.2 Running the Test Suite

```bash
PYTHONPATH=src pytest tests/unit
```

### 2.3 Quick Example

```python
from cc.core.models import ExperimentConfig, GuardrailSpec

config = ExperimentConfig(
    experiment_id="demo",
    n_sessions=10,
    attack_strategies=["baseline"],
    guardrail_configs={
        "world0": [GuardrailSpec(name="keyword", params={"terms": ["ban"]})],
    },
)
print(config.to_dict())
```

## 3. Key Research Questions

1. How can guardrail stacking reduce attack success while maintaining utility?
2. Which statistical estimators yield the most stable CC measurements under
   limited samples?
3. What attacker strategy diversity is necessary to confidently bound CC?
4. How do different environment hashes influence reproducibility across runs?
5. Can adaptive guardrails learn from failed defences without leaking data?

## 4. Technical Improvement Suggestions

- **Type Safety** – introduce `mypy` and `ruff` to enforce style and typing.
- **Configuration Validation** – migrate dataclasses to `pydantic` models for
  automatic validation and schema export.
- **Experiment Tracking** – integrate an experiment tracker (e.g. MLflow) to
  centralize metrics and artefacts.
- **Parallel Simulation** – leverage `asyncio` or multiprocessing to speed up
  large attack sweeps.
- **Reproducibility** – record package versions and environment hashes in every
  result payload.
- **Testing** – expand coverage for protocol edge cases and guardrail
  calibration routines.
- **Continuous Integration** – add linting and test jobs to CI to catch
  regressions early.

## 5. Implementation & Extension Guidelines

* Use dataclasses for all configuration objects and provide `to_dict` methods
  for serialization.
* Prefer dependency injection for guardrails and attackers to ease
  experimentation.
* Maintain reproducibility by logging environment hashes and RNG seeds.
* When adding guardrails, implement both scoring and calibration interfaces.

### 5.1 Adding a New Guardrail

1. Implement a class under `cc.guardrails` exposing a `score(text)` method.
2. Provide a calibration routine returning empirical FPRs.
3. Define a `GuardrailSpec` describing parameters and versioning.
4. Register the guardrail in the chosen `WorldConfig` guardrail stack.

```python
@dataclass
class LengthGuardrail:
    max_tokens: int

    def score(self, text: str) -> float:
        return 1.0 if len(text.split()) > self.max_tokens else 0.0
```

### 5.2 Creating an Attack Strategy

1. Add a new subclass under `cc.core.attackers` implementing a `generate` method.
2. Register its name and default parameters in experiment configuration files.
3. Provide unit tests that validate `to_dict` serialization of the new strategy.

### 5.3 Contributing Analyses

1. Place new statistical routines in `cc.analysis` with clear docstrings.
2. Accept and return `AttackResult`/`CCResult` dataclasses for consistency.
3. Write benchmarks under `examples/` demonstrating usage and performance.

## 6. Action Items

- [ ] Integrate static analysis and linting (`mypy`, `ruff`).
- [ ] Document attack strategy schemas in the `docs/` directory.
- [ ] Provide examples for multi-guardrail experimentation.
- [ ] Benchmark alternative bootstrapping methods for CC estimation.
- [ ] Add CI workflows executing tests and style checks on pull requests.