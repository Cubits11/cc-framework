 Research & Development Guide for CC Framework

This guide provides an architectural overview, research directions, and implementation practices for the CC Framework. It is intended for engineers and researchers onboarding to the project.

## 1. Architectural Overview

- **Core Protocol (`src/cc/core/protocol.py`)**
  - Implements the *Two-World* experiment protocol used to measure composability of guardrails.
  - Handles session orchestration, guardrail stack construction, experiment logging, and result aggregation.
- **Attackers (`src/cc/core/attackers.py`)**
  - Strategy interfaces and concrete implementations (e.g., `RandomInjectionAttacker`, `GeneticAlgorithmAttacker`).
  - Each attacker exposes `generate_attack`, `update_strategy`, and `reset` methods.
- **Guardrails (`src/cc/guardrails/`)**
  - `Guardrail` base class defines scoring and blocking APIs.
  - Concrete guardrails: `KeywordBlocker`, `RegexFilter`, `SemanticFilter`, and compositional helpers.
- **Experiment Runner (`src/cc/exp/run_two_world.py`)**
  - CLI for loading YAML configs, building components, running experiments, and producing audit logs.
  - Supports config overrides and automated analysis output.
- **Logging (`src/cc/core/logging.py`)**
  - Chainable JSONL audit logger used across the protocol and runner.
- **Models & Stats (`src/cc/core/models.py`, `src/cc/core/stats.py`)**
  - Dataclasses for experiment metadata and statistical utilities (bootstrap CI, J-statistic).
- **Documentation & Scripts**
  - `docs/` for architectural notes and design specs.
  - `scripts/`, `tools/` provide support utilities.

## 2. Key Research Questions

- **Guardrail Interaction**
  - How do different guardrail classes interact under composition? Which combinations yield constructive vs. destructive interference?
- **Attacker Modeling**
  - What attacker capabilities most effectively exploit guardrail weaknesses? Explore RL-based attackers or side-channel attacks.
- **Statistical Reliability**
  - How many sessions are required for stable CC estimates? Investigate variance reduction and sequential analysis techniques.
- **Utility Metrics**
  - Beyond success rate, what utility trade-offs (latency, false positives) arise from guardrail stacks?
- **Calibration & Adaptation**
  - Can guardrails auto-calibrate to maintain target false-positive rates over time?   

## 3. Technical Improvements & Best Practices

- **Testing & CI**
  - Maintain high test coverage with unit, integration, and e2e tests under `tests/`.
  - Use `pre-commit` hooks for linting and formatting (`pre-commit run --files <file>`).
- **Configuration Management**
  - YAML configs under `src/cc/exp/configs/` should remain minimal; prefer defaults in code.
  - Provide sample configs for common experiment setups.
- **Type Safety & Documentation**
  - Use type hints and docstrings throughout core modules.
  - Keep `docs/` updated with new features and APIs.
- **Performance**
  - Profile attacker generation and guardrail evaluation loops.
  - Consider caching strategies and vectorized operations for large experiments.
- **Reproducibility**
  - Record random seeds and git commits for every experiment via the audit logger.

## 4. Implementing & Extending the Framework

### Adding a New Guardrail

1. Create a subclass of `Guardrail` in `src/cc/guardrails/`:
   ```python
   from cc.guardrails.base import Guardrail

   class LengthGuardrail(Guardrail):
       def score(self, text: str) -> float:
           return len(text)

       def blocks(self, text: str) -> bool:
           return len(text) > 500
   ```
2. Expose the guardrail in `TwoWorldProtocol._create_guardrail` mapping.
3. Add configuration example in `src/cc/exp/configs/`.
4. Write unit tests under `tests/unit/` and run `pytest`.

### Creating a Custom Attacker

1. Subclass `AttackStrategy` in `src/cc/core/attackers.py`.
2. Implement `generate_attack`, `update_strategy`, and `reset` methods.
3. Register the attacker in `src/cc/exp/run_two_world.py:create_attacker`.
4. Provide documentation and example config.

### Running Experiments

```bash
python -m cc.exp.run_two_world \
  --config experiments/configs/smoke.yaml \
  --set protocol.epsilon=0.02 protocol.T=-0.03 \
  --n 200
```
- Results are written to `results/analysis.json` and audit logs to `runs/audit.jsonl` by default.

## 5. Action Items & Roadmap

- [ ] Benchmark existing guardrails on larger datasets and record CC statistics.
- [ ] Prototype RL-based attacker to compare against GA and random baselines.
- [ ] Implement utility-aware scoring to capture trade-offs.
- [ ] Automate generation of experiment dashboards using results files.
- [ ] Draft publication-ready experiments based on gathered metrics.

---
This guide should evolve alongside the codebase. Contributions and updates are welcome.