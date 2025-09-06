# CC Framework Developer Manual

Welcome to the internal guide for engineers working on the cc‑framework. It
provides architectural background, research directions, and practical tips for
extending the codebase.

## 1. Architectural Overview

* **`cc.core`** – dataclasses for sessions, metrics, and shared constants.
* **`cc.guardrails`** – pluggable safeguards such as keyword filters and
  semantic classifiers.
* **`cc.exp`** – experiment runners implementing the two‑world protocol.
* **`cc.analysis`** – utilities for computing the Composability Coefficient and
  generating figures.

The experiment flow is illustrated in
[architecture/protocol_sequence.md](architecture/protocol_sequence.md).

## 2. Research Roadmap

* **N‑way composition**: extend CC beyond pairwise guardrails.
* **Adaptive attackers**: evaluate deep RL and side‑channel strategies.
* **Theoretical bounds**: derive closed‑form limits for CC under specific
  guardrail families.

## 3. Best Practices

* Write configuration‑driven experiments—avoid hard‑coded parameters.
* Keep runs reproducible: seed RNGs and commit `audit.jsonl` outputs.
* Adhere to Black formatting and ruff linting; run `pre-commit run --all-files`
  before pushing.

## 4. Extending the Framework

### Adding a Guardrail

1. Implement class in `cc/guardrails/` inheriting from `BaseGuardrail`.
2. Register in experiment configs under `guardrails:`.

### Adding an Attacker

1. Implement attacker in `cc/core/attackers.py` or a new module.
2. Expose entry point through `cc.exp.run_two_world` or custom runner.

### Adding Metrics

1. Define metric in `cc.analysis.metrics`.
2. Update summary aggregation to include the new metric.

## 5. Further Reading

* [Experiments Guide](experiments-guide.md)
* [Reproducibility Notes](reproducibility.md)

This manual will evolve as the framework matures—contributions welcome!
