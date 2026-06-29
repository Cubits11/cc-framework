# CC Framework Documentation

CC-Framework is a research prototype for dependence-aware evaluation and audit
receipts for composed AI guardrails. It studies when individual guardrail
metrics are insufficient because the joint dependence structure of failures is
unknown.

The public claim boundary is documented in
[Public Research Framing](research/public-framing.md). In short: CC-Framework
does not prove deployed systems are safe. It helps show when composition claims
are underidentified unless dependence is measured, bounded, or explicitly
assumed.

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

* **Frechet cartography**: compute sharp finite-atom composition intervals from
  marginals and optional pairwise dependence evidence.
* **Independence regret**: compare observed composition risk with an explicit
  product-coupling baseline.
* **Witness distributions**: return endpoint joint laws that verify reported
  lower and upper bounds.
* **Correlation cliffs**: detect dependence-driven jumps in composed risk.
* **Claim-bounded receipts**: bind claims to evidence while separating evidence
  integrity from statistical validity.

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

* [Public Research Framing](research/public-framing.md)
* [CC Reports and Receipts](research/CC_REPORTS.md)
* [Experiments Guide](experiments-guide.md)
* [Reproducibility Notes](reproducibility.md)

This manual will evolve as the framework matures—contributions welcome!
