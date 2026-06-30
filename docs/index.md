# CC Framework Documentation

CC-Framework is a research prototype for dependence-aware partial
identification of composed AI guardrail failures. It studies when singleton
guardrail failure rates are insufficient because the joint dependence structure
of failures is unknown or only partially measured.

The public claim boundary is documented in
[Public Research Framing](research/public-framing.md). In short: CC-Framework
does not prove deployed systems are safe. It helps show when composition claims
are underidentified unless dependence is measured, bounded, or explicitly
assumed.

## 1. Architectural Overview

* **`cc.kernel.sensitivity`** – finite binary atom LP for sharp identified
  composition intervals and endpoint witnesses.
* **`cc.kernel.metrics`** – identified-set diagnostics, product-coupling
  baselines, independence regret, and bounded normalization helpers.
* **`cc.kernel.sample_complexity`** – Hoeffding-style finite-sample helper
  functions for singleton and pairwise Bernoulli rates.
* **`cc.evals.dependence_benchmark`** – benchmark ingestion and Paper 1 example
  summaries using the `Z_i=1` failure convention.

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

* State the binary event convention before interpreting any bound.
* Keep product-coupling calculations labeled as baselines, not default truth.
* Pair every reported sharp interval with endpoint witness checks.
* Use `make paper-smoke`, `make reproduce-paper`, and
  `make verify-paper-artifacts` for Paper 1 source and artifact checks.

## 4. Extending the Framework

### Adding a Guardrail

1. Implement class in `cc/guardrails/` inheriting from `BaseGuardrail`.
2. Register in experiment configs under `guardrails:`.

### Adding an Attacker

1. Implement attacker in `cc/core/attackers.py` or a new module.
2. Expose entry point through `cc.exp.run_two_world` or custom runner.

### Adding Paper-Facing Metrics

1. Define the estimand-level metric in `cc.kernel.metrics`.
2. Add focused unit tests under `tests/unit/kernel`.
3. Update `docs/theory/metric_taxonomy.md` and the paper artifact verifier if
   the metric appears in Paper 1 outputs.

## 5. Further Reading

* [Public Research Framing](research/public-framing.md)
* [CC Reports and Receipts](research/CC_REPORTS.md)
* [Experiments Guide](experiments-guide.md)
* [Reproducibility Notes](reproducibility.md)

This manual will evolve as the framework matures—contributions welcome!
