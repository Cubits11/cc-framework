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

Validation is split into explicit lanes in the
[Validation Matrix](validation_matrix.md). The main documentation track is
Paper Core v0.3; Enterprise Reference v0.1 is a separate evidence-integrity
reference architecture, not a deployment safety certification.

Generated outputs are governed by the
[Generated Artifact Boundary](release/ARTIFACT_BOUNDARY.md), which defines what
may be tracked as a release artifact, fixture, archive, or runtime-only output.
Source-surface disposition is tracked in the
[Repository Migration Manifest](release/MIGRATION_MANIFEST.md), and overlapping
audit/evidence ownership is summarized in
[Audit and Evidence Boundaries](architecture/AUDIT_EVIDENCE_BOUNDARIES.md).

## 1. Architectural Overview

* **`cc.kernel.strict`** – narrow Paper Core import surface for finite binary
  atom LPs, Frechet helpers, canonical metrics, sample-complexity utilities,
  interval constraint propagation, and endpoint witnesses.
* **`cc.kernel.sensitivity`**, **`cc.kernel.metrics`**,
  **`cc.kernel.frechet_classes`**, and **`cc.kernel.sample_complexity`** –
  implementation modules behind the strict paper-core surface.
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
* **Correlation cliffs**: characterize dependence-driven jumps in composed risk
  under declared models or measurements.
* **Claim-bounded receipts**: bind claims to evidence while separating evidence
  integrity from statistical validity.
* **Evidence-bound claims**: govern safety statements as scoped claims with
  boundaries, endpoint worlds, decay rules, receipts, non-claims, and review
  state.

## 3. Best Practices

* State the binary event convention before interpreting any bound.
* Keep product-coupling calculations labeled as baselines, not default truth.
* Pair every reported sharp interval with endpoint witness checks.
* Use `make paper-smoke`, `make reproduce-paper`, and
  `make verify-paper-artifacts` for Paper 1 source and artifact checks.
* Name the validation lane when reporting evidence, especially when optional
  dependencies caused enterprise, dashboard, vendor, serialization, experiment,
  or performance tests to skip.

## 4. Extending the Framework

### Adding a Guardrail

1. Implement class in `cc/guardrails/` inheriting from `BaseGuardrail`.
2. Register in experiment configs under `guardrails:`.

### Adding an Attacker

1. Implement attacker in `cc/core/attackers.py` or a new module.
2. Expose entry point through `cc.exp.run_two_world` or custom runner.

### Adding Paper-Facing Metrics

1. Define the estimand-level metric in `cc.kernel.metrics` and export it through
   `cc.kernel.strict` only if it belongs inside Paper Core.
2. Add focused unit tests under `tests/unit/kernel`.
3. Update `docs/theory/metric_taxonomy.md` and the paper artifact verifier if
   the metric appears in Paper 1 outputs.

## 5. Further Reading

* [Public Research Framing](research/public-framing.md)
* [Business and Audit Brief](research/business-audit-brief.md)
* [Evidence-Bound Claim Governance Memo](research/CLAIM_GOVERNANCE_OS.md)
* [Evidence Role Ontology](design-specs/evidence_role_ontology.md)
* [Validation Matrix](validation_matrix.md)
* [Generated Artifact Boundary](release/ARTIFACT_BOUNDARY.md)
* [Repository Migration Manifest](release/MIGRATION_MANIFEST.md)
* [Audit and Evidence Boundaries](architecture/AUDIT_EVIDENCE_BOUNDARIES.md)
* [CC Reports and Receipts](research/CC_REPORTS.md)
* [Experiments Guide](experiments-guide.md)
* [Reproducibility Notes](reproducibility.md)

This manual will evolve as the framework matures—contributions welcome!
