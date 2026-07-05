# Public API Contract

This document defines the v0.1 public package contract for the boundary-managed
repository. Importable code outside this contract may remain useful for research,
backward compatibility, examples, and provenance, but it is not promoted merely
because it ships in the wheel.

## Stable API

`cc.kernel.strict` is the stable paper-facing API. It is the supported import
surface for finite binary atom bounds, Frechet helpers, canonical metrics,
endpoint witnesses, infeasibility detection, and finite-sample helper routines.
Paper-facing examples, artifact generators, and artifact verifiers should import
kernel symbols through this module.

The supported stable kernel symbols are the names exported by
`cc.kernel.strict.__all__`, including `AssumptionSet`, `LinearConstraint`,
`LinearQuery`, `identified_region`, `frechet_bounds`, `fh_width`,
`independence_regret`, `independent_event_probability`, endpoint witness result
types, and finite-sample count-to-constraint helpers.

The evidence/reporting surface is stable only where it is used for release
evidence and claim-governance capsules:

- `cc.evidence` exported claim-envelope, role-ontology, decay, extremal-scenario,
  and confirmatory-protocol models and validators.
- `cc.reporting.cli` command behavior for `build-report` and
  `verify-claim-governance` with deterministic local inputs.

Those surfaces support replayable evidence records and governance verification.
They do not certify deployed systems, prove evidence validity, or make live
cloud claims.

## Importable But Experimental Or Backcompat

These modules may be present in the wheel and may be imported by older examples,
tests, or exploratory workflows, but they do not carry the stable compatibility
guarantees of `cc.kernel.strict`:

- `cc.kernel`: broad compatibility aggregate, including experimental causal,
  cliff, CCF-model, sequential, and stress modules.
- `cc.core`: protocol, legacy workflow, metrics, manifest, and model support.
- `cc.evidence`: package internals outside the named stable exported governance
  models and validators above.
- `cc.evals`: benchmark ingestion and local evaluation helpers.
- `cc.cartographer`: older atlas, audit, reporting, and methods workflow tools.
- `cc.reporting`: report builders and receipt helpers outside the named stable
  CLI behavior above.
- `cc.analysis`: figure, reporting, and historical analysis helpers.
- `cc.exp`: experiment runners and two-world workflow support.
- `cc.guardrails`: local toy guardrail implementations.
- `cc.io`, `cc.cli`, and `cc.utils`: storage, manifest, plotting, dashboard, and
  utility helpers.
- `cc._legacy`: retained only for backward compatibility and migration context.

Experimental/backcompat APIs may change without compatibility guarantees unless
a release note explicitly promotes a symbol into the stable contract.

## Reference Or Demo Only

These surfaces are reference, demo, or integration material rather than the core
package contract:

- `cc.enterprise` and `src/cc/enterprise`: moto-backed AWS evidence-integrity
  reference code, not a live AWS or production-readiness claim.
- Dashboard helpers and `apps/dashboard`: visual/reporting demo surface, not the
  research kernel.
- `cc.adapters` and vendor adapter code: interoperability references unless a
  release note promotes a specific adapter contract.
- `cc.redteam`: exploratory dependence-search and red-team utilities.
- `infra` and deployment integration material: reference infrastructure only.

## Not Part Of The Public Package Contract

The following repository contents are outside the public Python package contract
even when they are retained for provenance, reproducibility, or review:

- `notebooks`
- `experiments`
- generated `runs`, `results`, `checkpoints`, and dashboard build outputs
- deployment leftovers and local infrastructure output directories
- visual identity assets and demo media
- raw paper build outputs unless explicitly included in a release artifact
  manifest
- local virtual environments, dependency caches, and build directories

## Deprecation Policy

Stable APIs require a changelog or release-note entry before removal or
incompatible behavior changes. When practical, stable removals should include a
deprecation period or a documented replacement path.

Experimental and backcompat APIs may change without compatibility guarantees, but
release notes should call out changes that affect documented examples,
deterministic artifacts, or migration paths.

Public claims must stay evidence-bound. Documentation should state the
assumptions, inputs, and verification lane that support a claim, and should not
turn hashes, receipts, dashboards, adapters, or cloud references into safety,
validity, compliance, or deployment-readiness claims.
