# Strict Kernel Contract

## Purpose

This document defines the stable research-software contract for the strict kernel of CC Framework.

The goal is to prevent future cleanup work from silently changing scientific meaning while repairing typing, tests, packaging, and implementation details.

This contract is documentation-only. It does not change runtime behavior, source code, dependency groups, tests, or CI configuration.

## Kernel and non-kernel boundary

The strict kernel is the part of the repository whose behavior should be treated as scientifically meaningful and regression-sensitive.

Kernel code should include modules that define or compute:

- core CC metrics,
- Youden-style J quantities,
- Fréchet-Hoeffding bounds,
- dependence coordinates,
- confidence intervals and finite-sample envelopes,
- deterministic audit records,
- serialized result payloads consumed by downstream reports.

`src/cc/kernel/sensitivity.py` is strict-kernel pure math. It must remain free
of network calls, model API calls, dashboard dependencies, filesystem side
effects, AWS dependencies, and runtime guardrail invocation. It operates only
on declared finite binary event assumptions and linear queries.

Non-kernel code may include:

- exploratory scripts,
- one-off notebooks,
- temporary experiment runners,
- presentation/demo utilities,
- generated artifacts,
- local scratch outputs,
- legacy prototypes not used by the current strict path.

Future PRs should classify files before large typing or refactor work. If a file is part of the strict kernel, its semantics must be preserved or explicitly migrated.

## Metric naming contract

Metric names must not be reused for different meanings.

The following names should remain stable:

- `J`: a Youden-style quantity, usually true positive rate minus false positive rate.
- `J_combo`: the Youden-style quantity for the composed guardrail system.
- `CC`: a normalized compositional coefficient derived from composed performance and a declared denominator.
- `CC_max`: a maximum or stress-bound value only when the denominator and bound convention are explicitly stated.
- `FH bounds`: lower and upper Fréchet-Hoeffding feasible limits for a composed rate under fixed marginals.
- `lambda` or dependence coordinate: a normalized position inside the relevant FH interval.

A future code PR should not rename or reinterpret these quantities casually. If a metric meaning changes, the PR must update docs, tests, and serialized schema expectations together.

## CC denominator contract

Any CC-like metric must make its denominator explicit.

The denominator should answer:

- What is being normalized?
- Is the denominator based on single-rail J values, an FH width, a theoretical maximum, or another declared quantity?
- Can the denominator be zero?
- What happens when the denominator is zero or numerically unstable?

Future implementation work should avoid hidden denominator conventions. If two CC definitions are needed, they should use distinct names rather than overloading `CC`.

## Bounds return contract

Functions that compute bounds should return stable, documented shapes.

Acceptable future shapes include:

- a typed dataclass,
- a named tuple,
- a dictionary with documented keys,
- a Pydantic or schema-validated record if the project later adopts that pattern.

Bounds-returning functions should document:

- lower bound field,
- upper bound field,
- class label or condition,
- operating point or alpha cap,
- whether the bound is exact or an outer approximation,
- whether the value is empirical, analytic, or simulated.

Large typing repairs should not mix tuple ordering changes with unrelated fixes.

## Numeric policy

Kernel numeric code should define and preserve conventions for:

- floating-point tolerance,
- clipping probabilities to `[0, 1]`,
- handling NaN and infinity,
- zero-denominator behavior,
- confidence interval rounding,
- bootstrap seed behavior,
- deterministic versus stochastic calculations.

Numerical cleanup should prefer explicit helper functions over scattered local fixes.

## Serialization policy

Serialized outputs should be treated as contracts once consumed by reports, docs, dashboards, or audit tools.

Serializable kernel records should be:

- JSON-safe,
- deterministic where practical,
- explicit about metric names,
- explicit about seeds and configuration hashes when relevant,
- stable enough for regression tests.

Future PRs should avoid silently changing serialized field names or field meanings.

## Audit determinism policy

Audit records should support reproducibility.

Audit-producing code should record:

- Git SHA when available,
- config hash or configuration payload,
- random seed,
- command or runner identity when practical,
- input and output artifact paths when stable,
- metric payloads using the serialization policy above.

Audit logs should not depend on generated local paths unless those paths are intentionally part of the artifact contract.

## Generated artifact boundary

Generated outputs should not be treated as committed source unless a PR explicitly declares them as publication artifacts.

Normally excluded generated outputs include:

- `site/`,
- `results/`,
- `runs/`,
- `checkpoints/`,
- `audit_logs/`,
- regenerated paper figures,
- regenerated paper PDFs.

Documentation may describe how to generate these artifacts, but strict docs should not depend on missing generated files.

## Cleanup order

Future cleanup should proceed in this order:

1. keep strict docs building,
2. classify repo health and remediation tracks,
3. define strict-kernel contracts,
4. repair dependency extras,
5. restore pytest collection,
6. repair typing by subsystem,
7. refactor source code only after contracts are explicit.

This order reduces the chance that a typing or refactor PR accidentally changes research meaning.

## Non-goals

This document does not fix:

- mypy errors,
- pytest failures,
- dependency extras,
- runtime behavior,
- source-code typing,
- generated artifacts,
- paper figures,
- regenerated PDFs.

Those belong in separate follow-up PRs.
