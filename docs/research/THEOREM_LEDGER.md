# CC-Framework Theorem Ledger

## Canonical Paper 1 Theorems
- Theorem 1: Finite Atom LP Sharpness
- Theorem 2: Classical FH Recovery
- Theorem 3: Finite-Sample Outer Confidence Interval
- Theorem 4: Endpoint Witness Verification

## Finite Binary Partial Identification by Atom LP

Status: implemented and unit-tested in the strict kernel.

Statement: For K binary guardrail events, represent the unknown joint law as a
probability vector over the `2^K` atom simplex. Any linear query over the joint
law has sharp identified bounds obtained by linear programming subject to
declared linear equality and inequality constraints. Adding valid constraints
can only narrow the identified interval. Infeasible constraints flag
inconsistent assumptions.

Assumptions:
- The event space is finite and binary.
- Guardrail names define a deterministic atom ordering.
- All caller-supplied assumptions are linear equalities or inequalities over
  the atom probability vector.
- The feasible set is nonempty unless infeasibility is explicitly raised.

Implementation file: `src/cc/kernel/sensitivity.py`

Test file: `tests/unit/kernel/test_sensitivity.py`

Limitations:
- Finite binary event spaces only.
- Computational cost grows as `O(2^K)` atoms, so this is intended for small K
  strict-kernel verification and theorem validation, not large-scale arbitrary
  guardrail ensembles without structure.
- Bounds are only as valid as the declared linear assumptions.
- This is a partial-identification interval, not a claim that the true deployed
  distribution is safe.

## Finite-Sample Outer Confidence Interval

Status: implemented and unit-tested in the strict kernel finite-sample helpers.

Statement: If count-derived Bernoulli intervals cover all true estimated
moments simultaneously with probability at least `1 - alpha`, and declared
exact assumptions are true, then the atom-LP interval computed from those
intervals contains the true target query with probability at least
`1 - alpha`.

Assumptions:
- The labels, failure events, estimated moments, and target query are fixed
  before sampling and interval construction.
- Estimated moments are iid Bernoulli samples from the same named target
  population, unless a separately justified alternative is used.
- No uncorrected adaptive target selection is used.
- Label noise is not modeled.
- Policy caps are assumptions, not empirical estimates.

Implementation file: `src/cc/kernel/sample_complexity.py`

Test file: `tests/unit/kernel/test_finite_sample_constraints.py`

Artifact / documentation: `docs/theory/finite_sample_identification.md` and
`artifacts/paper/table_4_sample_complexity.csv`

## Archived / Future Theorems
- Impossibility of a Single Scalar Metric
- Correlation Cliff Theorem
- N-Rail Junction-Tree FH Prototype
- Fragility / Slack Theorem
- Evidence Ledger Integrity Theorem

## Retired Language
- “certifies safety”
- “proves system is safe”
- “one score summarizes safety”
- “product coupling” unless explicitly labeled as a baseline
