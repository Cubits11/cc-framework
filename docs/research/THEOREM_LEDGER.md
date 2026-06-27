# CC-Framework Theorem Ledger

## Canonical Paper 1 Theorems
- Theorem 1: Two-Rail FH Envelope Sharpness
- Theorem 2: Dependence Coordinate Representation
- Theorem 3: Finite-Sample Identification and Affine Stress
- Theorem 4: Completeness of Interval Release Triage

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
- Guarantees are only as valid as the declared linear assumptions.
- This is a partial-identification interval, not a claim that the true deployed
  distribution is safe.

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
- “independence baseline” unless explicitly labeled as assumption
