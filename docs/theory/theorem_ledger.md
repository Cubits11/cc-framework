# Theorem Ledger

This ledger links paper-facing mathematical claims to implementation and test
witnesses. It records non-claims as part of the contract.

## T1_SHARPNESS_FINITE_LP

Statement: For finite binary guardrail failures and any feasible linear
constraint set, the atom LP computes sharp lower and upper values for any
linear composition query.

Assumptions: finite binary atom space, deterministic atom ordering, linear
constraints, nonempty feasible set, and finite numerical tolerance.

Conclusion: every value between the LP endpoints is attained by some feasible
joint law.

Proof sketch: the feasible set is a compact convex polytope. The query is
linear, so LP optima attain endpoints. Convex mixtures of endpoint witnesses
attain intermediate values.

Proof status: proved by finite-dimensional convexity; computationally witnessed.

Implementation witness: `src/cc/kernel/sensitivity.py::identified_region`.

Test witness: `tests/unit/kernel/test_sensitivity.py`.

Failure mode / non-claim: this theorem does not account for finite-sample
uncertainty in estimated constraints.

## T2_CLASSICAL_FRECHET_SPECIAL_CASES

Statement: With exact singleton marginals and no side constraints, the atom LP
recovers the classical Fréchet bounds for AND and OR failures.

Assumptions: exact marginal probabilities `p_i` and no additional dependence
constraints.

Conclusion:

```text
L_and = max(0, sum_i p_i - (m - 1))
U_and = min_i p_i
L_or = max_i p_i
U_or = min(1, sum_i p_i)
```

Proof sketch: these are the classical Fréchet-Hoeffding inequalities for
finite Bernoulli events; the atom LP optimizes over the same coupling polytope.

Proof status: standard theorem; computationally witnessed for `m=2` and `m>2`.

Implementation witness: `src/cc/kernel/frechet_classes.py` and
`src/cc/kernel/sensitivity.py`.

Test witness: `tests/unit/kernel/test_classical_frechet_special_cases.py`.

Failure mode / non-claim: these formulas are not point estimates and do not
assume independence.

## T3_MONOTONIC_TIGHTENING_UNDER_FEASIBLE_REFINEMENT

Statement: If `F_2 subset F_1` are nonempty feasible sets, then every Boolean
composition query satisfies:

```text
L_phi(F_1) <= L_phi(F_2) <= U_phi(F_2) <= U_phi(F_1)
```

Assumptions: both feasible sets are nonempty and the second is a valid
refinement of the first.

Conclusion: valid side information cannot widen the identified interval.

Proof sketch: the lower endpoint is an infimum over a smaller set and the upper
endpoint is a supremum over a smaller set.

Proof status: proved by set inclusion; computationally witnessed.

Implementation witness: `src/cc/kernel/sensitivity.py::identified_region`.

Test witness: `tests/unit/kernel/test_monotonic_tightening.py`.

Failure mode / non-claim: inconsistent added constraints make the feasible set
empty; bounds are then undefined rather than tighter.

## T4_INFEASIBILITY_DETECTION

Statement: If the declared linear constraints admit no atom distribution, the
kernel raises an infeasibility error.

Assumptions: constraints are encoded as finite linear equalities or inequalities
over the atom simplex.

Conclusion: contradictory reported evidence is detected as an empty feasible
set.

Proof sketch: LP feasibility over the simplex is exact for the declared finite
linear constraint system up to numerical tolerance.

Proof status: computational witness.

Implementation witness: `src/cc/kernel/sensitivity.py::IdentificationInfeasibleError`.

Test witness: `tests/unit/kernel/test_sensitivity.py`.

Failure mode / non-claim: infeasibility diagnoses internal contradiction in the
declared constraints, not which upstream measurement caused it.

## T5_WITNESS_DISTRIBUTION_VERIFICATION

Statement: Endpoint solutions returned by the atom LP are witness
distributions for the reported lower and upper bounds.

Assumptions: solver status is optimal and returned vectors are finite atom
probability distributions.

Conclusion: witnesses reconstruct the declared constraints and achieve the
reported endpoint objective values.

Proof sketch: optimal LP solutions lie in the feasible set and optimize the
linear objective by construction.

Proof status: computational witness.

Implementation witness: `IdentificationResult.lower_solution` and
`IdentificationResult.upper_solution`.

Test witness: `tests/unit/kernel/test_sensitivity.py`.

Failure mode / non-claim: numerical witnesses are not proof of empirical
correctness of the input constraints.

## T6_AUDIT_CHAIN_NOT_STATISTICAL_VALIDITY

Statement: Hash-chain or Merkle verification is orthogonal to statistical
validity.

Assumptions: an evidence or audit bundle contains serialized records plus a
cryptographic digest chain.

Conclusion: digest verification can provide tamper evidence for the recorded
bytes, but it does not prove that the data, sampling protocol, constraints, or
interpretation are correct.

Proof sketch: cryptographic integrity is a statement about byte identity and
append-only consistency; statistical validity depends on external sampling and
measurement assumptions.

Proof status: conceptual non-claim.

Implementation witness: existing evidence modules
`src/cc/evidence/merkle_log.py` and `src/cc/evidence/anchoring.py`.

Test witness: `tests/unit/evidence/test_transparency_log_adversarial.py`.

Failure mode / non-claim: this sprint does not implement new receipts, signing,
or paper reproduction artifacts.
