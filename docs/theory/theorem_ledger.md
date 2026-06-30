# Theorem Ledger

This ledger links paper-facing mathematical claims to implementation and test
witnesses. It records non-claims as part of the contract.

## Theorem-To-Test Summary

| Theorem / claim | Assumptions | Implementation | Tests | Artifact |
| --- | --- | --- | --- | --- |
| Finite atom LP sharpness | Finite binary atom space, linear constraints, nonempty feasible set | `src/cc/kernel/sensitivity.py` | `tests/unit/kernel/test_sensitivity.py` | `artifacts/paper/minimal_bounds.json` |
| FH recovery | Exact singleton marginals and no side constraints | `src/cc/kernel/frechet_classes.py`, `src/cc/kernel/sensitivity.py` | `tests/unit/kernel/test_classical_frechet_special_cases.py` | `artifacts/paper/table_1_classical_frechet_bounds.csv` |
| Witness verification | Optimal LP status and finite endpoint atom distributions | `IdentificationResult.lower_solution`, `IdentificationResult.upper_solution` | `tests/unit/kernel/test_sensitivity.py`, `tests/integration/test_verify_paper_artifacts.py` | `artifacts/paper/table_3_witness_verification.csv`, `artifacts/paper/minimal_witnesses.json` |
| Monotonic tightening with nonempty feasible refinement | Added constraints define a nonempty subset of the original feasible set | `AssumptionSet.with_*`, `identified_region` | `tests/unit/kernel/test_monotonic_tightening.py` | theorem ledger only |
| Finite-sample outer confidence interval | Fixed labels/query, iid Bernoulli counts from the target population, simultaneous moment coverage, true exact assumptions | `src/cc/kernel/sample_complexity.py` | `tests/unit/kernel/test_finite_sample_constraints.py` | `docs/theory/finite_sample_identification.md`, `artifacts/paper/table_4_sample_complexity.csv` |
| Receipt hash integrity as provenance only | Recorded bytes and canonical hash procedure match | `src/cc/reporting`, `src/cc/evidence` | `tests/unit/reporting/test_reporting.py`, `tests/unit/evidence/test_transparency_log_adversarial.py` | `examples/reporting/minimal_cc_report.json` |
| Product coupling baseline as non-theorem | Product coupling is explicitly chosen as a comparison baseline | `independent_event_probability`, `independence_regret` | `tests/unit/kernel/test_independent_probability_label_order.py` | `artifacts/paper/figure_2_independence_regret.png` |

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

## T6_FINITE_SAMPLE_OUTER_CONFIDENCE_INTERVAL

Statement: If the estimated Bernoulli moment intervals simultaneously cover
their true moments with probability at least `1 - alpha`, and every declared
exact assumption is true, then the atom-LP interval computed from exact
constraints plus those intervals contains the true target query with
probability at least `1 - alpha`.

Assumptions: finite binary atom space, fixed labels/events before sampling,
fixed target query before interval construction, iid Bernoulli samples from the
same named target population for each estimated moment, no uncorrected adaptive
post-selection, and no unmodeled label-noise claim.

Conclusion: the count-derived LP interval is an outer confidence interval for
the target composition risk under the stated assumptions.

Proof sketch: on the simultaneous coverage event, the true atom distribution
satisfies every interval constraint. If exact assumptions are true, the true
distribution is feasible for the relaxed LP. The LP lower endpoint is therefore
no larger than the true query value, and the upper endpoint is no smaller.

Proof status: finite-dimensional argument documented in
`docs/theory/finite_sample_identification.md`; computationally smoke-tested.

Implementation witness: `src/cc/kernel/sample_complexity.py`.

Test witness: `tests/unit/kernel/test_finite_sample_constraints.py`.

Failure mode / non-claim: this theorem does not certify deployment safety,
dataset representativeness, or validity after uncorrected adaptive target
selection.

## T7_AUDIT_CHAIN_NOT_STATISTICAL_VALIDITY

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

## T8_PRODUCT_COUPLING_BASELINE_NOT_TRUTH

Statement: Product coupling is a baseline calculation, not a theorem that the
guardrail failures are independent and not a default truth model.

Assumptions: singleton failure rates and a declared query are supplied, and the
caller explicitly asks for the product-coupling baseline comparison.

Conclusion: `independent_event_probability` and `independence_regret` are
diagnostics relative to the product baseline only.

Proof sketch: the product distribution is one constructed coupling from the
supplied marginals. Without an independence assumption, it need not be the true
joint law and need not lie at either identified endpoint.

Proof status: non-theorem / modeling-baseline warning.

Implementation witness: `src/cc/kernel/metrics.py`.

Test witness: `tests/unit/kernel/test_independent_probability_label_order.py`.

Failure mode / non-claim: product coupling baseline output is not evidence of
safety and is not used as a Paper Core theorem premise unless independence is
explicitly declared.
