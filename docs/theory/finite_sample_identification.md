# Finite-Sample Identification

This note states the paper-core finite-sample claim for the strict kernel. It is
about finite binary guardrail failure events and count-derived interval
constraints. It is not a deployment safety certificate and it does not make a
representativeness claim outside the named evaluation distribution.

## Formal Setting

Let `Z = (Z_1, ..., Z_m) in {0,1}^m`, where `Z_i = 1` means guardrail `i`
failed or allowed the named unsafe-pass event. The unknown joint law is the atom
probability vector `p` over `{0,1}^m`; in code this is the atom simplex used by
`cc.kernel.sensitivity.AssumptionSet`.

A target composition query is fixed before interval construction:

```text
q(p) = c^T p
```

where `c` is the coefficient vector in a `LinearQuery`. Exact assumptions are
linear equalities or inequalities in an `AssumptionSet` and must be true for
the theorem to apply. Estimated Bernoulli moments, such as singleton rates
`P(Z_i=1)` or pairwise rates `P(Z_i=1, Z_j=1)`, are converted into interval
constraints:

```text
ell_r <= a_r^T p <= u_r
```

The relaxed feasible set is the atom simplex intersected with all declared
exact constraints and all count-derived interval constraints. The strict kernel
then computes the LP interval:

```text
L = min c^T p
U = max c^T p
```

over that relaxed feasible set.

## Sampling Assumptions

Finite-sample claims require all of the following assumptions to be stated:

- The guardrail labels, binary events, estimated moments, and target query were
  fixed before sampling and before interval construction.
- The observations used for each estimated Bernoulli moment are iid draws from
  the same named target population, or a separately stated bounded or
  martingale-valid alternative is used. The current helper implements the iid
  Hoeffding route.
- Pairwise and singleton counts may use different sample sizes, but each count
  must estimate the moment named by its constraint.
- No adaptive post-selection of the target query is allowed unless the run is
  explicitly marked exploratory or uses a split-sample or multiplicity-corrected
  construction.
- Label noise is not modeled by the current theorem.
- Policy caps and exact constraints are modeling assumptions. They are not
  empirical estimates and do not receive confidence radii.

## Theorem

Assume every exact constraint in the declared `AssumptionSet` is true for the
target population. Assume the count-derived intervals cover all of their true
Bernoulli moments simultaneously with probability at least `1 - alpha` under
the stated sampling assumptions. Then, with probability at least `1 - alpha`,
the LP interval `[L, U]` computed from the exact constraints plus those
interval constraints contains the true target composition risk `q(p_true)`.
Equivalently, the count-derived strict-kernel interval is an outer confidence
set for the target query, conditional on the declared exact assumptions.

## Proof Sketch

On the simultaneous coverage event, every true estimated moment lies inside its
reported interval. If the exact assumptions are also true, the true atom law
`p_true` satisfies every constraint in the relaxed LP feasible set. Therefore
the LP minimum is no larger than `q(p_true)` and the LP maximum is no smaller
than `q(p_true)`. Thus `q(p_true) in [L, U]` whenever the simultaneous coverage
event occurs, so the interval has coverage at least `1 - alpha`.

## Non-Claims

- The interval is not a proof that the guardrail stack is safe.
- The interval is not representative outside the named sampling or evaluation
  distribution.
- The interval is not valid after uncorrected adaptive target selection.
- The interval is not a deployment safety certificate or regulatory claim.
- Product coupling is a baseline calculation only, not the true joint law.

## Practical Notes

- Wider intervals are the correct output under weaker assumptions or less
  information.
- Exact constraints are preserved exactly; count-derived constraints become
  intervals with method metadata.
- Pairwise counts can have a different `n` from singleton counts.
- Finite-sample intervals can become infeasible when exact assumptions or
  policy caps conflict with the data-derived intervals. In that case the LP
  should raise infeasibility rather than silently repair the inputs.
