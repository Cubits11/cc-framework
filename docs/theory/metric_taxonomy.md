# Formal Metric Taxonomy

This document defines the canonical metric language for the dependence-aware
guardrail composition kernel.

## Event Convention

The repo-wide convention is:

```text
Z_i = 1 means guardrail failure / unsafe pass.
```

All composition events are defined on the binary atom vector
`Z = (Z_1, ..., Z_m) in {0, 1}^m`.

- AND failure: all selected guardrails fail together.
- OR failure: at least one selected guardrail fails.
- k-of-m failure: at least `k` selected guardrails fail.
- Custom query: a validated `LinearQuery` over the atom ordering used by
  `cc.kernel.sensitivity`.

No metric in the canonical kernel silently inverts failure into success.

## Metric Categories

### Identified-Set Diagnostics

These are defined relative to the sharp identified interval
`[L_phi, U_phi]` returned by the atom LP.

`FH_width`

```text
FH_width = U_phi - L_phi
```

Domain: finite probabilities `0 <= L_phi <= U_phi <= 1`.

Undefined cases: invalid bounds raise `MetricDomainError`.

`FH_position`

```text
FH_position = (r_hat - L_phi) / (U_phi - L_phi)
```

Domain: finite probabilities, nonempty interval, and `r_hat` feasible inside
`[L_phi, U_phi]` up to tolerance.

Undefined cases: if `U_phi - L_phi <= tol`, return `None`. Values outside the
interval beyond tolerance raise `MetricDomainError`. Endpoint drift within
tolerance is snapped to the nearest endpoint, so returned values are always in
`[0, 1]`.

Interpretation: position inside this operator-specific identified interval. It
is not a causal parameter and not a universal dependence measure.

### Assumption-Comparison Diagnostics

These compare an observed or computed event risk to an extra modeling
assumption. They are not identified-set quantities.

`independent_event_probability`

For singleton failure rates `p_i`, product coupling is:

```text
pi_ind(z) = product_i p_i ** z_i * (1 - p_i) ** (1 - z_i)
r_ind = sum_z phi(z) * pi_ind(z)
```

Domain: explicit labels, a `LinearQuery` with dimension `2 ** len(labels)`, and
marginal keys that exactly match labels. There is no string parsing, no `eval`,
and no implicit atom ordering.

`independence_regret`

```text
Regret_ind = r_hat - r_ind
```

Domain: finite event probabilities in `[0, 1]`.

Interpretation: signed error made by using the product-coupling assumption for
the same composition event.

### One-World Normalization Diagnostics

`cc_gain`

```text
CC_gain = r_phi / max_i p_i
```

Domain: one-world composition failure risk `r_phi` and singleton failure risks
`p_i`. If `max_i p_i <= eps`, return `None`.

Interpretation warning: this is an operator-relative scale normalization. It is
not a causal effect, not a universal performance gain, and not meaningful
without specifying the composition operator.

### Two-World Movement Diagnostics

`cc_shift`

For baseline world `0` and deployed world `1`:

```text
composition_shift = r_phi^(1) - r_phi^(0)
singleton_shift_i = p_i^(1) - p_i^(0)
CC_shift = composition_shift / max_i abs(singleton_shift_i)
```

Domain: matching singleton labels or matching sequence positions across both
worlds. If all singleton shifts are `<= eps`, return `None`.

Interpretation: composed failure movement per largest singleton movement,
regardless of singleton direction. It is not defined in a one-world analysis.

## Deprecated Name Mapping

| Old name | Status | Canonical guidance |
| --- | --- | --- |
| `cc_max` | Deprecated; legacy formula preserved until v0.4. | Use `cc_gain` only when using failure-risk normalization with matching domain assumptions. |
| `cc_rel` | Deprecated; legacy formula preserved until v0.4. | Not canonical. Keep only for legacy report compatibility. |
| `delta_add` | Deprecated; legacy formula preserved until v0.4. | Use `independence_regret` only when comparing an event risk to an explicit product-coupling risk. |
| `delta_mult` | Deprecated; legacy formula preserved until v0.4. | No canonical replacement. |
