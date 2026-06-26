# Common-Cause-Failure Point Models

This note documents `cc.kernel.ccf_models`, which adds classical
common-cause-failure (CCF) point models beside the assumption-free
Frechet-Hoeffding (FH) envelope in `cc.kernel.frechet_classes`.

Let `F_i` be the event that guardrail or component `i` fails during one demand
or mission interval, and let `q_i = P(F_i)`.  For a redundant one-out-of-`m`
success criterion, the composed system fails when all rails fail:

```text
P_sys = P(F_1 and ... and F_m).
```

With only the marginal probabilities known, the FH envelope is

```text
max(0, sum_i q_i - (m - 1)) <= P_sys <= min_i q_i.
```

The CCF models below replace that envelope with a point estimate by adding
strong exchangeability and common-cause structure.  The point estimate is useful
as a sensitivity analysis or as a fault-tree basic-event calculation, but it is
not assumption-free.

## Basic Event Notation

The classical CCF parameterizations assume a homogeneous common-cause component
group (CCCG): the rails are similar enough to share one total component failure
probability `Q_t`.  The code therefore accepts a vector of guardrail-level
failure probabilities but requires all entries to be equal within tolerance.

`Q_k^(m)` denotes the probability of a basic event involving a specific subset
of exactly `k` components in a group of size `m`.  For example, with three
components, `Q_2^(3)` is the probability of a specified pair failure such as
`AB`, not the probability of any pair failure.

For the one-out-of-`m` success criterion, `partition_failure_probability`
evaluates the standard top-event partition polynomial:

```text
B_0 = 1
B_n = sum_{k=1}^n C(n - 1, k - 1) Q_k B_{n-k}
P_sys = B_m.
```

For `m = 3`, this gives the familiar expression

```text
P_sys = Q_1^3 + 3 Q_1 Q_2 + Q_3.
```

## Assumption Table

| Model | Parameters | Structural assumption encoded | Basic-event formula used |
| --- | --- | --- | --- |
| Beta-factor | `Q_t`, `beta` | A constant fraction `beta` of each component failure probability is assigned to a common cause that fails the entire CCCG. All other failures are single-component independent-cause basic events. Intermediate-size CCF events are ruled out. | `Q_1 = (1 - beta) Q_t`; `Q_k = 0` for `1 < k < m`; `Q_m = beta Q_t`. |
| Multiple Greek Letter (MGL) | `Q_t`, `beta`, `gamma`, ... | Conditional sharing chain: given that a component failure is shared with at least one other component with probability `beta`, it is shared with additional components according to the subsequent Greek-letter conditional probabilities. Parameters are tied to the CCCG size. | Set `rho_1 = 1`, `rho_2 = beta`, `rho_3 = gamma`, ..., `rho_{m+1} = 0`; `Q_k = Q_t prod_{i=1}^k rho_i (1 - rho_{k+1}) / C(m - 1, k - 1)`. |
| Alpha-factor | `Q_t`, `alpha_1`, ..., `alpha_m` | Event-frequency ratios: `alpha_k` is the fraction of common-cause basic events in the CCCG that involve exactly `k` component failures. The conversion depends on the testing scheme. | Staggered: `Q_k = alpha_k Q_t / C(m - 1, k - 1)`. Non-staggered: `Q_k = k alpha_k Q_t / (C(m - 1, k - 1) alpha_t)`, where `alpha_t = sum_k k alpha_k`. |

## FH Consistency

A CCF point estimate is admissible only if it lies inside the FH envelope
computed from the same marginal failure probabilities.  Otherwise no Bernoulli
joint distribution with those marginals can have that all-fail probability; the
CCF parameter, testing scheme, or rare-event approximation is inconsistent with
the supplied guardrail rates.

For the homogeneous CCCG used by these models, the FH envelope simplifies to

```text
max(0, m Q_t - (m - 1)) <= P_sys <= Q_t.
```

The beta-factor, MGL, and alpha-factor conversions all satisfy the component
marginal identity

```text
Q_t = sum_{k=1}^m C(m - 1, k - 1) Q_k.
```

The partition recurrence chooses the block containing a distinguished component,
so the resulting `P_sys` is bounded above by `Q_t` whenever the partition
polynomial is a valid probability.  The lower FH endpoint is automatically zero
in the rare-failure regime

```text
Q_t <= (m - 1) / m.
```

Therefore, under valid parameters, a homogeneous CCCG, the matching testing
scheme, and `Q_t <= (m - 1) / m`, the CCF point estimate is guaranteed to fall
inside the FH envelope.  If `Q_t > (m - 1) / m`, the lower FH endpoint is
positive and the guarantee requires the additional inequality

```text
P_sys >= m Q_t - (m - 1).
```

`assert_within_fh_envelope(point_estimate, failure_rates)` implements this as a
hard assertion.  Regression tests call it for valid examples and also include a
deliberate high-probability beta-factor rare-event approximation that falls
below the FH lower bound.

## Model Recommendation

`recommend_model(evidence)` maps dependence evidence to a modeling strategy:

| Evidence available | Recommendation | Reason |
| --- | --- | --- |
| None beyond marginal failure rates | `FH-only` | CCF parameters are unidentified; report the assumption-free envelope. |
| Pairwise correlation only | `FH+CCF-point-estimate` | Pairwise dependence can tighten FH but does not identify higher-order common-cause structure. Use the CCF point estimate as explicit sensitivity analysis and run the FH consistency check. |
| Full historical co-failure counts | `full empirical estimation` | Estimate the joint law or basic-event probabilities directly, with uncertainty for sparse cells. Use CCF models as diagnostics or regularizers. |

## Published Regression Example

The golden test reproduces the three-component worked example in ReliaSoft
HotWire Issue 125, "The Parametric Models for Common Cause Failure Analysis",
which uses a group of three identical components where the system succeeds if
at least one component functions.  The example gives

```text
Q_t = 0.05
alpha = (0.980, 0.013, 0.007)
```

and, using the staggered alpha-factor equations,

```text
Q_1 = 0.049
Q_2 = 0.000325
Q_3 = 0.00035.
```

NUREG/CR-5485 Appendix E gives the same three-component top-event form:

```text
Q_s = Q_1^3 + 3 Q_1 Q_2 + Q_3.
```

The regression computes

```text
Q_s = 0.049^3 + 3 * 0.049 * 0.000325 + 0.00035
    = 0.000515424
R_s = 1 - Q_s = 0.999484576,
```

which reproduces the published displayed system reliability `0.9995`.

Sources:

- ReliaSoft, "The Parametric Models for Common Cause Failure Analysis",
  HotWire Issue 125:
  <https://help.reliasoft.com/articles/content/hotwire/issue125/hottopics125.htm>
- U.S. NRC, NUREG/CR-5485, "Guidelines on Modeling Common-Cause Failures in
  Probabilistic Risk Assessment":
  <https://nrcoe.inl.gov/publicdocs/CCF/NUREGCR-5485_Guidelines%20on%20Modeling%20Common-Cause%20Failures%20in%20PRA.pdf>
