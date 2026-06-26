# Correlation Cliffs as Tail-Dependence Cliffs

This note makes the "correlation cliff" claim precise.  The cliff is not a
large Pearson correlation.  It is a rare-event co-failure transition governed by
copula tail-dependence coefficients.

## Setup

Let `M_A` and `M_B` be miss or failure scores for two guardrails.  After applying
their marginal distribution functions, write

```text
U = F_A(M_A),        V = F_B(M_B),
```

so `(U, V)` has uniform margins and copula `C`.  If larger scores are worse, a
joint extreme co-failure is an upper-tail event:

```text
P(U > 1 - epsilon, V > 1 - epsilon).
```

If smaller scores are worse, use the lower-tail event:

```text
P(U <= epsilon, V <= epsilon) = C(epsilon, epsilon).
```

The copula tail-dependence coefficients are

```text
lambda_L = lim_{q -> 0+} P(V <= q | U <= q)
         = lim_{q -> 0+} C(q, q) / q,

lambda_U = lim_{q -> 1-} P(V > q | U > q)
         = lim_{q -> 1-} (1 - 2q + C(q, q)) / (1 - q).
```

Both lie in `[0, 1]`.  A positive coefficient means that once guardrail `A` is in
an extreme miss state, guardrail `B` remains extreme with non-vanishing
probability even as the miss marginal becomes rarer.

## Definition

Fix the relevant tail side `s in {L, U}` and let

```text
p_comp(epsilon) = P(two guardrails miss together at marginal miss rate epsilon).
```

For lower-tail misses, `p_comp(epsilon) = C(epsilon, epsilon)`.  For upper-tail
misses, `p_comp(epsilon) = 1 - 2(1 - epsilon) + C(1 - epsilon, 1 - epsilon)`.

The guardrail pair is:

```text
asymptotically sub-critical    if lambda_s = 0,
asymptotically tail-coupled    if lambda_s > 0.
```

Equivalently:

```text
lambda_s = 0  =>  p_comp(epsilon) = o(epsilon),
lambda_s > 0  =>  p_comp(epsilon) ~ lambda_s * epsilon.
```

An operational cliff is defined relative to a pre-registered critical value
`lambda_crit in (0, 1)`: the pair crosses the cliff when the supported value of
`max(lambda_L, lambda_U)` crosses `lambda_crit`.  This is the regime used by
`cc.kernel.cliff.cliff_certificate`:

```text
sub-critical    CI upper endpoint < lambda_crit
super-critical  CI lower endpoint > lambda_crit
critical        bootstrap CI intersects lambda_crit
```

This makes the statement falsifiable: more data can move the bootstrap interval
entirely below or above the threshold.

## Why Pearson Correlation Is Not Enough

Pearson correlation averages linear association over the whole distribution.
Extreme co-failure is controlled by the limiting conditional probability in the
tail.  These can disagree sharply.

For example, a Gaussian copula with correlation `rho = 0.9` has strong average
association but

```text
lambda_L = lambda_U = 0        for every |rho| < 1.
```

Thus high Pearson correlation may create finite-threshold overlap, but it does
not imply asymptotic tail co-failure.  Conversely, a Student-t copula can have
the same linear correlation as a Gaussian copula while having strictly positive
tail dependence because the shared radial shock is heavy-tailed.  The co-failure
question is therefore a copula-tail question, not an average-correlation
question.

## Worked Copula Examples

### Gaussian Copula

For the bivariate Gaussian copula with correlation `rho`,

```text
lambda_L = lambda_U = 0,       |rho| < 1.
```

Only the degenerate comonotonic limit `rho = 1` has
`lambda_L = lambda_U = 1`.  This family is the canonical "false cliff" example:
finite rare-event probabilities can grow quickly as `rho` approaches one, but
the asymptotic tail coefficient remains zero for every non-degenerate Gaussian
copula.

### Clayton Copula

For `theta > 0`,

```text
C_theta(u, v) = (u^{-theta} + v^{-theta} - 1)^{-1/theta},

lambda_L = 2^{-1/theta},
lambda_U = 0.
```

With `theta = 2`,

```text
lambda_L = 2^{-1/2} = 0.7071,
lambda_U = 0.
```

Clayton is lower-tail coupled: it is appropriate when low-score extremes encode
misses or when variables have been transformed so failures sit in the lower
tail.

### Gumbel Copula

For `theta >= 1`,

```text
C_theta(u, v)
  = exp(-(((-log u)^theta + (-log v)^theta)^{1/theta})),

lambda_L = 0,
lambda_U = 2 - 2^{1/theta}.
```

With `theta = 2`,

```text
lambda_U = 2 - sqrt(2) = 0.5858,
lambda_L = 0.
```

Gumbel is upper-tail coupled: it is the natural example when large scores mean
both guardrails are in their extreme failure state.

### Student-t Copula

For a bivariate Student-t copula with correlation `rho` and degrees of freedom
`nu`,

```text
lambda_L = lambda_U
         = 2 * t_{nu+1}(
             -sqrt(((nu + 1) * (1 - rho)) / (1 + rho))
           ),
```

where `t_{nu+1}` is the univariate Student-t CDF.  With `nu = 4` and `rho = 0.5`,

```text
lambda_L = lambda_U ~= 0.2532.
```

As `nu -> infinity`, the Student-t copula approaches the Gaussian copula and the
tail coefficient goes to zero.  At finite `nu`, the shared heavy-tailed radial
shock creates symmetric upper and lower co-failure risk.

## Implementation Contract

`src/cc/kernel/cliff.py` exposes:

- `estimate_tail_dependence(joint_samples)`: rank-transforms bivariate samples
  to pseudo-observations and estimates finite-threshold lower and upper tail
  dependence, with optional percentile bootstrap CIs.
- `fit_copula_family(marginal_samples, joint_samples)`: fits Gaussian, Clayton,
  Gumbel, and Student-t copulas by pseudo-likelihood and ranks candidates by AIC
  or BIC.
- `cliff_certificate(estimate, ci)`: turns a tail-dependence estimate and
  bootstrap CI into a falsifiable regime statement.

The empirical threshold estimator is useful for direct data summaries.  For
asymptotic claims under a named family, use `fit_copula_family` and the selected
family's closed-form `lambda_L` and `lambda_U`.

## Reproducible Simulation

The controlled simulation suite is script-based, not notebook-based:

```text
PYTHONPATH=src .venv/bin/python scripts/generate_correlation_cliff_copula_simulation.py
```

It sweeps dependence parameters for Gaussian, Clayton, Gumbel, and Student-t
copulas, estimates composed miss probability for rare guardrail-pair failures,
computes finite-difference derivatives, and writes a reproducible CSV plus a
figure under `docs/theory/figures/`.
