# Anytime-Valid Sequential Testing for Guardrail Misses

This note replaces the old ROPE/Bayes-factor stopping heuristic with an
anytime-valid e-process.  The goal is to allow continuous monitoring without
pre-committing a fixed sample size while still controlling false alarms.

## Null Hypothesis

Let `X_t in {0, 1}` be the monitored composed-system miss indicator.  In the
guardrail setting, `X_t = 1` means the composed guardrail stack failed to block a
harmful example.  The null is one-sided:

```text
H0: P(X_t = 1 | F_{t-1}) <= p0 for every t.
```

The benchmark `p0` must be fixed before monitoring, or estimated from data
independent of the monitored sequence.  A common two-rail OR-stack benchmark is
the worst-case-independent co-miss rate:

```text
p0 = p_A_miss * p_B_miss.
```

The implementation accepts `p0` directly because each deployment must define the
right baseline for its composition rule and threat model.

## E-Process Construction

For a predictable bet `lambda_t in [0, 1 / p0]`, define

```text
M_t = product_{i=1}^t (1 + lambda_i (X_i - p0)).
```

The factor is nonnegative.  Under `H0`,

```text
E[1 + lambda_t (X_t - p0) | F_{t-1}]
  = 1 + lambda_t (E[X_t | F_{t-1}] - p0)
  <= 1.
```

So `M_t` is a nonnegative supermartingale with `M_0 = 1`.  The code uses a
finite uniform mixture over constant betting fractions, `lambda = fraction / p0`.
A convex mixture of nonnegative supermartingales is still a nonnegative
supermartingale.  The reported wealth `E_t` is therefore an e-process.

## Stopping Rule

For Type-I level `alpha`, stop and reject `H0` when

```text
E_t >= 1 / alpha.
```

For `alpha = 0.05`, the rejection threshold is `20`.

## Theorem

**Ville's inequality.** If `(E_t)` is a nonnegative supermartingale with
`E_0 <= 1`, then for every `c > 0`,

```text
P_H0(sup_t E_t >= c) <= 1 / c.
```

Setting `c = 1 / alpha` gives

```text
P_H0(exists t: E_t >= 1 / alpha) <= alpha.
```

Equivalently, for every stopping time `tau` adapted to the observed data,

```text
P_H0(tau < infinity and E_tau >= 1 / alpha) <= alpha.
```

This is the anytime-valid guarantee: looking after every sample, stopping early,
or continuing after an inconclusive look does not inflate Type-I error.

## Assumptions

- Outcomes are binary miss indicators.
- The null benchmark `p0` is pre-registered or estimated from independent data.
- Bets are predictable; they cannot depend on the current or future outcome.
- Under the null, the conditional miss probability is bounded by `p0` at each
  monitored time.
- The monitored sequence matches the unit of inference.  If the protocol skips
  baseline-world samples and monitors only protected-world samples, that
  filtering rule must be fixed or predictable from the past.

## Implementation

The implementation lives in `src/cc/kernel/sequential.py`.

Key APIs:

- `AnytimeBernoulliTester(null_rate=p0, alpha=alpha)`
- `update(outcome)` and `update_many(outcomes)`
- `calibrate_false_stop_rate(...)`
- `simulate_power_curve(...)`

All simulation functions require an explicit `numpy.random.Generator`.  No
global random state is used by the new kernel.

## Legacy Heuristic

`BayesianSequentialTester` in `src/cc/core/protocol.py` is retained only for
backward compatibility.  It is behind the explicit
`--legacy-bayesian-heuristic` flag in the two-world runner and emits a runtime
warning because its ROPE stopping rule is not anytime-valid and has no
continuous-monitoring Type-I guarantee.
