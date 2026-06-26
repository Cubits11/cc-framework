# Frechet Classes With Pairwise Dependence Side Information

This note documents `cc.kernel.frechet_classes`, the n-way replacement for the
pairwise-only guardrail-event Frechet-Hoeffding calculations.

## Binary Frechet Class

Let `A_1, ..., A_n` be binary guardrail events with indicators
`X_i = 1{A_i}` and marginal probabilities `p_i = P(X_i = 1)`.  A joint law on
`{0,1}^n` is a vector

```text
x = (x_w : w in {0,1}^n),        x_w >= 0,        sum_w x_w = 1.
```

The classical Frechet class fixes only the one-dimensional marginals:

```text
sum_w w_i x_w = p_i,             i = 1, ..., n.
```

For the conjunction and disjunction this gives the sharp n-way
Frechet-Hoeffding bounds:

```text
max(0, sum_i p_i - (n - 1)) <= P(all_i A_i) <= min_i p_i,

max_i p_i <= P(any_i A_i) <= min(1, sum_i p_i).
```

These are exactly the formulas used by the existing cartographer helpers when
there is no dependence information.

## Pairwise Side Information

If `P(A_i and A_j) = p_ij` is known for some pairs, each such fact adds another
linear equality:

```text
sum_w w_i w_j x_w = p_ij.
```

For binary indicators, Spearman's rho computed from midranks and Kendall's
tie-corrected tau-b both reduce to Pearson's phi coefficient:

```text
rho_ij = tau_ij = (p_ij - p_i p_j)
                  / sqrt(p_i (1 - p_i) p_j (1 - p_j)).
```

Therefore known `rho_ij` or tau-b is converted to:

```text
p_ij = p_i p_j
       + rho_ij sqrt(p_i (1 - p_i) p_j (1 - p_j)).
```

The implied `p_ij` must satisfy the bivariate Frechet interval

```text
max(0, p_i + p_j - 1) <= p_ij <= min(p_i, p_j).
```

Passing all constraints through the atom-vector representation makes the
improved bound an exact finite linear program:

```text
lower = min  f^T x
upper = max  f^T x

subject to:
  x >= 0
  1^T x = 1
  M x = m
  B x = b
```

Here `f_w = 1` when atom `w` satisfies the queried event and `0` otherwise,
`Mx=m` are the marginal constraints, and `Bx=b` are the known pairwise moments.
This is the finite Bernoulli version of the Frechet-class duality program
reviewed by Rueschendorf: optimize over all couplings with fixed marginals and
any additional prescribed moments.

## Monotonicity: Side Information Cannot Loosen Bounds

Let `S` be a set of constraints and let

```text
C_S = {x in Delta_{2^n} : x satisfies every constraint in S}
```

be the corresponding Frechet class.  Let `f(x)` be the event probability.  The
bound under `S` is

```text
L(S) = inf_{x in C_S} f(x),
U(S) = sup_{x in C_S} f(x).
```

Now add more valid side information, producing a larger constraint set `T` with
`S subset T`.  Every distribution satisfying all constraints in `T` also
satisfies all constraints in `S`, so

```text
C_T subset C_S.
```

Taking an infimum over a smaller feasible set can only increase the value:

```text
L(T) = inf_{x in C_T} f(x) >= inf_{x in C_S} f(x) = L(S).
```

Taking a supremum over a smaller feasible set can only decrease the value:

```text
U(T) = sup_{x in C_T} f(x) <= sup_{x in C_S} f(x) = U(S).
```

Thus each added valid side constraint either tightens the interval or leaves an
endpoint unchanged:

```text
[L(T), U(T)] subset [L(S), U(S)].
```

If the new constraints are inconsistent, then `C_T` is empty; the correct result
is not a wider interval but an infeasibility error.

## Numerical Tightness Checks

The script `scripts/benchmark_frechet_classes.py` performs two checks.

1. For `n in {2,3,4,5}`, it builds feasible random Bernoulli laws, extracts a
   partial set of pairwise Kendall/Spearman constraints, solves the analytic LP
   bounds, then samples observations from the lower- and upper-extremal feasible
   distributions.  The empirical event rates converge to the analytic
   endpoints as sample size grows.
2. It compares classical width against improved width on a grid of marginal
   probabilities using pairwise-independence side information.

The default output location is `docs/theory/figures/`:

```text
PYTHONPATH=src .venv/bin/python scripts/benchmark_frechet_classes.py
```

## References

- L. Rueschendorf, "Frechet-bounds and their applications," in *Advances in
  Probability Distributions with Given Marginals*, pp. 151-187, 1991:
  https://link.springer.com/chapter/10.1007/978-94-011-3466-8_9
- R. Fontana and P. Semeraro, "Characterization of multivariate Bernoulli
  distributions with given margins," arXiv:1706.01357:
  https://arxiv.org/pdf/1706.01357
