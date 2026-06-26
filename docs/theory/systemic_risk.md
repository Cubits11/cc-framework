# Systemic Risk Stress Tests for Guardrail Stacks

This note documents `cc.kernel.stress`, which adds a financial-risk-style
stress test for composed guardrails.  The analogy is portfolio risk under
correlation breakdown: assets that look only moderately dependent in ordinary
conditions can become much more dependent in crisis, and the damaging movement
is often asymmetric in the tail.  For guardrails, the corresponding object is
not asset return correlation but dependence among failure events.

Let

```text
F_i = event that guardrail i fails on a demand
q_i = P(F_i)
P_sys = P(F_1 and ... and F_m).
```

The baseline dependence structure is a joint law over the finite atom space
`{0, 1}^m` with fixed marginals `q_i`.  A stress test perturbs that joint law
while preserving the marginals:

```text
maximize      P_Q(F_1 and ... and F_m)
subject to    Q has marginals q_i
              distance(Q, Q_baseline) <= epsilon.
```

The implemented distances are:

- `metric="wasserstein"`: one-Wasserstein distance on binary atoms with
  normalized Hamming ground cost.
- `metric="kl"`: regularized relative entropy between the stressed and baseline
  atom laws.

The output is therefore a local worst-case composed failure probability at a
declared stress budget.  It is not the unconstrained Frechet-Hoeffding (FH)
upper bound unless the budget is infinite.

## Why Not Report Only FH?

The FH envelope is still the assumption-free benchmark:

```text
max(0, sum_i q_i - (m - 1)) <= P_sys <= min_i q_i.
```

It answers: "What is possible under any coupling with these marginals?"  That
is valuable but deliberately global.  A crisis stress test answers a different
question: "How bad can the composed stack get if the dependence structure moves
within a bounded neighborhood of the estimated baseline?"

`stress_test(..., StressBudget(float("inf")))` returns the FH upper endpoint as
the budget-to-infinity limiting case.  Finite budgets solve the constrained
problem above and report the remaining gap to that FH limit.

## CoVaR-Style Conditional Failure

Finance often asks for risk conditional on another institution being in distress
(`CoVaR`).  The guardrail analogue is:

```text
P(F_1 and ... and F_m | F_x).
```

`composed_risk_given_guardrail_X_failure(...)` reports this conditional
composed failure risk, along with the drop in effective protection:

```text
unconditional protection = 1 - P_sys
conditional protection   = 1 - P_sys|F_x
protection drop          = P_sys|F_x - P_sys.
```

For an all-rails-fail top event, this conditional risk is especially direct:
once `F_x` is known, it measures how often the remaining stack also fails under
the same dependence law.  If a stress budget is supplied, the measure is
computed on the worst-case stressed law returned by `stress_test`.

## Analogy to Financial Risk

The useful carry-over assumptions are:

- Dependence is part of the risk model, not a nuisance parameter.
- Baseline dependence estimated in calm conditions should not be treated as
  stable under stress.
- Tail dependence matters more than average correlation for rare composed
  failures.
- A bounded stress neighborhood is more informative than jumping directly to
  the worst mathematically possible coupling.
- Conditional systemic risk can be more operationally meaningful than
  unconditional risk when one layer has already failed.

The analogy breaks down in important ways:

- Guardrails do not have continuous returns.  The kernel works with finite
  Bernoulli failure atoms, so "copula" here means a fixed-marginal dependence
  law over binary events or an empirical binary copula, not a continuous market
  return distribution.
- Guardrail dependence may be adversarially induced.  Co-failure can arise
  because an attacker learns to target shared blind spots, not because of a
  naturally recurring market factor.
- There is no liquid market price for guardrail dependence.  Stress budgets are
  modeling choices or audit knobs, not calibrated option-implied quantities.
- Failure semantics are system-specific.  "All rails fail" is appropriate for a
  one-out-of-`m` protection stack, while other deployments may use different top
  events.
- Historical samples can be highly nonstationary.  A baseline dependence law
  estimated from old prompts may understate future adaptive attack dependence.

Despite those limits, the portfolio analogy is useful because it forces the
right question: not "what is the point estimate?" but "how fragile is the stack
to dependence moving in the bad direction?"

## Implementation Contract

`src/cc/kernel/stress.py` exposes:

- `BaselineDependence`: accepts an atom distribution, binary samples, or for two
  guardrails a pairwise dependence plus marginals.
- `StressBudget`: records the budget amount, metric, event, and solver
  tolerances.
- `stress_test(baseline_dependence, stress_budget)`: returns the worst reachable
  composed risk, realized distance, stressed atom law, baseline risk, and FH
  limit.
- `composed_risk_given_guardrail_X_failure(...)`: returns the CoVaR-style
  conditional composed risk and protection drop.

Pairwise-only baselines identify a unique binary copula for two guardrails.  For
larger stacks, pairwise moments do not generally identify the full joint law, so
the stress kernel requires an atom distribution or binary samples instead of
silently inventing higher-order dependence.

## Worked Case Study

The reproducible case study is script-based:

```text
PYTHONPATH=src .venv/bin/python scripts/generate_systemic_risk_case_study.py
```

It creates an exact synthetic two-guardrail dataset with:

```text
P(F_1) = 0.20
P(F_2) = 0.20
P(F_1 and F_2) = 0.08.
```

It compares:

- the empirical baseline copula,
- the module-4 beta-factor CCF point estimate fitted to the same co-failure
  rate,
- three finite Wasserstein stress budgets,
- the unconstrained FH upper limit.

Generated artifacts:

- [systemic_risk_case_study.csv](figures/systemic_risk_case_study.csv)
- [systemic_risk_case_study.md](figures/systemic_risk_case_study.md)
- [systemic_risk_case_study.png](figures/systemic_risk_case_study.png)

The generated comparison table is:

| Method | Budget | Risk | Protection | Increase | Gap to FH |
| --- | --- | --- | --- | --- | --- |
| Empirical copula baseline | 0.00 | 0.0800 | 0.9200 | 0.0000 | 0.1200 |
| CCF beta-factor point estimate | - | 0.0800 | 0.9200 | 0.0000 | 0.1200 |
| Budget-constrained stress eps=0.02 | 0.02 | 0.1000 | 0.9000 | 0.0200 | 0.1000 |
| Budget-constrained stress eps=0.06 | 0.06 | 0.1400 | 0.8600 | 0.0600 | 0.0600 |
| Budget-constrained stress eps=0.10 | 0.10 | 0.1800 | 0.8200 | 0.1000 | 0.0200 |
| Unconstrained FH upper limit | infinity | 0.2000 | 0.8000 | 0.1200 | 0.0000 |
