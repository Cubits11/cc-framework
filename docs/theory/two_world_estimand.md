# Two-World Incremental Guardrail Estimand

## Setup

The two-world protocol evaluates the incremental effect of composing guardrail
`B` onto a system that is already running guardrail `A`.

For prompt or attack unit `i`, let:

* `W_i = 0` mean the unit is evaluated in the baseline world: system plus
  guardrail `A`.
* `W_i = 1` mean the unit is evaluated in the composed world: system plus
  guardrails `A+B`.
* `Y_i(w)` be the potential outcome that would be observed for unit `i` under
  world `w`.

In the current protocol, `Y_i(w)` is the binary attack-success or harmful
pass-through indicator encoded by `AttackResult.success`. If lower attack
success is better, a negative effect means composition reduced harmful
pass-through.

Clusters are prompt batches, attack strategies, templates, users, or other
shared-generation groups. Let `G_i` denote the cluster for unit `i`. Units may
be arbitrarily correlated within a cluster, but clusters are treated as
independent sampling units for inference.

## Estimand

The primary estimand is the population average treatment effect over the prompt
population under evaluation:

```text
tau = E_P[Y_i(1) - Y_i(0)]
```

Equivalently, for a binary attack-success outcome:

```text
tau = Pr_P(Y_i(1) = 1) - Pr_P(Y_i(0) = 1)
```

This is the causal risk difference for adding `B` to a system already protected
by `A`. It is not the isolated effect of `A`, not the isolated effect of `B`,
and not a claim about prompts outside the declared evaluation population.

## Estimator

The point estimate is the difference in observed means:

```text
hat_tau = mean(Y_i | W_i = 1) - mean(Y_i | W_i = 0)
```

Uncertainty is estimated by a cluster bootstrap:

1. Resample clusters `G` with replacement.
2. Keep all observations within each selected cluster.
3. Recompute `hat_tau`.
4. Use the bootstrap distribution for the standard error and confidence
   interval.

This is the pre-specified default because prompt batches and attack templates
can induce within-cluster dependence that invalidates independent-observation
standard errors.

## Identifying Assumptions

Identification of `tau` from the two-world design requires:

1. **Consistency.** If unit `i` is assigned to world `w`, the observed outcome
   equals `Y_i(w)`.
2. **Exchangeability of world assignment.** Conditional on the declared design,
   `W_i` is independent of `(Y_i(0), Y_i(1))`. This is satisfied by randomized
   world assignment; otherwise it must be justified by measured covariates.
3. **Positivity.** Each prompt type or relevant covariate stratum in the
   evaluation population has nonzero probability of evaluation under both
   worlds.
4. **Non-interference across prompts.** A prompt's potential outcome is not
   changed by another prompt's world assignment.
5. **Stable outcome measurement.** `AttackResult.success` has the same semantic
   meaning in both worlds; only adding `B` to `A` changes.
6. **Independent clusters for inference.** Clusters are independent draws from
   the evaluation process. Within a cluster, arbitrary correlation is allowed
   and handled by the cluster bootstrap.
7. **Fixed evaluation population.** The target population `P` is the prompt or
   attack distribution declared before analysis. Adaptive generation must be
   logged as part of that design rather than silently mixed into a different
   target population.

## Interpretation

`tau > 0` means the composed world has a higher attack-success rate than the
baseline world on the evaluation population. For a harmful pass-through outcome,
this is evidence that adding `B` made the system worse.

`tau < 0` means the composed world has a lower attack-success rate. For the same
outcome, this is evidence that adding `B` reduced harmful pass-through.

The confidence interval is a cluster-robust uncertainty statement over the
declared prompt population and clustering scheme. It is not a proof of
generalization to new prompt distributions, new guardrail versions, or new
attackers unless those are part of the pre-specified evaluation population.
