# Two-World Analysis Plan

Experiment: `exp_1782533466`
Status: `completed`

## Estimand

`tau = E_P[Y_i(1) - Y_i(0)]`

Average effect, on the prompt population under evaluation, of composing guardrail B onto a system already running guardrail A.

- World 0: A-only baseline system
- World 1: A+B composed system
- Outcome: Attack success / harmful pass-through indicator as encoded by AttackResult.success
- Scale: risk difference; positive values mean higher success in world 1

## Identifying Assumptions

- Consistency: observed outcome equals the potential outcome for the assigned world.
- Exchangeability: world assignment is randomized or conditionally ignorable.
- Positivity: every evaluated prompt type has positive probability of both worlds.
- Non-interference across prompts: one prompt's assignment does not affect another prompt's outcome.
- Independent clusters: prompt/attack clusters are independent draws; arbitrary within-cluster correlation is allowed.
- Stable measurement: outcome definition and guardrail semantics are identical across worlds except for adding B.

## Pre-Specified Analysis

- Alpha: 0.05
- Confidence level: 0.95
- Causal estimator: cluster_bootstrap_ate
- Cluster variable: AttackResult.attack_strategy, falling back to strategy_type/unknown
- Bootstrap reps: 2000
- Bootstrap seed: 123
- Sequential test: disabled_fixed_sample
- Safe null rate: 0.65
- Planned sessions: 150 to 150

## Worlds

- World 0: (no description); guards=[]
- World 1: (no description); guards=['toy_threshold']

## Actual Run

- Actual sessions: 150
- Causal method: cluster_bootstrap_unavailable

## Deviations

- causal estimator method differed from plan: cluster_bootstrap_unavailable
