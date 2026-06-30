# Business and Audit Brief

CC Framework is a research prototype for evaluating composed AI guardrails as
an evidence, assumptions, and identification problem. Given declared binary
guardrail outcomes, where `Z_i = 1` means guardrail `i` failed or allowed an
unsafe pass, the framework uses measured singleton rates, optional dependence
constraints, and a stated Boolean composition event to compute the range of
composition risk consistent with that evidence. Its output is a claim-bounded
record: what was measured, what was assumed, what risk interval is identified,
what dependence remains unresolved, and what artifacts allow another reviewer
to inspect the run.

## Audit Analogy

Audit quality and guardrail evaluation both ask whether controls reduce risk
under measurable assumptions. An auditor would not conclude that a control
environment is effective solely because each control passed an isolated test;
the reviewer also asks about population, sampling, evidence quality, operating
context, complementary controls, common-cause failures, unresolved exceptions,
and whether the evidence supports the assertion being made. CC Framework applies
the same discipline to layered AI guardrails: individual guardrail scores do not
by themselves identify the risk of a composed system unless dependence among
failures is measured, bounded, or explicitly assumed.

## What The Framework Identifies

The framework identifies a narrow mathematical claim: the feasible interval for
a declared composition event under the supplied evidence and assumptions. It can
show, for example, that the probability of all guardrails failing at once is
bounded between lower and upper values given the observed marginal failure
rates. If additional pairwise or linear dependence evidence is supplied, the
interval may narrow. If dependence is not measured, the interval may remain
wide, and that width is itself audit-relevant evidence.

The framework also reports diagnostics that make assumptions visible. A
product-coupling or independence calculation may be shown as a baseline, but it
is not treated as truth. Endpoint witness distributions are used to demonstrate
that the reported lower and upper bounds are feasible relative to the declared
constraints. These witnesses verify the mathematics of the bound, not the
business validity of the dataset or the deployment decision.

## Enterprise Receipts

Enterprise receipts are deterministic evidence bundles for review. A receipt
binds the claim text, claim level, measurement outputs, calibration metadata,
assumptions, non-claims, audit artifacts, file hashes, and a canonical report
hash. This gives governance and audit reviewers a chain-of-custody style record:
they can check whether the reviewed files match the report and whether the
report's stated claim is narrower than the evidence.

A receipt is not an audit opinion. It does not prove labels are correct, the
sample is representative, the model is safe, the control is effective in
production, or a regulatory obligation has been met. It preserves inspectable
evidence and claim boundaries so that those judgments can be reviewed rather
than implied.

## Non-Claims

CC Framework does not prove deployed systems are safe, approve deployed models,
provide legal or regulatory compliance, guarantee future performance, prove
dataset representativeness, infer causality without causal assumptions, or make
cryptographic receipts equivalent to factual correctness. It does not replace
red teaming, human governance, acceptance criteria, or accountable risk
decisions.

## Concrete Example

Suppose an enterprise evaluates three guardrails on a fixed test set: an input
filter, an output filter, and a policy classifier. The observed miss rates are
2%, 3%, and 5%, and the business assertion under review is: "Layered controls
materially reduce the probability that an unsafe item passes all checks."

If failures were independent, the all-miss baseline would be:

```text
0.02 * 0.03 * 0.05 = 0.00003, or 0.003%
```

But independence is an assumption, not evidence. With only the three marginal
miss rates, the all-miss probability could still be as high as the smallest
miss rate, 2%, if the same cases defeat every guardrail. CC Framework would
report the identified interval under those inputs, record whether any dependence
evidence narrows it, and issue a receipt tying that claim to the run artifacts.

The reviewable conclusion is therefore limited: under the stated evaluation data
and assumptions, the all-miss risk lies in the reported interval. The framework
does not conclude that the deployed system is safe or that the control stack is
effective for future populations.
