# Non-Claims

This document is intentionally strict. It defines claims the project does not
make, why it does not make them, and what the project provides instead. Any
public description that implies one of these claims should be treated as out of
scope unless accompanied by new assumptions, evidence, review, and explicit
documentation.

The project does not claim:

- to prove AI systems are safe,
- to certify deployed models,
- to solve AI alignment,
- to infer causality without causal assumptions,
- to guarantee future performance,
- to eliminate distribution shift,
- to prove dataset representativeness,
- to make cryptographic receipts equivalent to truth,
- to replace red teaming,
- to replace human governance.

| Non-claim | Why the project does not claim it | What the project provides instead |
| --- | --- | --- |
| The project does not prove AI systems are safe. | The kernel reasons over declared binary events, constraints, and composition queries. A real system's safety depends on task scope, environment, users, model behavior, monitoring, governance, and failure modes not captured by a finite LP alone. | Sharp composition intervals conditional on supplied assumptions and evidence. |
| The project does not certify deployed models. | Certification requires an external standard, deployment context, acceptance criteria, operational controls, and accountable review. This repository does not define or operate that regime. | Mathematical diagnostics and audit artifacts that can inform review without replacing certification. |
| The project does not solve AI alignment. | Alignment is a broad scientific and governance problem. Bounding binary guardrail failures under unknown dependence is a narrower evaluation task. | A disciplined method for asking what composed guardrail evidence identifies and what it leaves unidentified. |
| The project does not infer causality without causal assumptions. | Dependence, overlap, and feasible joint laws are not causal effects. Causal interpretation requires interventions, exchangeability or design assumptions, and a defensible causal estimand. | Dependence-aware bounds and diagnostics that can coexist with causal analyses when those analyses are separately justified. |
| The project does not guarantee future performance. | Future deployments may differ in prompts, users, models, policies, tools, attacks, and monitoring. Static evidence does not determine future risk. | Reproducible artifacts and intervals for the evaluated setting, plus explicit assumptions that future use would need to revisit. |
| The project does not eliminate distribution shift. | Distribution shift is a property of the environment and sampling process, not something a composition-bound kernel can remove. | A way to report how composition conclusions depend on declared evidence and to avoid hiding uncertainty behind independence assumptions. |
| The project does not prove dataset representativeness. | Representativeness depends on sampling design, population definition, measurement quality, and external validity. The LP only uses the evidence it is given. | A clear separation between mathematical feasibility of bounds and empirical validity of the input data. |
| The project does not make cryptographic receipts equivalent to truth. | Hashes, Merkle proofs, and manifests can preserve byte identity and tamper evidence. They cannot prove that an experiment was well designed, that labels were correct, or that interpretation was valid. | Claim-bounded audit artifacts that make records inspectable and challengeable. |
| The project does not replace red teaming. | Red teaming searches for failures, threat-model gaps, and operational weaknesses. Static bounds over supplied evidence cannot discover every relevant failure mode. | A complementary analysis layer for quantifying dependence and composition risk in evidence produced by evaluations, including red-team exercises. |
| The project does not replace human governance. | Decisions about acceptable risk, deployment, user impact, legal obligations, and remediation require accountable human judgment. | Structured evidence, non-claims, and review-ready artifacts that help humans make narrower, better-audited decisions. |

## Evidence-Object Non-Claims

- A claim_decay artifact does not prove the system is currently safe; it defines
  when the claim should be rechecked, degraded, or expired.
- An extremal_scenario artifact does not prove the endpoint scenario is likely;
  it proves or records a feasible endpoint/fitted scenario under the stated
  assumptions.
- An exploratory red-team interval is not a confirmatory certificate unless
  validated by a separate confirmatory procedure.
- A receipt verifies artifact integrity, not statistical validity or deployment
  safety.
- A capsule that reproduces deterministically from its declared inputs does not
  establish that those inputs were measured rather than asserted. Changing an
  input and regenerating the chain yields a fully self-consistent capsule, a
  valid receipt, and a PASS verdict over a fabricated value. See
  `docs/research/epistemic-program/challenges.md` (CH-001).
- A scalar that happens to be expressible over a checked-in row count is not
  thereby derived from that matrix. The event, population, and numerator must
  be explicitly declared and checked before any derivation claim is made.
- A PASS verdict from `cc-report verify-claim-governance` means the
  evidence-bound claim package is internally consistent under the verifier
  rules. It does not mean the AI system is safe in deployment.

Exploratory red-team discovery can find candidate dependence cliffs. It does not
by itself certify a confidence interval. Confirmatory failure-matrix evidence
must be generated separately.

## Verifier-Mandated Non-Claims

The claim-governance verifier uses a small v0 non-claims engine. It looks for
stable substance rather than exact strings, so wording may vary, but the package
must preserve these boundaries when the corresponding evidence appears:

| Evidence role | Required boundary |
| --- | --- |
| receipt | Integrity checks do not prove statistical validity or deployment safety. |
| `claim_decay` | Decay policy does not prove the system is currently safe; it defines when a claim should be rechecked, degraded, or expired. |
| `extremal_scenario` | Endpoint or fitted scenarios do not prove the scenario is likely; they record feasible, fitted, stressed, or confirmatory scenarios under stated assumptions. |
| exploratory red-team evidence | Exploratory intervals are not confirmatory certificates unless validated by a separate confirmatory procedure. |

## Enforcement Guidance

- If a result is conditional on assumptions, state those assumptions.
- If a witness distribution is shown, say it witnesses an LP endpoint, not real
  deployment behavior.
- If a receipt is verified, say the bytes and hashes are consistent, not that
  the underlying safety claim is correct.
- If a dataset is used, identify the evaluated population and avoid implying
  representativeness beyond that scope.
- If a composition interval is wide, do not hide the width behind a point
  estimate or an independence calculation.
