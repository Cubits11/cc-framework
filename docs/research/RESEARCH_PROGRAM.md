# Research Program

This document defines the publication-facing research spine for CC-Framework. It
separates the implemented mathematical kernel from future research directions
and states the non-claims that should constrain public descriptions of the
project.

## Central Thesis

"Composed AI safety should be evaluated as a dependence-aware partial-identification problem, not as a product of independent guardrail scores."

Let `G_1, ..., G_m` denote a finite collection of guardrails. For each
guardrail, define a binary failure outcome `Z_i`, with the repository-wide
convention:

```text
Z_i = 1 means guardrail failure / unsafe pass.
```

The latent object is the unknown joint law

```text
\pi(z) = P(Z_1 = z_1, ..., Z_m = z_m),  z in {0, 1}^m.
```

Typical evaluation evidence identifies singleton marginals

```text
p_i = P(Z_i = 1)
```

and may identify optional pairwise constraints

```text
q_{ij} = P(Z_i = 1, Z_j = 1)
```

or intervals for those quantities. A composed safety question is represented as
a Boolean composition event `\phi(Z)`, such as all selected guardrails failing,
at least one selected guardrail failing, or a declared custom event over the
finite atom space.

Given the supplied assumptions and evidence, the kernel computes the sharp
identified interval

```text
[L_\phi, U_\phi]
```

over all joint laws `\pi` consistent with those inputs. The interval is a
statement about what is mathematically identified by the declared assumptions.
It is not a statement that the declared assumptions are empirically correct.

## Immediate Publishable Core

The first paper should be limited to the implemented, finite, binary
partial-identification setting:

- finite-atom Frechet/LP composition bounds,
- independence regret relative to an explicit product-coupling baseline,
- endpoint witness distributions returned by the LP,
- the correlation-cliff phenomenon as dependence sensitivity,
- claim-bounded reproducibility and non-theatrical receipts as discipline, not
  proof of safety.

The current kernel does not prove deployment safety. It computes sharp
composition intervals conditional on supplied assumptions and evidence.

The paper-sized contribution is the disciplined translation from composed
guardrail evaluation to finite partial identification: state the event, state
the evidence, solve the atom LP, expose endpoint witnesses, and report how far
an independence baseline can be from feasible dependence-aware conclusions.

## Ten Research Directions

### 1. Common-Cause Failure Analysis for AI Guardrails

- Name: Common-Cause Failure Analysis for AI Guardrails
- Status: Partially supported
- Formal object: A latent or observed factor `C` that induces dependence among
  guardrail failure variables `Z_1, ..., Z_m`, changing the feasible set for
  `\pi(z)` or the empirical distribution of failure atoms.
- Research question: When do nominally distinct guardrails fail because they
  share prompts, model blind spots, policy ambiguities, retrieval artifacts, or
  evaluation-set construction artifacts?
- What current CC kernel supports: Finite atom bounds, pairwise side
  constraints, endpoint witnesses, and common-cause model prototypes can expose
  whether observed marginal evidence is consistent with high-overlap failure.
- What is missing: A validated empirical design for discovering common causes,
  estimating their prevalence, and distinguishing common-cause structure from
  finite-sample coincidence.
- Possible paper contribution: A guardrail-specific adaptation of
  common-cause-failure diagnostics that maps latent failure modes to
  dependence-tightened composition intervals.
- Non-claim: The current kernel does not discover latent causes or establish
  causal structure by itself.

### 2. Frechet Cartography for Composed Safety Systems

- Name: Frechet Cartography for Composed Safety Systems
- Status: Implemented core
- Formal object: The feasible Frechet class of finite binary joint laws
  satisfying declared marginal, pairwise, and linear constraints.
- Research question: How large is the identified set for a composed guardrail
  event when dependence is unknown or only partially constrained?
- What current CC kernel supports: Exact finite-atom LP bounds for linear
  composition queries, classical Frechet special cases, arbitrary linear
  constraints, infeasibility detection, and endpoint solutions.
- What is missing: A complete publication artifact set with canonical examples,
  tables, runtime characterization, and paper-ready verification outputs.
- Possible paper contribution: A finite, auditable map from guardrail evidence
  to sharp composition intervals.
- Non-claim: Wide Frechet intervals are not model failures; they are evidence
  that the composition risk is underidentified under the supplied assumptions.

### 3. Independence Regret

- Name: Independence Regret
- Status: Implemented core
- Formal object: For a composition event `\phi`, the signed difference between
  an observed or selected event risk and the product-coupling baseline
  `r_ind = E_{\pi_ind}[\phi(Z)]`.
- Research question: How wrong can a product-of-independent-guardrails
  interpretation be when the joint law is not identified?
- What current CC kernel supports: Product-coupling event probability and
  `independence_regret` for validated finite `LinearQuery` objects with
  explicit labels.
- What is missing: Paper-grade examples that distinguish signed regret from
  interval width and from causal effect measures.
- Possible paper contribution: A simple diagnostic vocabulary for the error
  introduced by treating independent guardrail scores as a composition model.
- Non-claim: Independence regret is not a universal safety metric and does not
  show which dependence structure will occur in deployment.

### 4. Correlation Cliffs

- Name: Correlation Cliffs
- Status: Partially supported
- Formal object: A region in dependence-parameter or constraint space where a
  small change in dependence evidence produces a large movement in a composed
  event risk, interval endpoint, or diagnostic.
- Research question: Which guardrail compositions are stable under dependence
  perturbation, and which become fragile near feasible-set boundaries?
- What current CC kernel supports: Dependence-sensitive LP intervals, FH width,
  FH position, independence regret, and experimental scripts for correlation
  cliff demonstrations.
- What is missing: A final formal definition, robustness criteria, and
  empirical protocols that separate numerical artifacts from substantive
  dependence sensitivity.
- Possible paper contribution: A toy and empirical demonstration that
  dependence can induce abrupt changes in composition conclusions even when
  singleton rates are stable.
- Non-claim: A correlation cliff is not proof of an attack or of real-world
  deployment failure without an empirical link to the evaluated system.

### 5. Witness Distributions for Safety Bounds

- Name: Witness Distributions for Safety Bounds
- Status: Implemented core
- Formal object: Endpoint atom distributions `\pi_L` and `\pi_U` that satisfy
  the declared constraints and achieve `L_\phi` and `U_\phi`.
- Research question: Can reported sharp bounds be made reproducible and
  challengeable by exposing concrete feasible distributions at the endpoints?
- What current CC kernel supports: `IdentificationResult.lower_solution` and
  `IdentificationResult.upper_solution`, active constraints, assumption hashes,
  and classical Frechet distributions when requested.
- What is missing: A paper-facing witness verification table and stable
  artifact schema for saving endpoint witnesses across experiments.
- Possible paper contribution: A reproducibility protocol in which every sharp
  interval is accompanied by endpoint witnesses.
- Non-claim: Endpoint witnesses verify the mathematical LP result, not the
  representativeness or correctness of the evidence used as input.

### 6. Guardrail Portfolio Optimization Under Dependence Uncertainty

- Name: Guardrail Portfolio Optimization Under Dependence Uncertainty
- Status: Future research only
- Formal object: An optimization problem over candidate guardrail sets,
  constraints, costs, and robust composition-risk objectives.
- Research question: Which subset of guardrails should be selected when
  marginal performance, cost, and unknown dependence all matter?
- What current CC kernel supports: Evaluation of a declared portfolio once its
  guardrails, event, and constraints are supplied.
- What is missing: Decision variables, cost models, robustness criteria,
  computational scaling strategy, and empirical validation.
- Possible paper contribution: A robust portfolio-selection layer that chooses
  guardrails by worst-case or partially identified composition risk.
- Non-claim: The current repository does not optimize guardrail deployments or
  recommend production portfolios.

### 7. Adversarial Dependence Amplification

- Name: Adversarial Dependence Amplification
- Status: Future research only
- Formal object: An adversary or stress process that changes the dependence
  structure of `Z` while possibly holding some singleton rates approximately
  fixed.
- Research question: Can adversarial prompt selection increase shared failure
  overlap enough to invalidate conclusions drawn from independent or weakly
  dependent evaluations?
- What current CC kernel supports: Scoring declared dependence scenarios and
  bounding composition risk under supplied constraints.
- What is missing: A formal attack objective, search procedure, threat model,
  dataset controls, and tests for distinguishing amplification from sampling
  noise.
- Possible paper contribution: An adversarial evaluation protocol that searches
  for high-overlap failure atoms rather than only high singleton failure rates.
- Non-claim: Existing LP bounds do not generate adversarial examples or prove
  that an attacker can realize a worst-case coupling.

### 8. Sequential Dependence Bounds for Agentic Systems

- Name: Sequential Dependence Bounds for Agentic Systems
- Status: Future research only
- Formal object: A stochastic process `(Z_t, A_t, H_t)` over agent histories,
  actions, tool calls, and time-indexed guardrail failures.
- Research question: How should dependence-aware bounds extend when failures
  accumulate across trajectories rather than a single finite batch of binary
  outcomes?
- What current CC kernel supports: Static finite-atom reasoning and limited
  sequential utilities that should not be treated as a full agentic dependence
  theory.
- What is missing: Temporal estimands, filtration assumptions, stopping rules,
  path-dependent composition events, and scalable trajectory-level witnesses.
- Possible paper contribution: A sequential partial-identification framework
  for trajectory safety claims under unknown dependence.
- Non-claim: The current kernel is not a full sequential agent safety
  framework.

### 9. Semantic Fault-Line Analysis

- Name: Semantic Fault-Line Analysis
- Status: Future research only
- Formal object: A partition, embedding-derived slice, or semantic subgroup
  index `S` such that `P(\phi(Z) | S=s)` may differ sharply across slices.
- Research question: Where do aggregate composition bounds hide concentrated
  failure modes across semantic or population-defined subgroups?
- What current CC kernel supports: Conditional analysis if a subgroup is
  already declared and the corresponding marginal or joint evidence is supplied.
- What is missing: Slice discovery, multiplicity control, semantic validation,
  dataset governance, and subgroup-specific uncertainty procedures.
- Possible paper contribution: A statistically controlled method for finding
  semantic slices where dependence-aware composition risk is unusually high.
- Non-claim: Current aggregate bounds do not prove subgroup safety.

### 10. Non-Theatrical Safety Receipts

- Name: Non-Theatrical Safety Receipts
- Status: Partially supported
- Formal object: A claim-bounded artifact bundle linking assumptions,
  evidence, code identity, LP outputs, witness distributions, hashes, and human
  review status.
- Research question: How can safety-evaluation artifacts be made inspectable
  without turning auditability into a deployment-safety claim?
- What current CC kernel supports: Assumption hashes, endpoint witnesses,
  Merkle transparency primitives, evidence-bundle concepts, and draft assurance
  case schemas that mark claims for human review.
- What is missing: A manuscript-integrated reproduction workflow, stable
  receipt schema, verifier workflow, and governance process for accepting or
  rejecting claims. The paper-core artifact pipeline is implemented separately
  through `make reproduce-paper` and `make verify-paper-artifacts`.
- Possible paper contribution: A claim-bounded reproducibility discipline that
  ties mathematical outputs to audit artifacts while preserving non-claims.
- Non-claim: Receipts can support integrity and review; they do not make the
  recorded claim true or certify deployment safety.

## Implemented vs Future Boundary Table

| Concept | Mathematical foundation | Implemented now? | Requires new theory? | Requires new experiments? | Non-claim |
| --- | --- | --- | --- | --- | --- |
| Finite binary atom space | Probability simplex over `{0,1}^m` | Yes | No | No | Binary reduction may omit score-level behavior. |
| Frechet LP bounds | Linear programming over feasible couplings | Yes, implemented core | No for finite case | Paper examples still needed | Sharp conditional bounds are not deployment guarantees. |
| Classical Frechet special cases | Frechet-Hoeffding inequalities | Yes | No | No | Classical envelopes do not assume independence. |
| Marginal constraints `p_i` | Linear moment constraints | Yes | No | Empirical estimation protocols still needed | The kernel does not prove marginals are representative. |
| Pairwise constraints `q_{ij}` | Linear pairwise moment constraints | Yes, including intervals in the sensitivity layer | No for supplied constraints | Yes for estimating reliable pairwise evidence | Pairwise evidence does not identify the full joint law in general. |
| Boolean composition event `\phi` | Linear event functional over atoms | Yes | No | Examples and reporting discipline still needed | A declared event must match the system being evaluated. |
| Endpoint witness distributions | LP optimal solutions | Yes | No | Yes for saved artifact verification | Witnesses verify feasibility, not empirical correctness. |
| FH width and FH position | Identified-set diagnostics | Yes | No | Examples still needed | They are diagnostics, not causal effects. |
| Product-coupling baseline | Independent Bernoulli coupling | Yes | No | Examples still needed | Product coupling is a baseline, not a default truth model. |
| Independence regret | Difference from product-coupling event risk | Yes | No | Examples still needed | Regret does not identify the deployed joint law. |
| Correlation cliffs | Dependence sensitivity of event risk or endpoints | Partially supported | Yes, final definition needed | Yes | A cliff demonstration is not proof of realized harm. |
| Common-cause failure analysis | Latent common-cause dependence | Partially supported | Yes | Yes | The kernel does not infer causality. |
| Guardrail portfolio optimization | Robust optimization under dependence uncertainty | No | Yes | Yes | No current deployment recommendations are implied. |
| Adversarial dependence amplification | Threat-model-driven dependence search | No | Yes | Yes | Worst-case couplings are not automatically realizable attacks. |
| Sequential dependence bounds | Time-indexed partial identification | No, future research | Yes | Yes | Static bounds are not trajectory-safety guarantees. |
| Semantic fault-line analysis | Conditional or slice-specific identified sets | No, future research | Yes | Yes | Aggregate intervals are not subgroup guarantees. |
| Claim-bounded safety receipts | Cryptographic integrity plus claim review | Partially: evidence primitives and draft assurance schemas exist | Governance theory may be needed | Yes | Integrity artifacts do not prove safety or statistical validity. |
| Deployment safety certificate | External certification regime | No | Yes | Yes | The project does not certify deployed models. |

## Public Language Guide

Use:

- dependence-aware guardrail composition
- partial identification
- finite-atom Frechet bounds
- sharp composition intervals
- product-coupling baseline
- independence regret
- endpoint witness distributions
- claim-bounded audit artifacts

Avoid:

- proves safety
- certifies AI
- truth engine
- solves alignment
- magical kingdoms
- spiritual truth infrastructure
- guaranteed safe deployment
