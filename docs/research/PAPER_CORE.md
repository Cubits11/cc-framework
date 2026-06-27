# Paper Core

This document defines the first publishable paper and rejects scope creep. The
paper should be written as a focused mathematical and reproducibility
contribution, not as a product, platform, or deployment-safety claim.

## Working Title

"Sharp Composition Bounds for AI Guardrails Under Unknown Dependence"

Alternate titles:

- Dependence-Aware Guardrail Composition by Finite Partial Identification
- Frechet Bounds and Witnesses for Composed AI Safety Evaluation

## Paper Thesis

We introduce a partial-identification framework for composed AI guardrail
evaluation under unknown dependence. Given binary guardrail failure outcomes,
observable singleton marginals, optional pairwise or linear side constraints,
and a declared Boolean composition event, the method computes sharp finite-atom
composition intervals by linear programming, reports diagnostics for the error
of an independence baseline, and exposes endpoint witness distributions for
reproducible verification. The framework is conditional on supplied evidence
and assumptions; it does not establish deployment safety.

## Contributions

1. Formalization of guardrail composition as finite partial identification.
2. Sharp LP bounds for Boolean composition events.
3. Dependence-aware diagnostics including FH width, FH position, and independence regret.
4. Endpoint witness distributions for reproducibility.
5. Claim-bounded artifact discipline separating mathematical verification from deployment safety.

## Non-Contributions

- Not a deployment safety certificate.
- Not causal inference.
- Not a guarantee of dataset representativeness.
- Not a full sequential agent safety framework.
- Not a new copula theory.
- Not a new cryptographic truth system.

## Paper Outline

### 1. Introduction

- Motivate composed guardrail evaluation as a dependence problem.
- Show why singleton guardrail scores do not identify composed failure risk.
- State the finite partial-identification thesis.
- Preview sharp intervals, independence regret, and witnesses.

### 2. Problem Setup

- Define guardrails `G_1, ..., G_m` and binary failures `Z_i`.
- State the convention `Z_i = 1` for guardrail failure / unsafe pass.
- Define the unknown joint law `\pi(z)`, marginals `p_i`, optional pairwise
  constraints `q_{ij}`, and Boolean event `\phi(Z)`.
- Distinguish mathematical assumptions from empirical measurement assumptions.

### 3. Sharp Frechet Composition Bounds

- Present the finite atom simplex and linear constraint system.
- Define lower and upper bounds as LP optima for `E_\pi[\phi(Z)]`.
- Prove sharpness by compactness, linearity, and endpoint attainment.
- Recover classical Frechet special cases for AND and OR events.

### 4. Diagnostics

- Define FH width as identified-set size.
- Define FH position for observed or selected risks inside the interval.
- Define product-coupling event probability as an explicit baseline.
- Define independence regret and explain its signed interpretation.

### 5. Witnesses and Reproducibility

- Define endpoint witness distributions.
- Show how witnesses reconstruct constraints and endpoint objectives.
- Describe assumption hashes and active constraint reporting.
- Separate witness verification from empirical validity.

### 6. Experiments

- Include toy finite examples that recover classical special cases.
- Include dependence-sensitivity examples where singleton rates are fixed.
- Include a correlation-cliff toy demonstration if the artifact is stable.
- Report runtime and atom-scaling limitations for the explicit LP.

### 7. Related Work

- Cover Frechet classes, optimal transport or coupling bounds, and partial identification.
- Cover reliability and common-cause failure where directly relevant.
- Cover guardrail and AI safety evaluation practices.
- Cover reproducibility and audit artifacts without claiming certification.

### 8. Limitations

- State finite binary reduction limitations.
- State finite-sample and dataset-representativeness limitations.
- State that causal and sequential claims require additional assumptions.
- State computational scaling limits of explicit atom enumeration.

### 9. Conclusion

- Reiterate the dependence-aware partial-identification framing.
- Summarize sharp bounds, diagnostics, and witness artifacts.
- Identify future work without importing it into the first paper.
- Restate that the framework bounds claims rather than certifying systems.

## Required Figures/Tables

| Artifact | Purpose | Status |
| --- | --- | --- |
| Table 1: Classical Frechet special cases | Show AND and OR bounds from marginals alone. | Planned artifact |
| Table 2: Metric taxonomy | Distinguish identified-set diagnostics, product-baseline diagnostics, and movement metrics. | Planned artifact |
| Table 3: Witness verification | Demonstrate that endpoint witnesses satisfy constraints and achieve objectives. | Planned artifact |
| Figure 1: Frechet interval visualization | Visualize `[L_\phi, U_\phi]` and dependence-driven uncertainty. | Planned artifact |
| Figure 2: Independence regret under product-coupling baseline | Show signed deviation from the independence baseline. | Planned artifact |
| Figure 3: Correlation-cliff toy demonstration | Show abrupt dependence sensitivity in a controlled example. | Planned artifact |

Existing scripts and docs may support some of these artifacts, but the paper
should count an artifact as implemented only after it has a stable generation
command, deterministic inputs, saved output, and verification path.
