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
of an explicit product-coupling baseline, and exposes endpoint witness
distributions for reproducible verification. Count-derived simultaneous
confidence intervals yield outer confidence intervals for the target
composition risk under stated sampling assumptions. The framework is
conditional on supplied evidence and assumptions; it does not establish
deployment safety.

## Contributions

1. Formalization of guardrail composition as finite partial identification.
2. Sharp LP bounds for Boolean composition events.
3. Dependence-aware diagnostics including FH width, FH position, and independence regret.
4. Endpoint witness distributions for reproducibility.
5. Claim-bounded artifact discipline separating mathematical verification from deployment safety.
6. Finite-sample identification theorem converting count-derived simultaneous
   Bernoulli intervals into outer LP confidence intervals under stated sampling
   assumptions.
7. Explicit scaling analysis distinguishing closed-form marginal-only AND/OR
   bounds from the general `2^m` atom-LP path.

## Paper-Core Import Surface

Paper-facing examples, artifact generators, and artifact verifiers should import
kernel symbols through `cc.kernel.strict`. The broader `cc.kernel` package
remains backward compatible for experimental and legacy workflows, but it is
not the paper-core claim boundary.

## Non-Contributions

- Not a deployment safety certificate.
- Not causal inference.
- Not evidence of dataset representativeness.
- Not a full sequential agent safety framework.
- Not a new copula theory.
- Not a new cryptographic truth system.

## Paper Outline

### 1. Introduction

- Motivate composed guardrail evaluation as a dependence problem.
- Show why singleton guardrail scores do not identify composed failure risk.
- State the finite partial-identification thesis.
- Preview sharp intervals, finite-sample outer confidence intervals, product-baseline diagnostics, and witnesses.

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
- State that product coupling is a baseline, not truth.

### 5. Witnesses and Reproducibility

- Define endpoint witness distributions.
- Show how witnesses reconstruct constraints and endpoint objectives.
- Describe assumption hashes, active constraint reporting, and per-case
  verification contexts that bind labels, atom order, query coefficients, constraints,
  assumptions hash, and numerical tolerance.
- Separate witness verification from empirical validity.

### 6. Experiments

- Include toy finite examples that recover classical special cases.
- Include dependence-sensitivity examples where singleton rates are fixed.
- Include one LlamaGuard + deterministic keyword worked-example scaffold,
  separating CI fixtures from pinned real-model evidence.
- Include a dependence-sensitivity toy demonstration only if labeled illustrative.
- Report runtime and atom-scaling limitations for the explicit LP.

### 7. Related Work

- Cover Frechet classes, optimal transport or coupling bounds, and partial identification.
- Cover reliability and common-cause failure where directly relevant.
- Cover guardrail and AI safety evaluation practices.
- Cover reproducibility and audit artifacts without claiming deployment approval.

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
| Table 1: Classical Frechet special cases | Show AND and OR bounds from marginals alone. | Implemented and verified in `artifacts/paper/table_1_classical_frechet_bounds.csv`. Generated by `scripts/reproduce_paper.py`; checked by `scripts/verify_paper_artifacts.py`. |
| Table 2: Metric taxonomy | Distinguish identified-set diagnostics, product-baseline diagnostics, and movement metrics. | Implemented and verified in `artifacts/paper/table_2_metric_examples.csv`. Generated by `scripts/reproduce_paper.py`; checked by `scripts/verify_paper_artifacts.py`. |
| Table 3: Witness verification | Demonstrate that endpoint witnesses satisfy constraints and achieve objectives. | Implemented and verified in `artifacts/paper/table_3_witness_verification.csv`, with detailed witnesses in `minimal_witnesses.json`. |
| Figure 1: Frechet interval visualization | Visualize `[L_\phi, U_\phi]` and dependence-driven uncertainty. | Implemented and hash-verified in `artifacts/paper/figure_1_fh_interval.png`. |
| Figure 2: Independence regret under product-coupling baseline | Show signed deviation from the product-coupling baseline. | Implemented and hash-verified in `artifacts/paper/figure_2_independence_regret.png`; product coupling is labeled as a baseline, not truth. |
| Figure 3: Dependence-sensitivity toy demonstration | Show dependence sensitivity in a controlled example. | Implemented as an illustrative toy artifact in `artifacts/paper/figure_3_correlation_cliff_toy.png`; not empirical deployment evidence. |
| Table 4: Finite-sample Bernoulli intervals | Show simultaneous Bernoulli-rate radii, sample sizes, and the outer-confidence construction assumptions. | Implemented and verified in `artifacts/paper/table_4_sample_complexity.csv`; theorem note in `docs/theory/finite_sample_identification.md`. |
| Table 5: Runtime scaling | Show closed-form versus atom-LP scaling. | Implemented and verified in `artifacts/paper/table_5_runtime_scaling.csv`. |
| Figure 4: Runtime scaling | Visualize explicit atom variables as guardrail count grows. | Implemented and hash-verified in `artifacts/paper/figure_4_runtime_scaling.png`. |

The artifact bundle also includes `minimal_bounds.json`,
`minimal_witnesses.json`, `minimal_bundle.json`, `environment.json`,
`benchmark_example_summary.json`, and `manifest.json`. The verifier checks
required filenames, schemas, file hashes, metrics, assumption hashes,
verification context fields, and LP endpoint witnesses. An artifact should be promoted into the
paper only when those checks pass.
