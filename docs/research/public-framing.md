# Public Research Framing

CC-Framework should be described as a research program in dependence-aware
statistical identification for composed AI safety systems.

The project does not claim to discover new mathematical fields. It combines
existing literatures, including Frechet-Hoeffding bounds, copula theory,
partial identification, reliability and common-cause failure, conformal or
anytime-valid uncertainty, probabilistic verification, and AI safety
evaluation.

The central claim is narrower and stronger:

```text
Composed AI safety should be studied as a problem of statistical dependence
under partial identification.
```

## Core Thesis

Modern AI systems often compose multiple safety mechanisms: input filters,
output filters, policy classifiers, retrieval filters, refusal policies, judge
models, tool-permission systems, and trajectory monitors. These mechanisms are
usually evaluated one at a time, then interpreted as redundant when layered.

That interpretation is mathematically unsafe unless the joint distribution of
their failures is measured, bounded, or explicitly assumed.

Let `Z_i = 1` mean that guardrail `i` fails, misses, or permits an unsafe pass.
The object of interest is not only the marginal failure probability
`P(Z_i = 1)`, but the joint law over failure patterns:

```text
pi(z) = P(Z_1 = z_1, ..., Z_m = z_m),    z in {0,1}^m.
```

In realistic evaluations, `pi` is rarely identified. We may observe marginals
and sometimes pairwise dependence evidence, but not the full coupling. This
naturally leads to a Frechet-class or partial-identification view: compute the
set of joint distributions consistent with the evidence, then bound the risk of
a Boolean composition event.

## First-Paper Core

The strongest publishable subset is intentionally small:

1. Frechet cartography for composed guardrail failure.
2. Independence regret against a product-coupling baseline.
3. Endpoint witness distributions for reproducible bounds.
4. Correlation cliffs as dependence-driven composition-risk changes.
5. Claim-bounded audit receipts that separate evidence integrity from safety
   validity.

The crisp paper thesis is:

```text
We introduce a partial-identification framework for composed AI guardrail
evaluation under unknown dependence. Given marginal and optional pairwise
evidence, we compute sharp finite-atom Frechet bounds for Boolean composition
events, quantify independence error, and provide endpoint witness
distributions for reproducible verification.
```

## Research Directions

| Direction | Foundation | Contribution status |
|---|---|---|
| Common-cause guardrail failures | Reliability engineering and latent common-cause failure | AI-safety adaptation and diagnostics |
| Frechet cartography | Frechet classes and finite linear programming | Immediate paper core |
| Independence regret | Product-coupling model misspecification | Simple diagnostic framing |
| Correlation cliffs | Dependence sensitivity and copula tail behavior | Proposed term; needs definitions and experiments |
| Witness distributions | Linear programming certificates | Reproducibility protocol |
| Guardrail portfolio optimization | Risk and reliability portfolio selection | Follow-up optimization paper |
| Adversarial dependence amplification | Adversarial evaluation | Multi-layer attack objective |
| Sequential dependence bounds | Probabilistic verification and temporal risk | Harder future theory |
| Semantic fault-line analysis | Subgroup and slice diagnostics | Needs multiple-testing controls |
| Non-theatrical safety receipts | Auditability and provenance | Engineering and governance contribution |

These are research directions, not named kingdoms or established subfields.
They become research contributions when formalized into definitions, algorithms,
benchmarks, theorems, falsifiable experiments, and reproducible artifacts.

## Public Language

Use:

- dependence-aware guardrail composition
- partial identification
- finite-atom Frechet bounds
- sharp composition intervals
- product-coupling baseline
- independence regret
- endpoint witness distributions
- claim-bounded audit receipts

Avoid:

- truth engine
- truth infrastructure
- proves safety
- certifies AI
- solves alignment
- discovers hidden reality
- new field unless the statement is carefully qualified

## Non-Claims

CC-Framework does not prove that an AI system is safe. It shows that many
safety-composition claims are underidentified unless dependence is measured or
bounded.

Endpoint witnesses demonstrate mathematical feasibility relative to supplied
constraints. They do not certify dataset representativeness, causal validity,
future deployment safety, semantic correctness, or absence of unmeasured
confounding.

Audit receipts bind claims to evidence, code, assumptions, and computation
traces. They do not turn evidence integrity into statistical validity.
