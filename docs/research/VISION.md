# CC-Framework — Research Vision

> **Status: vision (aspirational).** This document describes where the framework
> could go, not what it currently proves. Implemented status lives in
> [`RESEARCH_PROGRAM.md`](RESEARCH_PROGRAM.md), [`ROADMAP.md`](ROADMAP.md), and
> [`THEOREM_LEDGER.md`](THEOREM_LEDGER.md); the boundaries in
> [`NON_CLAIMS.md`](NON_CLAIMS.md) remain binding. **Nothing here is a capability
> claim.** Each pillar names the seed that exists today and the maximal form it
> could take; a pillar with no shipped kernel is a research target, not a result.

## Thesis

CC-Framework today is scoped as *a deterministic reporting primitive that bounds
the joint failure of composed binary guardrails under dependence uncertainty.*
The elevation is to make it **the measurement-and-certification calculus for
safety-under-composition**: the partial-identification engine that turns marginal
guardrail evidence into signed, replayable, worst-case-correlation-robust
joint-risk certificates — across filters (space) **and** across agent steps
(time) — and, from those bounds, designs the stack.

The framework already refuses to certify safety. The vision does not weaken that
refusal; it makes the honest object — the identified interval under adversarial
dependence — sharper, sourced, certified, and empirically grounded.

## Why now — the empirical anchor

A late-2025 result co-authored across OpenAI, Anthropic, and Google DeepMind
found adaptive attacks bypassing published defenses at success rates above ~90%,
against defenses first reported near 0%. Universal and transferable adversarial
suffixes generalize across prompts and across model families. The operational
reading is exact: **composed defenses fail together, not independently, and an
adversary induces the correlation on purpose.** The independence assumption that
most composition analysis silently ships is not merely imprecise — it is the
specific lie the state-of-the-art attack exploits.

That is CC-Framework's thesis, now validated by the frontier labs. It moves the
framework from "a mathematically neat idea" to "the required measurement calculus
for the current state of the attack."

## The object, restated

Model each guardrail failure (unsafe pass) as a binary variable `Z_i = 1`. Given
marginals `p_i = P(Z_i = 1)` and a set of linear constraints, the finite-atom LP
returns the sharp identified interval `[L_phi, U_phi]` for a target event `phi`
(e.g. joint compromise of a defense-in-depth stack), with endpoint witness
distributions. For two guardrails the joint-failure quadrant is bounded by the
Frechet-Hoeffding inequalities

```
max(0, p_A + p_B - 1)  <=  P(Z_A = 1, Z_B = 1)  <=  min(p_A, p_B)
```

against the independence point value `p_A * p_B`. The distance between the
independence value and the upper bound is `independence_regret` — the honest
measure of how badly the product assumption under-reports adversarial joint
failure. For a homogeneous stack of `m` filters each at rate `p`, independence
predicts `p^m` while the worst-case corner permits `min_i p_i = p`: **stacking
`m` filters can buy nothing under adversarial correlation.** The stronger each
filter looks alone (small `p`), the larger the relative regret (`~ 1 / max_i
p_i`). That is the correlation cliff, and it is the thing this framework exists
to bound.

---

## Seven pillars

Ordered by leverage. Pillars **I** and **V** are the immediate next deliverables
(see below); the rest sequence after them.

### Pillar I — Close the estimation loop *(IMMEDIATE)*

**Seed:** Wilson intervals on marginals and `sample_complexity.py` Hoeffding
helpers. **Maximal form:** the framework's own stated key limitation is that the
marginals `p_i` require justification independent of the composition analysis.
Own it. Propagate marginal (and pairwise-constraint) estimation uncertainty
*through* the finite-atom LP to produce a **confidence region for the identified
interval itself** at level `1 - alpha` — a bound on the bound. The honest object
is not `[L, U]` but a confidence band around `[L, U]`, and `independence_regret`
reported *with* a CI rather than as a point. Statistical inference for
partial-identification bounds under *estimated* marginals is thin in the
literature; this is a gap the framework can own. **First step:** a sample-
complexity statement for the width of the joint-failure interval as a function of
per-marginal `n`, and a two-stage (estimate -> bound) uncertainty pass wired into
the report.

### Pillar II — Signed composition certificates

**Seed:** claim envelopes, Merkle logs, receipts, endpoint witnesses. **Maximal
form:** every CC computation emits a canonical, signed, replayable *certificate*
carrying the query event `phi`, the input marginals with their CIs and
provenance, the constraint set, the interval `[L, U]`, and the **LP witness**
(the primal endpoint distribution achieving each bound plus the dual
multipliers). A skeptic recomputes `L` and `U` from the witness without trusting
the producer. Measurement becomes *auditable* measurement. **First step:** a
`cc.certificate.v1` schema binding a report to its witness distribution and dual
certificate, with an independent replay checker.

### Pillar III — The Grand Unification *(the crowning result)*

**Seed:** the Frechet-Hoeffding calculus already used here over filters, and the
same union bound used by the sibling Ghost-Ark semantic gate over trajectory
steps. **Maximal form:** these are the *same object* — `P(any-of-k failure under
unknown dependence)` — over different index sets. CC bounds the intersection
(all-fail) quadrant over **filters in space**; Ghost-Ark bounds the union
(any-step-fails) `min(1, sum p_i)` over **steps in time**. Unify them into one
calculus instantiated for (a) parallel filters, (b) sequential agent steps, and
(c) the product lattice **filters x steps** — a defense-in-depth stack evaluated
at every step of an agent trajectory, with correlation permitted both across
filters and across time. This is the mathematical statement of the joint research
program — *Verifiable Agent Governance under Correlated Guardrail Failure* — and
it is what makes CC the shared measurement spine of both repositories rather than
a sibling of one. **First step (theorem-ledger discipline):** state the
unification as a formal conjecture with its index-set abstraction and the exact
Boole-Frechet bounds it specializes to; prove it before it is claimed.

### Pillar IV — From measurement to design (ROC-aware minimax)

**Seed:** binary events at fixed thresholds; correlation-cliff studies.
**Maximal form:** lift each guardrail from a binary event to its ROC curve, then
solve for the **threshold vector that minimizes worst-case-correlation joint
failure** subject to a false-positive / latency / cost budget. CC becomes a
*design* calculus — how to stack and tune `m` filters for minimum
honestly-bounded joint risk — with the correlation cliff as a design constraint
(avoid the threshold region where dependence detonates the bound). **First
step:** a two-guardrail minimax threshold solver over supplied ROC samples.

### Pillar V — The Correlation Atlas *(IMMEDIATE)*

**Seed:** the two-setting diagnostics and `cc_shift`. **Maximal form:** theory
says how badly independence *could* lie; measure how badly it *does*. A versioned,
reproducible benchmark of **real co-failures** — open and production guardrails
(e.g. Llama Guard-class filters, prompt-shields, regex gates, LLM judges) run on
*shared* jailbreak / attack corpora — quantifying observed phi-correlation and
`independence_regret` in the wild. This converts the library from "here are the
bounds you could compute" into "here is how badly the product assumption
under-reports joint failure on real stacks, measured." It is the empirical
keystone and the most citable single artifact the program can produce. **First
step:** a minimal atlas over two or three guardrails on one public jailbreak
corpus, reporting per-pair marginals, observed joint rates, phi, and the
independence-vs-worst-case gap, each row emitting a Pillar II certificate.

### Pillar VI — Scale the identified set

**Seed:** the exact `2^m`-atom LP (sharp, exponential). **Maximal form:** a
hierarchy — exact LP (small `m`) -> marginal-polytope / Sherali-Adams
relaxations (medium `m`) -> factor-copula and vine structures (large `m`) —
adopting *improved* Frechet-Hoeffding bounds that incorporate partial model
information (distance-to-reference, low-dimensional marginals). Sharp where
feasible, certified-outer where necessary, at realistic stack sizes. **First
step:** a marginal-polytope outer relaxation with a witnessed duality gap versus
the exact LP on small instances.

### Pillar VII — Adversarial dependence steering

**Seed:** the worst-case (comonotone) corner the upper bound already reports.
**Maximal form:** make the adversary first-class — an attacker who *selects
inputs to drive the empirical dependence toward the Frechet-upper corner*, i.e.
the correlation cliff as an attack objective rather than an accident — together
with a defender's certificate that holds even under adversarial dependence
steering. This connects CC directly to the transferable-attack result that
motivates the field. **First step:** a steering-attack model and a certified
guarantee that survives it, on a synthetic two-guardrail instance.

---

## The two immediate deliverables

The narrative that sells the program is **Pillar I + Pillar V together**:

- **Pillar V (Atlas)** produces the empirical shock — *independence under-reports
  joint guardrail failure by a measured factor on real stacks* — landing exactly
  as the frontier labs prove composed defenses fail correlatedly.
- **Pillar I (Estimation)** makes that number *honest* — reported as a confidence
  band, not a point, closing the framework's own stated limitation and the same
  "where do the marginals come from" question raised against the sibling
  Ghost-Ark semantic gate.

Together they are a self-contained measurement paper:
**"How badly does independence lie about composed AI defenses? A measured
partial-identification study."** Every Atlas row carries a Pillar II certificate,
so the empirical claim is replayable, not asserted.

## Sequencing (leverage x feasibility)

1. **Now:** I (estimation loop) + II (certificates) + V (Atlas) — small surface,
   closes the stated gap, produces the citable result.
2. **Next:** III (unification) + IV (design) — the unification is the *idea* that
   ties the doctoral thesis together; the design calculus is its first payoff.
3. **Then:** VI (scale) + VII (adversarial) — the ambitious tail.

## Claim discipline (binding)

This vision does not relax a single non-claim. The bounds remain conditional on
the supplied marginals and constraints. The Atlas is **descriptive**, not
predictive: a measured regret on one cohort is not a guarantee about another. A
certificate proves **replay of a computation**, not safety. The unification, until
it is in the theorem ledger with a proof, is a **conjecture**. The framework's
purpose is to bound what the evidence supports and to name what it does not — and
to remain, deliberately, the opposite of the institutional overclaiming it was
built to expose.

## Sibling program

CC-Framework is the measurement science; **Ghost-Ark**
(`github.com/Cubits11/ghost-ark`) is the AWS-native evidence/control plane whose
semantic gate consumes exactly the marginals this framework calibrates and
bounds. Pillar III is the formal bridge between them.
