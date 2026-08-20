# Downstream Contracts

> **Scope note.** Every change described here is made **in cc-framework**. This
> document describes what the other repositories need and what this repository
> must offer; it does not propose edits to them. Where a downstream change would
> be required to complete a loop, it is named as *their* decision, not scheduled
> here.
>
> **Non-claim.** Nothing in this document establishes that any downstream system
> is safe, correct, or compliant. A contract that carries lineage is a contract
> that can be checked — not a contract that is true.

Three consumers, three different failures of the current library, three
different fixes.

| Consumer | Relationship | Current state | What cc-framework owes it |
|---|---|---|---|
| [ghost-ark](#ghost-ark) | upstream producer *and* parallel implementer | reimplemented the calculus in TypeScript; declared a binding ingest rule this repo cannot honour | an ingest module and a conformance corpus |
| [vinctura](#vinctura) | rejected adopter | read the source, rejected the API, wrote 214 lines of JS; has a dated request this repo cannot serve | an ROC-free composition surface and a cross-language guard |
| [b2b-spatial](#b2b-spatial-intelligence-engine) | skeptic | demoted the core contribution to "engineering" | a defensible answer, and instrumentation it can actually use |

---

## The shared finding

All three relationships fail the same way. cc-framework's mathematics is sound
and its packaging makes the mathematics unreachable. Every consumer independently
concluded that reimplementing was cheaper than depending — and **each was right**,
given the API they were offered.

The upgrade is therefore not "add features." It is: make the thing that already
works reachable from outside Python, in a shape that matches how the calculus is
actually used.

---

## ghost-ark

`PSUCyberSecurityLab/ghost-ark` — the AWS-native evidence and control plane.
Public. 501 TypeScript files, 169 test files, 156 markdown documents.

### What Ghost-Ark already decided

`docs/research/CLAIM_EVIDENCE_MATRIX.md` carries a rule stated as binding:

> CC-Framework must not consume naked binary labels from Ghost-Ark. Binary
> variables must be tied to a discretization rule, threshold, comparator,
> calibration digest, scoring digest, validity window, and parent evidence
> lineage.

and a layer table that assigns the claim boundary at each hop:

| Layer | Boundary |
|---|---|
| Ghost discretization rule receipt | defines how a score becomes a binary variable; does not prove the score is valid |
| Ghost binary observation | records applying the rule; does not prove the threshold is optimal |
| CC evidence bundle | provides CC-compatible binaries and provenance; does not prove safety |
| **CC-Framework bounds report** | **computes what follows under stated assumptions; does not validate upstream data collection** |
| Ghost signed claim envelope | binds evidence, assumptions, result digest; does not widen the claim |

The object is specified: `ghost.discretization_rule_receipt.v1`, with a monotonic
risk invariant (`higher_is_riskier` admits only `>=` and `>`), seventeen required
fields, and eleven verification preconditions.

**cc-framework implements none of it** ([F-11](FINDINGS_REGISTER.md#f-11)). The
rule binds a repository that cannot honour it.

### The parallel implementation

`packages/research-frontier/src/ccCorrelation.ts` computes, in TypeScript:
per-variable failure counts and rates, 95% Wilson intervals, `n00/n01/n10/n11`
co-failure tables, observed joint rates with Wilson intervals, empirical phi, and
**pairwise Fréchet lower and upper bounds**. It requires a complete rectangular
grid and rejects mixed cohorts, non-binary values, and absent stationarity
declarations.

It is a careful implementation of cc-framework's calculus by someone who could
not call cc-framework. It has never been compared against the Python.

### What cc-framework ships

**C1 — `cc.ingest.discretization`** (owner [W6](EPISTEMIC_UPGRADE_PLAN.md#w6))

A fail-closed reader for `ghost.discretization_rule_receipt.v1` implementing all
eleven preconditions:

| Precondition | Refusal on violation |
|---|---|
| binary domain | value is exactly 0 or 1 |
| failure semantics | 1 means guardrail failure / unsafe pass |
| bounded score domain | finite lower and upper bounds declared |
| threshold legality | threshold inside the declared domain |
| signed comparator | comparator included in the rule digest |
| monotonic risk invariant | comparator direction matches score polarity |
| calibration digest | present |
| scoring digest | scoring function, model, or policy digest present |
| temporal validity | observation timestamp inside the rule validity window |
| parent lineage | observation references a parent receipt |
| stationarity declaration | cohort declares whether joint dependence is assumed stable |

A marginal arriving without lineage is **refused**, not defaulted. The negative
corpus carries one fixture per precondition, and each must be rejected with a
distinguishable reason — a reader that rejects everything for the same reason is
not enforcing eleven rules, it is enforcing one.

`declared_reference` means the observation carries receipt identifiers. It does
**not** mean cc-framework verified that receipt or its signature. That non-claim
travels with the ingested object.

**C2 — the conformance corpus** (owner [W5](EPISTEMIC_UPGRADE_PLAN.md#w5))

`conformance/cc-kernel-v1/` — language-agnostic cases with pinned expected
values, covering pairwise FH bounds, Wilson intervals, phi, and the n-ary
conjunction and union bounds. Ghost-Ark's TypeScript either passes it or it is
not a CC kernel. Running it there is Ghost-Ark's decision; publishing it is this
repository's obligation.

**C3 — the canonicalization treaty** (owner [W3](EPISTEMIC_UPGRADE_PLAN.md#w3))

[F-04](FINDINGS_REGISTER.md#f-04) and [F-05](FINDINGS_REGISTER.md#f-05) mean a CC
report canonicalized here and re-canonicalized in Ghost-Ark's TypeScript can
disagree — five of six number forms diverge from RFC 8785, and integers above
2^53 do not survive `JSON.parse`. The treaty is a written, versioned
`cc.canonical.v1` profile stating its relationship to RFC 8785, plus a schema
constraint bounding receipt-covered integers to the IEEE-754 safe range.

This must be decided **before** an independent verifier is written. Ghost-Ark's
E1 census found that the kernel is set by the parser, not the canonicalizer; the
same corollary applies here and cannot be fixed downstream.

### What this does not create

No claim that Ghost-Ark's evidence is valid, that its scores are calibrated, or
that a CC report over Ghost-Ark observations says anything about deployed
safety. The ingest module checks that evidence carries its lineage. Whether the
lineage describes a well-designed measurement is outside every layer in the
table above.

---

## vinctura

`Cubits11/vinctura` — graduation photography, and underneath it a provenance
kernel and a verifiable-credential register. Private.

Vinctura's `CLAIMS.md` opens with the strictest rule in the program:

> Every public claim this repository makes, with the command that falsifies it or
> an explicit limitation. **No orphan claims.** A sentence that no command can
> demonstrate is deleted or given a command.
>
> Limitations come first, in this file as in every other.

cc-framework should adopt this rule ([W2](EPISTEMIC_UPGRADE_PLAN.md#w2)). It is
stronger than anything currently enforced here.

### Failure 1 — the API rejected

`scripts/compose-bounds.js` records, in its header, why cc-framework was not used
([F-08](FINDINGS_REGISTER.md#f-08)): `cc.core.composition_theory` operates on ROC
point sets and bounds the Youden J statistic, which presumes each guardrail is a
classifier with a threshold and an operating curve. Vinctura's four controls are
deterministic refusal rules — `SELF_REPORTED` refuses iff `loggedBy === memberId`.
No threshold, no operating point, no false-positive rate.

> Forcing a deterministic refusal rule into an ROC shape would produce numbers
> with the form of a measurement and none of the content.

The consumer was right, and identified exactly what transfers: "the FH inequality
is applied directly to the events. **That is the part that transfers**; the ROC
machinery is not."

**C4 — `cc.compose`** (owner [W5](EPISTEMIC_UPGRADE_PLAN.md#w5))

A surface whose signature contains no ROC concept at all:

```python
compose_bounds(
    marginals: Mapping[str, float],       # P(E_i) — event probabilities, not TPR/FPR
    event: Literal["all", "any"] | LinearQuery,
    constraints: Sequence[Constraint] = (),
) -> Interval                              # sharp [L, U], plus witnesses
```

Deterministic predicates are first-class inputs. Detectors reach the same surface
by supplying their operating point as a marginal — the ROC path becomes an
optional adapter *on top of* the event calculus, never underneath it.

**Acceptance gate.** Not "the API exists." The gate is: Vinctura's four-control
result — including the sensitivity analysis over assumed detection rates — is
reproduced from `cc.compose` in a cc-framework test, and their 214 lines could be
deleted. If it cannot be reproduced, the surface is still wrong.

The test must also reproduce Vinctura's **correct refusal**: there is no
countermonotone regime for n > 2, because the FH lower bound is not a copula in
dimension ≥ 3 though it remains pointwise sharp. A composition API that silently
offers a "countermonotone" option for four events is wrong, and Vinctura noticed
before this repository did.

### Failure 2 — the guard that cannot be reached

`docs/research/program/ultracode/UC-10-KERNEL-BRIDGE.md` §4.3
([F-10](FINDINGS_REGISTER.md#f-10)):

> Optional stopping invalidates the density estimate entirely — and the sibling
> project `cc-framework` already refuses confidence claims on post-selection
> intervals at `src/cc/kernel/cliff.py:321`. **Use that.** [...] Making one
> repository's guardrail catch another repository's error is the strongest
> possible demonstration that the guardrail is real.
>
> That last point is the single most elegant thing available in this program. Do it.

The guard exists and is correct. It is reachable only from Python, in-process.

**C5 — `cc-guard`** (owner [W5](EPISTEMIC_UPGRADE_PLAN.md#w5))

Two deliverables, because one is not enough:

1. A `cc-guard` subcommand reading JSON on stdin and writing a verdict on
   stdout — `{"regime": "discovery-only", "confidence_claim": null, "reason": ...}` —
   exposing the provenance-tagged guards without a Python API.
2. The same logic as a **pure-data decision table** in the conformance corpus, so
   a JavaScript caller can enforce the rule with no Python process at all. A
   subprocess dependency is a weaker bridge than a table.

**Acceptance gate.** Vinctura's G3 gate — `npm test -- probe.post-selection-refused` —
passes against cc-framework's guard, by either route. Whether they wire it is
their call; that it *can* be wired is this repository's obligation.

### Failure 3 — the kernel probe

UC-10 specifies a `kernel-probe` measuring `|ker C|` for a canonicalization
function `C : Events → Records`, with `cheapestCollisionCostRatio` as the
security-relevant quantity — pre-registered sampling design, committed stopping
rule, and results reported whether or not they are interesting.

cc-framework's canonicalizer is a **valid target**, and
[F-03](FINDINGS_REGISTER.md#f-03) is a member of its kernel found by hand in
under an hour. The adversarial corpus in [W3](EPISTEMIC_UPGRADE_PLAN.md#w3)
should be built in a shape a probe can consume: declared intent per class,
machine-readable verdicts, no confidence interval over a curated census.

That last constraint is the pleasing part. Vinctura's probe wants to route its
density estimate through cc-framework's post-selection refusal; cc-framework's
own corpus is a census and must therefore carry no interval. **The guard applies
to the repository that wrote it.**

---

## b2b-spatial-intelligence-engine

`Cubits11/b2b-spatial-intelligence-engine` ("Assay") — hardware-attested capture
provenance via Nitro Enclaves, App Attest, StrongBox, NIST beacons, C2PA/JUMBF.
Private.

### The skeptic's verdict

`docs/latent-research-program.md` runs a kill-ledger over the program's research
concepts. The relevant row ([F-18](FINDINGS_REGISTER.md#f-18)):

> **Fréchet ceiling + reachability sharpening** — "Adversarial robustness ≠
> average-case robustness" — standard in ML security since 2014. **DEMOTED to
> engineering.** The mathematics is 1935; the sharpening is a known distinction.
> Survives as reporting practice, not as science.

This is correct and cc-framework should say so first. See
[W9](EPISTEMIC_UPGRADE_PLAN.md#w9).

What survives the demotion is worth more than what it removes: reporting practice
that a hostile reviewer trusts is exactly what an evidence platform needs, and
"survives as reporting practice" is a *use case*, not a dismissal.

### Where the calculus actually applies

Assay composes several independent checks before issuing a verdict: device
certificate chain validation, NIST beacon temporal freshness, offline Merkle CRL
revocation, and PCR0 enclave measurement. Its own README frames the value as
replacing probabilistic detection with deterministic attestation — and the
composed question is exactly CC's object:

> Given per-check evasion marginals, what is the sharp bound on an artifact
> passing **all** checks while being fabricated, under unknown dependence between
> the checks?

These checks are deterministic predicates, not classifiers — the same shape as
Vinctura's controls, and the same reason `cc.compose` (C4) is the surface that
fits. An attacker who compromises a signing key correlates several checks at
once; independence across them is precisely the assumption an adversary attacks.

**C6 — the deterministic-predicate composition path** is C4. No separate
deliverable; the same surface serves both consumers, which is the argument for
building it once and building it right.

### The three-valued verdict

`schemas/vinctura_spatial_verdict.v1.schema.json` carries a lesson cc-framework
should learn rather than teach:

```
"class": ["WITHIN_DECLARED_BOUND", "EXCEEDS_DECLARED_BOUND", "INDETERMINATE"]
"INDETERMINATE is not a pass. Relying parties MUST NOT collapse it to either other value."
```

and, on the convenience boolean:

> Deliberately NOT named 'compliant' [...] A boolean asserting legal compliance
> would be a claim Vinctura cannot support. Consumers MUST branch on 'class';
> this field exists so that a null-unaware consumer fails loudly rather than
> silently reading INDETERMINATE as a pass.

**C7 — an explicit indeterminate state in the CC report**
(owner [W6](EPISTEMIC_UPGRADE_PLAN.md#w6))

Today, a CC report either carries an interval or an error is raised. There is no
first-class "the evidence does not identify this" verdict that survives
serialization. Insufficient evidence, an infeasible constraint set, and a
refused post-selection claim are all *indeterminate*, and each should serialize
as such with a machine-readable reason — never as `[0, 1]`, which a careless
consumer reads as a computed bound.

The design constraint is Assay's, and it is right: a null-unaware consumer must
fail loudly rather than silently read indeterminate as a pass.

### The convergence worth naming

Assay ships a research module titled `01-canonicalization-collapse.html`.
Ghost-Ark's E1 measured five unintended kernel members in its own pipeline.
cc-framework's canonicalizer silently merges Unicode-distinct keys
([F-03](FINDINGS_REGISTER.md#f-03)).

**Three repositories in one program are working on canonicalization collapse.
Two are studying it. One has it and had not looked.** W3 closes that, and the
corpus it produces is the artifact the other two can measure against.

---

## Contract summary

| ID | Deliverable | Serves | Workstream | Acceptance |
|---|---|---|---|---|
| C1 | `cc.ingest.discretization` | ghost-ark | W6 | eleven preconditions, one negative fixture each, distinguishable refusals |
| C2 | `conformance/cc-kernel-v1/` | ghost-ark, vinctura | W5 | pinned expected values; adversarial section; runs in CI |
| C3 | `cc.canonical.v1` profile + integer bound | ghost-ark | W3 | written RFC 8785 relationship; safe-range constraint in schema |
| C4 | `cc.compose` (ROC-free) | vinctura, b2b-spatial | W5 | Vinctura's four-control numbers reproduce; no countermonotone option for n>2 |
| C5 | `cc-guard` CLI + decision table | vinctura | W5 | Vinctura's G3 gate passes by either route |
| C6 | deterministic-predicate path | b2b-spatial | W5 | same surface as C4 |
| C7 | indeterminate verdict state | b2b-spatial | W6 | serializes with reason; never `[0, 1]` |

## What none of this establishes

The contracts make evidence checkable. They do not make it true. No consumer
should read a passing contract check as establishing that a guardrail works, a
threshold is right, a cohort generalizes, an artifact is authentic, or a system
is safe. Each contract carries its own non-claim above, and those non-claims
travel with the artifacts, not merely with this document.
