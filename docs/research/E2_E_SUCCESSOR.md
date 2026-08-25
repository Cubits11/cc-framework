# E2-E — successor specification (NOT a protocol; NOT activated)

**Status: SPECIFICATION ONLY. No candidates named. No population named. Not activated.**

`E2-VNEXT` remains sealed and terminally **NOT ACTIVATED**, unmodified, at `c06b9b2`.
This document does not edit it, extend it, or rescue it. It states what a successor would
have to be, and **why it differs** — each difference traced to a documented epistemic
correction in [`E2_VNEXT_ADMISSIBILITY_COURT.md`](E2_VNEXT_ADMISSIBILITY_COURT.md).

## 1. Why a new identifier exists

| Correction | Court ref | Consequence for the successor |
|---|---|---|
| The decision-owner gate made a measurement contingent on a decision that does not exist, for a quantity the protocol itself says is not a safety quantity | §7 | **Court R is removed entirely.** E2-E measures; it decides nothing |
| The gate battery evaluated 6 of 8 declared gates and authorized retrieval while skipping the runtime lock | §3 | The gate list is **executable and machine-checked against the design**, or there is no study |
| `artifact_digest_and_runtime_lock` was a name with no content | §3, §5 | A runtime lock is **content, sealed pre-retrieval**, and is the *first* gate, not the sixth |
| Δ was not identified as detector dependence under template expansion | §4 | The **independent unit must be identified before the population is named** |
| No common "allows" event existed across two incompatible output contracts | §5 | **Output-contract compatibility is an eligibility criterion**, screened before candidates are named |
| A completeness floor borrowed from a power calculation made PASS unreachable | §6 | Completeness is derived from the decision rule, or there is no decision rule |
| The strict lineage gate is unsatisfiable by the entire field | §8 | The out-of-sample **claim** is dropped; lineage becomes a declared limitation |

## 2. What E2-E is

> On one frozen population, with two frozen, fully-specified detector configurations, what
> is the joint failure structure — and where does it sit inside the marginal-only Fréchet
> interval that an independence assumption would have licensed?

**Court E only.** No decision owner. No `r_star`. No `f_star`. No PASS/FAIL/HOLD.
No deployment, prevalence, generalization, or mechanistic claim.

**The scientific content does not require any of those.** If a risk assessor holding only
marginals would have been materially wrong about *this frame*, that is an existence
demonstration, and existence claims do not need external validity. Contamination is
irrelevant to it: memorization is itself a perfectly real cause of correlated guardrail
failure. What the claim *does* require is that Δ be **identified** — which is exactly what
E2-VNEXT's population could not deliver.

## 3. Eligibility criteria — frozen before any candidate is named

**Population.** Per-record **cluster/template identifier** (or provable absence of
template expansion) · per-item English text · explicit per-item taxonomy · stated
generation provenance · immutable revision · license permitting local evaluation ·
disclosed construction of the benign frame to the same standard as the attack frame.

**Guards.** Public, ungated, immutable artifact · license permitting local evaluation ·
text-level prompt-injection detection · **compatible output contracts across the pair**
(both continuous, or a predeclared and defensible mapping to a common event) ·
artifact-attested class mapping · **exactly one executable configuration selected from the
repo tree by digest** · reproducible local inference with no mandatory network service ·
enumerable training sources sufficient for an overlap audit over *disclosed* data.

**Pair ranking** (frozen before scoring): ontology match · lineage transparency ·
reproducibility · output comparability · architectural diversity · license · artifact
stability. **Never:** expected dependence, expected performance, expected correlation, or
any prior anecdotal output. **Selection must remain defensible if the result is zero.**

## 4. The runtime lock — gate one, not gate six

Sealed and hashed **before any retrieval**, machine-checked by the preflight, containing
per arm: exact artifact path + SHA-256 of every component invoked · dtype and quantization ·
execution provider and thread count · library versions · tokenizer digest · max sequence
length and **windowing/pooling policy** · any router/cascade/sink selection · input framing
bytes · batch size · seed policy · determinism verification method.

**Rationale.** In E2-VNEXT the two candidates admitted ~48 mutually non-equivalent
detectors from the frozen bytes, and because one arm max-pools over sliding windows and the
other does not, item length would have entered the two arms through different mechanisms.
Δ would have partly measured a harness mismatch. **A revision SHA pins a repository, not a
detector.**

## 5. Analysis, if it ever activates

- **Unit.** The template (or the verified-absent cluster), never the row. The resolution
  floor is derived in **units**, not items. If the unit cannot be identified, there is no study.
- **Census semantics.** With a censused frame and deterministic arms, Δ is **exact**; a
  confidence interval is meaningless. The legitimate inferential object is a
  **permutation/Fisher test on item identity** — *would this concordance arise if the two
  outcome vectors were randomly paired?* — reported as a model-based null about identity,
  **never** as sampling uncertainty.
- **Materiality.** Any band must be checked against the **feasible** Δ region at the
  observed margins before interpretation. If the band is infeasible, the finding is
  *"materiality band infeasible at observed margins"* — which is informative, because it
  means independence was necessarily approximately right.
- **Two-stage sealed reveal.** Stage 1: marginals, Fréchet interval, independence plug-in —
  hashed and sealed. Stage 2: the joint. Then ask where reality landed in the marginal-only
  interval. This is the cleanest test of CC's actual motivation, and sealing stage 1 first
  is what prevents reconstructing the marginal-only view after seeing the joint.
- **Completeness.** Derived from the reporting rule, not borrowed from a power calculation.
  Missing entries are never silently coerced to safe, blocked, or miss.

## 6. Predeclared: all three outcomes are publishable

1. **Joint materially differs from independence** — interpret narrowly, this frame only.
2. **Independence adequate within the materiality band** — a real negative result.
3. **No common operating point can be defensibly defined** — possibly the most practically
   important finding, and the one E2-VNEXT actually reached. **Never rescued by inventing
   a mapping.**

A narrow Fréchet interval is a legitimate result, not a failed study. CC's thesis has the
most headroom where guards are worst and the least where they are best; that must be
committed to in advance so it cannot be relabelled afterwards.

## 7. Forbidden rescues (carried forward, plus new)

Carried: substituting a candidate after the frozen screen fails · downgrading a lineage
failure to a passed gate · changing ontology, split, revision, or dedup after a score
exists · reading attack outputs while setting thresholds · replacing a missing item or
retrying a score · using Δ as a decision trigger · calling a detector miss a compromise risk.

**New:** changing any runtime-lock field after retrieval · selecting a cascade head, sink,
precision variant, or windowing policy after any score exists · substituting the analysis
unit after seeing the cluster structure · introducing a decision cap after activation ·
**counting rows as independent units.**

## 8. Status

**NOT ACTIVATED. No candidates named. No population named.** The single authorized next
operation is stated in the court's §14: obtain template identity from a benchmark author by
metadata request only. Every other design choice waits on that answer.
