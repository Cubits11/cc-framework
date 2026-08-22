# E2 — Shared-Item Guardrail Measurement Contract

> Not a system that knows more. A system that makes it harder for any of us to
> pretend we know more than the evidence permits.

**Status: FROZEN — preregistered before dataset selection. No conforming
dataset has been collected. E2 is UNTESTED.**

This document is the measurement contract for the successor to
[E1](E1_DEPENDENCE_EVIDENCE_STUDY.md). It is deliberately frozen *before* any
candidate dataset, ontology, or benchmark has been inspected.

## Why this is frozen now

E1 is synthetic. Its central limitation is not mathematical weakness but
**ontological closure**: the population, dependence structure, event,
measurement process, and ground truth were all constructed by us. E2 is where
reality begins fighting back, and it introduces every unknown at once:

```text
observed result = f(guardrail, items, ontology, annotation,
                    missingness, version, sampling)
```

If the dataset is chosen first and these terms are defined afterwards, the
result is a garden of forking paths wide enough to produce any conclusion. The
pristine opportunity to preregister measurement logic disappears the moment we
inspect candidate distributions, missingness patterns, or benchmark
peculiarities. That clock is already running, which is why this contract is
frozen ahead of the E1 publication rather than after it.

**E2 is not "E1 with real rows."** Carrying E1's clean binary ontology forward
unexamined would be the wrong research move. A substantial part of E2's purpose
is to discover *which abstractions from E1 survive contact with real
measurement*.

## Conformance is executable

| Artifact | Role |
| --- | --- |
| [`schemas/cc.e2_observation_row.v1.json`](https://github.com/Cubits11/cc-framework/blob/febeba5c9193e33d468fc4694df002cdd5a8420e/schemas/cc.e2_observation_row.v1.json) | Frozen row schema, including conditional validity rules |
| [`scripts/validate_e2_observations.py`](https://github.com/Cubits11/cc-framework/blob/febeba5c9193e33d468fc4694df002cdd5a8420e/scripts/validate_e2_observations.py) | Conformance validator and disclosure audit |
| [`examples/e2/observations.example.jsonl`](https://github.com/Cubits11/cc-framework/blob/febeba5c9193e33d468fc4694df002cdd5a8420e/examples/e2/observations.example.jsonl) | Minimal conforming reference dataset |

```bash
python scripts/validate_e2_observations.py examples/e2/observations.example.jsonl
```

The validator distinguishes **violations** (block, exit 1) from **warnings**
(never block, always disclosed). Warnings are not defects to be cleared; they
are the findings a reader must see before believing any dependence estimate.
Conformance is not a clean bill of health.

---

## Layer A — Target population

"Real guardrail data" is not a population. The inferential target must name
which of these is being estimated:

| # | Inferential target | E2 status |
| --- | --- | --- |
| 1 | These exact system versions on these exact items | **Permitted** |
| 2 | These system versions over an item-generating process | Permitted only with a declared sampling mechanism |
| 3 | A broader class of guardrails | **Not permitted from E2 alone** |
| 4 | Guardrail dependence in general | **Not permitted from E2 alone** |

E2 initially claims (1), and (2) only where the sampling frame is explicit. The
population declaration must state the named population, time window,
inclusion/exclusion rules, and whether the sampling mechanism can support any
intended generalization at all.

## Layer B — Shared-item unit

**This is the single most important design decision in E2.**

If two guardrails are evaluated on different items, observed dependence becomes
inseparable from item composition. For guardrails `G1` and `G2` the contract
requires paired observations on the same item `i`:

```text
Y[i,1], Y[i,2]     for the same item i
```

The **item**, not aggregate benchmark performance, is the epistemically
privileged unit. Marginal reports without row alignment are insufficient for
studying failure dependence and are rejected.

The validator enforces this directly: if any guardrail lacks a row for an item
another guardrail covers, that is a conformance violation. Absence must be
**recorded as a row with a missingness code**, never omitted. Silence and
missingness are different claims.

## Layer C — Event ontology

The temptation is to collapse everything to pass/fail because E1 is
mathematically clean. That may destroy scientifically relevant structure.

The contract therefore separates observation from reduction:

```text
R[i,g]  = raw response          (preserved, ordinal, uncollapsed)
Y[i,g]  = h(R[i,g])             (derived binary event, h is versioned)
```

Raw outcomes are recorded on an ordinal scale — `allow`, `warn`, `soft_block`,
`hard_block`, `error` — with optional provider-specific detail retained. The
binary event used by the identification kernel is a **separate, versioned
reduction**. Changing `h` requires a new `normalizer_version`, and exactly one
`h` may be in force per study.

This means the collapse can later be questioned without corrupting the source
record. `normalized_outcome` is always a *derived* field, never an observed one.

## Layer D — Version identity

"Guardrail X" is not an experimental unit. The unit is:

```text
G = (provider, system, version, policy, configuration, date)
```

Externally hosted systems drift. Without version identity, repeated E2 runs may
silently become experiments on different mechanisms.

The contract distinguishes the **declared** version, the **observable** version
identifier, the **execution timestamp**, the **configuration hash**, and the
**policy snapshot**. Crucially, `unknown` is a valid recorded value — it must
never be silently omitted. An unknown version is itself a finding, and the
validator reports it as a limitation on any replication claim.

Guardrails that share a base model, vendor, training corpus, or upstream
moderation layer must declare a common `shared_dependency_group`. Two wrappers
around the same judge model are not two independent mechanisms, and observed
dependence between them may be mechanism-induced rather than a property of
composition.

## Layer E — Missingness

This is where real experiments most often become epistemically dishonest by
accident. Suppose:

```text
Y[i,A] = 1        Y[i,B] = missing
```

*Why* missing? These are not interchangeable:

| Code | Meaning |
| --- | --- |
| `observed` | A decision was returned and parsed |
| `timeout` | No response within the frozen budget |
| `execution_error` | Transport or provider error |
| `unparseable_output` | Response returned but not interpretable under `h` |
| `unsupported_input` | Guardrail declined the input type |
| `quota_exceeded` | Budget or rate limit |
| `upstream_moderation` | Filtered **before** reaching the guardrail |
| `provider_refusal` | Provider declined to process |
| `excluded_pre` | Excluded under a rule frozen before outcomes were visible |
| `excluded_post` | Excluded after outcomes were visible |

`upstream_moderation` is especially load-bearing: content filtered before it
reaches the guardrail induces dependence between any guardrails sharing that
upstream layer. Treating it as ordinary missingness would attribute a shared
infrastructural artifact to the composition itself.

`excluded_post` must never be relabelled as predeclared. The schema enforces
this: `predeclared_exclusion: true` is only valid with `excluded_pre`.

The primary estimand is computed on the complete-case population:

```text
{ i : missingness_code[i,A] = missingness_code[i,B] = observed }
```

and the preregistered **missingness sensitivity analysis is mandatory, not
optional**. Complete-case analysis can itself manufacture or suppress
dependence, so a complete-case-only result is not a permitted E2 output.

## Layer F — Row schema

Observations are stored at the least-collapsed defensible level. Pair-level and
study-level tables are **derived deterministically downstream**, never recorded
directly. One row is one `(item, guardrail, replicate)` observation:

```text
schema_version
study_id                 run_id
item_id                  item_source          item_source_version   item_hash
guardrail_id             guardrail_version    policy_version        configuration_hash
shared_dependency_group
raw_input_hash           raw_output_hash
raw_outcome              raw_outcome_detail
normalized_outcome       normalizer_version
execution_timestamp      execution_status     missingness_code
predeclared_exclusion    exclusion_reason
replicate_id             seed_if_applicable
annotator_id             reference_label      reference_label_uncertainty
```

`replicate_id` exists because guardrails may be stochastic. Disagreement across
replicates of the same `(item, guardrail)` is a property of the instrument that
no single-shot estimate represents, and the validator surfaces it.

A single reference label with no uncertainty representation is a **declared
limitation**, not a neutral default.

## Layer G — Primary estimand

Correlation must not become the scientific object merely because the project is
called CC. For two binary guardrails:

```text
pA   = P(A = 1)
pB   = P(B = 1)
p11  = P(A = 1, B = 1)
```

Independence would imply `p11 = pA * pB`. The preregistered primary estimand is
the **excess joint-failure term**:

```text
Δ = p11 - pA * pB
```

reported with its finite-sample uncertainty, alongside the B0/I0/I1/I2/I3
evidence ladder computed on the same frozen rows.

The motivating question is not "are these systems statistically correlated?"
but:

> **Does treating these mechanisms as independent materially misestimate joint
> failure?**

That is the question `Δ` answers, and it connects E1's synthetic result
directly to E2's measurement. E1 established that the answer can be large in
either direction and that pairwise evidence need not settle it; E2 asks whether
that possibility is realised by real instruments.

---

## Negative controls

At minimum, and preregistered:

- permute guardrail columns independently while preserving marginals;
- duplicate one guardrail's output as a synthetic common-cause control;
- inject controlled label noise;
- compare complete-case against missingness-sensitive analyses;
- recompute the target event with an independently written calculator.

## Success criteria

E2 succeeds only if the population is explicit; data rights are clear; all
guardrails process the same items; the failure ontology survives adversarial
review; versions and thresholds are frozen; missingness is represented rather
than dropped; the analysis replays; the strongest surviving objection is
documented; and the final public sentence remains narrower than the evidence.

## Practical defeaters

E2 must **Narrow**, **Hold**, or **Stop** if no credible failure label can be
defined; shared-item outcomes cannot be obtained; the guardrails are not
meaningfully distinct; missingness dominates the result; the interval is too
wide to affect any bounded decision; direct joint measurement is consistently
simpler and more informative; dependence evidence costs more than the decision
can justify; or the result changes radically under modest, defensible label
choices.

**Each of those outcomes is a valuable research result, not a failure to
report.**

---

## The evidence ladder

CC research claims are staged. Passing an earlier rung confers no authority at
a later one.

| Rung | Meaning | Status |
| --- | --- | --- |
| **E0** | Mathematical proposition | Established |
| **E1** | Controlled synthetic realization | **Complete — decision: Narrow** |
| **E2** | Shared-item empirical guardrail pilot | **Frozen contract, UNTESTED** |
| **E3** | Multi-system / multi-dataset replication | Not started |
| **E4** | Prospective external evaluation | Not started |
| **E5** | Operational decision consequence | Not started |

The phrase "we validated CC" is **prohibited** at every rung. It names no
population, no event, and no evidence regime.

## The claim inheritance rule

Evidence does not monotonically broaden claims. A later experiment may broaden,
narrow, split, invalidate, or leave a claim unchanged. Formally, claims do not
inherit by default:

```text
C(E_{n+1})  ⊉  C(E_n)        (no automatic inheritance)
C(E_{n+1})  =  EarnedClaims(E_{≤ n+1})
```

Every enlargement of scope must be **separately earned** and separately
recorded in the claim ledger. This exists to prevent the classic
research-program failure:

```text
synthetic → pilot → "validated framework" → marketing claim
```

where no one can afterwards identify the exact inferential jump that was never
justified.

> **Claims do not accumulate by narrative momentum.**

## The four-level record ladder

Every CC research document must keep these distinct. Most category errors in
empirical reporting are collapses between adjacent levels:

| Level | Definition | Example from E1 |
| --- | --- | --- |
| **Observed** | What execution directly produced | 126 of 128 replicates covered |
| **Derived** | What deterministic analysis computes | Wilson 95% lower limit `0.944819` |
| **Interpreted** | What the authors infer | The frozen simulation rule was met |
| **Claimed** | What the project asserts externally | Generator-scoped diagnostic only |

The invariant is:

```text
claim strength  ≤  verified evidence strength
```

always, and at every level of the ladder.
