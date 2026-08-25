# Decision authority criteria — and the E2 authority search

**Status:** frozen standard, written **before** any candidate authority was accepted.
**Result of applying it on 2026-08-24: `NO EXTERNAL CAP FOUND`.**

A person or institution is not a decision authority because they are knowledgeable,
prominent, or interested. This document fixes what would have to be true instead.

## 1. The nine load-bearing requirements

A candidate qualifies only if **all nine** hold. Any single failure disqualifies.

| # | Requirement | Failure looks like |
|---|---|---|
| 1 | **Standing** — actually controls the declared action | a commentator, a standards body that writes guidance but ships nothing |
| 2 | **Precedence** — the decision process existed *before* E2 outcomes | a rule written after seeing a number |
| 3 | **Metric fit** — the declared joint detector-miss / benign false-block quantity genuinely enters their decision | they care about attack success, uptime, or a single-model FPR instead |
| 4 | **Cap independence** — the cap arose independently of anything measured here | a threshold reverse-engineered to make the result interesting |
| 5 | **Action differential** — crossing the cap produces a *different action* | "we would review it" |
| 6 | **Material consequence** — that action affects evaluation, deployment, escalation, testing, procurement, or research allocation | the action is to write a sentence in a paper |
| 7 | **Freezability** — the policy/cap/action can be versioned and hashed | a verbal understanding |
| 8 | **Not the author in costume** — the authority is not the study author inventing a threshold that makes the experiment interesting | see §2 |
| 9 | **Null-survival** — the decision rule would still exist if E2 returned an inconvenient null | the rule only has content when Δ ≠ 0 |

### Semantic alignment (requirement 3, expanded)

An external threshold is admissible only with alignment on **all four** of:
**same event · same unit · same decision context · same performance quantity.**

Explicitly forbidden inferences — each is inadmissible no matter how often it is repeated:

- *NIST says evaluation matters, therefore NIST is the decision owner.*
- *OWASP says prompt injection is dangerous, therefore the threshold is externally justified.*
- *A model card reports accuracy, therefore that is the operational cap.*
- *A benchmark uses 5% somewhere, therefore 5% is authoritative here.*
- *A vendor SLA promises 99.9%, therefore 0.1% is the miss budget.* (Availability ≠ correctness.)

**A framework that requires a threshold but does not itself set one is a DELEGATOR. A
delegator restates the blocker; it does not discharge it.**

## 2. Internal authority — legitimate, but classify it correctly

CC may own an internal research-stage decision, e.g. *"if the measurement lands outside the
predeclared materiality band, authorize a second preregistered pair; if inside, close the
higher-order dependence line."* This can be valid if genuinely predeclared and if it
survives requirement 9 — and it does: a null **closes a line**, which is a real consequence
for a finite research budget.

But it must be labelled for exactly what it is.

| An internal research decision **can** establish | It **cannot** establish |
|---|---|
| the measurement changed the author's research allocation | that practitioners care |
| a preregistered rule was followed | operational significance |
| the study was falsifiable in advance | deployment relevance |

Inflating the left column into the right is prohibited.

## 3. Cap origin test

For every proposed cap: **why this exact number?**

**Acceptable origins:** an existing external policy · an independently justified loss
function · a pre-existing experiment-stage budget · a formal error-cost model · a declared
evaluation requirement · a risk tolerance owned by a real authority.

**Weak origins (disqualifying for Court R):** a round number · aesthetically convenient ·
derived after seeing expected model behaviour · inherited from an unrelated benchmark ·
chosen because the power calculation works · chosen because it creates a reversal.

**Applied to E2:** `r_star` — no defensible origin. `f_star` — no defensible origin.
**±0.020** — defensible **only** as an ex-ante analytical materiality convention tied to the
design's own resolution target. Labelled as such it is honest; called a practitioner
threshold it is not.

**A design-side floor also applies.** The caps are not free parameters: the sealed design's
own arithmetic makes PASS unreachable unless `r_star` ≳ 0.264 at the declared completeness
floor, and `f_star` ≳ 0.034 given the calibration rule. Any attestation must be checked
against these floors before acceptance — `preflight.py` currently validates only
`0 < cap < 1` and would accept an attestation declaring an unpassable study.

## 4. The search — fourteen candidates, all inadmissible

Primary documents were downloaded and scanned in full. **No search snippet was treated as
evidence.** Absence is recorded as `NO PUBLIC EVIDENCE FOUND`, never as "does not exist."

| Authority | Policy | Date | Metric | Cap | Fit | Admissible |
|---|---|---|---|---|---|---|
| NIST | AI RMF 1.0 (AI 100-1) | 2023-01 | **zero numeric percentages in 1,947 lines** | NONE | delegator | **NO** |
| NIST | GenAI Profile (AI 600-1) | 2024-07 | **zero numeric percentages in 3,097 lines** | NONE | FP/FN scoped to *content provenance* | **NO** |
| NIST | Adversarial ML (AI 100-2e2025) | 2025-03 | qualitative "very low FN/FP" | NONE | network/malware anomaly detection | **NO** |
| OWASP | LLM Top 10 v2025, LLM01 | 2024-11-18 | **zero numeric percentages in 1,931 lines** | NONE | awareness list, owns no decision | **NO** |
| MITRE | ATLAS (full `ATLAS.yaml`, 9,636 lines) | continuous | numbers only inside third-party case studies | NONE | HTTP C2 traffic classifier | **NO** |
| EU Commission | GPAI Code of Practice, Safety & Security | 2025-07-10 | **zero numeric percentages in 1,826 lines** | NONE | model-level systemic risk | **NO** |
| EU | AI Act Art. 15 | 2024-06-13 | accuracy *declared by provider* | NONE | never mentions prompt injection | **NO** |
| UK AISI | Making safeguard evaluations actionable | 2025-05-29 | poses thresholds as the **open problem** | NONE | self-describes as unresolved | **NO** |
| NIST/CAISI | Agent hijacking evaluations | 2025-01 | reported ASR 11%→81% | NONE | end-to-end hijack ≠ detector miss | **NO** |
| Anthropic | Browser-use prompt injection | n.d. | reported 23.6%→11.2% | NONE | adaptive best-of-100 attacker | **NO** |
| Microsoft | Prompt Shields + Azure SLA | 2025-11-21 | **99.9% availability only** | none on detection | uptime ≠ correctness | **NO** |
| Lakera | Guard L1–L4 levels | accessed 2026-08-24 | ordinal labels, "very low" | NONE | its own detector, not this pair | **NO** |
| ISO/IEC | 42001:2023 / 23894:2023 | 2023 | paywalled, HTTP 403 | `NO PUBLIC EVIDENCE FOUND` | management-system standard | **NO** |
| CSA | AI Controls Matrix v1.1, AIS-15 | 2026 | registration-gated | `NO PUBLIC EVIDENCE FOUND` | maps only to already-verified delegators | **NO** |

### Verdict

**`NO EXTERNAL CAP FOUND`.** No cap was synthesized from vibes, and none was inferred from
a document that merely says thresholds matter. Court R is therefore **untested**, not
failed — a legitimate result, and the honest one.

**The correct public language:**

> No external decision authority with a pre-existing applicable joint-miss / false-block
> policy was found. E2 therefore tests Court E only. Decision relevance remains untested.
