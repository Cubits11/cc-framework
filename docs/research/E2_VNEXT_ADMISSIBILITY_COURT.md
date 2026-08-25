# E2-VNEXT — pre-loader admissibility court

**Convened:** 2026-08-24 · **Subject:** `codex/e2-vnext` @ `c06b9b2` ·
**Court branch:** `codex/e2-vnext-admissibility` (this document)

> **VERDICT: RED — NOT ACTIVATED BEFORE LOADER.**
> Sub-verdict **E — no defensible study with the current candidates.**
> The sealed protocol stands unmodified. No loader may run.

This court did not repair the previous stop. It verified it, then found that the
stop was **under-determined**: two blockers were recorded, and there are **seven**.
Four of the five new ones are properties of the chosen systems and population that
no protocol wording can fix.

---

## 1. Baseline — what was verified before anything was adjudicated

Every claim the prior execution made was treated as a hypothesis and checked.

| Prior claim | Verdict | How checked |
|---|---|---|
| branch `codex/e2-vnext`, commit `c06b9b2` | CONFIRMED | parent is `eb03f83`, tip of `feat/claim-package-compiler`; branch is local-only, never pushed |
| 14 sealed artifacts exist | CONFIRMED | `git show --stat` |
| checksums pass | CONFIRMED | all 8 manifest hashes + 2 self-checksums recomputed independently |
| 24 relevant tests pass | CONFIRMED | 24 = `test_e2_dryrun`(5) + `test_e2_observation_contract`(14) + `test_e2_vnext`(5) |
| completed-pair target 2,025 | CONFIRMED, RE-DERIVED | see §6 — independent derivation, not a re-run of their code |
| no model artifact accessed (VNext) | CONFIRMED | HF cache empty (4.0K); no weights/parquet tracked |
| no source row inspected (VNext) | CONFIRMED | population absent from disk entirely |
| E2-FIELD-TEST-001 unmodified | CONFIRMED but MISLEADING | see §2 |
| attack frame = 2,750 | **UNVERIFIED, AND THE WRONG QUANTITY** | see §4 |

**The frozen artifacts remain byte-identical.** All 8 manifest hashes re-verify after
every change made on this branch.

---

## 2. The prior report's framing was materially incomplete

The report to this court said: *no model artifact accessed; no model loader invoked;
no source row inspected.* Those are **true of E2-VNEXT** and **false of the program**.

`E2-FIELD-TEST-001` is not an abstract identifier. It exists as untracked files in a
*different worktree* — `cc-framework-e2`, branch `exp/e2-real-guardrails` — and ran
**three hours before** the VNext preflight was recorded:

- downloaded the S-Labs dataset CSVs, ProtectAI `model.safetensors` (**737 MB**), and
  the PromptPurify release tarball — **all still on disk**;
- executed `prepare_items.py`, `analyze.py`, `run_protectai.py` (`__pycache__` proves it);
- **invoked the PromptPurify no-input smoke loader, which FAILED.**

To the field test's great credit, its own addendum says so plainly: *"The protocol
header's statement that no candidate had been installed or loaded is historical. A
no-input local loader attempt occurred."* It refused to repair itself. That was correct.

**What is genuinely clean:** the VNext population `3nesdeniz/agentic-prompt-injection-5k`
and Guard B `wolf-defender` have **never been retrieved** — verified by filesystem search
across `/tmp`, `~/.cache`, and every worktree. No score, prediction, or threshold exists
for any candidate in either study. **Outcome-blindness is intact.** That is the one thing
that matters most, and it survives.

**What is not clean:** PromptPurify is a shared candidate. Its loader was already known
broken under a frozen configuration when VNext selected it as Guard A, and
`CANDIDATE_SCREEN.json` does not mention this. This leaks no *outcome* information — but
it is decision-relevant feasibility information the program already held.

---

## 3. THE GATE BATTERY DOES NOT IMPLEMENT ITS OWN DECLARED ORDER

This is the most serious finding, and it is verified mechanically, not argued.

`FROZEN_DESIGN.activation_order` declares **8** gates and calls the order "non-negotiable."
`preflight.py` evaluates **6**, and they are not the same six:

| # | Declared in design | Evaluated by preflight |
|---|---|---|
| 1–5 | (five gates) | same five |
| 6 | `artifact_digest_and_runtime_lock` | **NEVER EVALUATED** |
| 7 | `no_input_local_loader_smoke_test` | **NEVER EVALUATED** |
| 8 | `calibration_then_benign_audit_then_attack_scoring` | **NEVER EVALUATED** |
| — | *(not in the design)* | `one_outcome_blind_candidate_pair_selected` — **added** |

Worse: `artifact_digest_and_runtime_lock` appears **exactly once in the entire frozen
design** — as a bare string inside `activation_order`. It is not a key. There is no
runtime lock: no file digest, no dtype, no execution provider, no library version, no
batch size, no windowing policy, no seed. **It is an empty label.**

And on PASS, `preflight.py` sets
`allowed_next_stage = "artifact_retrieval_and_no_input_loader"` — it **authorizes artifact
retrieval having skipped the gate that was supposed to precede it.** No test asserts
`gate_order == activation_order`; neither string appears in the test file.

**Consequence.** Had the decision-owner and lineage gates passed, this preflight would
have returned `ACTIVATION_READY` and licensed retrieval with the runtime specification
blank. **The study was protected from an under-specified activation only by the accident
that two earlier gates happened to fail.** A fail-stop architecture whose stop was
load-bearing by luck has a defect in the battery, not a limitation in the write-up.

Two further gates are **mis-passed** on the record:

- `frame_capacity_for_delta_resolution` = PASS on `2750 >= 2025`. It compares **items**
  against a floor the design's own variance formula defines over **independent units**.
  Under template expansion (§4) these are not the same quantity. Should read UNRESOLVED.
- `maintenance_and_rights_eligibility` = PASS. Guard A's pinned revision
  `8a035c79…` is an **orphan commit with no common ancestor to `main`**, reachable only
  through the owner-mutable annotated tag `v0.0.1`, and GC-eligible if that tag moves.
  That is precisely the "silent-update path" the criterion requires be **absent**.
  Guard B's pinned revision was also already superseded (v2 shipped 2026-08-18) at the
  time of the 2026-08-24 screen.

---

## 4. Δ is not identified as detector dependence

Four of six reviewers converged on this independently. It is the deepest defect.

The dataset card at the pinned revision declares the attack items were built from
**human-designed templates "authored with slot variables and expanded deterministically
with scattered mixed-radix sampling," then "deduplicated and split-isolated by template."**

Split-isolation *by template* is an admission that templates recur across many items —
otherwise the constraint is vacuous. The released schema is
`[id, text, label, class, attack_family, technique, severity, language]`: **no
`template_id`.** The author demonstrably holds template identity (they used it as the
isolation key) and withheld it from the release. The template count is disclosed nowhere.

Two consequences, both on the estimand rather than on precision:

1. **Bias.** Two near-deterministic classifiers behave near-identically across slot
   variants of one template. A single template both arms allow contributes its whole
   expansion block to `p_11`. Doubling one missed template's slot cardinality moves Δ
   arbitrarily **with zero change to either detector.** Δ is therefore a weighted average
   whose weights are one person's expansion budget — a property of the corpus, not of the
   guard pair. "Departure from independence" and "detector dependence" are not licensed.
2. **Variance.** `V_Δ` and the n=2,025 floor assume iid paired Bernoulli draws. The
   effective unit is the template; the design effect is unbounded from the public record.
   At an illustrative T=150 templates the smallest resolvable |Δ| is ≈0.086, not 0.020 —
   the study would report a "resolved" Δ that is noise, from a gate it believes it cleared.

`FROZEN_DESIGN.ontology.data_source_requirement` lists six per-item requirements. A
cluster/template identifier is not among them. **Nothing in the sealed design would have
caught this.**

---

## 5. There is no common "allows" event

Entailment (i) SOURCE LABEL → STUDY EVENT: **APPROXIMATE** (the card's four families do
support the narrow construct; the label is the author's assertion, not an attack-success
oracle).

Entailment (ii) MODEL OUTPUT → STUDY EVENT: **UNSUPPORTED.** The arms have incompatible
declared output contracts:

- **Guard A** emits a continuous [0,1] softmax (config-attested), the vendor explicitly
  delegates the threshold, and ships a **ternary** policy (block ≥0.95, advisory ≥0.85) as
  one half of a two-part product with a four-way sink selector. Its SDK **max-pools over
  500-char windows at 250 stride**, so allow-probability falls mechanically with length.
- **Guard B** declares a **binary** contract only — `argmax(logits)` — with no ROC, no AUC,
  no score semantics. Its `config.json` contains **no `id2label`**, so the artifact does not
  attest which logit is "injection." Its repo ships a **four-head L2 cascade** plus fp16 and
  mixed-precision ONNX variants **with a different tokenizer**, and a router that
  deliberately excludes the L3 transformer from most inputs.

That is roughly **48 mutually non-equivalent detectors** constructible from the frozen
bytes. `FROZEN_DESIGN` predeclares none of them, and `forbidden_rescues` covers ontology,
split, revision, and dedup — **not harness configuration.** Because Guard A windows and
Guard B does not, and item length varies by family, **item length enters the two arms
through different mechanisms** — so Δ would partly measure a harness mismatch.

Compounding it: Guard B's own training-source list includes `S-Labs/prompt-injection-dataset`
— *the exact source this protocol rejected* for mixing jailbreak personas — plus
`allenai/wildjailbreak` and `jackhhao/jailbreak-classification`. Its decision boundary is
fit to injection ∪ jailbreak and **cannot be un-fit by narrowing the frame.** Cov(A,B) over
an injection-only frame partly measures the coincidence of two vendors' label ontologies.

---

## 6. Power audit — the arithmetic is right and the unit is wrong

Independently re-derived from first principles, without importing `power_design.py`:

- The influence function is `(A−p_A)(B−p_B) − Δ`, so Δ = Cov(A,B).
- `V_Δ = p_A p_B (1−p_A)(1−p_B) + (1−2p_A)(1−2p_B)Δ − Δ²` — **verified analytically**
  against direct 4-cell enumeration, max abs error **1.4e-17**.
- At `p_A=p_B=.5`: n = **2025**. A 2000×2000 grid over all feasible margins and both
  signs returns worst case **2025**. **The number survives.**

But three things around it do not:

**(a) The completeness floor contradicts the decision rule.** At exactly 2,025 of 2,750,
725 items are unresolved, and the PASS rule imputes **every unresolved item as a joint
miss**. The conservative upper joint-miss rate therefore starts at **26.36%** before a
single real miss. **PASS is arithmetically unreachable for any `r_star` < 0.2636:**

| `r_star` | complete pairs actually needed | declared floor | reachable? |
|---|---|---|---|
| 0.01 | 2723 (99.0%) | 2025 (73.6%) | NO |
| 0.05 | 2613 (95.0%) | 2025 (73.6%) | NO |
| 0.10 | 2475 (90.0%) | 2025 (73.6%) | NO |

2,025 is a superpopulation power number for Δ, silently repurposed as a completeness floor
for `θ_miss`. The two quantities are unrelated. This would have surfaced only *after*
scoring — the whole measurement spent to return a structurally guaranteed HOLD.

**(b) The ±0.020 materiality band can lie outside the feasible region.** Δ is sharply
bounded by the margins. If both guards miss under ~2% of injections, **|Δ| can never reach
0.020** and the band is wholly infeasible. The negative half is infeasible for every margin
below ~0.2, so the two-sided test spends α on a mathematically impossible alternative.

**(c) The caps are not free parameters.** Given the 1%-of-500 calibration rule and a 500-item
audit, an expected serial false-block rate of ~2% yields a 95% upper bound of ~**0.034**.
So `f_star` must exceed ~3.4% or FAIL/HOLD is certain by construction. `preflight.py`
validates only `0 < cap < 1` — it would accept an attestation that declares an unpassable study.

**Census semantics.** If the frame is censused and both guards are deterministic, Δ is
**exact** and a confidence interval is meaningless. The legitimate inferential object is a
permutation/Fisher test on *item identity* — "would this concordance arise if the two
outcome vectors were randomly paired?" — which is a real and interesting null, but it is
**not** sampling uncertainty and must never be narrated as such.

---

## 7. Court E vs Court R — **DECOUPLE**

Answering the eight questions put to the court:

1. **Could E2-E be informative with no decision owner?** Yes. Δ on a frozen population is a
   measurement. Nobody needs to be acting on it for it to be true.
2. **Does requiring an external owner create unnecessary activation failure?** Yes — it
   already did. It is Blocker 1.
3. **Is the action gate part of the estimand or downstream interpretation?**
   **Downstream interpretation.** `θ_miss`, Δ, and the marginals are fully defined without
   `r_star`. The cap only converts a measured number into an action.
4. **Would separating reduce overclaim risk?** Yes, substantially — bundling creates
   pressure to *invent* an owner in order to activate, which is the exact failure mode.
5. **Does separation create decision-shopping risk?** **Yes, genuinely** — measure first,
   then shop for a cap that makes the result look consequential.
6. **Can it be controlled by registering R later?** Only partially, and the control must be
   stated as a real cost: any R registered after E2-E's numbers exist can **never** claim an
   outcome-blind cap. R is legitimate later *only* if its cap has a provable pre-E2-E date.
7. **Does the sealed protocol explicitly bind them?** **Yes, structurally.**
   `activation_order[0]` is the decision-owner gate, and `preloader_invariant` says a gate
   failure **"ends E2-VNEXT."** Protocol §2: *"Before any artifact retrieval, a named
   decision owner must add a nonempty decision_attestation.json."*
8. **Is a new protocol version required?** **Yes.** The binding is in the sealed text, so it
   cannot be undone by editing without breaking the seal.

The decisive internal evidence is the program's own precedent: **E1 measured the same class
of quantity — departure from a product baseline — and activated with a research-scope
decision rule, no external owner, no `r_star`, no `f_star`.** E2-VNEXT holds the
real-detector version of E1 to a standard E1 was never held to, for a quantity the protocol
itself says is *not* a safety quantity. `preflight.py` records the absence as gate **FAIL**
rather than **NOT_APPLICABLE**, converting "no decision is pending here" into "the
measurement is forbidden."

**Requiring a predeclared consequence is legitimate. Requiring a deployment owner with
safety caps, for a study that disclaims deployment scope, is the error.**

---

## 8. Overconstraint audit — is CC's rigor improving E2 or making it impossible?

Both, in different places. Classified honestly:

| Gate | Class | Why |
|---|---|---|
| Outcome-blind candidate selection | **NECESSARY** | cheap, removes the largest researcher degree of freedom |
| Ontology narrowing to 4 named families | **NECESSARY** | prevents the field-test relabelling failure |
| Frozen population / immutable revision | **NECESSARY** | without it nothing is reproducible |
| Missingness taxonomy, no silent coercion | **NECESSARY** | prevents manufactured dependence |
| Runtime/artifact digest lock | **NECESSARY — AND MISSING** | §3; the one gate that would have caught §5 is an empty label |
| Template/cluster identifier requirement | **NECESSARY — AND ABSENT** | §4; must bind *claims*, not stop the study |
| Decision owner as **gate zero** | **OVERCONSTRAINING** | §7; a category error for a non-safety quantity |
| Strict out-of-sample lineage (`unknown private ⇒ FAIL`) | **OVERCONSTRAINING as a stop; NECESSARY as a claim limit** | see below |
| `no silent-update path` | **OVERCONSTRAINING** | demands proof of a negative about someone else's infrastructure; every hosted artifact has one. Achievable standard is content-digest pinning, which the study already does |
| `minimum_complete_pairs = 2025` as a frame-eligibility floor | **OVERCONSTRAINING AND PERVERSE** | it *selects against* construct validity — see below |

**The lineage gate is a permanent non-activation machine — and swapping guards cannot fix
it.** Of 24 candidates screened on primary sources, **zero** publish both record-level
training data and a training cutoff. Exactly one (PIGuard, ACL 2025, MIT) publishes its
actual training rows as a file; exactly one (ModernGuard-1) publishes a training window,
and its data is proprietary. The gate also discriminates on **disclosure, not
independence**: a vendor with a full manifest showing 40% overlap would *pass*, while a
vendor with zero disclosure and genuinely 0% overlap *fails*. And its only positive test —
normalized exact matching — is blind to template-level overlap, which is the dominant
channel here. **It is simultaneously too strict to admit anything and too weak to mean
anything.**

**The 2,025 floor actively selects for the worst populations.** Every agentic
prompt-injection benchmark with an actual attack-success oracle is far smaller —
AgentDojo has 629 security cases, InjecAgent has 1,054. A floor of 2,025 *independent*
attack items therefore **excludes every frame with an oracle and admits only synthetic
slot-expanded corpora** — precisely the frames on which a "miss" is an author's label
rather than a security event, and precisely the expansion property that destroys Δ's
construct validity. A census is a census at any size. The lower arm of that rule should go.

---

## 9. The six reviewers

| Reviewer | Verdict | Strongest objection (one line) |
|---|---|---|
| Applied security | **FAIL** | Δ is a property of the author's expansion budget; the frame-capacity gate mis-passed on a row count |
| Statistical design | PASS WITH LIMIT | math verified correct; the **unit of analysis** is wrong and unrecoverable from this source |
| Contamination | PASS WITH LIMIT | descriptive claim survives (lineage IRRELEVANT to it); generalization and mechanism are **FATAL**, three times over |
| Ontology / construct | **FAIL** | no common "allows" event exists, and the seal contains nothing that could define one |
| Decision relevance | PASS WITH LIMIT | G0 is a category error; the protocol certifies its own gate unsatisfiable; **DECOUPLE** |
| Reproducibility | **FAIL** | a revision SHA pins a repository, not a detector; ~48 undeclared harness choices; the lock gate is empty |

**3 FAIL · 3 PASS WITH LIMIT · 0 declared individually fatal to a narrow descriptive study.**

That last point matters and must not be over-read. No single reviewer would block a
*narrow descriptive census*. But **FAIL is not averaged away**, and three independent
FAILs — on construct validity, ontology, and reproducibility — each of which alone
prevents the measured numbers from meaning what the protocol says they mean, are jointly
disqualifying for *this* configuration.

---

## 10. Contamination adjudication

For each intended claim, is unknown lineage FATAL, LIMITING, or IRRELEVANT?

| Claim | Verdict | Reason |
|---|---|---|
| **(1) Descriptive** — these two frozen systems behaved this way on this frozen frame | **IRRELEVANT** | a census of a finite frame is a measurement, not an inference. Contamination changes what the number *means*, not whether it *is* the number |
| **(2) Generalization** to unseen injections | **FATAL** | and not merely unsupported — affirmatively contra-indicated |
| **(3) Mechanistic** — shared failure = shared vulnerability | **FATAL**, three times independently | (a) both guards touch the same public injection vocabulary — Guard A's recipe string `v13-struq+supcon+distill` names **StruQ**, whose training construction *is* the `rag_document_embedded` family; (b) template expansion (§4) breaks it even with perfect lineage; (c) harness asymmetry (§5) manufactures Cov(A,B) out of item length |
| **(4) Deployment-rate** estimation | **LIMITING only because already dead** | 55.7% synthetic attack base rate, no loss model, both revisions superseded. Protocol already forbids it — correctly |

**Unknown lineage is therefore NOT the thing that blocks E2.** The narrow descriptive claim
survives it. What blocks E2 is §3, §4, and §5 — the empty runtime lock, the unidentified
unit of analysis, and the absent common event. This is the opposite of what the prior
report concluded, and it is the court's central correction.

**On temporality:** the dataset repo was created 2026-08-11, after both model revisions.
That dates the **repository, not the text**. The declared provenance — human-designed
templates over families public since ~2022-23 — defeats the inference outright. Both sides
of the comparison are undated: neither guard publishes a cutoff. **EARLIEST PROVABLE
AVAILABILITY: UNKNOWN**, structurally.

---

## 11. Decision authority — **NO EXTERNAL CAP FOUND**

Fourteen candidate authorities examined; six verified by downloading the **primary
document** and scanning the full text locally. No search snippet was treated as evidence.

**Four of the six contain ZERO numeric percentages anywhere in their full text:**
NIST AI 100-1 (1,947 lines), NIST AI 600-1 (3,097 lines), OWASP LLM Top 10 v2025
(1,931 lines), EU GPAI Code of Practice Safety & Security (1,826 lines). *A document
containing no numbers cannot supply a numeric cap.* That is a property of the text, not
an inference.

Every candidate falls into one of three modes, and none escapes:
- **Delegators** — require a threshold and explicitly assign its selection to someone else
  (NIST AI 100-1: *"Human judgment should be employed when deciding… the precise threshold
  values"*; EU AI Act Art. 15(3): accuracy levels *"shall be declared"* by the provider).
  These restate Blocker 1; they do not resolve it.
- **Reporters** — publish measured outcomes after the fact, never a predeclared bound
  (NIST/CAISI agent hijacking 11%→81%; Anthropic 23.6%→11.2%).
- **Unverifiable** — paywalled or gated (ISO/IEC 42001, CSA AICM): recorded
  **NO PUBLIC EVIDENCE FOUND**, never "does not exist."

Microsoft Prompt Shields' only numeric commitment is **99.9% service availability** —
whether the API responds, not whether it decides correctly. Lakera's L1–L4 levels are
ordinal labels; *"very low"* is not a number.

**Conclusion: `NO EXTERNAL CAP FOUND`. Verdict C is unavailable.** No cap was synthesized.

**Which caps have defensible origins?** `r_star`: none. `f_star`: none. The **±0.020**
band is defensible **only as an ex-ante analytical materiality convention** — declared
before any outcome, tied to the design's resolution target — and must be labelled exactly
that. It is **not** a practitioner threshold and cannot support Court R.

---

## 12. Candidate disposition — REPLACE, but not into this protocol

The incumbent pair sits near the **bottom** of the observed transparency distribution, not
the middle. Guard A names zero training datasets ("curated internal corpora"); its only
pretraining-corpus pointer, `models/l5e/CORPUS_LICENSES.json`, is a **404 at every revision
and mirror**; its recipe string names an **undisclosed distillation teacher**; its backbone
is a **private local path** (`/Users/ssk/…/latest`) that the vendor's own build record does
not pin. Guard B names zero datasets beyond "public sources plus internally generated."

Materially better options exist on lineage alone: **PIGuard** (publishes its training rows
as a file), **deepset/deberta-v3-base-injection** and **fmops** (single enumerable public
corpus each), **protectai-v2** (seven named datasets, provable 2024-04-21 weight freeze).
**Retaining the incumbents cannot be justified on lineage grounds; sunk investment is the
better explanation than merit.**

**But replacement must not happen inside E2-VNEXT.** `forbidden_rescues` bars substitution
after a frozen screen fails, and that bar is correct. Replacement belongs to a successor
study whose screen is frozen before any candidate is named. Note also that swapping guards
**cannot clear Blocker 2** — the out-of-sample gate fails on the **corpus** side too.

---

## 13. Verdict

**RED · E — NO DEFENSIBLE STUDY WITH THE CURRENT CANDIDATES.**

Not AMBER: seven blockers, not one. Not BLACK: no public claim exceeds the evidence —
`site/` and `docs/` contain no reference to E2-VNEXT, PromptPurify, or Wolf Defender, and
`NON_CLAIMS.md` is intact. Not D alone: the decisive defects (§4 template expansion, §5
incompatible output contracts, Guard A's orphan revision) are properties of the **chosen
systems and population**, which is the definition of sub-verdict E.

### The seven blockers

| # | Blocker | Status | Fixable by protocol text? |
|---|---|---|---|
| 1 | No decision owner / no external cap | CONFIRMED; `NO EXTERNAL CAP FOUND` | **Yes** — by decoupling R (§7) |
| 2 | Incomplete training lineage | CONFIRMED; structural to the whole field | **Yes** — by dropping the out-of-sample claim |
| 3 | Guard A loader already known broken | **NEW** — undisclosed in the screen | Yes — new runner, new study |
| 4 | Δ unidentified: template expansion, no `template_id` | **NEW** — 4 reviewers converged | **No** — needs the author to publish template IDs |
| 5 | No common "allows" event; ~48 harness configs | **NEW** | **No** for this pair — contracts are incompatible |
| 6 | Gate battery skips 3 of 8 gates; runtime lock is an empty label | **NEW** — verified mechanically | Yes — but the seal must be versioned |
| 7 | Completeness floor makes PASS unreachable; materiality band can be infeasible | **NEW** | Yes |

### What CC may believe after this court — exactly five things

1. **E2-VNEXT is terminally NOT ACTIVATED**, and now for seven documented reasons rather
   than two. The original stop was correct but under-determined.
2. **Outcome-blindness is intact.** The population and Guard B were never retrieved; no
   score exists anywhere in the program. Nothing here requires a contamination reset.
3. **The lineage blocker is not what stops E2.** It is irrelevant to the narrow descriptive
   claim and fatal only to generalization and mechanism — claims CC was already not making.
   What stops E2 is the unidentified unit of analysis, the absent common event, and the
   empty runtime lock.
4. **No external decision authority exists** for this quantity, and this is now established
   by primary-source full-text verification rather than assumed. Court R is untested, not
   failed. That is a legitimate and publishable finding.
5. **Court E does not need Court R**, and the program's own E1 precedent proves it. The
   coupling was an error of protocol construction, not a requirement of science.

### What must never be claimed on this record

No Δ, no joint detector-miss rate, no false-block rate, no dependence result, no
independence result, no guardrail comparison, no security finding. **Activation failure is
not an empirical null.** There is no measurement here — only an admissibility stop.

---

## 14. Successor and next operation

The sealed protocol **is not edited**. The successor is specified separately as **`E2-E`**
(see [`E2_E_SUCCESSOR.md`](E2_E_SUCCESSOR.md)) — a pure descriptive paired-measurement
study with Court R removed to its own future registration. It exists because of the
documented epistemic corrections in §3–§7, not as a convenience patch.

**Exactly one authorized next operation:**

> Obtain, from the benchmark author, a per-record `template_id` (or a published distinct-
> template count per family) at a **pinned dataset revision** — by metadata request only,
> with no row retrieval. Blocker 4 is the only one of the seven that is both decisive and
> resolvable by a party other than CC, and every downstream design choice depends on its
> answer. If template identity cannot be obtained, `E2-E` must select a different
> population before naming any candidate.

**No loader may run. No artifact may be retrieved. No outreach may be sent without
explicit authorization.**
