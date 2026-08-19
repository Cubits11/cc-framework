# Epistemic Upgrade Plan — cc-framework

> **Status: plan.** Nothing in this document is implemented. It describes work
> proposed for v0.4, grounded in measurements taken at commit `3e22c39` on
> 2026-08-19.
>
> **Non-claim.** This plan does not assert that cc-framework is safe, correct,
> production-ready, or fit for any deployment. It does not claim the work
> described here will succeed, and it does not relax a single boundary in
> [`NON_CLAIMS.md`](../research/NON_CLAIMS.md) or the
> [claim-boundary manifest](../claims/CLAIM_BOUNDARY_MANIFEST.md). A plan is a
> statement of intent. Only the acceptance gates below can turn any of it into
> evidence.

| | |
|---|---|
| **Scope** | `Cubits11/cc-framework` only. No change is proposed to any other repository. |
| **Measured baseline** | [BASELINE_MEASUREMENTS.md](BASELINE_MEASUREMENTS.md) |
| **Defects found** | [FINDINGS_REGISTER.md](FINDINGS_REGISTER.md) — 20 findings, 5 at S1 |
| **Scored dimensions** | [COMMITTEE_SCORECARD.md](COMMITTEE_SCORECARD.md) — 12 dimensions |
| **Target tiers** | [VERIFICATION_LADDER.md](VERIFICATION_LADDER.md) |
| **Consumer obligations** | [DOWNSTREAM_CONTRACTS.md](DOWNSTREAM_CONTRACTS.md) — C1–C7 |

---

## 1. The finding that organizes everything

cc-framework's mathematics is **exact**. The LP recovers Fréchet–Hoeffding to
1.11e-16 worst case. It reproduces the correlation cliff in twenty seconds:
eighteen guardrails at p = 0.1 each, composed as a conjunction, bound to
`[0, 0.1]` — where independence would predict 1e-18. Eighteen filters buy
nothing under adversarial dependence. That result is the thesis, and it works.

**And three separate projects reimplemented it rather than depend on this
repository.** Ghost-Ark in TypeScript, Vinctura in JavaScript, and one of them
wrote down exactly why:

> The obvious move was to call `cc.core.composition_theory` [...] it operates on
> ROC POINT SETS and bounds the Youden J statistic. That is a DETECTOR framing —
> it assumes each guardrail is a classifier with a threshold and an operating
> curve. Vinctura's controls are not classifiers.
>
> — `vinctura/scripts/compose-bounds.js`

A consumer read the source, found the mathematics correct and the ontology wrong,
and wrote 214 lines of JavaScript instead. That is not a documentation gap. It is
an API that models the wrong thing.

So the scorecard reads: mathematical correctness **8.9**, consumability **2.6**,
cross-implementation agreement **0.0**. The plan closes the second and third
without touching the first.

### The three failures, stated once

1. **Reachability.** The calculus is correct and unreachable — wrong ontology in
   the API, Python-only, no conformance corpus. Nobody can use it, so everybody
   rewrites it. → W5, W6
2. **Enforcement.** The prose discipline scores 8.8; the enforcement scores 2.1.
   Strict typing is declared for 90 files and enforced on 7. The claim-boundary
   manifest is validated by nothing. Everything holding this repository honest is
   a person remembering to be honest. → W1, W2
3. **Unexamined foundations.** The canonicalization kernel — which every signed
   artifact routes through — silently merges Unicode-distinct keys, diverges from
   RFC 8785 on five of six number forms, and has never been attacked. One hour of
   probing found four defects. → W3

---

## 2. What this plan refuses to claim

Stated before the workstreams, because a plan that leads with ambition and buries
its boundaries has already failed the discipline it proposes.

- **The mathematics is not new.** Fréchet–Hoeffding is 1935. A sibling repository
  already demoted "Fréchet ceiling + reachability sharpening" to engineering, and
  it was right ([F-18](FINDINGS_REGISTER.md#f-18)). This plan does not claim
  novelty for the bounds. The contribution is elsewhere — see W9.
- **No downstream integration is promised.** Whether Ghost-Ark runs the
  conformance corpus, or Vinctura wires the guard, is their decision. This plan
  commits only to making both *possible*.
- **No empirical claim is created by planning one.** The Correlation Atlas (W7)
  is the highest-value item here and it does not exist. Until it runs, empirical
  grounding stays at 2.4 and should be quoted as 2.4.
- **Completing every workstream would not establish safety.** It would establish
  that this repository does what it says. Whether what it says is worth anything
  is answered by someone outside the project, and no amount of internal work
  substitutes for that.
- **This plan is one excavation pass.** Absence of a finding is not evidence of
  absence. `redteam/dependence_search.py` — 709 statements at 65.6% coverage,
  performing the search whose post-selection bias `cliff.py` refuses to certify —
  was **not examined** and is the most likely home of a subtle statistical defect.

---

## 3. Workstreams

Ten workstreams. Each states its findings, deliverables, acceptance gate, and
what completing it does *not* establish. A workstream without a falsifiable gate
is a wish; every gate below is a command.

---

### W0 — Make the documented path work {#w0}

**Findings:** [F-01](FINDINGS_REGISTER.md#f-01), [F-20](FINDINGS_REGISTER.md#f-20)
· **Effort:** hours · **Prerequisite for everything else**

A stranger following the install instructions gets a red suite. That is the first
thing an external reviewer sees, and it is not a real failure — it is a missing
dependency in the `[test]` extra. Nothing else on this plan matters if the front
door is broken.

**Deliverables**

1. Add `build`, `setuptools`, `wheel` to `[project.optional-dependencies].test`.
2. CI job installing `.[test]` **specifically** — not `.[dev]` — and running the
   suite. The extra strangers are told to use becomes the extra that is tested.
3. Delete `tests/regression/week2/test_week2_deliverables.py::test_unit_tests_pass`.
   It shells out to run the whole unit suite and masks real failures — it is how
   F-01 presented, as an unrelated `assert 1 == 0`. If the intent was "week-2
   deliverables exist," assert that directly.
4. `scripts/upgrade_baseline.py` — regenerates every table in
   BASELINE_MEASUREMENTS.md into `docs/upgrade/baseline.json`. The plan's own
   numbers become reproducible, or the plan fails its own standard.

**Acceptance gate**

```bash
git clean -xdf && uv venv .venv && uv pip install -e '.[test]' && pytest
# expect: 0 failed, 0 errors
python scripts/upgrade_baseline.py --check   # regenerated numbers match committed
```

**Does not establish.** Anything about correctness. This is a packaging fix.

---

### W1 — Make the type discipline true {#w1}

**Findings:** [F-02](FINDINGS_REGISTER.md#f-02) · **Effort:** 1 week setup, then continuous

`pyproject.toml` declares `strict = true` over `packages = ["cc"]`. At that scope
mypy reports **279 errors in 47 files**. CI enforces 7 files. The configuration
asserts a discipline the repository does not have — claim inflation in TOML.

Do not delete `strict = true`. Do not silence 279 errors in one pass; a
thousand-line `type: ignore` commit is how strictness dies quietly.

**Deliverables**

1. `docs/upgrade/typing-ratchet.json` — per-module current error count, committed.
2. CI job running mypy at the declared scope, failing only if a module's count
   **increases**. New code is strict from birth; existing debt is visible and
   frozen.
3. Burn-down order: `kernel/` → `reporting/` → `evidence/` → the rest. The
   modules that emit signed artifacts get types first.
4. Remove the 7-file list from CI and pre-commit once the ratchet is live.

**Acceptance gate**

```bash
mypy                                    # declared scope
python scripts/check_typing_ratchet.py  # no module regressed
```

with `kernel/` and `reporting/` at zero errors by the end of the workstream.

**Does not establish.** That the code is correct. Types constrain shape, not
meaning. A well-typed wrong bound is still wrong.

---

### W2 — Make the claim discipline executable {#w2}

**Findings:** [F-12](FINDINGS_REGISTER.md#f-12), [F-15](FINDINGS_REGISTER.md#f-15)
· **Effort:** 1–2 weeks

The prose scores 8.8 and the enforcement scores 2.1. This workstream closes the
gap without weakening the prose.

**The design constraint that matters.** A naive port of Ghost-Ark's scanner would
fire on 70 uses of "guarantee" in this repository, **almost all of them negated** —
`does not guarantee`, `no guarantee of`. It would be switched off within a week,
and switching off a claim gate is worse than never having one. So negation
handling and an allowlist are day-one requirements, not refinements.

**Deliverables**

1. `scripts/check_claims.py` — forbidden-phrase scanner with:
   - negation-aware matching (a phrase inside "does not X" is not a violation);
   - an allowlist with a **written reason per entry**, following Ghost-Ark's
     pattern — policy documents and the scanner itself legitimately quote
     forbidden wording;
   - scanning of `.md`, `.py`, `.tex`, `.yml`, `.json`, `.sh`, and `Makefile`,
     because reviewer-facing claim text lives in build files too;
   - exclusion of generated output, so the verdict describes the committed tree
     rather than local run state.
2. Close the two residual manifest gaps ([F-12](FINDINGS_REGISTER.md#f-12), corrected):
   resolve path-like tokens in `supporting_tests_or_commands`, and assert the
   Markdown and JSON claim-id sets are equal. The validator already runs (via
   pytest) and already checks `supporting_files` existence — an earlier draft of
   this plan said otherwise and was wrong.
3. Resolve the two theorem ledgers. `docs/theory/theorem_ledger.md` governs; the
   other becomes a pointer or is deleted. Add a test asserting exactly one.
4. **Adopt Vinctura's rule:** *no orphan claims — every public sentence carries
   the command that falsifies it or an explicit limitation.* Apply to README
   first, then `docs/index.md`. This is the strongest claim rule in the program
   and it comes from a consumer, not from here.

**Acceptance gate**

```bash
make claims        # scanner + manifest validation, both clean
pytest tests/unit/docs/     # every manifest row's files and tests exist
```

plus: **every README claim maps to a command in the manifest.**

**Does not establish.** That the claims are true. A scanner catches known
overclaim phrasings. It cannot detect a number that was fabricated, a citation
that does not say what it is claimed to say, or a subtler overclaim in unfamiliar
words.

---

### W3 — Attack the canonicalization kernel {#w3}

**Findings:** [F-03](FINDINGS_REGISTER.md#f-03) **S1**, [F-04](FINDINGS_REGISTER.md#f-04) **S1**, [F-05](FINDINGS_REGISTER.md#f-05) **S1**, [F-06](FINDINGS_REGISTER.md#f-06), [F-07](FINDINGS_REGISTER.md#f-07)
· **Effort:** 2–3 weeks · **Highest severity**

Every signed artifact this repository emits routes through
`canonical_json_bytes`. One hour of probing found four defects, including a
silent two-keys-to-one collapse with no error raised. Ghost-Ark found five
unintended kernel members in its own pipeline by building a 31-class census; this
repository has never run the equivalent.

**Deliverables**

1. **Fail closed on key collision.** After NFC normalization, compare key-set
   cardinality. If it shrank, raise `CanonicalJSONError` naming both keys. Detect,
   never resolve.
2. **Strict duplicate-key rejection on read.** An `object_pairs_hook` that raises
   on repeated keys, applied to the report reader, evidence-bundle reader, and
   claim-envelope reader.
3. **Normalize `-0.0` to `0.0`** — with a positive control that genuinely distinct
   near-zero values stay distinct. A rule that rejects honest documents is a
   trade, not a fix.
4. **`docs/architecture/CANONICAL_PROFILE.md`** — `cc.canonical.v1`, versioned,
   stating its relationship to RFC 8785 explicitly. Either adopt JCS number
   serialization or declare a non-JCS profile with a written rationale. Both are
   defensible; silence is not. **Decide before an independent verifier is
   written.**
5. **Bound receipt-covered integers** to the IEEE-754 safe range in the schema, or
   carry them as strings. A JS `JSON.parse` collapses `2^53+1` before any
   verifier code runs — the kernel is set by the parser, and no downstream fix
   reaches it.
6. **`tests/canonical_corpus/`** — an adversarial census in Ghost-Ark's E1 shape:
   declared intent per class (`distinct` / `equivalent`), machine-readable
   verdicts (`sound`, `unintended-kernel`, `over-discrimination`, `fail-closed`,
   `sound-by-rejection`, `rejection-asymmetry`), and **positive controls** so
   strictness is a fix rather than a trade. Seed classes: the six probed for this
   plan, plus duplicate-key-nested-in-array, empty-key-repeated, lone surrogate,
   deep nesting, large-document-single-byte, safe-integer-neighbours.
   **Provenance is `census` — no confidence intervals, exact counts only.**

**Acceptance gate**

```bash
make canon-corpus     # every class matches its declared verdict
make receipt-replay   # capsule rebuilds byte-identically
```

with **zero classes at `unintended-kernel`** and **zero at
`rejection-asymmetry`** — the second is what distinguishes a fix from a trade.

**Binding rule:** never weaken the corpus to make a test pass. A class that starts
failing is a kernel regression.

**Does not establish.** That the canonicalizer has no collisions — only that it
has none in the declared classes. The corpus is a census and its size is an
authoring decision, so it carries no interval and no coverage claim.

---

### W4 — Deepen the test floor {#w4}

**Findings:** [F-13](FINDINGS_REGISTER.md#f-13), [F-14](FINDINGS_REGISTER.md#f-14)
· **Effort:** 3–4 weeks

Coverage is 69.91% with no gate. The `cc-report` CLI — the product surface, the
thing a reviewer actually runs — is at **0%**. `core/stats.py` is at **38% across
756 statements**: the statistics engine, in a repository about statistical
honesty, is its least-tested large component. Untested statistical code does not
fail loudly; it returns a plausible number.

**Deliverables**

1. **CLI golden tests** — build a report, verify it, verify a tampered copy is
   rejected, verify claim-governance PASS and FAIL paths. Target `reporting/cli.py`
   from 0% to >80%.
2. **Property-based tests on `core/stats.py`** using Hypothesis (already a
   declared dependency, currently ungated): interval coverage, monotonicity,
   boundary behaviour. Properties, not examples — the failure mode here is a
   plausible wrong number, which example tests are poor at catching.
3. **Property-based kernel tests** over the marginal simplex, replacing the
   six-point census with sampled invariants: FH recovery, monotone tightening
   under added constraints, `L <= U` always, infeasibility raises.
4. **Coverage floor at the measured 69.91%**, ratcheting upward. A floor set below
   the current value is theatre.
5. **Mutation testing** on `kernel/` and `reporting/`, score published in the
   scorecard. The current score is *unknown*, not low, and that distinction is
   worth removing.
6. **Dual certificates** from the LP, emitted with the primal witness, so a
   skeptic verifies optimality without re-solving. Cheapest available move on
   formal readiness.

**Acceptance gate**

```bash
pytest --cov=src/cc --cov-fail-under=69.91
make kernel-properties && make kernel-duals && make mutation
```

with `reporting/cli.py` > 80%, `core/stats.py` > 70%, and a published mutation
score.

**Does not establish.** Correctness. Higher coverage means more code was executed,
not that it was executed correctly. Mutation score bounds the suite's sensitivity
on the mutated subset, nothing wider.

---

### W5 — Make the kernel consumable {#w5}

> **DELIVERED 2026-08-19.** Consumability **2.6 → 6.8**, cross-implementation
> agreement **0.0 → 6.5**. Evidence in
> [§3.5 Delivery record](#w5-delivery-record) below; scores and their limits in
> [COMMITTEE_SCORECARD.md](COMMITTEE_SCORECARD.md#consumability-as-a-library).
> C4, C5, C6 and the C2 corpus are shipped. C1, C3, C7 remain with W3 and W6.

**Findings:** [F-08](FINDINGS_REGISTER.md#f-08) **S3**, [F-09](FINDINGS_REGISTER.md#f-09) **S3**, [F-10](FINDINGS_REGISTER.md#f-10) **S3**
· **Contracts:** [C2, C4, C5, C6](DOWNSTREAM_CONTRACTS.md#contract-summary)
· **Effort:** 4–6 weeks · **Highest leverage**

Three reimplementations, one written rejection, one dated unfillable request.
This workstream is why the repository scores 2.6 on consumability against 8.9 on
mathematics.

**Deliverables**

1. **`cc.compose`** — marginals in, sharp bounds out, **no ROC concept anywhere in
   the signature**:

   ```python
   compose_bounds(
       marginals: Mapping[str, float],
       event: Literal["all", "any"] | LinearQuery,
       constraints: Sequence[Constraint] = (),
   ) -> Interval
   ```

   Deterministic predicates are first-class. Detectors reach the same surface by
   supplying an operating point as a marginal — the ROC path becomes an optional
   adapter *above* the event calculus, never underneath it. This inverts the
   relationship that caused the rejection.

2. **`conformance/cc-kernel-v1/`** — language-agnostic corpus:
   `(marginals, constraints, query) → [L, U]` with exact expected values, plus an
   adversarial section (infeasible sets, degenerate marginals, boundary values,
   `p=0`, `p=1`, near-machine-epsilon marginals). Any implementation in any
   language either passes or is not a CC kernel. Run in CI here; published for
   the others.

3. **`cc-guard`** — a stdin/stdout JSON subcommand exposing the provenance-tagged
   guards (post-selection refusal first), **plus** the same logic as a pure-data
   decision table in the corpus, so a JS caller can enforce it with no Python
   process. A subprocess dependency is a weaker bridge than a table.

4. **A second implementation, in this repository, in another language.** Small,
   dependency-free, deriving from the published corpus and not from the Python
   source. Without it, Tier 3 of the ladder cannot exist and
   cross-implementation agreement stays at 0.0.

5. **Differential harness** — randomized `(marginals, query)` cases through both
   implementations, comparing to a declared tolerance. Divergence is recorded as
   a finding, never reconciled by fiat.

**Acceptance gate**

```bash
make conformance-publish && make differential-agreement && make differential-fuzz
pytest tests/acceptance/test_vinctura_four_control.py   # their numbers, from cc.compose
```

The Vinctura acceptance test is the real gate: **their four-control result,
including the sensitivity analysis over assumed detection rates, reproduced from
`cc.compose`, such that their 214 lines could be deleted.** It must also
reproduce their correct refusal — no countermonotone regime for n > 2, since the
FH lower bound is not a copula in dimension ≥ 3 though it stays pointwise sharp.
If the API silently offers a countermonotone option for four events, it is wrong,
and the consumer noticed before this repository did.

**Does not establish.** That downstream projects will adopt it. That is their
decision. The gate is that adoption becomes *possible* and that a real consumer's
published numbers reproduce.

#### Delivery record {#w5-delivery-record}

Measured on 2026-08-19. Every row is a command.

| Deliverable | Shipped as | Evidence |
|---|---|---|
| ROC-free surface | `src/cc/compose/` | 34 unit tests; a test greps the public signatures for `roc`, `youden`, `tpr`, `fpr`, `threshold`, `operating_point` and fails if any reappears |
| Conformance corpus | `conformance/cc-kernel-v1/` | 24 accept + 8 reject cases, `SPEC.md` (256 lines), manifest with digests, 1e-12 tolerance |
| Second implementation | `verifiers/node/cc_compose_verify.mjs` | zero-dependency Node, written from `SPEC.md`; 24/24 accept, 8/8 reject |
| Differential harness | `scripts/differential_compose.py` | 23,000 randomized cases across 6 seeds, **0 disagreements** |
| Cross-language guard | `src/cc/cli/guard.py`, `cc-guard` | stdin/stdout JSON **and** a pure-data decision table; table-vs-code agreement asserted |
| Acceptance gate | `tests/acceptance/` | an external consumer's **published** four-control result reproduced: `[0, 0.01]`, independence `1.2e-5`, **833×**, all three scenarios, their sensitivity finding |

Enforced by `make conformance`, `make differential`, `make acceptance`,
`make test-compose`, and two new CI jobs.

**Numbers.** Suite 689 → **808 passing**, 0 failing. Coverage 69.91% → **70.13%**.
`cc/compose/_bounds.py` **97.09%**, `cc/cli/guard.py` **97.56%**. The closed form
agrees with the finite-atom LP to **3.3e-16** over 600 randomized cases.

**The fuzzer earned its keep on its first run.** `dependence="countermonotone"`
with exactly one event: the Python raised `IndexError`, the Node silently
returned `NaN`. Both wrong, differently; the curated corpus had not thought to
ask. Both fixed, and pinned as `reject-countermonotone-one-event`. That is the
argument for randomized differential testing over a corpus alone.

**Honest limits.** The Node implementation and the Python reference were
authored in the same project — a wrong specification yields two implementations
wrong together. This is a differential-testing instrument, not an independent
replication, and both the verifier output and the corpus manifest say so in
their non-claims. The one genuinely non-same-author check is the external
oracle, and it is **one** oracle on **one** scenario family. Neither the corpus
nor the fuzzer covers the constrained LP path. No downstream project has adopted
any of this yet: the obstacle is removed, the adoption is theirs.

**Also fixed in passing.** [F-01](FINDINGS_REGISTER.md#f-01): `build`,
`setuptools`, and `wheel` added to the `[test]` extra, plus a CI job that
installs `.[test]` specifically, so the extra strangers are told to use is the
extra that is tested.

---

### W6 — Contract the boundaries {#w6}

**Findings:** [F-11](FINDINGS_REGISTER.md#f-11), [F-19](FINDINGS_REGISTER.md#f-19)
· **Contracts:** [C1, C3, C7](DOWNSTREAM_CONTRACTS.md#contract-summary)
· **Effort:** 3–4 weeks

Ghost-Ark declares a binding rule that cc-framework must not consume naked binary
labels. cc-framework has no ingest module, no schema, and no test for it. The rule
binds a repository that cannot honour it.

**Deliverables**

1. **`cc.ingest.discretization`** — fail-closed reader for
   `ghost.discretization_rule_receipt.v1` implementing all eleven preconditions
   (binary domain, failure semantics, bounded score domain, threshold legality,
   signed comparator, monotonic risk invariant, calibration digest, scoring
   digest, temporal validity, parent lineage, stationarity declaration). A
   marginal without lineage is refused, not defaulted.
2. **Negative corpus** — one fixture per precondition, each rejected with a
   **distinguishable reason**. A reader that rejects everything for the same
   reason enforces one rule, not eleven.
3. **Versioned resolvable schemas** — `cc.<object>.v<N>` identifiers, version in
   `$id` rather than `title`. A schema whose version lives in its title can change
   without its identifier changing.
4. **Indeterminate verdict state** (C7) — insufficient evidence, infeasible
   constraints, and refused post-selection claims all serialize as
   `INDETERMINATE` with a machine-readable reason, **never as `[0, 1]`**, which a
   careless consumer reads as a computed bound. The design constraint is Assay's
   and it is right: a null-unaware consumer must fail loudly rather than silently
   read indeterminate as a pass.
5. **`declared_reference` non-claim travels with the object** — carrying receipt
   identifiers is not verifying them, and the ingested artifact must say so.

**Acceptance gate**

```bash
make ingest-corpus     # every precondition fixture refused, reasons distinct
pytest tests/unit/evidence/test_indeterminate_serialization.py
```

**Does not establish.** That upstream discretization was appropriate, the
threshold well chosen, or the score calibrated. The contract checks that evidence
carries its lineage, not that the lineage is wise.

---

### W7 — Ground it empirically {#w7}

**Findings:** [F-17](FINDINGS_REGISTER.md#f-17) · **Effort:** 4–8 weeks
· **Highest value, lowest certainty**

Empirical grounding scores **2.4**. No `p_i` in this repository was measured
against a production guardrail. Every marginal is supplied, synthetic, or
assumed. VISION names the Correlation Atlas as the empirical keystone and the
most citable single artifact the program can produce; it does not exist. The
adapters that would collect it do exist and have never been run in a campaign.

**Deliverables**

1. **Turn on the dark lanes.** Scheduled workflow setting `CC_RUN_EXPERIMENTS=1`
   and `CC_RUN_PERF=1` with optional extras installed, publishing artifacts. Nine
   tests have never executed in CI. Allowed to be red without blocking merges — a
   lane that must stay green gets weakened until it does.
2. **Executable reporting rules**, adopted from Ghost-Ark and enforced in code,
   not prose:
   - no point estimate without dispersion (p50 with IQR, never a bare p50);
   - no proportion without a denominator, no rate without its control arm;
   - **no confidence interval over a curated census** — a hand-authored corpus is
     the whole population and its size is an authoring decision;
   - no interval below n = 30;
   - intent declared before results, pinned by a test;
   - state the host;
   - report what was not measured.

   These become assertions in `cc.core.stats`, in the shape of Ghost-Ark's
   `reportProportion` / `assertCensusReporting`. cc-framework enforces the
   post-selection rule already; these are the rest of the family.
3. **Minimal Correlation Atlas** — two or three real guardrails on one public
   jailbreak corpus. Per-pair marginals, observed joint rates, phi, and the
   independence-versus-worst-case gap. Each row emits a certificate. Pre-register
   the cohort and the stopping rule before the first measurement.
4. **Repeated-measurement timing** replacing the single-shot scaling table in the
   baseline, with p50 and IQR and a named host.

**Acceptance gate**

```bash
make atlas    # produces a signed, replayable atlas artifact
```

with: pre-registration digest published **before** results; every rate carrying
its denominator; no interval over a census; cohort, stationarity declaration, and
non-claims present on every row; **and results reported whether or not they are
interesting.** A finding of "independence was approximately right on this cohort"
is a real result and gets published.

**Does not establish.** That a measured regret on one cohort predicts another. The
Atlas is **descriptive, never predictive**. A number measured on one corpus of
jailbreaks says nothing about the next one, and the artifact must say so on its
face.

---

### W8 — Hygiene {#w8}

**Findings:** [F-16](FINDINGS_REGISTER.md#f-16) · **Effort:** days · **Do during a release**

17.8 MB of Blender renders tracked in git; `.git` is 43 MB against a 54 MB
working tree. Clone cost for external reviewers, no verification benefit.

**Deliverables.** Renders to release assets or LFS, one small preview retained.
Archived JSONL checkpoints reviewed. `src/cc/_legacy/` either documented with a
removal version or deleted.

**Acceptance gate.** `.git` under 15 MB; `git clone --depth 1` under 20 MB.

**Does not establish.** Anything. This is housekeeping and is scheduled last on
purpose.

---

### W9 — Reframe the contribution {#w9}

**Findings:** [F-18](FINDINGS_REGISTER.md#f-18) **S3** · **Effort:** 1 week
· **Do first, it is nearly free**

A sibling repository's research ledger demoted the core contribution:

> **Fréchet ceiling + reachability sharpening** [...] **DEMOTED to engineering.**
> The mathematics is 1935; the sharpening is a known distinction. Survives as
> reporting practice, not as science.

This is correct. Any framing that positions cc-framework's value as "we compute
Fréchet–Hoeffding bounds" is answering a criticism that has already been made and
sustained *from inside the program*.

**The move is to concede it first, above the fold.** A criticism you state
yourself cannot be used against you.

**What is genuinely not 1935:**

| Contribution | Status |
|---|---|
| Statistical inference for partial-identification bounds under **estimated** marginals — a confidence band around `[L, U]`, not a point interval | thin in the literature; VISION Pillar I; **open** |
| **Measured** co-failure dependence on real guardrail stacks — how badly independence lies in practice | nobody has published it; VISION Pillar V; **open** |
| A conformance corpus making three independent implementations agree | engineering, but engineering nobody has done for this calculus |
| **Refusing to certify** — the post-selection guard at `cliff.py:321` | a design commitment most statistical software does not make; a consumer called it "the single most elegant thing available in this program" |

**Deliverables**

1. README rewritten: concede the theorem, claim the measurement. Lead with
   limitations, following Vinctura's convention.
2. `docs/research/CONTRIBUTION_BOUNDARY.md` — what is classical, what is
   engineering, what is open. Cite the demotion by name rather than working
   around it.
3. VISION.md pillars re-sequenced against this framing. Pillars I and V are the
   contribution; II and III are the infrastructure that makes them checkable.

**Acceptance gate.** The README states, in its first screen, that the underlying
inequality is classical and names what is not. Reviewed by someone who has read
the demotion.

**Does not establish.** That the open problems will be solved. Naming a gap
honestly is not filling it.

---

## 4. Sequencing

Ordered by leverage × prerequisite, not by size.

### Phase 0 — Credibility (weeks 1–2)

**W0** (documented path works) + **W9** (reframe). Both are nearly free and both
are prerequisites for anyone taking the rest seriously. A reviewer who hits a red
suite, or a framing already publicly demoted, stops reading.

### Phase 1 — Foundations (weeks 2–6)

**W3** (canonicalization) + **W2** (claim enforcement) in parallel. W3 carries
every S1 finding and everything signed depends on it. W2 stops the discipline
from depending on memory. Start **W1** (typing ratchet) here — it is cheap to
start and pays continuously.

### Phase 2 — Reachability (weeks 5–12)

**W5** (consumable kernel) + **W6** (contracts). The highest-leverage work, and it
needs W3 finished first: a conformance corpus over a canonicalization that
silently collapses keys would pin the defect into the contract.

### Phase 3 — Depth (weeks 8–16, overlapping)

**W4** (test floor). Overlaps Phase 2; the CLI golden tests and property tests can
start as soon as W0 lands.

### Phase 4 — Evidence (weeks 12–24)

**W7** (empirical grounding). Highest value and lowest certainty, so it goes last
and is allowed to fail. It depends on W5 (certificates per Atlas row) and W2
(reporting rules enforced).

### Anytime

**W8** (hygiene), during a release.

```
wk:  1  2  3  4  5  6  7  8  9 10 11 12 ... 16 ... 24
W0   ██
W9   ████
W3      ████████████
W2      ████████
W1      ████████████████████████████████████████████  (continuous ratchet)
W5              ████████████████████████
W6                  ████████████████
W4                      ████████████████████████
W7                              ████████████████████████████
W8                                                    ██
```

---

## 5. What "done" looks like

At the end, the scorecard should read:

| Dimension | Now | Target |
|---|---:|---:|
| Mathematical correctness | 8.9 | 9.4 |
| Canonicalization integrity | 3.8 | 8.5 |
| Claim discipline — enforcement | 2.1 | 8.5 |
| Consumability | 2.6 | 8.5 |
| Cross-implementation agreement | 0.0 | 8.0 |
| Empirical grounding | 2.4 | 6.5 |

and these statements should be true, each checkable by a command:

- A stranger clones, runs one command, and gets a green suite.
- A stranger recomputes an interval from its witness without trusting the
  producer, and checks the dual certificate without re-solving the LP.
- An implementation in another language passes the published corpus, and a
  differential harness proves it agrees.
- The canonicalizer fails closed on every declared collision class, with zero
  `unintended-kernel` and zero `rejection-asymmetry` verdicts.
- No claim reaches `main` without a manifest row naming a file and test that
  exist.
- A consumer computes composition bounds from marginals with no ROC concept in
  sight, and a real consumer's published numbers reproduce from the library.
- At least one number in the repository was measured on a real guardrail, and is
  reported with its denominator, its cohort, and what it does not predict.

None of that establishes safety. It establishes that this repository does what it
says — which is the only thing a repository can establish about itself.

---

## 6. The one question, per feature

Adapted from Ghost-Ark's frontier questions. Every proposed change to
cc-framework must answer all ten before it merges:

1. What claim is being introduced, and at which claim-boundary level?
2. What evidence supports it, and where does that evidence live?
3. What command replays it?
4. What conformance case or malicious fixture disproves it?
5. Which boundary fails closed when it is violated?
6. Which non-claim accompanies it?
7. Which reviewer, outside this project, could reproduce it?
8. Which downstream contract does it satisfy or break?
9. Is the number a census or a sample — and does its interval match?
10. What assumptions remain invisible?

If these cannot be answered, the design is incomplete.
