# Committee Scorecard — cc-framework, August 2026

> Modelled on Ghost-Ark's committee scorecard. Scores are **judgements, not
> measurements** — they are defensible only because each one names the evidence
> that would raise it and the evidence that would lower it. A score whose
> movement conditions are unstated is meaningless.
>
> Passing tests are evidence. Passing tests are not correctness. Passing tests
> are not safety. Passing tests are not production readiness.

Baseline facts: [BASELINE_MEASUREMENTS.md](BASELINE_MEASUREMENTS.md).
Defects: [FINDINGS_REGISTER.md](FINDINGS_REGISTER.md).

---

## Summary

| # | Dimension | Now | Target (v0.4) | Workstream |
|---|---|---:|---:|---|
| 1 | Mathematical correctness | **8.9** | 9.4 | W4 |
| 2 | Canonicalization & provenance integrity | **8.2** | 8.5 | W3 |
| 3 | Claim discipline — prose | **8.8** | 9.2 | W2 |
| 4 | Claim discipline — enforcement | **4.0** | 8.5 | W2 |
| 5 | Test depth | **6.4** | 8.5 | W4 |
| 6 | External reproducibility | **4.2** | 8.0 | W0, W7 |
| 7 | Consumability as a library | **6.8** | 8.5 | W5 |
| 8 | Cross-implementation agreement | **6.5** | 8.0 | W5 |
| 9 | Empirical grounding | **2.4** | 6.5 | W7 |
| 10 | Statistical honesty machinery | **7.9** | 9.0 | W4, W7 |
| 11 | Evidence-governance architecture | **7.6** | 8.5 | W6 |
| 12 | Formal readiness | **3.1** | 4.5 | W4 |

**Committee verdict.** cc-framework is approximately:

- **8.4 / 10** as a *mathematical* research artifact — the kernel is exact, the
  theorem ledger names witnesses, the non-claims are unusually disciplined.
- **4.1 / 10** as a *verifiable evidence* artifact — the canonicalization kernel
  has an unexamined collapse, nothing enforces the claim manifest, and no
  independent implementation exists to disagree with it.
- **6.8 / 10** as a *consumable library* — was 2.6. W5 shipped an ROC-free
  composition surface, a published conformance corpus, an independent Node
  implementation, and a cross-language guard. A real consumer's published
  numbers now reproduce from the library.

The gap between the first and third numbers is the whole plan. W5 has closed
most of it; W6 and the empirical lanes remain.

---

## 1. Mathematical correctness — 8.9 {#mathematical-correctness}

*Measures: whether the computed bounds are the bounds claimed.*

**Evidence supporting.** FH recovery exact to 1.11e-16 worst case across six
marginal configurations. `kernel/frechet_classes.py` 98.87% covered. Theorem
ledger T1–T6 with proof status, implementation witness, test witness, and a
per-theorem non-claim. Infeasibility raises rather than returning a plausible
number. Monotonic tightening tested. LP is sharp by finite-dimensional convexity,
computationally witnessed.

**What would raise it.** Property-based tests over the marginal simplex rather
than six chosen points (the current evidence is a census, not a sample, and
carries no interval). Dual certificates emitted alongside primal witnesses so a
skeptic can verify optimality without re-solving. Exact rational arithmetic for
small instances as a cross-check on the float path.

**What would lower it.** Any marginal configuration where the LP and the closed
form disagree beyond solver tolerance. A degenerate constraint set that returns
`optimal` on an empty feasible region.

---

## 2. Canonicalization & provenance integrity — 8.2 {#canonicalization-provenance-integrity}

*Measures: whether a receipt identifies the document it claims to.*

> **Revised upward from 3.8 on 2026-08-19**, on delivered evidence. Every S1
> finding in this dimension is fixed. See
> [CANONICAL_PROFILE.md](../architecture/CANONICAL_PROFILE.md).

**Evidence supporting.** `cc.canonical.v2` implements RFC 8785: ECMAScript
`Number::toString` number forms, UTF-16 code-unit key ordering, no
normalization, and one declared narrowing (the IEEE-754 safe integer range)
documented rather than left implicit. 15 of 15 probed number forms conform.

The census at `scripts/canonicalization_probe.py` runs 14 declared-intent
classes across both profiles and reports, under v2: **12 sound, 1 fail-closed,
1 sound-by-rejection** — zero `unintended-kernel`, zero `rejection-asymmetry`,
zero `over-discrimination`. Four positive controls pass, which is what makes the
result a fix rather than a trade. `tests/unit/canonical/` holds all of it with 57
tests, and a further test asserts v1 *still* carries its defects, so the legacy
profile cannot be silently "fixed" out from under historical receipts.

Duplicate keys are refused on read at any nesting depth, wired into the report
CLI, the claim-governance readers, and the Merkle log.

Migration was done without breaking history: verification dispatches on the
profile each receipt declares, and every regenerated capsule artifact was diffed
with hashes, hash-derived ids, and the profile identifier scrubbed. All eleven
were byte-identical under that scrub — only hashes moved.

**What still lowers it.** No fuzzing of the canonicalizer: every class was
authored by hand, so the census establishes that the declared classes behave as
declared and nothing about the rest of the input space. **No cross-language
differential test on receipts** — the corpus establishes agreement on the
composition kernel, but nothing yet re-canonicalizes a CC report in another
language and compares digests, so "cross-language verifiable" currently
describes v2's design rather than a demonstrated result. No external reviewer
has attacked either profile.

**What would raise it further.** A receipt-level differential against a
non-Python JCS implementation. A fuzzer over the canonicalizer.

**What would lower it.** A collision found by anyone outside the project.

---

## 3. Claim discipline — prose — 8.8 {#claim-discipline-prose}

*Measures: whether the words stay inside the evidence.*

**Evidence supporting.** `NON_CLAIMS.md` is a genuine artifact — ten non-claims,
each with why-not and what-instead. The C0–C5 claim-boundary manifest maps every
public claim to level, lane, support, tests, files, and non-claim. `VISION.md`
labels itself aspirational in its first line and refuses to relax a single
non-claim. `cliff.py` refuses confidence claims on post-selection intervals *in
code*, with a comment explaining the statistics. Legacy metrics emit
`FutureWarning` naming what they are not.

This is, in places, stronger than the flagship's. The C0–C5 lattice has no
Ghost-Ark equivalent.

**What would raise it.** Adopting Vinctura's stricter rule — *no orphan claims:
every public sentence carries the command that falsifies it, or an explicit
limitation* — and applying it to the README. Conceding the 1935 point
([F-18](FINDINGS_REGISTER.md#f-18)) above the fold instead of leaving it for a
critic to make.

**What would lower it.** Any README sentence a reader cannot map to a command.

---

## 4. Claim discipline — enforcement — 4.0 {#claim-discipline-enforcement}

*Measures: whether the discipline survives an author in a hurry.*

> **Revised upward from 2.1 on 2026-08-19.** The original score rested partly on
> the claim that `validate_claim_boundary_manifest.py` was "wired into nothing —
> not CI, not the Makefile, not a test." The first two were right and the third
> was wrong: it is called by
> `tests/unit/docs/test_claim_boundary_manifest.py`, which runs in the pytest
> suite CI executes on four Python versions. See the correction note in
> [F-12](FINDINGS_REGISTER.md#f-12).

**Evidence supporting.** The claim-boundary manifest **is** enforced, and more
thoroughly than most such artifacts: the validator checks required keys, unique
ids, claim-level resolution, non-empty non-claims per claim, and the existence
on disk of every `supporting_files` path. It passes with zero errors across all
eight declared claims. `permission_compiler.py` carries a forbidden-phrase list
and is 95.28% covered. The claim-governance verifier checks non-claim substance
rather than exact strings. `check_artifact_boundary.py` runs in CI.

**What lowers it.** No forbidden-phrase scanner in CI at all — the gate
Ghost-Ark runs on every file has no counterpart here. Two divergent theorem
ledgers with no governing rule ([F-15](FINDINGS_REGISTER.md#f-15)). Strict typing
declared for the `cc` package and enforced on 7 of 90 files
([F-02](FINDINGS_REGISTER.md#f-02)) — claim inflation expressed in build
configuration. Two narrow manifest gaps remain unenforced
([F-12](FINDINGS_REGISTER.md#f-12)): test paths are not checked for existence,
and the Markdown and JSON manifests are not cross-checked.

The prose is 8.8 and the enforcement is 4.0. The gap is real but narrower than
first reported: what is missing is a phrase-level scanner and honest typing,
not the claim ledger itself.

**What would raise it.** A claim scanner with negation and allowlist handling
from day one — a naive port of Ghost-Ark's would fire on 70 negated uses of
"guarantee" and be disabled within a week. The typing ratchet. Closing the two
manifest gaps.

**What would lower it.** A published claim that no artifact supports, merged
green.

---

## 5. Test depth — 6.4 {#test-depth}

*Measures: whether the suite would notice a defect.*

**Evidence supporting.** 689 tests, 108 files, unit/integration/e2e/regression/
performance/experiment separation, adversarial tests for the transparency log,
fail-closed semantics tested, selective-inference guard tested, 69.91% coverage
with branch coverage on.

**What lowers it.** The report CLI at 0% ([F-13](FINDINGS_REGISTER.md#f-13)).
`core/stats.py` at 38% over 756 statements
([F-14](FINDINGS_REGISTER.md#f-14)) — the statistics engine, in a repository
about statistical honesty. No coverage gate. No mutation testing, so the score is
*unknown*, not low. Hypothesis is a declared dependency with no gate. A
regression test that shells out to the whole unit suite and masks real failures
([F-20](FINDINGS_REGISTER.md#f-20)). Nine tests that have never run in CI
([F-17](FINDINGS_REGISTER.md#f-17)).

**What would raise it.** Coverage floor at the measured 69.91%, ratcheting.
Mutation testing on `kernel/` and `reporting/` with a published score. Property-
based tests on the statistics module. CLI golden tests.

**What would lower it.** A defect found downstream that the suite could have
caught.

---

## 6. External reproducibility — 4.2 {#external-reproducibility}

*Measures: whether a stranger can reproduce the artifacts.*

**Evidence supporting.** `scripts/reproduce_paper.py`,
`scripts/verify_paper_artifacts.py`, a claim-governance capsule with expected
outputs and a manifest, `examples/minimal/`, a devcontainer, a Dockerfile,
`make reproduce-*` targets, pinned artifact manifests.

**What lowers it.** The documented install path produces a red suite
([F-01](FINDINGS_REGISTER.md#f-01)) — the first thing a stranger encounters is a
failure that is not a real failure. No single-command reproduction. No replay
manifest in Ghost-Ark's sense. **No stranger has ever done it**; every adversary
in this tree was written by the author of the code it attacks.

**What would raise it.** `make verify` that goes green from a clean clone on a
clean machine, tested in CI from `.[test]` specifically. A reviewer Dockerfile.
One external person reproducing the paper artifacts and saying so in writing.

**What would lower it.** An external reviewer failing to reproduce.

---

## 7. Consumability as a library — 6.8 {#consumability-as-a-library}

*Measures: whether a downstream project can depend on this instead of rewriting it.*

> **Revised upward from 2.6 on 2026-08-19**, on delivered evidence rather than
> intent. What moved it is listed below; each item is a command.

**Evidence supporting.** `cc.compose` takes named marginals and returns a sharp
interval, with no ROC, Youden, threshold, or operating-point concept anywhere in
its signature — pinned by a test that greps the signatures for that vocabulary,
so the surface cannot drift back to the shape a consumer already walked away
from. Deterministic predicates are first-class; detectors reach the same surface
through `marginal_from_operating_point`, which points *inward*.

The result object carries what qualifies the number: the independence baseline,
the understatement factor, the binding event, the marginal provenance
(`measured` / `assumed` / `supplied`), and the non-claims. A bound over assumed
rates cannot be serialized without the word `assumed` attached.

`cc-guard` exposes the inference guards as a stdin/stdout JSON subcommand *and*
as a pure-data decision table, so a non-Python caller needs no Python process.
A test asserts the table and the implementation agree rule by rule, and another
asserts `cc-guard` reaches the same verdict as `cc.kernel.cliff.cliff_certificate`
in both directions — the CLI can neither permit what the kernel refuses nor
refuse what it permits.

**The acceptance gate passed.** `tests/acceptance/test_external_consumer_reproduction.py`
reproduces an external consumer's *published* four-control result from
`cc.compose`: interval `[0, 0.01]`, independence baseline `1.2e-5`,
understatement factor `833×`, all three published scenarios, and their
sensitivity finding that improving a weak control moves the upper bound by
`0.00pp`. Ten tests. Their 214 lines of JavaScript could be deleted.

**What still lowers it.** No ingest path for the contract Ghost-Ark declares
binding ([F-11](FINDINGS_REGISTER.md#f-11)) — that is W6. The constrained LP path
is not exposed through `cc.compose`; only the closed form is. And no downstream
project has actually adopted any of this yet: the obstacle is removed, the
adoption is theirs to make.

**What would raise it further.** W6's `cc.ingest.discretization`. Side
constraints on the `cc.compose` surface. A downstream repository importing it in
anger.

**What would lower it.** A fourth reimplementation appearing anyway — which
would mean the surface is still the wrong shape.

---

## 8. Cross-implementation agreement — 6.5 {#cross-implementation-agreement}

*Measures: whether independent implementations produce the same answer.*

> **Revised upward from 0.0 on 2026-08-19.** There was nothing to score; now
> there is.

**Evidence supporting.**

- `conformance/cc-kernel-v1/` is published: 24 accept cases with pinned exact
  values, 8 reject cases with typed refusal reasons, a manifest with digests and
  a declared 1e-12 tolerance, and a 256-line normative `SPEC.md`. Every accept
  case is cross-checked against the finite-atom LP at generation time, and the
  build **refuses to write a case** the closed form and the LP disagree on.
- `verifiers/node/cc_compose_verify.mjs` is a zero-dependency Node
  implementation written from `SPEC.md` and the JSON, not from the Python. It
  passes 24/24 accept and 8/8 reject.
- `scripts/differential_compose.py` fuzzes both sides on randomized inputs,
  including malformed ones, requiring agreement on the answer *or* the refusal.
  **23,000 cases across six seeds, zero disagreements** — and it earned its
  keep immediately by finding a real bug (see below).
- Three of these run in CI as pytest cases, so agreement is enforced rather than
  demonstrated once.

**The fuzzer found a bug on its first run.** `dependence="countermonotone"` with
exactly *one* event: the Python raised `IndexError`, the Node silently returned
`NaN`. Both were wrong, differently, and the curated corpus had not thought to
ask. Both are fixed, and the case is now pinned as
`reject-countermonotone-one-event` with its provenance recorded in the corpus.
That is the argument for randomized differential testing over a corpus alone.

**Why this is 6.5 and not 8.0.** The honest limit: **the Node implementation and
the Python reference were authored in the same project.** A specification that
is wrong yields two implementations that are wrong together. This is a
differential-testing instrument, not an independent replication, and both the
verifier's own output and the corpus manifest say so in their non-claims.

The one check here that is *not* same-author is the external oracle: an
outside project's **published** numbers, produced independently for its own
purposes before this corpus existed, reproduced exactly. That is one oracle, on
one scenario family.

**What would raise it.** Ghost-Ark's TypeScript `ccCorrelation.ts` running this
corpus. A third implementation by someone who has not read either of these. More
external oracles. Extending the corpus to the constrained LP path, which neither
the corpus nor the fuzzer currently covers.

**What would lower it.** A disagreement found by anyone outside the project — or
a case quietly weakened to make an implementation pass.

---

## 9. Empirical grounding — 2.4 {#empirical-grounding}

*Measures: whether any number here came from a real system.*

**Evidence supporting.** A dependence benchmark, adapter implementations for
Llama Guard / NeMo / Guardrails-AI, a rails demo, correlation-cliff copula
simulations, week-by-week experiment memos.

**What lowers it.** **No `p_i` in this repository was measured against a
production guardrail.** Every marginal is supplied, synthetic, or assumed. The
adapters exist; no measurement campaign has run through them. The experiment lane
has never executed in CI ([F-17](FINDINGS_REGISTER.md#f-17)). VISION names the
Correlation Atlas as the empirical keystone and the most citable artifact
available; it does not exist.

**What would raise it.** The Atlas: two or three real guardrails on one public
jailbreak corpus, reporting per-pair marginals, observed joint rates, phi, and
the independence-versus-worst-case gap, each row emitting a certificate. Even a
small one changes this score more than any amount of engineering.

**What would lower it.** Publishing a measured-looking number that came from a
simulation.

---

## 10. Statistical honesty machinery — 7.9 {#statistical-honesty-machinery}

*Measures: whether the code refuses claims the data cannot support.*

**Evidence supporting.** This is the repository's most distinctive asset. The
post-selection refusal at `cliff.py:321` returns `discovery-only` with no
confidence claim, and the comment explains why: *"A discovery is a hypothesis;
only held-out data can certify it."* Confirmatory/exploratory provenance is a
first-class type. Wilson intervals rather than normal approximations. Legacy
metrics warn about what they are not. Non-finite bootstrap samples are dropped
with a warning rather than silently.

A downstream repository independently identified this guard as *"the single most
elegant thing available in this program."* External parties do not say that about
statistical software often.

**What would raise it.** Adopting Ghost-Ark's reporting rules as executable
assertions: no proportion without a denominator, no interval over a curated
census, no interval below n = 30, no point estimate without dispersion. Ghost-Ark
enforces these in `reportProportion` / `assertCensusReporting`; cc-framework
enforces the post-selection rule and not the others. Confidence bands around
`[L, U]` under estimated marginals — VISION Pillar I — would raise it furthest.

**What would lower it.** Any path that emits a confidence interval over a
hand-authored corpus.

---

## 11. Evidence-governance architecture — 7.6 {#evidence-governance-architecture}

*Measures: whether evidence carries its own boundaries.*

**Evidence supporting.** Claim envelopes, role ontology, decay/hazard policy,
extremal scenarios, confirmatory protocol, Merkle log with adversarial tests,
permission compiler, a governance capsule with expected artifacts and a manifest.
9,946 lines, mostly 80–95% covered. Non-claims are carried *inside* signed
payloads, not merely alongside them.

**What lowers it.** No ingest contract for the upstream that feeds it
([F-11](FINDINGS_REGISTER.md#f-11)). Two schemas with a non-resolvable `$id` and
version in the title ([F-19](FINDINGS_REGISTER.md#f-19)). The governance verdict
is only as good as the canonicalization beneath it — dimension 2 at 3.8 caps this
one.

**What would raise it.** The discretization ingest module. Versioned resolvable
schema identifiers. An evidence-window contract in Ghost-Ark's sense.

---

## 12. Formal readiness — 3.1 {#formal-readiness}

*Measures: distance to machine-checked proof.*

**Evidence supporting.** Theorem ledger with explicit proof status per theorem
and honest labelling of what is "standard theorem, computationally witnessed"
versus proved here. Explicit invariants in the kernel. LP sharpness follows from
finite-dimensional convexity, which is a proof, just not a mechanised one.

**What lowers it.** No mechanised proof of any kind. No TLA+, Alloy, Lean, or
Coq. No exact-arithmetic cross-check of the float LP path. Dual certificates are
not emitted, so LP optimality is asserted by the solver rather than independently
checkable.

**What would raise it.** Emitting dual certificates — the cheapest real move,
turning "the solver said optimal" into "here is the certificate, check it
yourself." An exact rational LP for small `m` as an oracle. A Lean statement of
the FH recovery theorem is a stretch goal, not a v0.4 target.

**What would lower it.** Claiming formal verification. The word does not appear
in this repository today; keep it that way.

---

## The committee's closing question

For every score above: *what evidence would raise it, and what evidence would
lower it?* Both are stated for all twelve. Where a future revision cannot answer
both, that score should be deleted rather than defended.
