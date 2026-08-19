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
| 2 | Canonicalization & provenance integrity | **3.8** | 8.5 | W3 |
| 3 | Claim discipline — prose | **8.8** | 9.2 | W2 |
| 4 | Claim discipline — enforcement | **2.1** | 8.5 | W2 |
| 5 | Test depth | **6.4** | 8.5 | W4 |
| 6 | External reproducibility | **4.2** | 8.0 | W0, W7 |
| 7 | Consumability as a library | **2.6** | 8.5 | W5 |
| 8 | Cross-implementation agreement | **0.0** | 8.0 | W5 |
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
- **2.6 / 10** as a *consumable library* — three downstream repositories
  reimplemented its calculus rather than depend on it, and one of them wrote
  down why.

The gap between the first and third numbers is the whole plan.

---

## 1. Mathematical correctness — 8.9

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

## 2. Canonicalization & provenance integrity — 3.8

*Measures: whether a receipt identifies the document it claims to.*

**Evidence supporting.** Canonical form is deterministic, sorted, compact,
UTF-8, rejects NaN/Infinity, rejects non-string keys, rejects non-JSON-native
types, and excludes the receipt hash from its own preimage. 91.89% covered.
That is a real, careful implementation.

**What lowers it, now.** [F-03](FINDINGS_REGISTER.md#f-03): two Unicode-distinct
keys silently become one, with no error, inside the function every signed
artifact routes through. [F-04](FINDINGS_REGISTER.md#f-04): five of six number
forms diverge from RFC 8785. [F-05](FINDINGS_REGISTER.md#f-05): integers above
2^53 cannot survive a JS verifier. [F-07](FINDINGS_REGISTER.md#f-07): duplicate
keys accepted last-wins on read.

The score is not 3.8 because the code is careless. It is 3.8 because the code is
careful and **has never been attacked**. Ghost-Ark found five unintended kernel
members in its own pipeline by building a 31-class census. cc-framework has not
run the equivalent, and the one pass performed for this plan found four issues
in under an hour.

**What would raise it.** A committed adversarial corpus with declared intent per
class. Collision detection that fails closed. A documented, versioned
`cc.canonical.v1` profile stating its relationship to RFC 8785. Strict duplicate-
key rejection on read.

**What would lower it.** A collision found by anyone outside the project.

---

## 3. Claim discipline — prose — 8.8

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

## 4. Claim discipline — enforcement — 2.1

*Measures: whether the discipline survives an author in a hurry.*

**Evidence supporting.** `permission_compiler.py` carries a forbidden-phrase list
and is 95.28% covered. The claim-governance verifier checks non-claim substance
rather than exact strings. `check_artifact_boundary.py` runs in CI.

**What lowers it.** No claim scanner in CI at all
([F-12](FINDINGS_REGISTER.md#f-12)). `validate_claim_boundary_manifest.py` is
wired into nothing — not CI, not the Makefile, not a test. Two divergent theorem
ledgers with no governing rule ([F-15](FINDINGS_REGISTER.md#f-15)). Strict typing
declared and enforced on 7 of 90 files ([F-02](FINDINGS_REGISTER.md#f-02)) — claim
inflation expressed in build configuration.

The prose is 8.8 and the enforcement is 2.1. **Everything holding this repository
honest is currently a person remembering to be honest.**

**What would raise it.** The manifest validator in CI. A claim scanner with
negation and allowlist handling from day one — a naive port of Ghost-Ark's would
fire on 70 negated uses of "guarantee" and be disabled within a week. A test
that fails when a manifest row names a file that does not exist.

**What would lower it.** A published claim that no artifact supports, merged
green.

---

## 5. Test depth — 6.4

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

## 6. External reproducibility — 4.2

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

## 7. Consumability as a library — 2.6

*Measures: whether a downstream project can depend on this instead of rewriting it.*

**Evidence supporting.** Installable, four console entry points, a lazy-loading
kernel aggregate, `docs/api.md`, four downstream-facing design specs.

**What lowers it.** Three independent reimplementations of the core calculus
([F-09](FINDINGS_REGISTER.md#f-09)). A written rejected-adoption report from a
real consumer naming the ROC framing as the obstacle
([F-08](FINDINGS_REGISTER.md#f-08)). A dated, explicit request from a consumer to
route through `cliff.py:321` that cannot be honoured across a language boundary
([F-10](FINDINGS_REGISTER.md#f-10)). No ingest path for the contract Ghost-Ark
declares binding ([F-11](FINDINGS_REGISTER.md#f-11)).

This is the lowest non-zero score and the highest-leverage one. The mathematics
is 8.9 and nobody can use it.

**What would raise it.** `cc.compose` with marginals in and bounds out, no ROC
concept in the signature. `cc-guard` as a stdin/stdout JSON surface. Vinctura's
214 lines deleted and their numbers reproduced from the library.

**What would lower it.** A fourth reimplementation.

---

## 8. Cross-implementation agreement — 0.0

*Measures: whether independent implementations produce the same answer.*

There is nothing to score. No second implementation exists inside this
repository, no conformance corpus is published, and the two external
implementations have never been compared to this one on a single case.

Zero is the honest score. It is also the easiest to move: Ghost-Ark already
operates E5 (cross-language verifier agreement) and E7 (differential fuzz)
successfully. The machinery exists; it has never been pointed here.

**What would raise it.** `conformance/cc-kernel-v1/` published with exact expected
values and an adversarial section. A second implementation in another language
inside this repository. A differential fuzz harness comparing them. Agreement
demonstrated against Ghost-Ark's TypeScript and Vinctura's JavaScript.

**What would lower it.** Nothing. It cannot go lower.

---

## 9. Empirical grounding — 2.4

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

## 10. Statistical honesty machinery — 7.9

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

## 11. Evidence-governance architecture — 7.6

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

## 12. Formal readiness — 3.1

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
