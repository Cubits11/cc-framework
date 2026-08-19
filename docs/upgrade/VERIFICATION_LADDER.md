# Verification Ladder — cc-framework

> Modelled on Ghost-Ark's repository verification ladder. Each tier states what
> passing establishes **and what it does not**. A tier that cannot name its
> non-claims is decoration.
>
> Tiers marked **[planned]** do not exist yet and are workstream deliverables.
> They are listed here so the target shape is legible, not to imply they run.

---

## Why a ladder rather than "run the tests"

`pytest` returning 0 currently means one thing: no assertion in this repository
disagreed with this repository. That is regression resistance. It is not
correctness, it is not agreement with any other implementation, and it is
certainly not safety.

A ladder separates those. A reviewer who runs Tier 1 knows exactly which of them
they have obtained.

---

## Tier 0 — Fast validation

```bash
make fmt-check          # ruff format --check
make lint               # ruff check
make types              # [planned] mypy at declared scope, against the ratchet
make claims             # [planned] claim scanner + boundary-manifest validation
make docs-check         # [planned] required docs exist and cross-references resolve
```

**Establishes.** The tree is formatted, lints clean, introduces no new type
errors beyond the recorded ratchet, contains no forbidden claim phrasing outside
the allowlist, and every claim-manifest row names a file and test that exist.

**Does not establish.** That any computation is correct. Tier 0 is a text and
configuration gate.

Runtime target: under 60 seconds.

---

## Tier 1 — Kernel correctness

```bash
make test-kernel                    # existing
make kernel-conformance             # [planned] the published corpus, in-repo
make kernel-properties              # [planned] Hypothesis over the marginal simplex
make kernel-duals                   # [planned] verify emitted dual certificates
```

**Establishes.** The LP recovers Fréchet–Hoeffding on the tested family;
infeasible constraint sets raise rather than returning a number; monotone
tightening holds; every conformance case in `conformance/cc-kernel-v1/` matches
its pinned expected interval; sampled marginals satisfy the declared invariants;
emitted dual certificates verify against their primal solutions.

**Does not establish.** That the marginals fed to the kernel are valid,
representative, or measured. That an implementation in another language agrees
(that is Tier 3). That the result means anything about a deployed system.

Current status: the first target exists and passes; the rest are W4/W5
deliverables.

---

## Tier 2 — Receipt and canonicalization integrity

```bash
make canon-corpus                   # [planned] the adversarial canonicalization census
make receipt-verify-sample          # [planned] verify a committed golden report
make receipt-verify-corpus          # [planned] every malicious fixture must be REJECTED
make receipt-replay                 # [planned] rebuild artifacts, assert byte-identity
make test-reporting                 # existing
```

**Establishes.** Canonical bytes are deterministic and stable; every class in the
adversarial corpus produces the declared verdict; documents that must collapse do
and documents that must stay distinct do; malicious fixtures fail closed;
a committed capsule replays byte-identically.

**Does not establish.** Semantic truth of the report. That the numbers inside are
correct. That the producer was honest — a receipt binds bytes, not intentions.

This tier is where [F-03](FINDINGS_REGISTER.md#f-03) through
[F-07](FINDINGS_REGISTER.md#f-07) get caught, permanently.

**The rule that makes this tier real:** *never weaken the corpus to make a test
pass.* A corpus class that starts failing is a regression in the kernel, not a
problem with the corpus.

---

## Tier 3 — Cross-implementation agreement

```bash
make conformance-publish            # [planned] emit conformance/cc-kernel-v1/
make differential-agreement         # [planned] Python vs. second implementation
make differential-fuzz              # [planned] randomized cases, both sides, compare
```

**Establishes.** Two independently written implementations, on the same inputs,
produce the same intervals to a declared tolerance — and where they diverge, the
divergence is recorded as a finding rather than reconciled by fiat.

**Does not establish.** That either is correct. Two implementations can share a
misreading. Agreement is necessary, not sufficient.

**Why this tier matters most.** Three implementations of this calculus exist
across the program ([F-09](FINDINGS_REGISTER.md#f-09)) and none has ever been
compared to another. Ghost-Ark treats breaking verifier agreement as a critical
architectural event. Here, agreement has never been established, so it cannot yet
be broken.

Breaking agreement, once established, is a critical architectural event here too.

---

## Tier 4 — Downstream contract conformance

```bash
make ingest-corpus                  # [planned] discretization receipts: accept/reject
make guard-contract                 # [planned] cc-guard decision table vs. implementation
make compose-acceptance             # [planned] reproduce Vinctura's published numbers
```

**Establishes.** A `ghost.discretization_rule_receipt.v1` missing lineage, or
carrying a comparator inconsistent with score polarity, or outside its validity
window, is **refused**; the guard decision table matches the Python
implementation case-for-case; the composition surface reproduces a real
consumer's published bounds.

**Does not establish.** That the upstream discretization was appropriate, that the
threshold was well chosen, or that the score was calibrated. The contract checks
that evidence carries its lineage, not that the lineage is wise.

---

## Tier 5 — Full baseline

```bash
make test                           # the whole suite
make cov                            # coverage against the floor
make mutation                       # [planned] mutation score on kernel/ and reporting/
```

**Establishes.** Repository consistency, regression resistance, coverage at or
above the recorded floor, and a published mutation score.

**Does not establish.** Correctness. This is the tier most likely to be
misquoted; state its boundary whenever it is cited.

---

## Tier 6 — Empirical lanes (scheduled, not per-commit)

```bash
CC_RUN_EXPERIMENTS=1 CC_RUN_PERF=1 make test-empirical   # [planned]
make atlas                                               # [planned] Correlation Atlas run
```

**Establishes.** The experiment and performance lanes execute and report; measured
co-failure dependence on the declared cohort is published with its provenance and
its non-claims.

**Does not establish.** That a measured regret on one cohort predicts another.
The Atlas is **descriptive**, never predictive. A number measured on one corpus
of jailbreaks says nothing about the next one.

Runs on a schedule, publishes artifacts, and is allowed to be red without
blocking merges — a lane that must stay green will be quietly weakened until it
does.

---

## Independent verification boundary

Once Tier 3 exists, the following hold, borrowed directly from Ghost-Ark and
adapted:

- A standalone verifier must not import `cc` internals. It re-derives from the
  published corpus and the schema.
- Independent verification must recompute the interval, not accept a reported one.
- Conformance failures must be deterministic — the same input yields the same
  verdict on every run.
- Malicious fixtures must fail closed, never "best effort."
- Unicode-sensitive canonicalization must remain stable across releases.
- **Never weaken verifier strictness to make tests pass.**

---

## What no tier establishes

No tier on this ladder, at any level, establishes that:

- an AI system is safe,
- a guardrail is effective,
- a threshold is appropriate,
- a marginal is representative,
- a dependence structure is stable,
- a cohort generalizes,
- a deployment decision is correct,
- a compliance obligation is met.

The ladder measures whether this repository does what it says. Whether what it
says is worth anything is a separate question, answered by
[COMMITTEE_SCORECARD.md](COMMITTEE_SCORECARD.md) and, ultimately, by someone
outside the project.

---

## North star

cc-framework succeeds when a hostile reviewer can say:

> I do not trust the author.
>
> I do not trust the README.
>
> I do not trust the theorem ledger.
>
> But I can recompute the interval from the witness.
>
> I can check the dual certificate without re-solving the LP.
>
> I can run the conformance corpus against my own implementation.
>
> I can find the canonicalization collisions they already found, and no others.
>
> I can see which marginals were measured and which were assumed.
>
> I can read what they refused to claim, and check that the code refuses it too.
>
> And I can determine exactly what cc-framework does and does not establish
> without asking anyone.

Everything else is secondary.
