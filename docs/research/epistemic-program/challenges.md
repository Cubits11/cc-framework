# Challenges

Adversarial participation, run as public research rather than as marketing.

A challenge is an invitation to falsify something specific in this repository.
Results are published whether or not they are flattering, and the uncomfortable
ones are published first.

## Completed

### CH-001 — Break the canonical page's drift guard

**Challenge.** Can the canonical page be made to display `BYTES MATCH` over a
different underlying fixture, without causing the drift guard to fail?

**Answer: yes.** Two attempts were run in a detached worktree at `ceeca6b`.

#### Attempt A — repair the consistency chain

Forge `expected/calibration.json` (`realized_fpr` 0.041666666667 → 0.011111111111),
recompute its digest, patch `manifest.expected.json`, and patch the page's
embedded copy and digest to match.

| Guard | Result |
| --- | --- |
| `tests/unit/docs/test_canonical_page_fixture.py` | **4 passed** — the forgery is internally consistent |
| `tests/integration/test_claim_governance_capsule.py` | **failed** — regeneration from declared inputs produced the original file |

The drift guard alone does not detect this. The capsule integration test does,
because it rebuilds the artifact rather than comparing it to itself.

*(A first run of this attempt reported a false failure. The attack script used
`re.sub` with a replacement string containing `\n`, which Python interprets as a
newline, corrupting the page instead of forging it. The tooling failed, not the
guard. It is recorded here because a challenge result produced by a broken
attacker is worse than no result.)*

#### Attempt B — move the forgery upstream

Change the **declared input** (`inputs/capsule_config.json`, same field, same
value), regenerate the capsule with `build_capsule.py`, refresh all eleven
expected artifacts and the manifest from the regenerated outputs, then update the
page.

| Guard | Result |
| --- | --- |
| `tests/unit/docs/test_canonical_page_fixture.py` | **4 passed** |
| `tests/integration/test_claim_governance_capsule.py` | **5 passed** |
| Governance verdict in the regenerated manifest | `pass` |
| Report receipt | new, valid, `c1643673596fab68…` |
| Canonical page in a real browser | `BYTES MATCH` over `"realized_fpr": 0.011111111111` |

Every check green. Every hash correct. The displayed number is fiction.

#### What CH-001 establishes

The guard chain establishes **deterministic reproducibility from declared
inputs**. It cannot distinguish a measured input from an asserted one. This is
rung 4 of the ladder in
[evidence scope inflation](evidence-scope-inflation.md).

#### What CH-001 does not establish

- Not that the capsule is broken. It does exactly what it says, and Attempt A
  shows the chain catching the naive forgery.
- Not a general result about reproducible builds. One repository, one chain.
- Not that an attacker could do this to the published repository — the forgery
  requires commit access, and a reviewer reading the diff would see the input
  change. The finding is about what the *green checks* prove, not about
  compromise.

#### What changed because of it

- The limitation is now stated on the canonical page itself, not only here.
- The drift guard's docstring states what it does not establish.
- A new entry in `docs/research/NON_CLAIMS.md`.
- Closing rung 4 requires an anchor outside the repository. There is none today,
  and the documents now say so rather than implying otherwise.

### CH-002 — Close rung 4

**Challenge.** Propose a mechanism that would let an outside reviewer
distinguish a measured input from an asserted one, without requiring them to
trust the author.

**Answer: no repository-local mechanism can, and the reason is uncomfortable —
deterministic reproducibility is what makes the forgery cheap.**

#### The argument

Let `R` be a repository controlled by author `A`, and let `V` be any
verification procedure whose inputs are entirely contained in `R`. `V` cannot
distinguish "input `x` was measured" from "input `x` was asserted by `A`".

`V` is a function of `R`'s contents, so `A` can compute it. To place a chosen
value `x'` into a state that `V` accepts, `A` does not need to invert anything:
the generator's outputs are a deterministic function of its inputs, and the
checks verify exactly that functional relationship. So `A` edits `x'`,
regenerates, and `V(R') = holds` by construction. CH-001 is this argument
executed rather than asserted.

The uncomfortable part is that the property being exploited is the one we want.
Reproducibility guarantees that a change to an input propagates consistently
through every downstream artifact, hash, and receipt. That is exactly why the
capsule is valuable — and it applies to honest and dishonest inputs equally:

> **Reproducibility amplifies consistency, not truth.**

A less reproducible pipeline would be *harder* to forge coherently, and worse in
every other respect. This is not an argument against reproducible builds. It is
an argument for knowing which of the two things they establish.

#### One cheap partial defence, and its limit

`realized_fpr` is `0.041666666667`, which is exactly `1/24`. The capsule ships
`inputs/failure_matrix.csv`, which has exactly 24 rows. The declared statistic
has a denominator structure matching the declared sample size — but nothing
checks the relationship, because the summary is a *declared input* rather than a
*derived* one.

CH-001's forgery set it to `0.011111111111`, which is `1/90`. That is not
expressible as a count over 24 observations. A **derivability check** — requiring
declared statistics to be expressible over the declared sample size — would have
caught it, costs almost nothing, and needs no external party.

It does not close rung 4. An attacker who picks `2/24` instead of `1/90` passes.
It raises the cost of forgery from *any number* to *any number consistent with
the declared n*, which is a real improvement and a bounded one.

**It is deliberately not implemented.** The columns of the failure matrix give
rates of 9/24, 12/24, and 9/24; none is 1/24. So `realized_fpr` is plausibly a
false-positive count over the same 24 prompts, and plausibly a quantity from a
different sample that happens to share a denominator. Implementing a check that
assumes the first reading would encode an unverified assumption about what the
field means — the same error class this whole program is about, committed in the
act of defending against it. The correct next step is to establish what the field
denotes, then check it.

#### What would actually close it

Ranked by cost, each closing a different threat:

| Anchor | Closes | Leaves open |
| --- | --- | --- |
| Timestamp the input at collection (RFC 3161, a transparency log) | Fabricating or revising an input *after seeing results* | Fabricating it at collection time |
| Signature by the measuring instrument or a second party | Binding the value to someone other than its writer | Trust in that party |
| Independent replication | Measurement itself | Nothing — but it is a social process, not a mechanism |

Only the third establishes measurement, and it is definitionally not
repository-local. The first is the cheapest and maps precisely onto
preregistration: freeze the protocol, and the input, before the observation.

#### Convergence worth noting

This is the same shape as the attestation boundary in the
[eight-week plan](../future-expansion/eight-week-plan.md): an enclave attests a
measurement, App Attest validates an app instance, C2PA binds an assertion to an
artifact — and none of them establishes that the assertion is true. Rung 4 is
that boundary, met from the reproducible-build direction instead of the
hardware-attestation direction. Two tracks, one wall.

#### Status

Partially answered. The impossibility argument is stated and demonstrated; the
anchors are named and **none is implemented**. CH-002 stays open for anyone who
can refute the argument or implement the first anchor.

## Open

### CH-003 — Make the film show something false

The film renders whatever is in `window.__VERDICT` and stamps `ILLUSTRATION`
when nothing is injected. Find a path that produces an un-stamped frame
displaying a verdict that no verifier issued.

### CH-004 — Find a reserved-vocabulary bypass

The report validator rejects reserved overclaim vocabulary in claim statements
and evidence roles. Construct a report that passes validation and still reads,
to an ordinary reader, as a safety claim. Metadata, field names, and alternate
routes all count.

### CH-005 — Widen an interval by adding evidence

The kernel's intervals should tighten monotonically as constraints are added.
Find declared evidence whose addition makes the reported interval wider, or an
argument for why the monotonicity test does not cover a real case.

### CH-006 — A docstring contract, falsified by a property test

**Not opened by us.** Hypothesis found it while the full suite was running at
the end of this session, and it is recorded here because a challenge programme
that only publishes the challenges it chose is not a challenge programme.

`ModelBase.migrate()` is documented as best-effort, and its test states the
contract explicitly:

> migrate() should be best-effort and never throw on arbitrary old dicts.

It throws. Given `{"updated_at": ""}` it raises a `ValidationError`, because
`migrate()` passes arbitrary values straight into `model_validate` and
`updated_at` carries a validator that rejects a non-numeric timestamp.

| Fact | Status |
| --- | --- |
| Reproduces on `main` | Yes, identically — verified in a detached worktree |
| Touched by this branch | No. `src/cc/core/models.py` is unmodified here |
| Deterministic in CI | No. The failing example lives in a gitignored `.hypothesis/` database, so CI may or may not draw it |

#### The interesting part

The obvious fix — make `migrate()` swallow invalid values for known fields — is
**silent repair**, which this repository rejects everywhere else. The report
validator refuses rather than repairing. The governance verifier fails closed.
A migration that quietly drops a malformed timestamp would hide exactly the kind
of data problem the rest of the codebase is built to surface.

So the defect is plausibly in the *contract*, not the code:

| Option | Effect | Cost |
| --- | --- | --- |
| A — make `migrate()` best-effort for real | Honours the docstring | Silent repair; contradicts the repository's fail-closed posture |
| B — narrow the contract | `migrate()` must not throw on **unknown** keys; a known field with an invalid value fails closed, as it should | Changes a documented contract and a test in the core model layer |

**Recommendation: B**, with the Hypothesis strategy excluding known field names
and an explicit test asserting that an invalid known field *does* raise.

**Not applied.** This is a semantic decision in the core model layer, made on a
pre-existing bug that is unrelated to this branch's work. Changing it
unilaterally would be its own kind of scope inflation — repairing something
quietly because it was inconvenient to the session's green build.

## How results are handled

1. Reproduce it here, in a detached worktree, with the commands recorded.
2. Publish the result — including a result that makes this project look worse.
3. Change the artifact, the limitation text, or both.
4. Record what changed, so the challenge's effect is inspectable too.

A challenge that produces no change to any artifact is recorded as such. That is
also a result.

## Non-claims

- A challenge going unanswered is not evidence that a property holds.
- The list of open challenges is not a claim that these are the only weaknesses.
  They are the weaknesses currently visible from the inside, which is exactly the
  vantage point that missed rung 4.
