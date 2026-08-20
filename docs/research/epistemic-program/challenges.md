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

## Open

### CH-002 — Close rung 4

Propose a mechanism that would let an outside reviewer distinguish a measured
input from an asserted one, without requiring them to trust the author. Signature
over the input at collection time, an independent measuring party, a
transparency log outside the author's control — or an argument that no
repository-local mechanism can do it, which would also be a result.

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
