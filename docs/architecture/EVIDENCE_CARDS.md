# Evidence Cards

> **Status: implemented.** `cc.evidence_card` emits `cc.evidence_card.v1` cards
> and a `cc.site_evidence_manifest.v1` bundle from the claim-boundary manifest.
>
> **Non-claim.** A card lists a claim and the command that tests it. It does not
> establish that the claim is true. A card carrying `verdict: pass` establishes
> only that the named command exited successfully on some host — nothing about
> deployment, safety, or the world.

An evidence card is a claim together with everything a reader needs in order to
**disagree** with it: how it was operationalized, which command reproduces it,
what would falsify it, what it assumes, and what it explicitly does not claim.

This repository emits cards. It does not host an Atlas, a site, or any surface
that renders them — those live elsewhere. What is here is the instrument.

---

## The object

```
claim → maturity → source revision → artifact digests → command
      → result → falsifier → counterevidence → assumptions → non-claims
```

Two fields are required that most claim registries treat as optional:

- **`falsifier`** — what observation would show the claim is wrong. A claim with
  no falsifier is an assertion, not evidence, and `EvidenceCard.__init__`
  refuses it.
- **`non_claims`** — what the card explicitly does not establish. Also required.

Both are now mandatory *in the claim-boundary manifest itself*, enforced by
`scripts/validate_claim_boundary_manifest.py`. Every claim in this repository
carries a falsifier because the validator will not accept one that does not.

---

## Three labels that never collapse

```
evidence_state      local-only | aws-synth-only | aws-live | illustrative
verdict             pass | fail | unverifiable | not-run
publication_state   draft | released | superseded | retracted
```

They answer different questions and **none implies another**:

- A `pass` that is `local-only` says nothing about deployed behaviour.
- `aws-live` evidence can record a `fail` — provenance is not quality.
- A `retracted` card may still carry a `pass`: the run happened, and the claim
  was withdrawn anyway.
- `released` says a card was reviewed for release, not that its claim is true.

This orthogonality is enforced, not merely documented:

| Enforcement | Where |
|---|---|
| No `status`, `score`, `is_ok`, `badge`, `health`, `overall` on the card | a test asserts each is absent |
| No aggregate field in the site manifest | a test asserts absence there too |
| No single-label renderer | `render_labels` returns all three or raises |
| Schema defines no aggregate | asserted against the committed schema |
| All 64 label combinations constructible | a test builds every one |

The last row is the important one. If a future change ever makes one label
constrain another — "a retracted card cannot be passing", say — that test
fails. The point of a failure museum is that withdrawn claims keep their
results visible.

Every label value ships with a **meaning string** rendered beside it, so a bare
token in a UI is never left to interpretation:

> `not-run` — The command has not been executed for this card. No result is
> claimed.

---

## `not-run` is the default, and that is the design

A card cannot acquire a passing verdict by being written confidently. The
generator emits `not-run` unless `--run` is passed, and `--run` executes the
commands and records what actually happened.

The committed cards in `evidence-cards/` are always the **not-run scaffold**. A
test asserts this: `pass` and `fail` counts must both be zero in the committed
manifest. Verdicts are host-specific, and a repository that ships someone's
laptop results as published evidence is doing the thing this object exists to
prevent.

---

## `unverifiable` is not `fail`

The distinction earned itself on the first run of the generator.

`make test-kernel` exited non-zero because its dependency-install step could not
reach the network. The kernel tests never executed. The first version of the
harness mapped any non-zero exit to `fail`, and so reported a **failure nobody
observed**.

The harness now never guesses. A non-zero exit becomes `fail` only when it can
be attributed to a check running and not succeeding — pytest exit code 1, whose
meaning is documented. Codes 2–5 mean usage error, internal error,
interruption, or no tests collected; in each the check did not run. Everything
else is `unverifiable`, with the reason recorded.

Per-command outcomes are preserved rather than collapsed:

```
[pass=1, unverifiable=1] pytest tests/unit/kernel/test_classical_frechet_special_cases.py
  -> exit 0: 6 passed in 0.63s | make test-kernel -> exit 2: ...
  | exit not attributable to a check running
```

The card verdict is the worst outcome present — conservative — but a reader can
still see that the tests passed and only the `make` target was unrunnable.

---

## What the generator does

```bash
python scripts/build_evidence_cards.py            # emit the not-run scaffold
python scripts/build_evidence_cards.py --run      # execute commands, record results
python scripts/build_evidence_cards.py --check    # verify the committed scaffold is current
```

It reads `docs/claims/claim_boundary_manifest.v0.1.json`, binds each claim's
supporting files by SHA-256, records the git revision (with a `-dirty` suffix
when the working tree differs — a card bound to a revision the tree does not
match is bound to the wrong thing), and writes one card per claim plus the site
manifest.

Because cards bind their supporting files **by digest**, editing any of those
files makes the committed cards stale. Regenerate as the last step before
committing; `--check` runs in CI and will catch it otherwise. That staleness is
the feature working: a card whose artifact digests no longer match the tree is
describing a repository that no longer exists.

Under `--run`, commands are executed only if they match a small allowlist
(`pytest `, `python `, `make `). A card generator that shells out arbitrary
strings from a data file is a code-execution surface, not an evidence tool.
Anything outside the allowlist is recorded `unverifiable` with that as the
reason.

---

## Current state

Eight claims, from the claim-boundary manifest:

| Label | Counts |
|---|---|
| `evidence_state` | `local-only` 7, `aws-synth-only` 1 |
| `verdict` | `not-run` 8 |
| `publication_state` | `draft` 8 |

The single `aws-synth-only` card is the enterprise reference, whose evidence is
moto emulation. Its assumptions say so on the card: *"Emulated evidence is never
live evidence, and no live-AWS claim rests on this lane."* Nothing in this
repository is `aws-live`, and nothing should claim to be.

All eight are `draft`. Release is a human decision in a publication workflow,
not a side effect of generation.

---

## What this does not do

Stated so absence is not read as capability:

- **No Atlas, site, or renderer.** This repository emits cards. Anything that
  displays them is a separate surface with its own design and its own review.
- **No hosting, no AWS, no deployment.** Nothing here provisions or publishes
  anything.
- **No cross-repository aggregation.** The cards describe *this* repository's
  claims. Combining cards from several projects into one view is a different
  problem with different failure modes, and none of them are solved here.
- **No verdict earned by generation.** The committed scaffold claims no results.
- **No commercial material.** Pricing, go-to-market, and venture planning are
  out of scope for a research repository, and the sibling institutional
  repository enforces the equivalent rule with a test.

## Binding rules

- **Never add a composite status.** Not to the card, the manifest, or the
  schema. The tests are the tripwire; do not edit them to make one pass.
- **Never commit run verdicts.** The committed scaffold is `not-run`.
- **Never let a claim into the manifest without a falsifier.** The validator
  refuses it; do not relax the validator.
- **`unverifiable` is never rounded to `fail` or `pass`.** It is its own fact.
