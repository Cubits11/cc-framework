# Evidence Scope Inflation

## The error

A verification establishes something narrow. A reader converts it into something
broad, without noticing that a conversion happened.

The canonical page's check makes the whole sequence visible in about four
seconds. The machine observes exactly one thing:

```text
sha256(bytes_in_the_box) == digest_recorded_in_the_manifest
```

The reader's mind, unprompted, performs three more steps:

```text
1.  hash(A) == manifest_hash        →   A is the file that was recorded
2.  A is the recorded file          →   the contents of A are accurate
3.  the contents of A are accurate  →   the claim A represents is true
```

Step 1 is licensed. Steps 2 and 3 are not. Nothing computed anything about
whether `0.041666666667` is a real false-positive rate. The number could have
been typed in.

**Bytes verified. Claim unresolved.**

That sentence is the whole discipline compressed into three words and a
concession. A less careful project ships `VERIFIED ✓`.

## Why it is worth building a laboratory around

The error is not exotic. It is the default behaviour of an attentive, honest
reader. Nobody has to be fooled. The inflation happens in the gap between what a
mechanism computes and what a person needs to know, and the mechanism is usually
silent about the gap because it does not know the gap exists.

Every mechanism worth trusting has this shape:

| Mechanism | Actually establishes | Routinely read as |
| --- | --- | --- |
| A digest match | These bytes are those bytes | This artifact is legitimate |
| A signature | A key signed this | The signer is honest and the content is true |
| A green CI check | The declared tests passed on this commit | The change is correct |
| An enclave attestation | This measurement ran in that enclave | The computation was correct |
| Provenance metadata | An assertion is bound to an artifact | The assertion is true |
| A governance PASS | The package is internally consistent under verifier rules | The system is safe |
| A reproducible build | The output follows from the declared inputs | The inputs describe reality |

The right-hand column is where products live. The left-hand column is where
truth lives. The distance between them is the territory.

## The ladder does not stop at three rungs

The three jumps above are the ones a reader makes. There are more, and they are
made by *builders* — including by this repository.

[CH-001](challenges.md) attacked this project's own guards to find the next rung.
The result, in full there and in summary here:

- Editing an evidence artifact and repairing the manifest and the page is caught,
  because regeneration from declared inputs disagrees. **Consistency alone is not
  enough, and the capsule knows it.**
- Editing the *declared input*, regenerating the whole chain, and refreshing the
  goldens passes **every** check. Governance verdict `pass`. A new, valid receipt.
  The page displays `BYTES MATCH` over a false-positive rate that was simply
  asserted.

So the next rung is:

```text
4.  the chain reproduces from declared inputs
        →   the declared inputs describe a real measurement
```

Also unlicensed. Every hash in that chain did its job correctly. Not one of them
reaches back to the world. The capsule establishes **deterministic
reproducibility from declared inputs** — which is genuinely valuable, and is not
measurement.

This is the same error as steps 2 and 3, one level up and committed by someone
who had already written the non-claims document. That is the point worth
sitting with: knowing about scope inflation does not immunize you against it.
Only a mutation test finds the next rung.

## The general form

Every verification is a function with a domain and a codomain.

```text
verify : artifact → {holds, fails}     over a specific property P
```

Scope inflation is silently widening `P` after the fact. The defence is not
better hashes. It is making `P` travel attached to the result, at the same size,
in the same frame, at the same moment.

Four things make that happen in practice:

1. **Co-located limitation.** The boundary is printed where the result is
   printed — not in a footnote, not in a linked document, not on a second page.
   On the canonical page, the limitation sits directly under the verdict and
   names the specific number it does not vouch for.
2. **A mutation affordance.** Give the reader a button that breaks it. A result
   the reader has personally made fail is a result they understand the scope of.
   Nobody who has clicked *Mutate one byte* believes the check is about truth.
3. **Refusal vocabulary.** Make the overclaim unsayable in the artifact itself:
   reserved words rejected by the report validator, a language quarantine over
   paper prose, forbidden *upgrades* enumerated in the claim manifest.
4. **An external anchor, or an honest admission that there is none.** Rung 4
   cannot be closed from inside the repository. It needs a signature over the
   input at collection time, an independent party who performed the measurement,
   or a transparency log outside the author's control. Until one exists, the
   correct move is to say so — which is what this document does.

## What this is not

It is not a claim that hashing is useless. The check is real, the digest match
is real, and tamper evidence is worth having.

It is a claim about *placement*: a mechanism that establishes something narrow
should be presented at the size of the thing it establishes. Most of the value
in this program comes from insisting on that, out loud, in public, including
when the resulting screen is less impressive than a green tick.

## Non-claims

- This document does not establish that scope inflation is common, only that it
  is available — and that this project committed a rung of it while explicitly
  trying not to.
- CH-001 is a demonstration on one repository's own guards. It is not a general
  result about reproducible-build systems.
- Naming the four countermeasures is not evidence that they work. Three of them
  are implemented here; the fourth, an external anchor, is not.
