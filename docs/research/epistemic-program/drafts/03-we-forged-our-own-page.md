# Draft 03 — We forged our own verified page

**Surface:** challenge. **Points at:** CH-001.

---

We published a page that verifies an evidence file in your browser and fails
closed if you change one character.

Then we tried to forge it. It worked. Here is exactly how, and what it means.

---

## Attempt one: repair the consistency chain

Change the false-positive rate in the evidence file from `0.041666666667` to
`0.011111111111`. Recompute its digest. Patch the manifest to record the new
digest. Patch the page's embedded copy to match.

Everything is now internally consistent, and the drift guard that exists
specifically to catch this **passes — 4 tests green.**

But the capsule integration test fails. It does not compare the artifact to
itself; it *rebuilds* the artifact from the declared inputs and finds the
original value. Caught.

Good. That is what a reproducible build is for.

## Attempt two: move the forgery upstream

So don't forge the output. Forge the input.

Change the same field in `inputs/capsule_config.json`. Regenerate the whole
chain with the project's own build script. Refresh all eleven expected artifacts
and the manifest from the regenerated outputs. Update the page.

| Check | Result |
| --- | --- |
| Drift guard | 4 passed |
| Capsule integration | 5 passed |
| Governance verdict | `pass` |
| Report receipt | new, valid |
| The page, in a real browser | **BYTES MATCH** over `"realized_fpr": 0.011111111111` |

Every check green. Every hash correct. The number is fiction.

## What this actually means

Not that the capsule is broken — it does exactly what it claims. The finding is
about what the green checks *prove*:

> The chain establishes deterministic reproducibility from declared inputs.
> It cannot distinguish a measured input from an asserted one.

That is a fourth unauthorized inference, one level above the three a reader
makes about a digest — and this one is committed by the *builder*. By someone
who had already written the project's non-claims document. By us.

Knowing about scope inflation does not immunize you against it. Only a mutation
test finds the next rung.

## What changed

The page now says this on the page, not in a document nobody opens. The
repository's non-claims file has a new entry. The drift guard's own docstring
states what it does not establish.

Closing the gap needs an anchor outside the repository: a signature over the
input at collection time, an independent party who performed the measurement, or
a transparency log outside the author's control. **We don't have one.** That is
now written down instead of implied.

If you can close it, or show that no repository-local mechanism can, that's
CH-002 and it's open.

Full write-up, commands, and the exact worktree state:
`docs/research/epistemic-program/challenges.md`
