# The daily epistemic loop

One working loop for the research program, small enough to finish daily.
The completion criterion is never commit count. It is:

> **One claim became more inspectable today.**

## The sequence

1. **ORIENT** — choose one claim and one uncertainty. Name the claim by its
   registry or ledger ID where one exists; name the uncertainty as a
   question a measurement could move. If no single claim/uncertainty pair
   can be named, the day starts with naming one, not with building.
2. **ESTABLISH** — add one evidence-bearing artifact: a run, a pinned
   document, a dataset row that conforms to a frozen contract, a
   reproduction. Bind it (revision, hash, path) the moment it exists;
   unbound evidence is a story.
3. **ATTACK** — run one falsifier, control, or adversary against the day's
   claim. A control that could not have embarrassed the claim does not
   count. Record the outcome even when — especially when — it is
   unflattering.
4. **TRANSLATE** — create one visual frame from the day's epistemic
   operation, in the web-system grammar
   (`visual_identity/web_system/README.md`): the operation's verb (reveal,
   constrain, compare, collapse, lock, expire) decides the motion or the
   still. A frame that answers none of the epistemic questions is not made.
5. **RECONCILE** — run the relevant validation lanes; re-pin what moved;
   update the ledger/status surfaces; write the next question down. The
   loop ends with the registries true, not with the editor closed.

## Rules of the loop

- Steps may be small. A one-line conformance failure honestly recorded
  beats a large untested feature.
- The loop never upgrades a claim's rung. E-ladder promotions are separate,
  deliberate events with their own review — see
  [RESEARCH_PROGRAM](research/RESEARCH_PROGRAM.md).
- Skipped days are skipped, not backfilled. The git history is the honest
  cadence record.
- "Inspectable" means a stranger could check it: a command they can run, a
  pin they can resolve, a witness they can verify, a non-claim that tells
  them where to stop.

## Public surface

The site's `/now/` page carries the current question and the loop's public
state. It is owner-attested and dated; operational notes stay private, and
nothing automates private notes onto the public page.

## Non-claims

The loop is a working discipline, not evidence. Completing it daily
establishes nothing about the quality of the claims it touches — the
falsifiers do that, when they fail to kill.
