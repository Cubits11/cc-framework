# Before You See It - Director's Cut

## Logline

The moment a seductive story is forced to become a test.

## Thesis

```text
A story can start a question.
Evidence must finish the answer.
```

Campaign line: **Keep the wonder. Check the claim.**

The film honors curiosity without letting it impersonate evidence. It does not
mock the story. It makes the standard of proof impossible to evade. That
distinction is the whole brief: tenderness first, then ruthlessness.

## Placement in the visual world

This is **Chamber 00** of the Claim Observatory - the room before the Arrival
Hall. Every later chamber assumes a claim already exists as a bounded artifact.
Chamber 00 is where the claim is born: the instant an intuition stops being a
feeling and is written down in a form that can lose.

It inherits the [World Bible V3](../claim_observatory/WORLD_BIBLE_V3.md) laws
without exception, including Law 10 - *the world must make overclaiming
uncomfortable*.

## Beat sheet

Runtime 15.000s. 1920x1080. 60fps. Silent master.

| Time | Beat | Visual | On-screen copy |
| --- | --- | --- | --- |
| 0.00-2.00 | **The lure** | An electric-blue notebook in darkness, lit warm and almost oracular from above left. Gold embers drift. A slow 4.5% push-in. | `THIS FEELS TRUE.` |
| 2.00-3.40 | **The freeze** | Hard cut: the light goes cold, the push-in stops dead, and the embers quantize into a 12x7 grid of candidate matches - `BLUE NOTEBOOK`, `CRACKED SCREEN`, `CURLY HAIR`, `VENDING MACHINE`, `FELT LIKE MOMENTUM`. The notebook drains of color and becomes one cell among eighty-four. | `BUT WHAT WOULD COUNT?` |
| 3.40-6.50 | **The ledger** | The grid dissolves. Four tablets snap in on a 0.6s beat, the row re-centering as each one lands. | `CLAIM` `FALSIFIER` `CONTROL` `NON-CLAIM` |
| 6.50-9.50 | **The lock** | A fifth slot appears, dashed and empty, labeled `RESULT`. A protocol bar locks a timestamp and a hash beneath the row while that slot is still empty. Three seconds of near stillness. | `NAME IT BEFORE YOU SEE IT.` |
| 9.50-11.80 | **The disagreement** | The slot fills with `INCONCLUSIVE`, not a green tick. The falsifier's rose accent flares once - it did its job. At 10.80 the result becomes `NEXT EXPERIMENT`. | `LET THE RESULT DISAGREE.` |
| 11.80-15.00 | **The lockup** | The five accents compress into a ledger rule under the wordmark. | `CC-FRAMEWORK` / `Keep the wonder. Check the claim.` |

The emotional center is the empty dashed slot at 6.50-9.50. Everything before it
is seduction; everything after it is consequence. The protocol is frozen while
the answer is still unknown, on screen, in front of the viewer.

## Voiceover

Quiet, calm, unsentimental. Never wry. Never triumphant.

> "An intuition can start a question. Before you see an answer, name what would
> count. Then let the result disagree."

Timing: line 1 lands over the freeze (2.2-3.3), line 2 over the lock
(6.8-9.2), line 3 over the disagreement (9.8-11.4). The lockup plays silent.

## Sound design

The master ships silent; `render_film.py` will mux an audio track if one is
supplied. The spec, if it is scored:

- 0.00-2.00: a low warm drone, close and intimate, a little too pleasant.
- 2.00: **total silence for 400ms.** The cut is the loudest moment in the film
  and it is made of nothing.
- 2.40-3.40: a fine granular texture as the embers quantize.
- 3.55, 4.15, 4.75, 5.35: four dry mechanical snaps, no reverb tail.
- 7.85: one clack as the protocol locks.
- 9.70: no sting. The result arrives without a reward sound. This is deliberate.
- 11.80-15.00: the drone returns, resolved a fifth lower.

## Color law

| Token | Hex | Means |
| --- | --- | --- |
| Ground | `#0E0E10` | The room |
| Lure amber | `#E9A23B` | Reflective and proposed material, unresolved pressure |
| Protocol white | `#EDF2F7` | Structure, procedure, what was fixed in advance |
| Evidence blue | `#7FA8D9` | Evidence and replayable structure |
| Confirmatory cyan | `#5FD3D0` | Confirmatory evidence under a pre-registered protocol |
| Invalidation rose | `#D9534F` | Invalidation, rejection, the falsifier |
| Pale | `#C9CEDA` | Integrity only; a result that has not separated from its control |

**Deliberate deviation from the original brief.** The brief reserved emerald for
a real verifier result. This film uses no green at any point. World Bible V3
lists "green checkmark as final truth" under forbidden imagery and states plainly
that there is no green safety color. Confirmatory cyan carries that role instead,
and it appears exactly once - on `NEXT EXPERIMENT`, which is a commitment, not a
verdict. Continuity with the existing world outranks the brief.

## Typography

- Thesis lines: Bitstream Charter, falling back to Liberation Serif and Georgia.
  Wide tracking, generous size, never bold.
- Protocol and labels: DejaVu Sans Mono, uppercase, 0.24em tracking.
- Wordmark: Liberation Sans, 0.44em tracking.

Only fonts present in a standard Linux render environment are used, so the film
renders identically on CI without shipping font binaries.

## Production rules

1. Start sensual and human. End exact and calm. Never the reverse.
2. The oracle imagery is original and abstract. No tarot card art, no recording
   or likeness from any source creator, no borrowed footage.
3. The blue notebook is an easter egg from
   [the source ledger](../../docs/research/future-expansion/source-ledger.md).
   It is not proof of anything, and the film demotes it to one cell in a grid on
   purpose.
4. One honest `INCONCLUSIVE` moment is mandatory. It is the anti-marketing flex:
   the repository is willing to preserve uncertainty in its own advertising.
5. No terminal footage. The hero is the conceptual transformation, not code.
6. No verdict the film was not given. See below.

## The verdict rule

The result card renders whatever is in `window.__VERDICT` and nothing else. With
no verdict injected, the card falls back to an illustrative `INCONCLUSIVE` and
**stamps `ILLUSTRATION` in the corner of the frame** for the whole beat.

That tag is not decoration. A film about not overclaiming cannot itself display
an unearned result, so the film labels its own unverified moment on screen.

To show a real one, pass a verifier's own output:

```bash
# a real result, produced by a real run, displayed verbatim
python3 visual_identity/before_you_see_it/render_film.py \
    --cut ghost-ark --verdict verdict.json
```

```json
{ "state": "FAIL CLOSED", "detail": "mutated receipt rejected\nverifier: independent", "color": "#D9534F" }
```

For the Ghost-Ark cut this is the intended use: run the malicious-corpus
verifier, take its actual output, and let the film show a mutation failing
closed. An unmutated decorative checkmark would violate both repositories'
north star.

## Poster frame

Pulled from t=7.60s - the frozen protocol with the empty result slot.

```text
WHEN DOES A SIGN
BECOME EVIDENCE?
BEFORE YOU SEE IT.
```

## Cuts

| Cut | Wordmark | Tagline | Footer |
| --- | --- | --- | --- |
| `cc-framework` | CC-FRAMEWORK | Keep the wonder. Check the claim. | Not a safety proof. A method for making claims checkable. |
| `ghost-ark` | GHOST-ARK | Evidence you can inspect. | Replay the receipt. Inspect the non-claims. |

Same film, same timing, same frames - only the final lockup changes.

## Non-claims

- The film is a statement of method. It is not evidence, and it reports no
  measurement.
- The result card is an illustration of a protocol outcome unless a verdict was
  injected, in which case it displays that verdict verbatim and drops the
  `ILLUSTRATION` tag.
- The timestamp and hash in the protocol bar are set dressing at fixed values.
  They are not a real receipt over a real artifact.
- Neither lockup asserts that the named project has proven anything. Both name a
  standard the project holds itself to.
