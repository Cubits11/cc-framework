# Distribution Packet 001

The first packet. Everything in it points at one inspectable object.

## The object

The canonical page: `visual_identity/canonical_page/index.html`.

## The hook

> We made the green check break.

## The demonstration

Change one character in the box. The verification fails closed. Restore it and
the verification returns. Nothing is simulated; the browser is computing a real
SHA-256 over the bytes you can see.

## The surprising point

The successful hash still does not establish that the value inside the file is
true.

```text
Bytes verified. Claim unresolved.
```

## The philosophy

Evidence should expose the boundary of what it establishes.

## The invitation

> Bring one claim that matters.

## The copy

The whole announcement, without inflation:

```text
We built a browser check that verifies 189 bytes against a manifest.
Change one character and it fails closed.
It still cannot tell you whether the number inside those bytes is true.
That's the point.
```

No revolutionary framework. No changing AI safety forever. The restraint is the
differentiator, and it is also honest, which is why it is sustainable.

## Derived posts

Each points back to the same object. None is standalone.

| # | Angle | Core line | Ends at |
| --- | --- | --- | --- |
| 1 | The demonstration | A green check should have a falsifiable path to red. | The page. |
| 2 | The boundary | Integrity is not truth: these bytes are those bytes, and that is all. | The limitation text. |
| 3 | The self-attack | We tried to forge our own verified page. It worked. | [CH-001](challenges.md). |
| 4 | The method | Name the claim, the falsifier, the control, and the non-claim — before the result exists. | The film. |
| 5 | The invitation | Bring one claim that matters. Here is exactly what you get back. | [Claim intake](claim-intake.md). |

Post 3 is the one that will travel furthest, because publishing a successful
attack on your own verification is rare enough to be surprising and cheap enough
to be honest. It should not be held back for a better moment. It *is* the moment.

## Sequencing

Ship 1 and 2 with the page. Hold 3 until CH-001's write-up and the page's
updated limitation are both live, so the post lands on an artifact that already
reflects the finding — never on a promise to fix it later. Then 4, then 5.

## Rules for every public artifact in this packet

Each one carries three things, visibly:

```text
CLAIM     what is being asserted
EVIDENCE  what supports it, and where to inspect that
LIMIT     what remains unestablished
```

If a post cannot carry all three, it is not ready. This is the point at which
the marketing becomes a demonstration of the philosophy rather than a departure
from it.

## Non-claims

- This packet has not been published. Writing it is not distributing it.
- No claim is made about how any of it will perform.
- The ordering is a judgment about honesty and readiness, not about reach.
