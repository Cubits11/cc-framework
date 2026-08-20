# Draft 01 — The demonstration

**Surface:** aphorism + experience. **Points at:** the canonical page.

---

We built a browser check that verifies 189 bytes against a manifest.

Change one character and it fails closed.

It still cannot tell you whether the number inside those bytes is true.

That's the point.

---

## Longer version

There is a real evidence file in our repository: a calibration record, 189
bytes, holding a false-positive rate. There is a real digest for it, recorded in
a manifest that was generated when the artifact was built.

The page puts both in front of you and recomputes the digest in your browser,
over exactly the bytes you can see. No server. No API. `crypto.subtle.digest`,
the same primitive your browser uses for TLS.

Then it gives you a button that breaks it.

That button is the whole design. A verification you have personally made fail is
a verification whose scope you understand. Everyone who clicks it stops
believing that the green state means the file is *correct*, because they have
just watched what the check is actually sensitive to: bytes, and nothing else.

**A green check should have a falsifiable path to red.** Most don't. Not because
anyone is lying, but because the path is buried three systems away from the
person reading the result.

Claim: these bytes hash to that digest.
Evidence: recomputed in your browser, over bytes you can edit.
Limit: says nothing about whether 0.041666666667 is a real false-positive rate.

Try to break it: `visual_identity/canonical_page/index.html`
