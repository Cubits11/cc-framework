# Draft 02 — Integrity is not truth

**Surface:** philosophical. **Points at:** the limitation text.

---

A digest match establishes exactly one thing:

> These bytes are those bytes.

Here is what happens next, in an honest reader's head, in under a second and
without any awareness that it happened:

```text
1.  hash(A) == manifest_hash        →   A is the file that was recorded
2.  A is the recorded file          →   the contents of A are accurate
3.  the contents of A are accurate  →   the claim A represents is true
```

Step 1 is licensed. Steps 2 and 3 are not. Nobody computed anything about
whether the number in the file is real. It could have been typed in.

I call this **evidence scope inflation**, and the thing that makes it worth
building a laboratory around is that it requires no bad actors. It is the
default behaviour of a careful, honest reader. The inflation happens in the gap
between what a mechanism computes and what a person needs to know — and the
mechanism is usually silent about the gap, because it does not know the gap
exists.

Every mechanism worth trusting has this shape:

| Establishes | Routinely read as |
| --- | --- |
| A key signed this | The signer is honest and the content is true |
| The declared tests passed on this commit | The change is correct |
| This measurement ran in that enclave | The computation was correct |
| An assertion is bound to an artifact | The assertion is true |
| The package is internally consistent | The system is safe |

The right column is where products live. The left column is where truth lives.
The distance between them is the interesting part.

The fix is not better cryptography. It is **placement**: make the boundary
travel attached to the result, at the same size, in the same frame, at the same
moment. Not a footnote. Not a linked document. Directly under the verdict,
naming the specific thing it does not vouch for.

Bytes verified. Claim unresolved.
