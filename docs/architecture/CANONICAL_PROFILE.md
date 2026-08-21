# Canonicalization Profiles

> **Status: implemented.** `cc.canonical.v2` is the default for artifacts
> written by this repository. `cc.canonical.v1` is retained read-only.
>
> **Non-claim.** A canonicalization profile determines which documents share a
> receipt identity. It does not establish that a document is true, that its
> producer was honest, or that the system it describes is safe. A receipt binds
> bytes, not intentions.

A receipt identifies a document only up to the **kernel** of its canonicalizer:
the set of distinct documents that receive the same canonical bytes. Every
member of that kernel is a pair of documents a receipt cannot tell apart. The
job of a profile is to keep that kernel equal to JSON equality — no larger, and
no smaller.

Every receipt names its profile in `receipt.canonicalization_method`, and
verification dispatches on that name rather than assuming the current default.

---

## The two profiles

| | `cc.canonical.v1` | `cc.canonical.v2` |
|---|---|---|
| Identifier | `json.dumps(sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False); receipt.canonical_hash excluded` | `cc.canonical.v2/RFC8785; receipt.canonical_hash excluded` |
| Status | **read-only**, historical | **default** for new artifacts |
| Basis | Python `json.dumps` conventions | RFC 8785 (JCS) |
| Key order | Unicode code point | **UTF-16 code unit** |
| Unicode normalization | NFC applied to keys and values | **none** |
| Number form | Python `repr` | **ECMAScript `Number::toString`** |
| Integer domain | unbounded Python `int` | **IEEE-754 safe range** |
| Unpaired surrogates | emitted | **refused** |
| Cross-language verifiable | no | yes |

---

## Why v1 was replaced

An adversarial census — `scripts/canonicalization_probe.py`, modelled on
Ghost-Ark's E1 provenance-kernel census — found two `unintended-kernel` classes
in v1: documents a consumer needs distinguished that received identical bytes.

### The silent key merge

v1 applied `unicodedata.normalize("NFC", …)` to every mapping key and wrote the
results into a fresh dict. Two byte-distinct keys sharing an NFC form silently
became one:

```python
>>> canonical_json_bytes({"é": 1, "é": 2}, profile=LEGACY_SORT_KEYS)
b'{"\xc3\xa9":2}'
```

Two keys in, one key out, **no exception**. The receipt then attested faithfully
to a document with a field missing. The same collapse occurred nested inside an
object, which a guard inspecting only top-level keys would have missed.

Recorded as [F-03](../upgrade/FINDINGS_REGISTER.md#f-03).

### Bytes no other language could agree on

v1 emitted Python `repr` number forms — `1.0`, `-0.0`, `1e-07` — which diverge
from RFC 8785 on five of six probed forms. A verifier written in any other
language computes different bytes for the same document, so independent
verification was impossible: not because of a bug, but because the two sides
never agreed on what the bytes are.

v1 also emitted Python integers of unbounded size. A JavaScript `JSON.parse`
collapses values beyond 2⁵³ before any verifier code runs — the kernel is set by
the parser, and no downstream fix reaches it.

Recorded as [F-04](../upgrade/FINDINGS_REGISTER.md#f-04) and
[F-05](../upgrade/FINDINGS_REGISTER.md#f-05).

---

## What v2 does

### It does not normalize

RFC 8785 is explicit that Unicode normalization is the **producer's**
responsibility, not the canonicalizer's. v2 follows it.

This is the fix for F-03, and it is a fix by construction rather than by patch:
two keys that differ in Unicode form are two keys, which is what JSON says they
are, so there is nothing left to collide. A canonicalizer that mutates content
is not a canonicalizer.

Producers that would rather refuse such a document can call
`assert_no_confusable_keys` before hashing. It is deliberately **not** on the
hash path — v1's mistake was exactly that a content-altering rule lived inside
canonicalization.

### Numbers follow ECMAScript

`Number::toString`, as RFC 8785 requires. Python's `repr` already yields the
shortest round-tripping digits; what v2 supplies is the *formatting* of those
digits.

| Value | v1 emitted | v2 emits |
|---|---|---|
| `1.0` | `1.0` | `1` |
| `-0.0` | `-0.0` | `0` |
| `1e-7` | `1e-07` | `1e-7` |
| `100.0` | `100.0` | `100` |
| `1e20` | `1e+20` | `100000000000000000000` |
| `1e21` | `1e+21` | `1e+21` |

The `-0.0` case is not academic: `scipy.optimize.linprog` returns a negative
zero at a lower bound of zero, so under v1 two numerically identical reports
could receive different receipts.

### Keys sort by UTF-16 code unit

Not by code point. The orders disagree above the BMP: U+10000 encodes as the
surrogate pair D800 DC00, so it sorts *before* U+FFFD under UTF-16 and *after*
it under code point. Comparing big-endian UTF-16 bytes reproduces the required
order exactly.

### One declared narrowing: the integer domain

RFC 8785 is defined over JSON numbers, which are IEEE-754 doubles. Python's
`int` has no such bound, so a profile must supply one.

v2 refuses any integer with `|n| > 2**53 - 1`. That bound is the standard
`MAX_SAFE_INTEGER` — the largest *n* for which both *n* and *n+1* are exactly
representable — which conservatively also refuses 2⁵³ itself. Floats are
unaffected: they are already doubles, so `1e300` serializes normally.

**This is a narrowing of JCS, and it is declared here rather than left as an
undocumented difference.** A caller needing a larger integer must carry it as a
string. Refusing is the fail-closed choice: emitting bytes that cannot survive a
round trip through a conforming parser would produce a receipt no one else can
check.

---

## Migration

The change altered every receipt hash in the repository. Two things made that
safe to do:

1. **Verification dispatches on the declared profile.** A pre-migration receipt
   names v1 and is recomputed under v1. Recomputing it under RFC 8785 would
   report a mismatch for a report that is in fact intact — the opposite of what
   a receipt is for. Breaking historical receipts to fix the canonicalizer would
   trade one integrity failure for another.
2. **The regenerated artifacts were diffed before being accepted.** Every file
   in the claim-governance capsule was compared with hashes, hash-derived ids,
   and the profile identifier scrubbed. All eleven were byte-identical under
   that scrub: no content, no claim text, and no non-claim changed. Only the
   hashes moved.

`cc.canonical.v1` is not deprecated-but-available. It is **read-only**: nothing
in the repository writes it, and a test asserts that its known defects are still
present. If v1 were ever "fixed", every pre-migration receipt would silently
become unverifiable.

---

## Verification

```bash
make canon-probe        # the adversarial census, both profiles
pytest tests/unit/canonical
```

The census gates on v2 only. v1 is **expected to fail** — that is why it was
replaced — and a test asserts it still does.

Current v2 census: **12 sound, 1 fail-closed, 1 sound-by-rejection**; zero
`unintended-kernel`, zero `rejection-asymmetry`, zero `over-discrimination`; 15
of 15 number forms conformant.

### One intent was corrected during this work

The class `int-vs-float-same-value` was originally declared with intent
`distinct`. That declaration described Python's type system, not JSON's: JSON
has exactly one number type, so `1` and `1.0` are the same JSON number, and a
consumer needing them distinguished is asking JSON for something it does not
provide. The intent is now `equivalent`.

Correcting a declaration because it was wrong about the domain is legitimate.
Correcting one to flatter a measured result is not. This was the former, and it
is recorded in the probe source and here rather than edited away.

---

## What a clean census does not establish

That the kernel has no other members. The corpus is **curated** — its coverage
is an authoring decision, not a measurement — and it carries no confidence
interval, because it is the whole population rather than a sample.

Known gaps, stated so absence is not read as a null result:

- **No fuzzing of the canonicalizer.** Every class was authored by hand.
- **No cross-language differential test on receipts.** `conformance/cc-kernel-v1/`
  establishes agreement on the *composition kernel*; nothing yet re-canonicalizes
  a CC report in another language and compares digests. That is the natural next
  step now that v2 makes it possible, and until it runs, "cross-language
  verifiable" describes the profile's design rather than a demonstrated result.
- **No external reviewer** has attacked either profile.

## Binding rules

- **Never weaken a corpus class to make a test pass.** A class that starts
  failing is a kernel regression.
- **Never change a profile in place.** A new canonicalization is a new profile
  identifier, so existing receipts keep verifying.
- **Never write v1.** It exists to read history.
