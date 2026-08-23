# The claim-package compiler — a portable claim that carries its own falsifier

**Status: implemented, tested, tamper-evident under adversarial challenge.**
Modules: `cc.evidence.claim_compiler`, `cc.evidence.claim_challenge`. CLI:
`compile-claim-package`, `verify-claim-package`, `challenge-claim-package`.

## What it is

A compiler that turns one already-verified `cc.report` into a **portable,
self-verifying, tamper-evident claim package**: the report copied byte-for-byte,
its bound evidence copied under `evidence/`, the governance audit, claim
envelope, lifecycle projection, and human-review projection regenerated, and a
manifest that records exactly what the package must contain. A recipient with no
access to the original run can re-verify the whole thing offline, and — this is
the point — can *disprove* its integrity claim without trusting the compiler.

It does deliberately less than its name invites. It does not decide whether a
claim is true, safe, deployable, or certified. Its one promise is narrow and
checkable: **if any byte of a bound surface is altered, verification falls to
`FAIL`.** Integrity is not validity, a `PASS` is not safety, and the package
says so on every surface.

## Fail-closed guarantees (each has a test)

The compiler refuses to build, and the verifier refuses to pass, in every case
below. `tests/unit/evidence/test_claim_compiler.py` is the evidence.

- Governance `FAIL` — never packaged.
- Expired decay at the compile time — refused (the capsule is expired by
  2026-08-23 and the compiler declines).
- Non-portable evidence paths (absolute, `..` traversal) — refused before any
  directory is created.
- A byte flipped in the report, any bound evidence file, or any generated
  surface (audit / envelope / lifecycle / review) — `FAIL`.
- A bound evidence file deleted — `FAIL`.
- **A manifest edited to lie** — the verifier re-derives every verdict-bearing
  manifest field (the artifact list, the subject-report binding, the
  generated-surface hashes, the non-claims, and the lifecycle/review
  projections) from the receipt-bound report, so a manifest that repoints an
  artifact hash or smuggles in a false non-claim (`"This system is certified
  safe."`) is rejected — even when the attacker also edits the file the manifest
  now points at.

The verifier has two modes. With no `now`, it reuses the package's recorded time
and checks *reproduction* of the recorded verdict. With a `now`, it performs a
*freshness* check at that time, whose verdict may legitimately differ (the
capsule's claim verifies at its recorded time and fails a freshness check once
its decay window has passed).

## The built-in falsifier

Every package ships `CHALLENGE.md` and records a `challenge_command`. The
challenge is a separate adversary module — the builder and the attacker share no
private assumptions:

```bash
python -m cc.reporting.cli challenge-claim-package .
```

It copies the package to a scratch directory (never touching the original),
flips one byte of **each** bound surface in turn, re-verifies, and records
whether the verdict fell to `FAIL`. A control run over the untouched copy must
reproduce the recorded verdict, so a harness that merely always fails cannot
pass, and a package missing a bound surface cannot pass either. The report's
`tamper_evident` is true only if the control reproduced *and* every mutated
surface was detected.

This is the portable, offline analogue of the site's "one real check" widget:
a recipient does not take the compiler's word that the package is tamper-evident
— they run the challenge and watch each mutation caught.

## How this invention was hardened

The first real compile revealed the compiler never self-verified: the manifest
hashed the governance audit as compact in-memory JSON while the verifier hashed
the indented on-disk file — two serializations, so the check could never pass.
Fixed. The first run of the built-in falsifier then found a genuine
tamper-evidence gap: a one-byte edit of `manifest.json` (landing in a non-claim
string) was **not** detected, because the manifest carried authority no verifier
re-derived. Closed by re-deriving every verdict-bearing manifest field from the
receipt-bound report. The adversary caught the builder twice; both catches are
now regression tests. That is the intended lifecycle of this subsystem — the
falsifier is not decoration, it is how the compiler earns its one claim.

## The honest boundary

- A `PASS` means the package is internally consistent under the verifier rules
  at the recorded time. It does not mean the AI system is safe in deployment.
- Package hashes bind copied bytes and report references; they do not prove
  statistical validity, data representativeness, label correctness, or
  sufficiency for a release decision.
- The compiler does not assign, transition, revoke, or approve a claim lifecycle
  state — the manifest records that absence explicitly rather than rebranding a
  governance verdict as a lifecycle state.
- The challenge demonstrates detection of the single-byte mutations it applies
  to each named surface; it is not a proof that no undetectable modification
  exists. `package_id` and `created_at` are pure input labels the verifier
  cannot re-derive and are, correctly, not verdict-bearing.
