# Claim-package compiler

**Status: implemented.** Modules: `cc.evidence.claim_compiler` and
`cc.evidence.claim_challenge`; CLI: `compile-claim-package`,
`verify-claim-package`, and `challenge-claim-package`.

The compiler copies one receipt-bound `cc.report` and its bound evidence into a
portable directory. It is an integrity and inspection tool, not a truth engine,
semantic validator, source authenticator, or assurance certificate.

## Do not collapse the verdicts

Every newly compiled package records three separate results. They answer
different questions and must not be read as substitutes for one another.

| Axis | Values | What it checks | What it does not establish |
| --- | --- | --- | --- |
| Integrity | `PASS` / `FAIL` | The copied report, evidence, manifest bindings, and generated package surfaces are byte/package-consistent. | The report's claim is true, the evidence was measured rather than asserted, or the package came from a particular author. |
| Entailment | `PASS` / `FAIL` / `NOT_CHECKED` | An explicitly declared structured upper- or lower-bound proposition against the report measurement interval. | Natural-language meaning, point claims, denominators, populations, confidence interpretation, or unstructured prose. |
| Independence | `NONE` | Whether this package contains independently established evidence. Claim-package v1 has none. | That a second invocation, same-author challenge, or public canonicalizer is an independent witness. |

An honest package can therefore report:

```text
Integrity: PASS
Entailment: FAIL
Independence: NONE
```

That combination means its bytes are consistent while its declared numerical
proposition is contradicted by the measurement interval. It is deliberately not
rewritten into an integrity failure.

The aggregate package result remains conservative: an integrity or governance
failure, or a structured-entailment failure, makes it `FAIL`; unresolved
governance review makes it `NEEDS_REVIEW`. An aggregate `PASS` still does not
establish semantic truth or real-world validity.

## The deliberately narrow interval check

The compiler never parses `claim.statement`. Free-text claim prose receives
`NOT_CHECKED`, even when package integrity is `PASS`.

An author who wants the one supported machine check must add a receipt-bound
structure under `claim.quantitative_proposition`:

```json
{
  "metric_family": "CC",
  "relation": "upper_bound",
  "threshold": 0.20
}
```

For `upper_bound`, the compiler checks `measurement.interval.upper <= threshold`.
For `lower_bound`, it checks `measurement.interval.lower >= threshold`. The
metric family must exactly match. This is intentionally a small comparison,
not natural-language entailment, source-data validation, external anchoring, or
statistical re-analysis.

## What integrity means

`verify-claim-package` replays the record at the package's fixed verification
time (or at an explicitly supplied time) and checks copied bytes, evidence,
generated surfaces, and re-derived manifest fields. It also re-derives the
report receipt hash recorded in the manifest and the reproducibility commands;
those values are not trusted merely because the manifest says them.

The verifier has no signing authority or external immutable anchor. A party
able to rewrite a report, recompute its public canonical receipt, and recompile
the package can produce a new internally consistent package. That reissue can
be byte-valid without proving provenance or interpretation. See
[`CLAIM_PACKAGE_COMPILER_ATTACK_REPORT.md`](CLAIM_PACKAGE_COMPILER_ATTACK_REPORT.md)
for the reproduced cases.

## The built-in byte-mutation challenge

Every package ships `CHALLENGE.md` and records:

```bash
python -m cc.reporting.cli challenge-claim-package .
```

The challenge copies the package to scratch space and applies one named minimal
byte mutation to each fixed surface and copied evidence artifact. It passes only
when the untouched control reproduces the recorded axes and every tested
mutation makes the **integrity** verdict `FAIL`. It neither evaluates semantic
entailment nor supplies independent evidence.

The challenge demonstrates detection for those named mutations only. It does
not prove that every possible modification is detectable, that a coherent full
reissue is impossible, or that a claim is true.

## Scope limits

- A canonical receipt binds a document under the declared profile; it is not an
  author signature or external timestamp.
- Legacy receipts are checked using their receipt-declared canonicalization
  profile, rather than the current default profile.
- `source_path`, `package_id`, and `created_at` are labels, not provenance
  evidence.
- No natural-language entailment, external anchoring, signatures, or
  independently run verifier is implemented by this package format.
