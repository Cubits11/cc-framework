# Claim-package compiler attack report

**Status:** empirical temporary-directory audit of `feat/claim-package-compiler`
at `eb03f83`, followed by the narrow fixes documented here. This is not an
external security certification, an independent review, or a claim that every
attack has been tested.

## Scope and classification

The audit used `examples/claim_governance_capsule/expected/cc_report.json` at
the recorded verification time `2026-01-02T00:00:00Z`. Its receipt-bound
measurement is `CC = 0.125` with interval `[0.041666666667, 0.166666666667]`
and reported `n = 24`. Tests copied inputs into temporary directories, used the
public `sha256_canonical` implementation when a reissue was part of the vector,
and ran the public compiler/verifier. No tracked capsule input was modified.

- **Courier** means a delivery-side party who can alter files but has no trusted
  signing credential. Claim-package v1 has no such credential, so a courier who
  can run the public compiler can make a coherent reissue.
- **Author** means a party controlling the source report and its interpretation.
- **DETECTED** means the relevant verifier rejected the attempted package or
  reported `FAIL` for the tested condition.
- **ACCEPTED** means the tested package passed the then-applicable verifier
  checks. It never means the claim was true.
- **INVALID TEST** would mean the proposed mutation did not exercise the named
  vector. **UNTESTED** would mean no observed result. Neither label is used for
  a mandated vector below; all were exercised in the stated fixture scope.

The baseline byte challenge detected all 16 named one-byte mutations and
reproduced its untouched control. That result concerns the challenge's named
mutations only; it is not evidence against coherent reissuance.

## Reproduced vectors

| Vector | Adversary | Pre-fix observed status | Post-fix status / boundary |
| --- | --- | --- | --- |
| Prose upper bound below the evidence interval (`"at most 0.01"` while upper interval endpoint is `0.166666666667`) | Author; courier able to reissue | **ACCEPTED** after recomputing the public receipt and package. | Free prose is explicitly `Entailment: NOT_CHECKED`, not semantic `PASS`. The same statement declared as `quantitative_proposition: {metric_family: CC, relation: upper_bound, threshold: 0.01}` is **DETECTED** as `Entailment: FAIL`, while integrity remains `PASS`. |
| Point estimate | Author | Changing the structured `measurement.point_estimate` to `0.99` outside its interval was **DETECTED** by the strict report model. A false point stated only in prose was **ACCEPTED**. | Direct interval invariant remains detected. Prose point claims remain `NOT_CHECKED`; v1 intentionally does not parse them. |
| Denominator (`n = 2400` asserted against a 24-row source) | Author | **ACCEPTED** when asserted in prose and reissued. | `NOT_CHECKED`. No row-count/denominator derivation or prose parser was added. |
| Population expansion (fixture population relabeled as worldwide production) | Author | **ACCEPTED** as a prose interpretation after reissue. | `NOT_CHECKED`. No population ontology or external sampling anchor was added. |
| Metric relabeling | Author | **ACCEPTED** when the prose metric label was changed after reissue. | Prose remains `NOT_CHECKED`. An explicit structured proposition whose `metric_family` differs from the report is **DETECTED** as entailment `FAIL`. |
| Confidence relabeling | Author | **ACCEPTED** when prose asserted a different confidence interpretation after reissue. | `NOT_CHECKED`. The compiler does not interpret confidence prose or recalculate intervals. |
| Removed non-claims | Author | Removing all non-claims from the non-diagnostic report was **DETECTED**. Replacing material scope non-claims with one generic non-claim was **ACCEPTED**. | Unchanged: the compiler preserves the report boundary but does not prove that a surviving generic boundary fully captures scope. |
| Contradictory report / manifest | Courier | Editing the package report without updating manifest bindings was **DETECTED**. | Still detected for an inconsistent package. The verifier now also re-derives `subject_report.canonical_receipt_sha256` and reproducibility commands, closing two manifest-only fields that had been **ACCEPTED** when changed alone. |
| Stale evidence | Author | The original fixture was **DETECTED** as expired at `2026-08-24`. Retiming the decay policy to the current date, updating its report binding, and reissuing the receipt was **ACCEPTED**. | Unchanged: without an external collection-time anchor, a coherent author reissue can restate freshness. |
| Valid reissued false interpretation | Author or credential-free courier | **ACCEPTED**: a report could be re-receipted with the official canonicalizer and recompiled under the original package identity. | A false **declared structured bound** yields `Integrity: PASS / Entailment: FAIL / Independence: NONE`. A false unstructured interpretation remains `NOT_CHECKED`. No origin/authorship distinction is established. |
| Official canonicalizer re-receipt | Author or courier | **ACCEPTED** by design: the canonical receipt verified the new bytes. | Still **ACCEPTED** as byte identity only. It is not a signature, timestamp, provenance proof, or semantic witness. |

## What changed

The compiler now records and displays the three axes separately:

```text
Integrity: PASS | FAIL
Entailment: PASS | FAIL | NOT_CHECKED
Independence: NONE
```

Only receipt-bound structured upper/lower propositions are compared against the
receipt-bound measurement interval. This closes the confirmed structured-bound
forgery without pretending that natural-language interpretation is solved.

The package verifier also now re-derives the manifest's recorded receipt hash
and reproducibility commands. A valid legacy report receipt is checked with its
declared canonicalization profile rather than being incorrectly recomputed with
the current default profile.

Regression coverage includes the interval-inconsistent structured upper bound,
unstructured `NOT_CHECKED`, integrity challenge behavior when aggregate
entailment fails, manifest receipt/command tampering, and legacy-profile
governance verification.

## Remaining limits

No natural-language entailment, source-data anchoring, author signatures,
external timestamps, population/denominator semantics, confidence semantics,
or independent witness is implemented by this patch. `Independence: NONE` is an
explicit absence of evidence, not a weak pass. A coherent reissue can be
internally consistent and still be a false or unsupported interpretation.
