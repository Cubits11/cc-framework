# Prompt 8 - Receipt Verification Upgrade

Use this as a Codex execution prompt.

```text
You are working in the cc-framework repository. Implement Prompt 8:
Receipt Verification Upgrade as layered integrity and provenance verification,
not safety verification.

Start by reading:
- docs/research/CC_REPORTS.md, especially receipt semantics
- docs/research/CLAIM_GOVERNANCE_OS.md, especially Section 7
- src/cc/reporting/canonical.py
- src/cc/reporting/report.py
- src/cc/evidence/anchoring.py
- src/cc/evidence/merkle_log.py
- src/cc/evidence/claim_governance.py
- tests that cover receipts, report building, anchoring, and governance

Objective:
Build receipt_verifier.py with layered integrity modes and integrate it with
Claim Governance verification. It should verify bytes, canonical hashes,
optional hash-chain or transparency anchoring, and witness signatures where
available, while preserving semantic humility.

Core object:
ReceiptIntegrityAudit(
    artifact_hashes_verified,
    canonical_report_hash_verified,
    previous_hash_verified,
    merkle_inclusion_verified,
    witness_signatures_verified,
    integrity_non_claims,
)

Integrity modes:
- local_hash_only
- canonical_hash
- hash_chain
- transparency_log
- witness_signature

Non-negotiable semantics:
- Receipt verification is integrity/provenance evidence only.
- It does not prove statistical validity, deployment safety, regulatory
  sufficiency, or data representativeness.
- Absence of transparency anchoring is not a failure unless the report claims
  transparency anchoring.
- Broken claimed anchoring is a failure.
- Missing optional anchoring may be NEEDS_REVIEW when the claim level or policy
  requires it.

Implementation requirements:
- Add src/cc/evidence/receipt_verifier.py unless existing modules already
  provide the right home.
- Reuse canonical hashing from src/cc/reporting/canonical.py.
- Reuse sha256_file and existing report/evidence structures.
- Integrate ReceiptIntegrityAudit into verify_claim_governance without changing
  cc.report.v0.3.1 unless absolutely necessary.
- Keep the verifier read-only.
- Include integrity non-claims in emitted audits.

Tests:
- Artifact mutation produces FAIL.
- Report mutation produces canonical hash mismatch when canonical hash is
  present.
- Missing Merkle proof triggers NEEDS_REVIEW only if anchoring was claimed or
  required by the mode.
- Broken Merkle proof triggers FAIL when anchoring is claimed.
- Receipt audit includes integrity-not-safety non-claim.
- local_hash_only mode does not claim transparency anchoring.

Docs:
- Update docs/research/CC_REPORTS.md or add docs/design-specs/receipt_verifier.md.
- Separate "what the receipt proves" from "what it does not prove".

Acceptance checklist:
- Layered modes are explicit.
- Optional unavailable services do not fail unclaimed guarantees.
- Claimed but broken integrity guarantees fail.
- Claim governance consumes the receipt audit.
- The final response reports files changed, public API, tests run, and remaining
  risks.
```

