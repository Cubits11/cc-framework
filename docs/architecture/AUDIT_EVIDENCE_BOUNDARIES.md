# Audit and Evidence Boundaries

Several modules use audit or evidence language, but they operate at different
layers. This note records the intended ownership so future cleanup does not
merge distinct responsibilities just because names overlap.

| Surface | Responsibility | Not responsible for |
| --- | --- | --- |
| `src/cc/cartographer/audit.py` | Hash-chained JSONL audit records for the cartographer search process. | Final evidence bundles, claim governance, cloud storage, or dashboard payloads. |
| `src/cc/core/audit_runner.py` | Orchestrating local runs and verifying attestation-style outputs around configured experiment commands. | Defining evidence roles, claim envelopes, or release artifact policy. |
| `src/cc/core/evidence_bundle.py` | Building and verifying deterministic evidence-bundle directories consumed by enterprise reference code and tests. | Claim-level governance semantics or Paper Core math. |
| `src/cc/evidence/` | Typed evidence governance: role ontology, claim envelopes, assurance cases, Merkle logs, anchoring, confirmatory protocol, decay, and extremal scenarios. | Running experiments or discovering dependence cliffs. |
| `src/cc/reporting/` | Canonical report payloads and receipts that can be bound into evidence governance. | Audit-chain storage or external deployment verification. |

## Layering Rules

- Search-process audit records may feed evidence bundles, but they are not
  themselves claim-governance decisions.
- Evidence bundles prove integrity of bytes and manifests. They do not prove
  statistical validity, deployment safety, compliance, or model correctness.
- Claim envelopes and governance audits may refer to reports, receipts, bundles,
  and roles, but they should not re-run cartographer search logic.
- Enterprise reference code may transport and verify bundles. It should consume
  core and evidence outputs rather than defining new math or new claim semantics.

When adding a new audit-like surface, state which layer owns it and which layer
consumes it. If a module crosses layers, document the direction of dependency
and add a focused test for the serialized contract it exports.

## Security Vocabulary

- A **digest** is a canonical hash of serialized content. It can identify byte
  changes when the canonicalization rule is stable.
- A **signature** is a key-backed attestation over a specific serialized
  payload. It identifies the signing key and payload, not the truth of the
  payload's research interpretation.
- A **receipt** is a structured evidence record that may include digests,
  signatures, schema versions, key identity, run context, and non-claims.
- A **chain** is an append-only sequence where each record binds the previous
  record hash or a previous root.
- **Auditability** means provenance can be inspected and replayed from recorded
  artifacts.
- **Validity** means the statistical, mathematical, or research claim is
  correct under its stated assumptions.

Hashes do not prove validity. Signatures do not prove statistical correctness.
Receipts do not prove safety. Enterprise verification endpoints transport and
check evidence-integrity records; they do not validate deployment safety.

## Minimum Evidence-Security Checks

Public evidence or receipt code should keep tests for canonical serialization
stability, reserved hash-key rejection, signature verification, replay/context
binding, Merkle inclusion or consistency tampering, and output path containment.
For current coverage, see:

- `tests/unit/core/test_evidence_bundle.py`
- `tests/unit/core/test_audit_runner.py`
- `tests/unit/evidence/test_transparency_log_adversarial.py`
- `tests/unit/cartographer/test_cartographer_audit_chain.py`
