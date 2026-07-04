# Prompt 9 - Human Review Artifact

Use this as a Codex execution prompt.

```text
You are working in the cc-framework repository. Implement Prompt 9:
Human Review Artifact as scoped review attestation, not automated approval
theater.

Start by reading:
- docs/research/CLAIM_GOVERNANCE_OS.md, especially Sections 8 and 9
- docs/research/CC_REPORTS.md
- docs/governance/model-guardrail-card-template.md
- src/cc/evidence/assurance_schema.py
- src/cc/evidence/claim_governance.py
- any prompt-roadmap files for ClaimEnvelope and Role Ontology if implemented
- existing governance tests

Objective:
Build human_review.py as a scoped, stale-sensitive human review artifact. The
verifier may reduce a human-review requirement only when the review artifact
matches the report hash, reviewed artifact hashes, claim level, authority
scope, and expiry rules.

Core object:
HumanReviewArtifact(
    review_id,
    reviewer,
    authority_scope,
    decision,
    reviewed_report_hash,
    reviewed_artifact_hashes,
    reviewed_claim_level,
    conditions,
    remaining_non_claims,
    expires_at,
    created_at,
)

Allowed decisions:
- approved_for_diagnostic_use
- approved_with_conditions
- requires_more_evidence
- rejected
- superseded
- withdrawn

Non-negotiable semantics:
- Human review does not upgrade evidence.
- Human review can only accept, reject, or conditionally authorize use of an
  evidence package within a scope.
- Review over the wrong report hash must be ignored or fail closed.
- Review over a partial artifact set cannot reduce review requirements for
  artifacts it did not review.
- Review expires.
- Review cannot approve a release claim above ontology support.

Implementation requirements:
- Add src/cc/evidence/human_review.py.
- Use strict typed models and deterministic serialization.
- Include a verifier/helper:
  - validate_review_against_report(...)
  - review_reduces_requirement(...)
  - review_state_at(...)
- Integrate with verify_claim_governance so matching review artifacts can affect
  required_human_review conservatively.
- Preserve non-claims and conditions in all audit outputs.
- Do not add UI or workflow theater.

Tests:
- Review over wrong report hash is ignored or fails closed.
- Review missing artifact set cannot reduce review requirement.
- Review becomes stale when artifact set changes.
- Review cannot approve release claim above ontology support.
- Review expires after expires_at.
- rejected, withdrawn, and superseded decisions do not authorize use.
- extra fields are rejected.

Docs:
- Add docs/design-specs/human_review.md.
- Explain authority scope, stale review, and why human review does not upgrade
  evidence.

Acceptance checklist:
- Review semantics are scoped and hash-bound.
- The verifier cannot use review to launder weak evidence into stronger claims.
- Tests cover stale, partial, mismatched, and over-scoped review.
- The final response reports files changed, public API, tests run, and remaining
  risks.
```

