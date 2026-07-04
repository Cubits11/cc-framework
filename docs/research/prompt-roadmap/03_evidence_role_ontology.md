# Prompt 3 - Evidence Role Ontology

Use this as a Codex execution prompt.

```text
You are working in the cc-framework repository. Implement Prompt 3:
Evidence Role Ontology as the type system for claim support.

Start by reading:
- docs/research/CLAIM_GOVERNANCE_OS.md
- docs/research/CC_REPORTS.md
- docs/research/NON_CLAIMS.md
- docs/research/prompt-roadmap/02_claim_envelope_boundary_envelope.md if present
- src/cc/evidence/claim_governance.py
- src/cc/evidence/decay.py
- src/cc/evidence/extremal_scenario.py
- tests that cover claim governance verification

Objective:
Build role_ontology.py as a conservative evidence type system. A role is not a
label; it is a typed permission set that says what support an artifact may
provide, what it must not imply, what non-claims it requires, when it goes
stale, and whether human review remains required.

Core object:
EvidenceRoleDefinition(
    role,
    semantic_class,
    supports,
    does_not_support,
    mandatory_non_claims,
    allowed_claim_levels,
    staleness_behavior,
    exploratory_status,
    confirmatory_status,
    required_fields,
    forbidden_fields,
    review_rules,
    invalidation_triggers,
)

Required v0 roles:
- claim_decay
- extremal_scenario
- receipt_integrity
- exploratory_redteam
- confirmatory_failure_matrix
- human_review_note
- measurement_evidence, if needed to represent existing generic artifacts
- unknown role handling, without giving unknown roles support power

Forbidden-field examples:
- exploratory_redteam forbids confirmatory_ci.
- claim_decay forbids live_status_as_signed_truth.
- receipt_integrity forbids deployment_safety_support.
- fitted_empirical_scenario forbids model_truth_claim.
- human_review_note forbids replacing artifact hashes it did not review.

Non-negotiable semantics:
- Role definitions must be machine-readable, not prose-only.
- Unknown roles may be accepted in read-only mode, but they cannot improve a
  verifier verdict, support strength, or claim level.
- Role support permissions should be consumed by ClaimEnvelope support edges
  when Prompt 2 exists.
- Role ontology must remain conservative: NEEDS_REVIEW is better than implying
  evidence supports a claim it cannot support.

Implementation requirements:
- Add src/cc/evidence/role_ontology.py.
- Use strict Pydantic models or frozen dataclasses consistent with this repo.
- Expose a small registry API:
  - get_role_definition(role)
  - classify_role(role)
  - validate_role_payload(role, payload)
  - support_permissions_for(role)
  - mandatory_non_claims_for(role)
- Integrate the ontology into src/cc/evidence/claim_governance.py, replacing or
  narrowing hard-coded role support tables where appropriate.
- Keep behavior backward-compatible for existing cc.report.v0.3.1 fixtures.
- Do not let this become a policy engine that approves deployment.

Tests:
- exploratory_redteam cannot support confirmatory claim level.
- human_review_note cannot reduce review requirement unless artifact hashes
  match the reviewed set.
- receipt_integrity contributes integrity_only support.
- unknown role cannot improve verdict.
- role ontology definitions reject extra fields.
- required_fields and forbidden_fields are enforced on sample payloads.
- mandatory non-claims implied by roles are detected by the verifier.

Docs:
- Add docs/design-specs/evidence_role_ontology.md or update an existing design
  spec if there is a better local home.
- Include a role support table with supports, does_not_support, review triggers,
  and required non-claims.

Acceptance checklist:
- Every role definition says what it supports and what it does not support.
- Forbidden fields exist and are tested.
- The verifier uses the ontology for at least the v0 known roles.
- Existing governance tests still pass.
- The final response reports files changed, public API, tests run, and remaining
  risks.
```

