# Prompt 2 - ClaimEnvelope / BoundaryEnvelope

Use this as a Codex execution prompt.

```text
You are working in the cc-framework repository. Implement Prompt 2:
ClaimEnvelope / BoundaryEnvelope as the typed intermediate representation for
evidence-bound claims.

Start by reading:
- docs/research/CLAIM_GOVERNANCE_OS.md
- docs/research/CC_REPORTS.md
- docs/research/NON_CLAIMS.md
- src/cc/evidence/claim_governance.py
- src/cc/evidence/decay.py
- src/cc/evidence/extremal_scenario.py
- schemas/cc_report.schema.json
- the relevant tests under tests/

Do not rewrite the report schema as the first move. Preserve
cc.report.v0.3.1 compatibility. The ClaimEnvelope should compile from existing
reports and evidence artifacts, not make the stable report format explode.

Objective:
Build ClaimEnvelope as the central typed IR of evidence-bound claims. It should
represent what is claimed, where the boundary is, what supports each claim
fragment, what invalidates support, and what governance state is currently
computed or attached.

Core design:
Create strict typed objects, using the repo's existing Pydantic v2 style:

ClaimEnvelope(
    identity,
    proposition,
    boundary,
    support_graph,
    governance_state,
)

BoundaryEnvelope(
    scope,
    assumptions,
    non_claims,
    defeaters,
    invalidation_conditions,
    review_requirements,
)

SupportGraph(
    evidence_refs,
    scenario_refs,
    decay_refs,
    receipt_refs,
    review_refs,
    support_edges,
)

SupportEdge(
    source_artifact_id,
    target_claim_fragment,
    relation,
    strength,
    non_claims,
)

Allowed support relations:
- supports
- bounds
- qualifies
- invalidates
- requires_review
- integrity_binds
- exploratory_suggests
- confirmatory_tests

Allowed support strengths:
- weak
- diagnostic
- confirmatory
- integrity_only

Non-negotiable semantics:
- Evidence attached to a report does not automatically support the whole claim.
- Receipt evidence can provide integrity support only.
- Decay evidence can support staleness or review pressure only.
- Extremal scenarios can support endpoint feasibility or counterfactual bounds,
  not likelihood.
- Human review can support scoped authorization of use, not the underlying
  statistical evidence.
- Unknown evidence roles must be preserved but cannot strengthen a claim.

Implementation requirements:
- Add a small module under src/cc/evidence/ for envelope models and compilation
  helpers. Prefer a name like claim_envelope.py unless local conventions point
  somewhere better.
- Include schema strings on serialized public artifacts.
- Include artifact_id, subject_ref, created_at or evaluated_at where relevant.
- Forbid extra fields on typed models.
- Provide deterministic JSON serialization helpers or reuse existing canonical
  serialization utilities.
- Add a compiler/helper that builds a ClaimEnvelope from a cc.report.v0.3.1
  report plus the existing governance audit outputs where available.
- Integrate lightly with verify_claim_governance: the verifier may emit or
  reference envelope support summaries, but keep the first implementation
  narrow and read-only.

Tests:
- receipt evidence cannot support statistical validity.
- claim_decay evidence cannot support deployment safety.
- extremal_scenario evidence cannot support likelihood.
- human review cannot support evidence it did not review.
- unknown evidence role is preserved but cannot improve verdict or support
  strength.
- support_edges survive round-trip serialization.
- extra fields are rejected.

Docs:
- Add or update docs/design-specs/claim_envelope.md.
- Explain proof meaning and non-proof meaning.
- Say explicitly that a PASS verifier result means internal consistency under
  verifier rules, not deployment safety.

Acceptance checklist:
- Files changed are narrowly scoped.
- New public models have schema names and strict parsing.
- The support graph encodes typed relations, not just lists of artifacts.
- Existing fixture reports still verify.
- Tests cover forbidden support upgrades.
- The final response reports files changed, APIs added, tests run, and remaining
  risks.
```

