# Prompt 4 - Confirmatory Evidence Protocol

Use this as a Codex execution prompt.

```text
You are working in the cc-framework repository. Implement Prompt 4:
Confirmatory Evidence Protocol as the bridge from exploratory discovery to
confirmatory evidence.

Start by reading:
- docs/research/CLAIM_GOVERNANCE_OS.md, especially Section 6
- docs/research/CC_REPORTS.md
- docs/research/DISCOVERY_REPORT.md
- src/cc/evidence/claim_governance.py
- src/cc/redteam/dependence_search.py
- existing tests for exploratory or claim governance behavior

Objective:
Build confirmatory_protocol.py as a pre-registration and confirmation protocol
layer. It should make timing, fixed endpoints, fixed analysis plans, stopping
rules, sample plans, and separation from adaptive discovery machine-checkable.

Core object:
ConfirmatoryProtocolPlan(
    protocol_id,
    hypothesis,
    discovery_ref,
    protocol_mode,
    preregistered_at,
    primary_endpoint,
    fixed_analysis_plan,
    sample_plan,
    stopping_rule,
    cluster_blocking,
    exclusion_rules,
    decision_rule,
    non_claims,
)

Protocol modes:
- held_out_matrix
- sample_split
- fixed_attack_suite
- pre_registered_retest
- cluster_blocked_confirmation

Design for, but do not over-implement in v0:
- selective_inference_adjusted
- anytime_valid_e_process
- conformal_risk_control

Required semantic check:
temporal_validity_check = plan.created_at < run.started_at

Non-negotiable semantics:
- Held-out is not valid if the held-out set was chosen after seeing the failure
  pattern.
- Adaptive discovery evidence cannot become confirmatory by renaming the role.
- Missing stopping rules must trigger NEEDS_REVIEW or FAIL depending on the
  surface being claimed.
- Clustered data without cluster blocking must trigger review.
- A confirmatory artifact must reference both the plan and the run.

Implementation requirements:
- Add src/cc/evidence/confirmatory_protocol.py.
- Define strict serializable models for protocol plans and confirmatory run
  references.
- Add validation helpers that can be called by claim governance verification and
  role ontology checks.
- Add a CLI hook only if it matches existing CLI patterns cleanly. Otherwise,
  expose the model and verifier first and document the future CLI.
- Keep v0 honest. Do not compute sophisticated inference unless the protocol
  actually supports it and tests prove the claim.

Tests:
- plan timestamp after run timestamp fails.
- plan timestamp equal to run timestamp fails unless a clear local convention
  already permits equality.
- adaptive discovery artifact cannot become confirmatory evidence by role rename.
- missing stopping rule triggers review/fail.
- missing cluster blocking with clustered data triggers review.
- confirmatory artifact must reference both plan and run.
- extra fields are rejected.

Docs:
- Add docs/design-specs/confirmatory_protocol.md.
- Include proof meaning and non-proof meaning.
- State that confirmatory validity depends on the protocol, not on report polish.

Acceptance checklist:
- Protocol models are strict and serializable.
- Timing/provenance separation is enforced.
- Verifier integration exists or is intentionally staged with documented hooks.
- Tests cover the adaptive-discovery firewall.
- The final response reports files changed, APIs added, tests run, and remaining
  risks.
```

