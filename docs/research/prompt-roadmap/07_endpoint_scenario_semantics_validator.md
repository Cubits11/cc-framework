# Prompt 7 - Endpoint Scenario Semantics + Validator

Use this as a Codex execution prompt.

```text
You are working in the cc-framework repository. Implement Prompt 7:
Endpoint Scenario Semantics + Validator as a semantic firewall around
ExtremalScenario artifacts.

Start by reading:
- docs/research/CLAIM_GOVERNANCE_OS.md, especially Section 4 and Lane B
- docs/theory/frechet_classes.md
- docs/theory/correlation_cliffs.md
- src/cc/evidence/extremal_scenario.py
- src/cc/evidence/claim_governance.py
- src/cc/kernel/frechet_classes.py
- tests that cover extremal scenarios and claim governance

Objective:
Build scenario_semantics.py so every endpoint, stress, fitted, or confirmatory
scenario declares what it proves, what it does not prove, required non-claims,
misuse flags, and confirmatory requirements.

Core object:
ScenarioSemanticAudit(
    scenario_id,
    kind,
    proof_meaning,
    non_proof_meaning,
    required_non_claims,
    misuse_flags,
    confirmatory_requirements,
)

Misuse flags:
- endpoint_interpreted_as_likely
- fitted_model_interpreted_as_true
- stress_path_interpreted_as_attacker_realizable
- confirmatory_matrix_missing_protocol
- lp_distribution_missing_feasibility_diagnostics

Non-negotiable semantics:
- Feasible does not mean probable.
- Fitted does not mean true.
- Stress does not mean attacker-realizable.
- A confirmatory failure matrix needs a protocol reference.
- LP feasibility diagnostics are part of the scenario's claim boundary.

Implementation requirements:
- Add src/cc/evidence/scenario_semantics.py or integrate into
  extremal_scenario.py if that is clearly cleaner.
- Validate proof meaning and non-proof meaning by ScenarioKind.
- Preserve strict parsing and deterministic serialization.
- Integrate with verify_claim_governance so semantic misuse becomes
  NEEDS_REVIEW or FAIL according to severity.
- Do not broaden mathematical behavior of the kernel unless a bug is discovered
  and tested.

Tests:
- Frechet endpoint missing likely-world non-claim triggers NEEDS_REVIEW.
- Fitted empirical scenario missing model-truth non-claim triggers NEEDS_REVIEW.
- Confirmatory failure matrix without protocol ref triggers FAIL or
  NEEDS_REVIEW, with a documented reason.
- Negative atom probability or infeasible residuals produce an infeasible audit.
- Semantic audit round-trips through serialization.

Docs:
- Add docs/research/extremal_scenario_semantics.md or update existing theory
  docs if that is the better home.
- Include a table mapping scenario kind to proof meaning and non-proof meaning.
- Include the high-dimensional warning: endpoint feasibility discipline matters;
  do not imply every lower-bound construction is a valid copula in higher
  dimensions.

Acceptance checklist:
- Scenario semantics are machine-checkable.
- Misuse flags are explicit and tested.
- Claim governance verification consumes semantic audit output.
- Existing scenario fixtures still work or are intentionally migrated.
- The final response reports files changed, public API, tests run, and remaining
  risks.
```

