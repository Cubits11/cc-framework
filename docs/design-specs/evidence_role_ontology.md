# Evidence Role Ontology

`EvidenceRoleDefinition` is the conservative type system for claim support.
An evidence role is not a label or a reviewer hint. It is a machine-readable
permission set that says what an artifact may support, what it must not
support, which non-claims must remain visible, which fields are required or
forbidden, when review remains required, and which events invalidate support.

The implementation lives in `src/cc/evidence/role_ontology.py` and is consumed
by `verify_claim_governance` and `ClaimEnvelope` support-edge validation.

## Public API

```python
from cc.evidence import (
    get_role_definition,
    classify_role,
    validate_role_payload,
    support_permissions_for,
    mandatory_non_claims_for,
)
```

The public schema string is `cc.evidence_role_ontology.v1`.

Unknown roles are preserved for read-only review, but they return no support
permissions. They cannot improve verifier verdict, support strength, or claim
level.

## Definition Shape

Each role definition includes:

- `role`
- `semantic_class`
- `supports`
- `does_not_support`
- `mandatory_non_claims`
- `allowed_claim_levels`
- `staleness_behavior`
- `exploratory_status`
- `confirmatory_status`
- `required_fields`
- `forbidden_fields`
- `review_rules`
- `invalidation_triggers`

Models are strict and reject extra fields.

## V0 Role Table

| Role | Supports | Does not support | Review or invalidation triggers | Required non-claims |
| --- | --- | --- | --- | --- |
| `receipt_integrity` | Report and artifact byte integrity via `integrity_binds` / `integrity_only`. | Statistical validity, deployment safety. | Hash mismatch invalidates integrity support. | Receipt integrity is not statistical validity or deployment safety. |
| `measurement_evidence` | Named report interval under the report scope. | Deployment safety, external validity, representativeness. | Scope shift requires review. | Measurement evidence is scoped and does not certify deployment safety. |
| `calibration_evidence` | Operating-point and calibration boundary. | Deployment safety, external validity. | Calibration-window shift requires review. | None in v0. |
| `claim_decay` | Time bounding, staleness review, freshness invalidation. | Deployment safety, statistical validity, confirmatory evidence. | Degraded requires review; expired or watched-version changes expire support. | Claim decay does not prove the system is currently safe. |
| `extremal_scenario` | Endpoint feasibility and counterfactual dependence bounds. | Likelihood, deployment realization, model-truth claims. | Infeasible scenarios invalidate; excluded fields or fitted-without-confirmation require review. | Extremal scenarios do not prove endpoint worlds are likely. |
| `exploratory_redteam` | Weak exploratory suggestions and confirmatory-firewall review pressure. | Confirmatory evidence, release claims, deployment safety. | Confirmatory fields in exploratory payloads invalidate the firewall; use above `diagnostic` requires review. | Exploratory red-team evidence is not a confirmatory certificate. |
| `confirmatory_protocol` | Confirmatory boundary support from a pre-registered plan and separate run. | Deployment safety, external validity, adaptive-discovery reuse. | Temporal order violations and adaptive reuse invalidate; missing stopping rules or clustered data without blocking trigger fail/review by surface. | Confirmatory validity depends on protocol/run separation, not report polish; it still does not certify deployment safety. |
| `confirmatory_failure_matrix` | Confirmatory tests and scoped confirmatory intervals. | Deployment safety, external validity. | Missing `confirmatory_ci` invalidates confirmatory support. | Confirmatory evidence is still scoped and does not certify deployment safety. |
| `fitted_empirical_scenario` | Diagnostic fitted-scenario bounds. | Model-truth claims, deployment safety. | Model-truth claims invalidate role semantics; unconfirmed fits require review. | A fitted empirical scenario does not prove the fitted model is true. |
| `human_review_note` | Weak scoped authorization of reviewed artifact hashes. | Statistical upgrades, deployment safety, unreviewed artifacts. | Partial artifact set requires review; hash replacement attempts invalidate review semantics. | Human review does not upgrade underlying statistical evidence. |
| `human_review` | Weak scoped authorization of reviewed artifacts. | Statistical upgrades, deployment safety. | Partial artifact set requires review. | Human review does not upgrade underlying statistical evidence. |
| `artifact`, `audit_log`, `figure_manifest` | No direct support edges. | Unstated claim support. | None in v0. | None in v0. |

## Payload Field Enforcement

`validate_role_payload(role, payload)` enforces role-specific fields before an
artifact can be used semantically. Examples:

- `exploratory_redteam` forbids `confirmatory_ci`.
- `confirmatory_protocol` requires both `plan` and `run`, and forbids adaptive
  interval fields such as `adaptive_search_ci`.
- `claim_decay` forbids `live_status_as_signed_truth`.
- `receipt_integrity` forbids `deployment_safety_support`.
- `fitted_empirical_scenario` forbids `model_truth_claim`.
- `human_review_note` forbids artifact hash replacement fields.

The verifier parses and validates semantic payload roles before role-specific
decay or scenario validation. Invalid exploratory, confirmatory, fitted, or
scenario payloads fail closed. Invalid decay and review-note payloads
conservatively require review unless strict mode escalates them.

## ClaimEnvelope Integration

`SupportGraph` validation consumes ontology permissions. A support edge is valid
only when:

- the source artifact role is known, or the edge is weak review-only support for
  an unknown role;
- the target fragment is not in the role's `does_not_support` markers;
- the edge relation and strength match one of the role's support permissions;
- human-review edges over `evidence.*` targets refer to reviewed artifact IDs or
  hashes.

Receipt evidence can therefore create `integrity_only` support, but cannot
support statistical validity. Decay evidence can qualify or invalidate
freshness, but cannot provide confirmatory support. Unknown roles are preserved
as review requirements only.

## Verifier Integration

`verify_claim_governance` uses the ontology to:

- classify known and unknown roles;
- validate semantic role payloads;
- reject role support above allowed claim levels;
- detect mandatory non-claims implied by roles;
- preserve unknown roles without support power;
- allow `human_review_note` to satisfy only the scoped release-claim review
  requirement, and only when the note covers the current non-review artifact
  hash set and matching claim level.

A PASS verdict still means internal consistency under verifier rules only. It
does not mean deployment safety, regulatory sufficiency, or statistical truth.
