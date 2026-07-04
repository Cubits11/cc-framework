# Confirmatory Evidence Protocol

`confirmatory_protocol.py` is the pre-registration and confirmation layer for
evidence that is stronger than adaptive discovery. It records a fixed plan, a
separate run, and machine-checkable reasons why the run may or may not support a
confirmatory claim surface.

The implementation lives in `src/cc/evidence/confirmatory_protocol.py`.

## Public API

```python
from cc.evidence import (
    ConfirmatoryProtocolArtifact,
    ConfirmatoryProtocolPlan,
    ConfirmatoryRunReference,
    ProtocolAuditStatus,
    temporal_validity_check,
    verify_confirmatory_protocol_artifact,
)
```

Schema strings:

- `cc.confirmatory_protocol.v1`
- `cc.confirmatory_protocol_audit.v1`

The artifact role is `confirmatory_protocol`.

## Core Objects

`ConfirmatoryProtocolPlan` records:

- `protocol_id`
- `hypothesis`
- `discovery_ref`
- `protocol_mode`
- `created_at`
- `primary_endpoint`
- `fixed_analysis_plan`
- `sample_plan`
- `stopping_rule`
- `cluster_blocking`
- `exclusion_rules`
- `decision_rule`
- `non_claims`

`created_at` is the pre-registration time. Payloads may use
`preregistered_at` as an input alias, but serialization uses `created_at`.

`ConfirmatoryRunReference` records:

- `run_id`
- `started_at`
- `completed_at`
- `artifact_id`
- `artifact_role`
- `source_role`
- `artifact_sha256`
- `primary_endpoint`
- `analysis_plan_id`
- adaptive-discovery and held-out-selection flags
- cluster-observation fields

`ConfirmatoryProtocolArtifact` must reference both the plan and the run. Missing
either one fails validation.

All models are strict Pydantic models and reject extra fields.

## V0 Protocol Modes

Supported in v0:

- `held_out_matrix`
- `sample_split`
- `fixed_attack_suite`
- `pre_registered_retest`
- `cluster_blocked_confirmation`

Reserved for future support:

- `selective_inference_adjusted`
- `anytime_valid_e_process`
- `conformal_risk_control`

Reserved modes parse as protocol modes but require review. The v0 verifier does
not claim to perform selective-inference, anytime-valid, or conformal risk
control calculations.

## Machine Checks

The required temporal check is:

```text
plan.created_at < run.started_at
```

Equality fails. A plan created after the run starts fails.

The verifier also checks:

- the artifact references both a plan and a run;
- adaptive or exploratory discovery artifacts are not reused as confirmatory
  evidence by role rename;
- held-out matrices were not selected after seeing the failure pattern;
- fixed endpoints match between plan and run;
- the run references the frozen analysis plan;
- missing stopping rules require review on diagnostic surfaces and fail stronger
  claim surfaces;
- clustered data without cluster blocking or cluster-robust design requires
  review;
- future protocol modes require review.

## Proof Meaning

A passing `confirmatory_protocol` artifact can support this narrow proposition:

```text
The attached confirmatory run is procedurally separated from discovery under
the declared v0 protocol checks.
```

It may support a scoped confirmatory edge in `ClaimEnvelope` for
`claim.confirmatory_boundary`, subject to the report's allowed claim level and
all other governance checks.

## Non-Proof Meaning

A passing protocol does not prove:

- deployment safety;
- external validity;
- dataset representativeness;
- label correctness;
- regulatory sufficiency;
- that a held-out matrix covers future adversaries;
- that sophisticated post-selection inference was performed;
- that report polish repairs an invalid protocol.

Confirmatory validity depends on the protocol and run separation, not on report
formatting quality.

## Governance Integration

`verify_claim_governance` treats `confirmatory_protocol` as a semantic evidence
role. It parses the payload, runs `verify_confirmatory_protocol_artifact`, and
adds a `confirmatory_protocols` block to the governance audit.

Mappings:

- protocol verifier `FAIL` -> governance `FAIL`;
- protocol verifier `NEEDS_REVIEW` -> governance `NEEDS_REVIEW`;
- protocol verifier `PASS` -> no additional review by itself.

`ClaimEnvelope` only emits a confirmatory support edge for a protocol artifact
that passed governance checks. Review-triggering protocol artifacts emit review
edges, and failed protocol artifacts emit invalidation edges.

If a report attaches `confirmatory_failure_matrix` evidence without any
`confirmatory_protocol` artifact, governance requires review because the matrix
has not been bound to both a plan and a run.

## CLI Status

No dedicated CLI is added in v0. The current clean integration point is
`cc-report verify-claim-governance`, which already verifies attached semantic
evidence artifacts. A future CLI should be a thin wrapper around
`verify_confirmatory_protocol_artifact`, not a separate policy engine.
