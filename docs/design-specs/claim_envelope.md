# ClaimEnvelope and BoundaryEnvelope

`ClaimEnvelope` is the typed intermediate representation for evidence-bound
claims. It compiles from existing `cc.report.v0.3.1` reports and optional
claim-governance audit outputs. It does not change the stable report schema.

## Purpose

A CC report is the stable receipt format. A claim envelope is the internal
governance object that makes the report's permitted interpretation explicit:

- what proposition is being claimed,
- which claim fragments evidence may support,
- what boundary, assumptions, non-claims, defeaters, and invalidators apply,
- which artifacts are bound to the claim,
- which typed support edges exist between artifacts and claim fragments,
- what governance state has been computed or attached.

The envelope exists to prevent accidental support inflation. Evidence attached
to a report is preserved, but it does not automatically support the whole claim.

## Public Schemas

The implementation lives in `src/cc/evidence/claim_envelope.py`. Role-specific
support permissions are defined in
[Evidence Role Ontology](evidence_role_ontology.md).

Public schema strings:

- `cc.claim_envelope.v1`
- `cc.boundary_envelope.v1`
- `cc.support_graph.v1`
- `cc.claim_envelope.support_summary.v1`
- `cc.claim_envelope.governance_state.v1`

The public compiler is:

```python
from cc.evidence import compile_claim_envelope

envelope = compile_claim_envelope(report, governance_audit=audit)
```

The deterministic helpers are:

```python
from cc.evidence import claim_envelope_sha256, claim_envelope_to_canonical_json
```

They use the same canonical JSON discipline as report receipts: sorted keys,
compact separators, UTF-8, and no non-finite floats.

## Support Graph Semantics

Support is represented by typed edges, not by loose artifact lists.

Allowed relations:

- `supports`
- `bounds`
- `qualifies`
- `invalidates`
- `requires_review`
- `integrity_binds`
- `exploratory_suggests`
- `confirmatory_tests`

Allowed strengths:

- `weak`
- `diagnostic`
- `confirmatory`
- `integrity_only`

Role boundaries in v1:

| Role | May support | Must not support |
| --- | --- | --- |
| `receipt_integrity` | Artifact and report byte integrity only. | Statistical validity, deployment safety. |
| `measurement_evidence` | The named report interval under the report scope. | Deployment safety or generalization beyond the report. |
| `calibration_evidence` | The operating point and calibration boundary. | Deployment safety or external validity. |
| `claim_decay` | Staleness, time bounding, freshness invalidation, review pressure. | Deployment safety, statistical validity, confirmatory evidence. |
| `extremal_scenario` | Endpoint feasibility and counterfactual dependence bounds. | Likelihood, deployment realization. |
| `human_review` / `human_review_note` | Scoped authorization of use. | Upgrading underlying statistical evidence or evidence not reviewed. |
| Unknown roles | Preservation and review triggers. | Any strengthening support. |

## Proof Meaning

A valid `ClaimEnvelope` proves only that the envelope is internally well typed:
its artifacts have stable identifiers, its support edges point to known sources,
and its role-specific edge restrictions are respected. When compiled with a
governance audit, it also records the verifier's current verdict, freshness
state, review requirement, and support summary.

Receipt support means byte integrity. Decay support means time-bounding or
review pressure. Extremal-scenario support means endpoint feasibility or
counterfactual bounds. Human-review support means scoped authorization, subject
to the reviewed artifact set.

## Non-Proof Meaning

A valid envelope does not prove:

- the AI system is safe in deployment,
- the evaluation population is representative,
- labels are correct,
- the claim remains fresh after decay triggers,
- an extremal endpoint is likely,
- a human reviewer upgraded weak evidence into strong statistical evidence,
- unknown evidence roles carry support power.

A `PASS` result from `cc-report verify-claim-governance` means the evidence-bound
claim package is internally consistent under the verifier rules. It does not
mean the AI system is safe in deployment.

## Verifier Integration

`verify_claim_governance` remains read-only over `cc.report.v0.3.1` packages. It
now attaches an `envelope_support` summary to the audit output. The summary is
derived from a compiled `ClaimEnvelope` and includes edge counts, strength
counts, unknown-role counts, and review-edge counts. The verifier does not write
an envelope artifact by default and does not mutate reports.

This keeps the migration narrow:

1. existing report fixtures remain valid,
2. existing receipt hashes do not change,
3. support semantics become machine-readable,
4. the evidence role ontology and future human-review artifacts can plug into
   the same support graph.
