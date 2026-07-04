# From Safety Scores to Evidence-Bound Claims

A claim governance architecture for compositional AI safety.

This memo names the next research spine for CC-Framework. It is a synthesis
document, not a new certification claim and not an implemented product spec. It
connects the finite-atom dependence kernel, endpoint witnesses, CC reports,
claim decay artifacts, non-claims, and human review into one governing object:
the evidence-bound claim.

North star:

```text
The future of AI assurance is not better safety scores; it is evidence-bound
claims that know their scope, their counterexamples, their expiration
conditions, and the worlds in which they fail.
```

## 1. Why Scores Are Insufficient

A single safety score hides the objects a reviewer actually needs:

- the evaluated population and binary event convention,
- the assumptions used to interpret the measurement,
- the dependence structure among guardrail failures,
- the endpoint worlds still feasible under the evidence,
- the artifacts and hashes that support the report,
- the conclusions the report explicitly refuses to make,
- the conditions under which the claim degrades or expires,
- the review state and accountable human decision.

CC-Framework's core thesis already points away from score-first language.
Singleton guardrail metrics do not identify composed risk unless dependence is
measured, bounded, or explicitly assumed. The honest output is therefore not a
single number. It is a bounded evidence structure.

The claim-governance layer makes that structure explicit. A safety statement
becomes a living object with boundaries, evidence, endpoint scenarios, decay
rules, receipts, non-claims, and review state.

## 2. The Evidence-Bound Claim

An evidence-bound claim is a scoped assertion whose permitted interpretation is
defined by the evidence it binds and by the conclusions it excludes.

Conceptually, the object is:

```text
claim
+ boundary
+ assumptions
+ evidence artifacts
+ endpoint scenarios
+ decay policy
+ receipt
+ non-claims
+ review state
+ computed verification state
```

The current repository already implements pieces of this object:

| Piece | Current anchor | Boundary |
| --- | --- | --- |
| Claim text and allowed claim level | `cc.report.v0.3.1`, documented in [CC Reports and Receipts](CC_REPORTS.md) | The report permits only the selected claim level and requires explicit non-claims above diagnostic use. |
| Evidence binding | report evidence artifacts and canonical SHA-256 receipt | Hashes bind bytes; they do not prove statistical validity or deployment safety. |
| Claim decay | `claim_decay` evidence role and decay models | The signed artifact records policy; live freshness is computed at verification time. |
| Endpoint scenarios | `extremal_scenario` evidence role and scenario models | Endpoint worlds show feasible or fitted scenarios, not likely deployment futures. |
| Non-claims | [Non-Claims](NON_CLAIMS.md) | Non-claims are part of the claim boundary, not after-the-fact caveats. |
| Human review | assurance schema review status | Structured evidence can require review; it should not auto-approve release claims. |

The next architectural step is not to inflate the report schema. It is to make
the governing idea legible: every safety statement should answer what it claims,
what evidence supports it, what worlds could still break it, when it goes stale,
and what it refuses to claim.

## 3. Boundary and Non-Claims

A claim without machine-readable boundaries is easy to misuse. A reviewer needs
to see not only the positive assertion, but also the excluded assertions.

Minimum boundary fields:

- evaluated population or run scope,
- binary event convention,
- composition event,
- measurement assumptions,
- dependence assumptions or dependence uncertainty,
- artifact set,
- allowed claim level,
- non-claims,
- review requirements,
- defeaters or unresolved challenges.

Future object:

```text
BoundaryEnvelope(
    scope,
    assumptions,
    non_claims,
    defeaters,
    unresolved_review_items,
    allowed_claim_level,
    review_status,
)
```

Design principle:

```text
A safety claim is only machine-verifiable if its non-claims are also
machine-readable.
```

Non-claims should be generated or required for each evidence role. Examples:

- This report does not claim independence between guardrails.
- This report does not claim deployment validity after model or policy changes.
- This report does not claim the fitted dependence model is true.
- This report does not claim an extremal endpoint is likely.
- This report does not claim exploratory red-team intervals are confirmatory.
- This report does not claim the claim remains fresh after decay triggers fire.

## 4. Endpoint Worlds as Counterfactual Evidence

Endpoint scenarios are the visual and mathematical heart of the project. They
show the concrete worlds that a single score hides.

An endpoint scenario does not say:

```text
this world is likely
```

It says:

```text
this world is feasible under the stated evidence and assumptions
```

That distinction is the claim boundary.

Useful scenario kinds:

| Scenario kind | Meaning | Non-claim |
| --- | --- | --- |
| Frechet endpoint | Lower or upper endpoint joint law for a declared event under supplied constraints. | Does not prove the endpoint occurs in deployment. |
| Stress endpoint | A dependence-stressed world under a declared perturbation budget. | Does not prove an attacker can realize the stress path. |
| Fitted empirical scenario | A scenario fit from measured evidence. | Does not prove the fitted dependence model is true. |
| Confirmatory failure matrix | A held-out or predeclared failure matrix used for confirmation. | Does not inherit validity from adaptive discovery. |

Endpoint worlds turn an interval into inspectable evidence. Instead of saying
only that a composed failure risk is bounded by `[L, U]`, the system can show
the lower and upper atom tables, feasibility residuals, dominant failure
patterns, and explicit exclusions.

Research question:

```text
Can a safety case be trusted if it cannot display the endpoint worlds
consistent with its own evidence?
```

## 5. Claim Decay

Time is part of safety evidence. A claim can become stale even when the original
receipt remains byte-valid.

Current discipline:

- the decay artifact records signed policy, issue time, watched versions, and
  optional configured covariates;
- the live state is not stored in the receipt;
- verifiers compute `fresh`, `degraded`, or `expired` against their clock and
  observed versions;
- configured hazard scoring is a non-calibrated heuristic unless separately
  validated.

Future research direction:

```text
Temporal Compositional Assurance
```

Question:

```text
How long does a compositional safety claim remain valid after the world changes?
```

Potential decay drivers:

- time since evaluation,
- model version drift,
- policy version drift,
- dataset shift,
- adversary-class shift,
- dependence-width uncertainty,
- fitted-dependence instability,
- confirmatory evidence age.

Hypothesis:

```text
A claim with a wide Frechet interval should decay faster than a claim with a
narrow, well-identified dependence structure.
```

This should begin as deterministic policy: fixed TTLs, version-watch invalidity,
and explicit review triggers. Configured risk scoring can remain clearly marked
as non-calibrated until real claim-failure data exist.

## 6. Exploratory Versus Confirmatory Evidence

Adaptive red-team search can discover cliffs, failure modes, and candidate
dependence structures. That does not automatically make post-search intervals
confirmatory.

Firewall:

- Exploratory evidence may guide hypotheses, triage, and future tests.
- Confirmatory evidence must come from a separately valid procedure.
- Receipts should preserve this distinction through evidence roles and
  non-claims.

Possible confirmatory designs:

- held-out confirmation sets,
- sample splitting,
- nested bootstrap designs,
- selective-inference adjustments,
- pre-registered red-team protocols,
- anytime-valid testing,
- conformal failure certificates.

Research question:

```text
When does red-team discovery become certifiable evidence?
```

## 7. Receipts and Artifact Binding

Receipts are integrity objects. They preserve the relationship among claim text,
evidence bytes, assumptions, non-claims, metadata, and canonical hashing.

A receipt can prove:

- the report validates against a declared schema,
- the canonical bytes reproduce the recorded hash,
- the named evidence files match their recorded SHA-256 hashes,
- the claim boundary and non-claims were part of the reviewed object.

A receipt cannot prove:

- production safety,
- statistical validity,
- dataset representativeness,
- label correctness,
- legal compliance,
- sufficiency of evidence for a release decision.

This is why the receipt belongs inside claim governance, not outside it. It
binds the object under review; it does not decide the review.

## 8. Human Review and Claim State

The future governance layer should treat claims as stateful. State changes
should be driven by evidence, version changes, review actions, and decay policy.

Candidate states:

- `draft`
- `issued`
- `fresh`
- `degraded`
- `expired`
- `challenged`
- `superseded`
- `withdrawn`
- `revalidated`
- `human_reviewed`
- `rejected`

Future object:

```text
ClaimStateTransition(
    claim_id,
    from_state,
    to_state,
    trigger,
    evidence_artifact_ids,
    evaluated_at,
    rationale,
    reviewer_id=None,
)
```

Transition triggers include:

- new model version,
- new policy version,
- new attack class,
- new evaluation run,
- new red-team cliff,
- new confirmatory evidence,
- human review,
- expired TTL,
- superseding claim.

This is the bridge from mathematical repository to governance operating system:
not a dashboard of scores, but a lifecycle for evidence-bound claims.

## 9. Future ClaimEnvelope

The longer-term unifying object can be a `ClaimEnvelope`: a typed container for
claim text, boundaries, evidence roles, endpoint worlds, decay policy,
verification results, and review state.

Candidate shape:

```text
ClaimEnvelope(
    claim_id,
    statement,
    allowed_claim_level,
    scope,
    assumptions,
    evidence_artifacts,
    endpoint_scenarios,
    decay_policy_ref,
    receipt_ref,
    non_claims,
    defeaters,
    review_status,
    state,
    state_history,
)
```

Migration path:

1. Keep `cc.report.v0.3.1` stable.
2. Continue attaching new evidence through evidence roles.
3. Use documentation and examples to teach the envelope concept.
4. Introduce `ClaimEnvelope` only when it removes real duplication across
   reports, assurance schemas, decay records, and review workflows.

## 10. Research Lanes

### Lane A - Claim Decay Theory

Question:

```text
How should safety claims expire under model drift, policy drift, adversary
drift, and dependence uncertainty?
```

Deliverables:

- `docs/research/claim_decay_theory.md`
- hardened deterministic decay policies
- simulation where claims with different dependence widths decay differently
- claim half-life visual, clearly marked as configured unless calibrated

### Lane B - Endpoint Scenario Semantics

Question:

```text
What does an extremal scenario prove, and what does it explicitly not prove?
```

Deliverables:

- `docs/research/extremal_scenario_semantics.md`
- definitions for feasible endpoint, fitted empirical scenario, stress-limit
  scenario, and confirmatory failure-matrix scenario
- non-claims for each scenario kind

### Lane C - Exploratory to Confirmatory Evidence

Question:

```text
When does red-team discovery become confirmatory AI safety evidence?
```

Deliverables:

- confirmation protocol memo
- evidence-role guidance for exploratory and confirmatory artifacts
- test fixtures that prevent adaptive confidence intervals from being promoted
  silently

### Lane D - Claim and Boundary Envelopes

Question:

```text
Can claims, assumptions, non-claims, defeaters, and review states share one
schema without overloading the report receipt?
```

Deliverables:

- `docs/design-specs/claim_envelope.md`
- `docs/design-specs/boundary_envelope.md`
- migration notes from `ClaimSummary`, assurance schemas, and evidence roles

### Lane E - Evidence Role Ontology

Question:

```text
What kind of claim can each evidence artifact support, and what kind of claim
can it not support?
```

Candidate roles:

- `measurement_evidence`
- `dependence_evidence`
- `boundary_evidence`
- `decay_evidence`
- `human_review_evidence`
- `confirmatory_evidence`
- `exploratory_evidence`
- `policy_evidence`
- `runtime_evidence`

Each role should answer:

- What claim level can this support?
- What non-claims are mandatory?
- Does it require human review?
- Is it exploratory or confirmatory?
- Can it become stale?

## 11. Demo and Visual Story

The public demo should show the life of a claim, not just the value of a metric:

1. A safety score appears.
2. The score resolves into a scoped claim.
3. The claim reveals evidence artifacts.
4. The evidence reveals endpoint worlds.
5. The endpoint worlds reveal non-claims.
6. A clock shows claim decay.
7. A red-team cliff appears as exploratory evidence only.
8. Confirmatory evidence upgrades or challenges the claim.
9. The receipt binds the artifact set.
10. Human review accepts, rejects, or asks for more evidence.

This visual story is the right public interface for the thesis. It makes the
epistemic shift visible: the point is not to sell a larger score, but to show
what the evidence can and cannot support.

## 12. One-Spine Architecture

The architecture should remain simple:

```text
Runtime system
  -> guardrail outputs
  -> measurement layer
  -> dependence bounds
  -> endpoint scenarios
  -> claim object
  -> boundary / non-claim object
  -> decay policy
  -> evidence manifest
  -> receipt / signature / transparency log
  -> human review
  -> claim state machine
```

The claim is not a marketing sentence placed after the computation. It is the
governed object inside the system.

## 13. Immediate Next Move

The next implementation is the executable verifier, not another prose layer.
`cc-report verify-claim-governance` reads an existing `cc.report.v0.3.1`
package, follows its evidence artifacts, and emits a
`cc/claim-governance-audit.v1` JSON verdict.

It checks:

- report readability and core structure,
- canonical receipt verification when possible,
- evidence artifact SHA-256 and byte counts,
- verification-time `claim_decay` freshness,
- `extremal_scenario` feasibility and exclusions,
- exploratory/confirmatory firewall boundaries,
- mandatory non-claims implied by evidence roles.

The v0 evidence-role support matrix is intentionally conservative:

| Role | Supports | Does not support | Review triggers |
| --- | --- | --- | --- |
| `claim_decay` | Time bounding, staleness review | Deployment safety, statistical validity | Expired, degraded, triggered versions |
| `extremal_scenario` | Dependence endpoint explanation, counterfactual feasibility | Likelihood, deployment realization | Excluded evidence fields, fitted evidence without confirmation |

Verifier verdicts are deliberately narrow:

- `PASS`: the evidence-bound claim package is internally consistent under the
  verifier rules.
- `NEEDS_REVIEW`: the package is readable, but conservative review triggers are
  present.
- `FAIL`: the package is unreadable, tampered, expired, malformed, infeasible,
  or has exploratory evidence leaking into a confirmatory surface.

A PASS verdict means the evidence-bound claim package is internally consistent
under the verifier rules. It does not mean the AI system is safe in deployment.

```bash
cc-report build-report ...
cc-report verify-claim-governance report.json \
  --out claim_governance_audit.json
```

Exit codes are `0` for `PASS`, `1` for `NEEDS_REVIEW`, and `2` for `FAIL`, so
the command can be used in CI without collapsing review-required and failed
states.

Do not scatter into broad product features yet. The strongest move is to keep
the spine executable, legible, and reviewable.
