# Future Expansion: Epistemic Research Upgrade

## Status

Method documentation. This package defines how a narrative, an intuition, or a
motivational source is converted into research objects that can be tested,
replicated, or discarded.

It adds no claim to the
[Claim Boundary Manifest](../../claims/CLAIM_BOUNDARY_MANIFEST.md), changes no
kernel behavior, produces no measurement, and is not evidence for anything. It
is a procedure for producing evidence later, under stated rules.

## Purpose

Turn subjective intuition into disciplined inquiry without pretending that
intuition is evidence.

A source narrative may remain meaningful as reflection or motivation. It must
not be used to predict revenue, demand, personal outcomes, technical
feasibility, or market timing. The operating rule is:

```text
A story may generate a question.
Only a registered observation may support an answer.
```

## The operating loop

```text
Narrative / intuition
  -> candidate hypothesis
  -> alternative explanation
  -> predeclared indicator
  -> reversible test
  -> observed result
  -> replication or control
  -> bounded decision
```

The loop is one-directional. A result may retire a hypothesis; it may not
retroactively edit the indicator that was predeclared to test it.

## Non-negotiable rules

1. No narrative-derived statement becomes a business, product, or technical
   claim.
2. Every experiment names a decision it can change and a result that would stop
   the work.
3. Results, interpretation, and non-claims remain separate fields.
4. No human research, customer outreach, external data collection, paid API
   use, or live cloud work occurs without explicit authorization and
   appropriate consent.
5. Tracks stay separate. Two projects may share epistemic methods without
   sharing customers, claims, data, or branding.

## Research foundations

- Personalized but broadly applicable descriptions can feel unusually accurate.
  When studying "resonance," use blinded or shuffled comparison material rather
  than assuming felt accuracy reflects predictive accuracy.
  [Mason & Budge, 2011](https://pubmed.ncbi.nlm.nih.gov/21315874/)
- Freeze protocols before observation. OSF registrations are designed as a
  fixed project state that allows a visible withdrawal record rather than
  silent revision.
  [OSF registration guidance](https://help.osf.io/article/330-welcome-to-registrations)
- Treat small early studies as feasibility work with explicit progression
  criteria, not as demonstrations of effectiveness.
  [CONSORT pilot and feasibility guidance](https://www.bmj.com/content/355/bmj.i5239)
- Attestation mechanisms have separate proof boundaries. AWS Nitro attests
  enclave measurements; Apple App Attest validates an app instance connecting
  to a server; C2PA validates association and integrity of signed assertions,
  not whether those assertions are true.
  [AWS Nitro](https://docs.aws.amazon.com/enclaves/latest/user/set-up-attestation.html),
  [Apple App Attest](https://developer.apple.com/documentation/DeviceCheck?changes=_3),
  [C2PA](https://spec.c2pa.org/specifications/specifications/1.1/specs/C2PA_Specification.html)

## Contents

| Document | What it is | When to use it |
| --- | --- | --- |
| [Source Ledger](source-ledger.md) | A worked Week 1 artifact: one narrative source classified statement by statement into reflection, forecast, causal assertion, action heuristic, empirical claim, and marketing funnel, with testable translations and disallowed inferences. | Whenever a narrative, pitch, reading, or founder intuition is proposed as a reason to act. |
| [Eight-Week Plan](eight-week-plan.md) | The schedule: convert, destroy boundaries, pre-register, build instruments, run reversible work, replicate, review adversarially, decide. | Starting a bounded research push where the deliverable is a dated evidence packet. |
| [Prompt Library](prompt-library.md) | Eleven prompts that produce research artifacts rather than predictions. | Drafting a ledger entry, a boundary audit, a prior-art sweep, a preregistration, a replay handoff, a decision memo, or a retraction. |

## Relationship to existing CC-Framework surfaces

This package does not create a parallel governance process. Where CC-Framework
already has a mechanism, the plan uses it:

| Plan artifact | Existing surface |
| --- | --- |
| Predeclared claim with owner, falsifier, decision, window, non-claim | [Claim Boundary Manifest](../../claims/CLAIM_BOUNDARY_MANIFEST.md) and `docs/claims/claim_boundary_manifest.v0.1.json` |
| Explicit non-claim per result | [Non-Claims](../NON_CLAIMS.md) |
| Evidence typed by what it can support | [Evidence Role Ontology](../../design-specs/evidence_role_ontology.md) |
| Exploratory result kept separate from confirmatory result | `src/cc/evidence/confirmatory_protocol.py` |
| Result that expires and must be rechecked | `src/cc/evidence/decay.py` |
| Replayable record of what was run | [CC Reports and Receipts](../CC_REPORTS.md) |
| Scoped human sign-off that does not upgrade evidence | `human_review_note` handling in `src/cc/evidence/claim_governance.py`; the dedicated artifact is still a planned [prompt-roadmap item](../prompt-roadmap/09_human_review_artifact.md) |
| Prompt-shaped execution specs | [Prompt Roadmap](../prompt-roadmap/02_claim_envelope_boundary_envelope.md) |

The one boundary worth restating: a governance PASS means an evidence-bound
package is internally consistent under the verifier rules. It does not mean the
underlying result is correct, and it does not mean a system is safe.

## Porting to sibling projects

The eight-week plan carries two project-specific tracks, written to be lifted
into their own repositories rather than run from here. When those repositories
are available as writable workspaces, copy this directory to:

| Track | Destination |
| --- | --- |
| B2B / Assay-Limina | `limina/docs/future-expansion/` |
| Vinctura | `docs/pivot/research/future-expansion/` |

Two conditions apply to that copy. First, reconcile the plan with whatever
experiment registry and claim register already exist in the destination
repository; do not create a second, competing set of metrics. Second, check the
destination working tree before writing, since uncommitted work in an unrelated
area is easy to bury.

## Non-claims

- This package does not establish that any narrative source is accurate,
  predictive, or causal.
- Classifying a statement as testable does not mean it has been tested.
- Completing the eight weeks does not produce certainty; the completion
  standard is fewer unsupported claims, better instruments, clearer stop
  conditions, preserved negative evidence, at least one independently
  replayable result, and a more honest next decision.
- Nothing here authorizes human-subject research, customer outreach, external
  data collection, or paid or live cloud work. Those require separate,
  explicit authorization.
