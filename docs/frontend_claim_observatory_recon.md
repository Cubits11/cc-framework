# Frontend Claim Observatory Recon

## Executive Summary

This repository does have a frontend. The active source frontend is
`apps/dashboard`, a Next.js 15, React 18, TypeScript app with three views:
Composition Risk, Assurance Case, and Verify. It consumes
`cc/enterprise-dashboard-bundle.v1`, not the newer claim-governance capsule
artifacts directly.

The repo also has a newer deterministic claim-governance capsule under
`examples/claim_governance_capsule`. That capsule produces `cc_report.json`,
`claim_governance_audit.json`, `claim_envelope.json`, `capsule_manifest.json`,
decay, confirmatory, extremal-scenario, calibration, bounds, and audit-log
artifacts. These are not wired into `apps/dashboard`.

The architectural split is real but not hostile. The enterprise dashboard is
an evidence-integrity and enterprise-reference viewer. The capsule is a local
semantic witness for evidence-bound claims. They are conceptually aligned and
technically separate.

The canonical abstraction for claim semantics should be `ClaimEnvelope`,
compiled from `cc.report.v0.3.1` plus an optional `ClaimGovernanceAudit`.
Enterprise bundles should adapt to the envelope, not replace it. The existing
dashboard should evolve into a Claim Observatory by adding a bridge adapter and
new claim-governance views, while preserving its current enterprise transport
and Merkle proof surfaces.

Recommended path: Option C, hybrid. First add a deterministic TypeScript
adapter and tests that let `apps/dashboard` consume the capsule artifacts.
Then add Claim Anatomy, Evidence Body, Non-Claims Wall, Decay Clock, and
Support Graph views inside the existing dashboard. Add a tiny static capsule
viewer only if a zero-build artifact is needed for papers or offline review,
and keep it explicitly non-competing with `apps/dashboard`.

North star: not safety scores. Evidence-bound claims.

## Baseline Test Status

Starting tree:

- `git status --short`: clean before this document was added.
- Editable install: `source .venv/bin/activate && python -m pip install -e ".[dev,docs,notebooks]"` succeeded.
- Baseline proof gate: `source .venv/bin/activate && make governance-proof-fast` succeeded.

`make governance-proof-fast` ran:

- governance CLI help checks.
- governance import checks.
- `ruff check src/cc/evidence src/cc/reporting tests/unit/evidence tests/unit/reporting tests/integration/test_claim_governance_capsule.py`.
- `ruff format --check` over the same paths.
- `pytest tests/unit/evidence/test_claim_governance.py tests/integration/test_claim_governance_capsule.py -q`, with 23 tests passing.
- `bash examples/claim_governance_capsule/reproduce.sh --verify-only`.
- `bash examples/claim_governance_capsule/reproduce.sh`.
- capsule inspection and `git diff --check`.

One raw inventory command initially hit generated/vendor noise in
`.venv-enterprise`, `apps/dashboard/.next`, `apps/dashboard/node_modules`, and
`infra/node_modules`. Source inventory below excludes generated dependency and
build output directories, while still inspecting generated capsule outputs
because they are first-class artifacts for this recon.

`mkdocs.yml` has no explicit `nav`; adding this file under `docs/` does not
require mkdocs nav edits.

## Existing Frontend Inventory

| Path | Technology | Purpose | Data model consumed | Active/tested | Claim-governance relation | Enterprise evidence relation | Crypto verification logic | Risks / unknowns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `apps/dashboard` | Next.js 15.5.19, React 18.3.1, TypeScript 5.5, CSS, lucide-react, Vitest/jsdom | Main dashboard with Composition Risk, Assurance Case, Verify | `EnterpriseBundle` in `apps/dashboard/lib/types.ts` | Tested by `apps/dashboard/tests/dashboard.smoke.test.tsx`; e2e via `tests/e2e/test_enterprise_smoke.py`; `make enterprise-smoke` installs deps and runs strict lane | Not directly connected to `ClaimEnvelope`, `ClaimGovernanceAudit`, or capsule manifest | Primary enterprise viewer for `cc/enterprise-dashboard-bundle.v1` | Client-side RFC6962-style inclusion and consistency proof verification in `apps/dashboard/lib/merkle.ts` | No schema validation, no claim-governance adapter, no non-claims wall, no decay UI, no signature/KMS verification in browser |
| `apps/dashboard/lib/merkle.ts` | Browser Web Crypto, TypeScript | Recompute leaf hashes, inclusion proofs, and consistency proof | `VerificationPayload` with records, canonical records, leaf hashes, proofs, trusted root | Covered by dashboard smoke test and backend/e2e generated bundle | None today | Verifies enterprise dashboard bundle proof payload | Yes: `sha256-rfc6962`; leaf prefix `0x00`, node prefix `0x01` | Trust boundary is root supplied inside uploaded bundle unless user separately verifies root provenance |
| `src/cc/enterprise/aws_reference.py` | Python, boto3/moto test surface | Local enterprise reference helpers; exports dashboard bundle; uploads and verifies with S3/KMS/DynamoDB | Evidence-bundle output directory, `cc/enterprise-dashboard-bundle.v1`, KMS attestation | Covered by `tests/integration/test_enterprise_aws_emulation.py` and e2e smoke | Could become the first Python-side bridge producer for claim-governance data | Central enterprise adapter and smoke generator | Uses RFC6962 Merkle log helpers; verifies KMS signature and sequence | Current exported bundle omits claim envelope and governance audit |
| `infra/` | AWS CDK TypeScript, Lambda Python | Minimum credible AWS deployment reference: S3 Object Lock, KMS signing, DynamoDB chain head, verify API | Enterprise dashboard bundles stored in S3 plus `enterprise_attestation` | CDK source present; emulation tests exercise equivalent API surface; `infra` package has `build` and `synth` scripts | None today | Enterprise deployment substrate | KMS Sign/Verify in CDK and Lambda; Lambda verifies inclusion and sequence | Lambda verifies inclusion and KMS but not full claim governance; no dashboard auth story in repo |
| `src/cc/evidence/assurance_schema.py` | Python Pydantic, JSON-LD/Markdown exporters | GSN-inspired assurance case schema and exports | Evidence bundle directory or payload | Covered by `tests/unit/evidence/test_assurance_schema.py`; used by enterprise export | Adjacent: supports defeaters, assumptions, evidence roles, review defaults | Feeds dashboard Assurance Case view | No direct crypto | Current dashboard only renders the tree; it does not expose role permissions or non-claim semantics |
| `examples/claim_governance_capsule` | Python script, Bash, deterministic JSON outputs | Regenerable local evidence-bound claim package | `cc.report.v0.3.1`, `ClaimGovernanceAudit`, `ClaimEnvelope`, manifest, bounds, scenarios, decay, confirmatory artifacts | Covered by `tests/integration/test_claim_governance_capsule.py` and governance proof gate | Primary claim-governance artifact set | Not connected to enterprise bundle or dashboard today | Hash manifest and report receipts; intentionally no fake package/signature layer | Needs viewer/adapter; generated outputs should not be hand-edited |
| `tools/week6_artifact.html` | Static HTML, React/Recharts via CDN, inline JS math | Older interactive research artifact for FH intervals, Bernstein CIs, lambda coordinates, adversarial stress | Inline state, URL state, JS calculations | No direct current test found; archived/tool artifact | No typed claim-governance fields | No enterprise bundle | No crypto | Uses "PASS" and "Compositional Safety Research OS" phrasing; should not become the claim observatory baseline without language review |
| `paper/figures/theorem1_visual.html` | Static HTML, CSS, MathJax, inline JS | Interactive visual abstract for Frechet-Hoeffding bounds | Inline calculators and visualization state | No direct test found | No direct claim-governance fields | No enterprise bundle | No crypto | Visual style is paper/demo oriented, not enterprise or claim-governance oriented |
| `notebooks/*.ipynb` | Jupyter notebooks | Tiny notebook stubs for toy validation, two-world experiments, scaling runtime | Notebook markdown only in current files | Not active dashboard surface | No direct capsule connection | No enterprise connection | No crypto | They are browser-facing artifacts but not a real UI surface |
| `docs/` | MkDocs Material | Documentation site | Markdown docs, generated figures, design specs | `mkdocs build --strict` is available; docs proof target exists | Strong design/spec grounding for claim envelope, role ontology, reports, non-claims | Enterprise docs in `README_ENTERPRISE.md` and `docs/validation_matrix.md` | Documents transparency log and receipt semantics | This recon doc should remain source-grounded and avoid becoming product fluff |
| `scripts/generate_figures/*`, `scripts/make_week*_figs.py` | Python, matplotlib | Figure generation for research docs/paper | CSVs and research result data | Regression tests cover some figure helpers | Indirect: paper evidence, not claim lifecycle | No enterprise bundle | No crypto | Not an interactive frontend |

Other relevant browser-facing or frontend-adjacent files:

- There is no repo-root JavaScript package. Dashboard dependencies live in
  `apps/dashboard/`; infrastructure dependencies live in `infra/`.
- `apps/dashboard/.next` and `apps/dashboard/node_modules` are local generated
  outputs/dependencies, not source.
- `infra/node_modules` and `infra/cdk.out` are generated/dependency outputs.

## Existing Dashboard Deep Read

### Framework And Version

`apps/dashboard/package.json` identifies:

- Next.js `15.5.19`.
- React and React DOM `18.3.1`.
- TypeScript `5.5`.
- `lucide-react` for icons.
- Vitest `3.2.6`, Vite `6.4.3`, jsdom, and Testing Library for tests.

There is no Tailwind config. Styling is a single CSS file:
`apps/dashboard/app/styles.css`.

There are no graph libraries in the dashboard. The Composition Risk view uses a
hand-written SVG chart.

### Current Routes And Views

The app has a single route:

- `apps/dashboard/app/page.tsx` renders `DashboardShell`.

`DashboardShell` has three tabs:

- `Composition Risk`: `apps/dashboard/components/views/CompositionRiskView.tsx`.
- `Assurance Case`: `apps/dashboard/components/views/AssuranceCaseExplorer.tsx`.
- `Verify`: `apps/dashboard/components/views/VerifyView.tsx`.

There is no route-level evidence bundle viewer, no claim envelope view, no
claim ledger, no decay view, no non-claims wall, no review workbench, and no
capsule manifest viewer.

### Current Data Model

The dashboard model is in `apps/dashboard/lib/types.ts`.

`EnterpriseBundle` fields:

- `schema`
- `bundle_id`
- `created_at`
- `composition_risk`
- `assurance_case`
- `verification`
- `attestation`
- `enterprise_attestation`
- `metrics`
- `manifest`

`CompositionRisk` fields:

- `composition_rule`
- `marginals`
- `envelope`
- `empirical`
- `cliff_certificate`

`AssuranceCase` fields:

- `id`
- `run_id`
- `top_claim`

`TreeClaim` supports:

- strategy
- contexts
- assumptions
- evidence
- defeaters
- subclaims
- review status

`VerificationPayload` supports:

- `hash_algorithm`
- `trusted_root`
- `tree_size`
- `records`
- optional `canonical_records`
- `leaf_hashes`
- `inclusion_proofs`
- `consistency_proof`

The dashboard does not model:

- `ClaimGovernanceAudit`.
- `ClaimEnvelope`.
- `BoundaryEnvelope`.
- `SupportGraph`.
- `SupportEdge`.
- capsule manifest files.
- `claim_decay` policy/freshness status.
- confirmatory protocol audit checks.
- role ontology support permissions and forbidden support.
- non-claims as first-class UI.

### Current Verification Model

The dashboard performs client-side proof checks:

- `verifyEnterpriseBundle` checks hash algorithm, leaf hashes, inclusion proofs,
  and consistency proof.
- `verifyInclusion` recomputes leaf hash, validates proof shape and sides, and
  checks against a trusted root argument.
- `verifyConsistency` implements RFC6962-style consistency verification.
- `leafHash` uses `0x00 || canonical_record`.
- internal node hash uses `0x01 || left || right`.

The dashboard does not perform:

- KMS signature verification.
- Ed25519 attestation verification.
- Ed25519 witness anchor verification.
- report receipt hash verification for `cc.report.v0.3.1`.
- evidence artifact byte/hash verification.
- claim-governance verification.
- deterministic replay of capsule outputs.

Backend and infrastructure verification exist elsewhere:

- `src/cc/enterprise/aws_reference.py` verifies inclusion, consistency, KMS
  signature, and sequence number.
- `infra/lambda/verify_handler.py` verifies inclusion, KMS signature, and
  sequence number.
- `src/cc/core/evidence_bundle.py` verifies manifest, metrics, transparency
  root, optional Ed25519 signature, and optional witness anchor.

### Current UI Language

Good bounded language already present:

- `README_ENTERPRISE.md`: explicitly says the architecture does not certify AI
  safety, compliance, guardrail correctness, policy correctness, or deployment
  readiness.
- `README_ENTERPRISE.md`: says evidence integrity is not deployment safety
  certification.
- Dashboard Composition Risk copy says bounds are under stated marginals and
  the plotted point is the empirical run estimate.
- Assurance Case renders defeaters visibly.

Language to handle carefully:

- `VerifyView` shows `Client verification: passed`. This is acceptable only if
  the UI makes clear this means Merkle proof verification, not claim validity.
- `Composition Risk` is a mathematical view, but "risk" can invite product
  users to treat the chart as a deployment decision. Future copy should anchor
  it to "bounded empirical claim" and "not deployment-safety proof".
- `tools/week6_artifact.html` contains older "Compositional Safety Research OS"
  and "PASS: qualifies for next-stage transfer tests" language. Do not reuse it
  for the claim observatory without rewriting the labels.

### Strengths

- The dashboard exists and is small enough to extend without fighting a large
  frontend architecture.
- It already has a tested upload path and local proof recomputation.
- It already visualizes the Frechet-Hoeffding envelope and empirical estimate.
- The assurance case model already includes assumptions, evidence, defeaters,
  and review status.
- `README_ENTERPRISE.md` sets strong non-claim boundaries.
- The enterprise smoke tests connect Python, AWS emulation, backend verification,
  and the dashboard.

### Weaknesses

- The dashboard only understands the enterprise bundle schema.
- The dashboard has no runtime schema validation or failure-mode-specific
  parsing.
- The claim-governance capsule is invisible to the dashboard.
- Non-claims are not first-class UI objects.
- Evidence roles are rendered as detail text, not as support permissions and
  forbidden support.
- Decay freshness is not shown.
- Confirmatory protocol checks are not shown.
- Human review state is only a generic tree detail when present.
- Browser verification does not verify root provenance, KMS, Ed25519 signatures,
  witness anchors, or `cc.report.v0.3.1` receipts.

### Philosophical Alignment

The dashboard is partly aligned. It shows evidence, defeaters, and proof
verification rather than a single safety score. But it is not yet aligned with
the stronger capsule thesis: typed support semantics for claims that can expire,
be challenged, be reviewed, replayed, and bounded by explicit non-claims.

### Overclaim Risks

- Users could interpret "passed" as "safe" unless the UI labels the exact
  verification surface.
- Users could interpret Merkle verification as claim verification.
- Users could interpret the assurance case tree as an accepted safety case,
  even though `assurance_schema.py` defaults claims, assumptions, and defeaters
  to human review and says it does not certify safety, compliance, or
  conformance.
- Users could interpret a fresh decay state as current deployment validity.

### Missing Screens

- Claim Ledger.
- Claim Anatomy.
- Evidence Body.
- Typed Support Graph.
- Non-Claims Wall.
- Decay Clock.
- Challenge/Defeater Surface.
- Human Review Workbench.
- Replay Console.
- Public Claim Receipt Page.
- Claim Lifecycle Ledger.

### Relationship To Current Claim-Governance Capsule

No direct relationship in code today. The capsule builds `ClaimEnvelope` and
`ClaimGovernanceAudit`, but `apps/dashboard` only consumes `EnterpriseBundle`.
The closest bridge point is `src/cc/enterprise/aws_reference.py`, which exports
the compact dashboard bundle.

## Claim-Governance Capsule Artifact Map

The deterministic capsule is implemented in:

- `examples/claim_governance_capsule/build_capsule.py`
- `examples/claim_governance_capsule/reproduce.sh`
- `examples/claim_governance_capsule/README.md`
- `examples/claim_governance_capsule/inputs/*`
- `examples/claim_governance_capsule/expected/*`
- `examples/claim_governance_capsule/outputs/*`

It is tested by:

- `tests/integration/test_claim_governance_capsule.py`
- `make governance-proof-fast`

`examples/claim_governance_capsule/reproduce.sh` was run by the baseline proof
gate and passed.

### Artifact Roles

| Artifact | Schema / version | Role | Identity fields | Claim fields | Non-claims | Verdict/status | Hash/receipt | Decay/freshness | Support graph | Review fields | Scenario / confirmatory fields |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cc_report.json` | `cc.report.v0.3.1` | Stable report receipt for one scoped claim | `report_id`, `run.run_id`, `created_at`, git/env metadata | `claim.statement`, `claim.allowed_claim_level`, measurement interval | `claim.non_claims` | claim level only; no verdict | `receipt.canonical_hash`, evidence SHA-256 and byte counts | References `decay_policy.json` as evidence role | Source input for compiled envelope | assumptions only; no human review artifact in capsule | Measurement, calibration, evidence artifacts |
| `claim_governance_audit.json` | `cc/claim-governance-audit.v1` | Read-only verifier result | `report_id`, `evaluated_at` | `claim_statement`, `allowed_claim_level` | `non_claims`; counts in `boundary` | `verdict`, `required_human_review`, `reasons` | `receipt.report_hash_verified`, artifact hash audit rows | `decay.present`, `decay.status`, `trigger_summary` | `envelope_support` summary | `required_human_review`, boundary unresolved gaps | `scenarios`, `confirmatory_protocols` audit details |
| `claim_envelope.json` | `cc.claim_envelope.v1` | Typed support IR | `identity.artifact_id`, `claim_id`, `subject_ref`, source report info | `proposition.statement`, fragments, allowed level | `boundary.non_claims` | `governance_state.verdict`, freshness, review requirement | source report hash and artifact refs | `governance_state.freshness_status`, `decay_refs` | Full `support_graph.support_edges` | `boundary.review_requirements` | `scenario_refs`, confirmatory edges |
| `capsule_manifest.json` | `cc.claim_governance_capsule_manifest.v1` | Deterministic package hash manifest | `capsule_id`, `report_id`, seed, fixed time | `governance_verdict` and report receipt | `pass_caveat` | `governance_verdict` | file SHA-256 and byte counts; input hashes; report receipt | none except fixed verification time | no full graph; points at files | none | files include confirmatory and scenario artifacts |
| `bounds.json` | `cc.capsule.bounds.v1` | Measurement evidence | source path/hash/rows, seed, guardrails | interval, observed event rate, metric family | two measurement non-claims | no verdict | bound by report and manifest hashes | none | becomes measurement evidence edge | none | pairwise joint probabilities and interval method |
| `calibration.json` | no schema field in current output | Calibration evidence | none beyond file hash | threshold, target FPR, alpha cap, realized FPR | none in file | `status: pass` | bound by report and manifest hashes | none | becomes calibration evidence edge | none | calibration window |
| `decay_policy.json` | `cc.claim_decay.v1` | Claim decay policy | `claim_id`, optional `claim_hash`, `issued_at` | evidence refs only | default decay non-claims | no live status in file | bound by report and manifest hashes | TTL policy, version watch set; live state computed by verifier | decay ref and staleness support edge | none | no scenario |
| `confirmatory_protocol.json` | `cc.confirmatory_protocol.v1` | Confirmatory protocol | `artifact_id`, `plan.protocol_id`, `run.run_id` | hypothesis, primary endpoint, analysis plan | protocol non-claims | audit status is in governance audit, not source file | bound by report and manifest hashes | none | confirmatory boundary edge | protocol/run separation fields | fixed analysis plan, sample plan, stopping rule, decision rule |
| `confirmatory_failure_matrix.json` | `cc.confirmatory_failure_matrix.v1` | Confirmatory evidence | `protocol_id`, `analysis_id` | event, guardrails, confirmatory CI | failure-matrix non-claims | no verdict | bound by report and manifest hashes | none | confirmatory failure-matrix edge | none | held-out matrix and confirmatory CI |
| `extremal_lower.json` | `cc.extremal_scenario.v1` | Extremal scenario | `scenario_id`, `kind`, source hash, endpoint | event probability and endpoint | scenario non-claims | `feasibility.is_feasible` | bound by report and manifest hashes | none | endpoint feasibility edge | excluded evidence fields can trigger review | atom table, feasibility residuals, top outcomes |
| `extremal_upper.json` | `cc.extremal_scenario.v1` | Extremal scenario | same as lower | same as lower | same as lower | same as lower | same as lower | none | endpoint feasibility edge | same as lower | same as lower |
| `audit_log.jsonl` | no top-level schema in capsule records | Reproducibility audit context | steps: load inputs, compute bounds, emit governance chain | no claim | no non-claims | no verdict | bound by report and manifest hashes | none | preserved as audit log, no support edge | none | step chain |

### Required Fields Called Out By This Recon

Observed values in the generated capsule:

- `claim_governance_audit.schema`: `cc/claim-governance-audit.v1`.
- `claim_governance_audit.verdict`: `pass`.
- `claim_governance_audit.allowed_claim_level`: `bounded_empirical`.
- `claim_governance_audit.required_human_review`: `false`.
- `claim_governance_audit.non_claims`: 17 entries.
- `claim_governance_audit.decay.status`: `fresh`.
- `claim_governance_audit.envelope_support`: 10 support edges, 1 integrity-only
  edge, 7 diagnostic edges, 2 confirmatory edges, 0 review edges, 0 unknown
  role refs.
- `claim_envelope.schema`: `cc.claim_envelope.v1`.
- `claim_envelope.identity`: source report id/hash/schema, subject ref,
  created/evaluated time.
- `claim_envelope.proposition`: statement, `bounded_empirical`, typed fragments.
- `claim_envelope.boundary.non_claims`: merged claim, role, decay, scenario,
  confirmatory, and measurement non-claims.
- `claim_envelope.support_graph.support_edges`: receipt integrity, measurement
  bounds, calibration qualifies, confirmatory tests, decay staleness, extremal
  endpoint bounds.
- `claim_envelope.governance_state.schema`:
  `cc.claim_envelope.governance_state.v1`.
- `claim_envelope.governance_state.verifier_schema`:
  `cc/claim-governance-audit.v1`.
- `claim_envelope.governance_state.verdict`: `pass`.
- `claim_envelope.governance_state.required_human_review`: `false`.
- `claim_envelope.governance_state.freshness_status`: `fresh`.
- `claim_envelope.governance_state.support_summary`: same compact summary as
  `claim_governance_audit.envelope_support`.
- `capsule_manifest.schema_version`:
  `cc.claim_governance_capsule_manifest.v1`.
- `capsule_manifest.files`: 11 generated files with filename, role, SHA-256,
  and byte count.
- `capsule_manifest.pass_caveat`: PASS means internal consistency under
  verifier rules; it does not mean the AI system is safe in deployment.

### Schema Bridge Table

| Claim Governance Capsule Field | Existing Dashboard / Enterprise Field Equivalent | Match | Required transformation | UI implication |
| --- | --- | --- | --- | --- |
| `cc_report.report_id` | `EnterpriseBundle.bundle_id` | Partial | Preserve report id as claim id; do not overwrite enterprise run bundle id if both exist | Display separate "claim id" and "bundle id" |
| `cc_report.schema_version` | `EnterpriseBundle.schema` | Partial | Support multiple schema strings in a discriminated loader | Upload should identify enterprise bundle vs capsule package |
| `cc_report.claim.statement` | `AssuranceCase.top_claim.statement` | Partial | Generate a claim anatomy object or map to top claim text | Top-level claim should be first-class, not buried in assurance tree |
| `cc_report.claim.allowed_claim_level` | none | Missing | Add `claimLevel` or load from envelope proposition | Show claim level badge: `bounded_empirical`, never safety score |
| `cc_report.claim.non_claims` | none | Missing | Preserve as boundary non-claims | Non-Claims Wall must be always visible |
| `cc_report.measurement.interval` | `composition_risk.envelope` | Partial | Map lower/upper to composition interval; preserve point estimate and method | Existing chart can be reused with clearer claim-bound labels |
| `cc_report.calibration` | `composition_risk` marginal/metric side panel partly | Partial | Add calibration panel with status/window/threshold | Avoid reading calibration `pass` as deployment pass |
| `cc_report.evidence.artifacts[*]` | none except enterprise `manifest` raw object | Partial | Normalize to artifact cards with role, path, sha256, bytes, status | Evidence Body should be artifact-first |
| `cc_report.receipt.canonical_hash` | `enterprise_attestation.bundle_sha256` and `verification.trusted_root` | Partial | Treat report receipt hash separately from enterprise object hash and Merkle root | Verify page needs "receipt hash", "artifact hashes", and "Merkle root" as distinct facts |
| `claim_governance_audit.verdict` | Verify view `result.ok` | Partial and dangerous | Keep separate: governance verdict vs Merkle proof verification | UI must say `PASS under verifier rules`, not `verified safe` |
| `claim_governance_audit.required_human_review` | Tree node `review_status` | Partial | Add governance review status at claim level | Review Workbench can start with read-only status |
| `claim_governance_audit.reasons` | Verify errors | Partial | Display as governance reasons, separate from cryptographic errors | Differentiate "cannot verify bytes" from "requires review" |
| `claim_governance_audit.boundary.mandatory_non_claims_missing` | none | Missing | Add boundary diagnostics | Missing non-claims should be blocking/review UI |
| `claim_governance_audit.decay.status` | none | Missing | Map to `fresh/degraded/expired` freshness status | Add Decay Clock |
| `claim_governance_audit.scenarios` | Composition Risk chart envelope | Partial | Use scenario ids, feasibility, and endpoint data to enrich chart | Let users inspect lower/upper endpoint worlds |
| `claim_governance_audit.confirmatory_protocols.audits[*].checks` | none | Missing | Add confirmatory protocol checklist | Challenge Mode should show firewall checks |
| `claim_governance_audit.envelope_support` | none | Missing | Add support summary tiles | Show support counts by relation/strength without scalar score |
| `claim_envelope.identity.source_report_hash` | none | Missing | Include in claim identity | Public receipt page should bind source report |
| `claim_envelope.proposition.fragments` | none | Missing | Render claim fragments and targetable support | Claim Anatomy should show what evidence can touch |
| `claim_envelope.boundary.assumptions` | `TreeClaim.assumptions` | Partial | Merge or cross-link assumptions from envelope and assurance tree | Assumptions should be navigable and challengeable |
| `claim_envelope.boundary.defeaters` | `TreeClaim.defeaters` | Partial | Normalize defeaters with source/status | Defeater surface can reuse assurance tree rendering |
| `claim_envelope.boundary.invalidation_conditions` | none | Missing | Add lifecycle conditions list | Lifecycle ledger needs trigger definitions |
| `claim_envelope.boundary.review_requirements` | `review_status` strings | Partial | Add structured review requirement rows | Human review UI should not infer approval |
| `claim_envelope.support_graph.evidence_refs` | `TreeEvidence` | Partial | Map refs to artifact cards; preserve role/path/status/hash | Evidence cards should show role permissions and forbidden support |
| `claim_envelope.support_graph.scenario_refs` | none | Missing | Add scenario group | Endpoint scenario cards |
| `claim_envelope.support_graph.decay_refs` | none | Missing | Add decay group | Decay policy viewer |
| `claim_envelope.support_graph.receipt_refs` | Verify root/hash displays | Partial | Treat receipt refs as integrity-only support | Receipt card should say integrity only |
| `claim_envelope.support_graph.support_edges` | none | Missing | New graph data structure in TS | Support Graph is the central new visual |
| `claim_envelope.governance_state` | Verify view result | Partial | Separate `governanceState` from proof verification state | Header should show both: proof check and governance verdict |
| `capsule_manifest.files[*]` | enterprise `manifest` raw object | Partial | Add capsule manifest parser and file hash verifier | Replay Console can list deterministic artifacts |
| `capsule_manifest.pass_caveat` | README enterprise non-claims | Partial | Surface caveat prominently | PASS label must always carry caveat |

## The Two Evidence Interfaces Problem

Yes, there are currently two parallel evidence-interface systems.

### Interface 1: Enterprise Dashboard System

Code and docs:

- `apps/dashboard`
- `src/cc/enterprise/aws_reference.py`
- `infra/`
- `infra/lambda/verify_handler.py`
- `README_ENTERPRISE.md`
- `tests/integration/test_enterprise_aws_emulation.py`
- `tests/e2e/test_enterprise_smoke.py`

Core object:

- `cc/enterprise-dashboard-bundle.v1`.

Main concepts:

- Composition risk.
- Assurance case tree.
- Merkle inclusion and consistency proofs.
- Enterprise KMS attestation.
- S3 Object Lock, DynamoDB chain head, verify endpoint.

### Interface 2: Claim-Governance Capsule System

Code and docs:

- `examples/claim_governance_capsule`
- `src/cc/evidence/claim_governance.py`
- `src/cc/evidence/claim_envelope.py`
- `src/cc/evidence/role_ontology.py`
- `src/cc/evidence/decay.py`
- `src/cc/evidence/confirmatory_protocol.py`
- `src/cc/evidence/extremal_scenario.py`
- `docs/design-specs/claim_envelope.md`
- `docs/design-specs/evidence_role_ontology.md`
- `docs/research/CC_REPORTS.md`
- `tests/integration/test_claim_governance_capsule.py`

Core objects:

- `cc.report.v0.3.1`.
- `cc/claim-governance-audit.v1`.
- `cc.claim_envelope.v1`.
- `cc.claim_governance_capsule_manifest.v1`.

Main concepts:

- Evidence-bound claims.
- Typed support roles.
- Support edges and claim fragments.
- Non-claims.
- Decay/freshness.
- Confirmatory firewall.
- Extremal scenario semantics.
- Deterministic replay.

### What Overlaps

- Both bind evidence artifacts by hash.
- Both care about replay/provenance.
- Both expose composition bounds and empirical measurements.
- Both use evidence and assumptions.
- Both have verification concepts.
- Both avoid deployment-safety certification in docs.

### What Differs

- Enterprise dashboard verification is mostly cryptographic/integrity and
  transport oriented.
- Claim governance verification is semantic, typed, and lifecycle oriented.
- Enterprise bundle is dashboard-shaped.
- Claim envelope is claim-shaped.
- Enterprise infra uses AWS KMS and DynamoDB chain-head metadata.
- Capsule intentionally avoids fake signatures and package layers.
- Dashboard UI is evidence/run oriented.
- Capsule artifacts are claim/review/non-claim oriented.

### Connection Diagnosis

Current state: conceptually aligned but technically separate.

They are not fully connected. They are not divergent in philosophy. The risk is
schema drift and duplicated "verification" language.

### Canonical Abstraction

For claim semantics, `ClaimEnvelope` should be canonical.

For enterprise transport and storage, `cc/enterprise-dashboard-bundle.v1` and
the KMS/DynamoDB/S3 infrastructure can remain canonical.

The dashboard should adapt to `ClaimEnvelope`; the envelope should not be
reshaped around the current dashboard.

### Required Bridge

A bridge adapter is needed.

Minimum bridge shape:

- TypeScript types for `ClaimGovernanceAudit`, `ClaimEnvelope`, `CapsuleManifest`,
  and `CcReport`.
- A parser that accepts either an enterprise dashboard bundle or a capsule file
  set.
- A normalized `ClaimObservatoryModel` with:
  - claim identity.
  - proposition.
  - boundary and non-claims.
  - support graph.
  - evidence artifact list.
  - freshness state.
  - governance verdict.
  - receipt/integrity facts.
  - optional enterprise proof facts.
- A Python exporter may also be useful in `src/cc/enterprise/aws_reference.py`
  so enterprise bundles can optionally include `claim_governance` and
  `claim_envelope` sections.

## Crypto / Ledger Infrastructure Map

| Existing mechanism | File/module | Purpose | Current consumers | Could support claim lifecycle ledger? | Required bridge work |
| --- | --- | --- | --- | --- | --- |
| RFC6962-style Merkle transparency log | `src/cc/evidence/merkle_log.py` | Append-only record log; inclusion and consistency proofs over JSON records | `src/cc/core/evidence_bundle.py`, `src/cc/enterprise/aws_reference.py`, tests, dashboard TS equivalent | Yes | Define claim lifecycle event records and decide which claim events are logged |
| Browser-side Merkle proof verifier | `apps/dashboard/lib/merkle.ts` | Client-side verification of enterprise dashboard bundle records | Dashboard Verify view | Yes, for viewing lifecycle proofs | Add model support for claim lifecycle records and distinguish root provenance |
| Ed25519 evidence bundle signing | `src/cc/core/evidence_bundle.py` | Optional local signing of evidence-bundle attestation | `verify_evidence_bundle`, adversarial transparency tests | Yes | Keep optional; do not add to deterministic capsule by default |
| Ed25519 witness anchoring | `src/cc/evidence/anchoring.py` | Independent witness co-signature over Merkle root checkpoint | Evidence bundle when witness configured; adversarial tests | Yes | Use for optional public claim ledger checkpoints, not local capsule replay |
| KMS enterprise signing | `src/cc/enterprise/aws_reference.py`, `infra/lib/cc-enterprise-stack.ts`, `infra/lambda/verify_handler.py` | Sign enterprise bundle checkpoint and verify stored bundle | Enterprise smoke and Lambda reference | Yes | Include claim envelope hash/report hash in signed checkpoint if enterprise claim ledger evolves |
| S3 Object Lock + versioning | `infra/lib/cc-enterprise-stack.ts`, emulated in `aws_reference.py` | Retained evidence bundle storage | Enterprise deployment reference | Yes | Store claim package/envelope artifacts with retention |
| DynamoDB chain-head sequence | `src/cc/enterprise/aws_reference.py`, `infra/lambda/verify_handler.py` | Monotonic sequence for uploaded bundle chain head | Enterprise tests and Lambda | Yes | Add claim id as partition or secondary index; avoid conflating run sequence with claim lifecycle sequence |
| Report canonical receipt | `src/cc/reporting/report.py`, `src/cc/reporting/canonical.py`, `schemas/cc_report.schema.json` | Canonical SHA-256 over report JSON excluding receipt hash | Claim governance verifier, capsule | Yes | Dashboard should verify and display report receipt separately from Merkle root |
| Evidence artifact hash audit | `src/cc/evidence/claim_governance.py` | Recompute evidence file SHA-256 and bytes | `verify_claim_governance` and capsule | Yes | Browser can verify when files are uploaded; backend can verify package directories |
| Local audit hash chain | `src/cc/cartographer/audit.py`, `src/cc/core/logging.py` | Tamper-evident JSONL chain | core logging, evidence bundle ledger | Maybe | Distinguish from RFC6962 log; avoid presenting as public transparency proof |
| Older audit runner Merkle root | `src/cc/core/audit_runner.py` | Minimal demo attestation over result lines with duplicate-last tree | `scripts/verify_attestation.py` | No, not preferred | Prefer `src/cc/evidence/merkle_log.py` for new lifecycle ledger |
| Capsule deterministic manifest | `examples/claim_governance_capsule/build_capsule.py` | Deterministic file/input hash manifest and pass caveat | Capsule tests and proof gate | Yes, for replay package integrity | Keep deterministic and unsigned unless explicitly packaged by a separate compiler |

Answers to required crypto questions:

1. RFC6962-style Merkle log: yes, in `src/cc/evidence/merkle_log.py` and
   mirrored in `apps/dashboard/lib/merkle.ts` and `infra/lambda/verify_handler.py`.
2. External witness anchoring: yes, optional Ed25519 witness anchoring in
   `src/cc/evidence/anchoring.py`.
3. Ed25519 signing: yes, evidence bundle attestations and witnesses.
4. KMS signing in enterprise infra: yes, `RSA_2048` KMS Sign/Verify with
   `RSASSA_PKCS1_V1_5_SHA_256`.
5. Attestation JSON flow: yes, local evidence-bundle attestation, enterprise
   KMS attestation, older audit-run attestation.
6. Dashboard verifies proofs client-side: yes, Merkle inclusion/consistency and
   leaf hashes only.
7. Claim-governance capsule uses any of this: it uses report receipts and file
   hashes, not Merkle/KMS/Ed25519. This appears intentional.
8. Boundary: respect the capsule README. Do not add crypto to the capsule unless
   optional, reused, clearly scoped, and separate from deterministic replay.

## 12-Stage Epistemic Pipeline Mapping

| Stage | Implemented? | Where | Artifact / schema | Tests | Missing | Philosophical ambiguity | Frontend implication |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1. World Event | Partly | `examples/claim_governance_capsule/inputs/failure_matrix.csv`; enterprise prompt fixtures in `aws_reference.py` | Binary failure rows and prompt rows | Capsule integration; enterprise smoke | Real-world sampling process is not modeled | Easy to confuse fixture population with deployment world | Show "evaluation population / fixture", not "world truth" |
| 2. Machine Observation | Partly | `src/cc/core/evidence_bundle.py`; guardrail adapters; capsule matrix | `results.jsonl`, `confirmatory_failure_matrix.json` | Evidence bundle tests; capsule tests | Label provenance and observation uncertainty are thin | Binary outcomes may look objective without label/process context | Evidence Body should show observation source and convention, including "1 means guardrail failure or unsafe pass" |
| 3. Evidence Artifact | Yes | `cc.report.v0.3.1` evidence list; capsule outputs; enterprise bundle | Evidence artifacts with role/path/sha256/bytes | Reporting and capsule tests | Dashboard artifact cards missing | Evidence may be treated as support for whole claim | Artifact cards should show role, hash, bytes, status, and what the role cannot support |
| 4. Cryptographic Receipt | Yes | `cc_report.receipt`; Merkle logs; KMS attestations; capsule manifest | `receipt.canonical_hash`, Merkle proofs, `enterprise_attestation`, manifest hashes | Reporting tests, transparency tests, enterprise tests | Dashboard does not verify report receipt; capsule not in dashboard | Receipt may be mistaken for statistical validity | Verify view should separate byte integrity, Merkle inclusion, root provenance, and governance |
| 5. Typed Evidence Role | Yes | `src/cc/evidence/role_ontology.py` | `EvidenceRoleDefinition`, evidence `role` fields | `tests/unit/evidence/test_role_ontology.py`, claim envelope tests | Dashboard does not expose role permissions | Role label may be read as informal metadata | UI should show "may support" and "must not support" for each role |
| 6. Claim Envelope | Yes | `src/cc/evidence/claim_envelope.py`; capsule output | `cc.claim_envelope.v1` | `tests/unit/evidence/test_claim_envelope.py`, capsule integration | No dashboard parser/view | Envelope validity may be overread as claim truth | Claim Anatomy should use envelope as canonical source |
| 7. Boundary / Non-Claims | Yes | `claim_envelope.boundary`, `cc_report.claim.non_claims`, role ontology mandatory non-claims | `cc.boundary_envelope.v1`; report claim | Claim governance and capsule tests | No Non-Claims Wall in dashboard | Non-claims can be hidden in JSON | Always-visible Non-Claims Wall with missing non-claim diagnostics |
| 8. Decay / Freshness | Yes | `src/cc/evidence/decay.py`; `claim_governance_audit.decay` | `cc.claim_decay.v1`; `fresh/degraded/expired` verifier state | `tests/unit/evidence/test_decay.py`, governance tests | No dashboard freshness UI; no calendar/version UX | Fresh can be mistaken for safe/currently valid | Decay Clock should say "fresh under policy", "degraded requires review", or "expired support" |
| 9. Challenge / Defeater Surface | Partly | `assurance_schema.py`, `ClaimEnvelope.boundary.defeaters`, governance unresolved gaps | Assurance case schema; boundary defeaters | Assurance schema tests | Capsule has no active defeaters; dashboard tree has no challenge workflow | Defeaters may look like optional annotations | Add Challenge Mode showing defeaters, invalidators, unresolved gaps, and review triggers |
| 10. Human Review / Governance | Partly | `ClaimGovernanceAudit.required_human_review`; role ontology human review roles | Governance audit; future `human_review_note` role | Governance and role ontology tests | No review artifact in capsule; no dashboard workbench | `required_human_review=false` may be overread as human approval | Show "review not required by narrow verifier", not "approved" |
| 11. Replayable Capsule | Yes | `examples/claim_governance_capsule/reproduce.sh`; manifest expected comparison | `cc.claim_governance_capsule_manifest.v1` | Capsule integration tests | Dashboard cannot load/replay capsule; no static viewer | Replay can be mistaken for external validity | Replay Console should list inputs, fixed time, seed, commands, expected hashes |
| 12. Public Claim Ledger / Claim Observatory | Not yet | Merkle/KMS/witness substrate exists; no claim lifecycle ledger | Future claim lifecycle records | No direct tests | No canonical claim event schema, no public UI | Ledger can imply authority if not carefully bounded | Claim Ledger should be append-only evidence history, not approval timeline |

## Decay: Mortal Claims Without Fake Precision

### What Exists

`src/cc/evidence/decay.py` implements strict claim-decay artifacts:

- `ClaimDecayRecord` stores policy, issued time, covariates, version watch set,
  evidence refs, notes, and non-claims.
- It deliberately does not store live freshness status.
- `evaluate_claim_decay` computes verification-time state from a verifier's
  clock and optional observed versions.
- TTL thresholds support `fresh`, `degraded`, and `expired`.
- Version watch changes force `expired`.
- A configured hazard policy exists, but its docstrings and serialized
  non-claims say it is a configured heuristic risk score, not a fitted or
  calibrated survival model.

The capsule uses a TTL policy:

- issued at `2026-01-01T00:00:00Z`.
- evaluated at fixed now `2026-01-02T00:00:00Z`.
- degraded after 30 days.
- expires after 60 days.
- audit status: `fresh`.

### Threshold-Based, Continuous, Or Hybrid

Current implementation is hybrid:

- primary UX state is discrete: `fresh`, `degraded`, `expired`;
- TTL thresholds are discrete;
- configured hazard computes a continuous heuristic half-life internally, but
  still maps to discrete degraded/expired states.

### Is Confidence Actually Decaying?

No. The implementation classifies freshness/support state. It does not
decrease a calibrated confidence score for the underlying empirical claim.

That is good. A continuous confidence decay display would be easy to overread
as a calibrated probability that the claim remains true.

### Does The Implementation Match A Mathematical Formula?

For TTL policies, yes: age in days is compared with degrade/expire thresholds.

For configured hazard policies, the formula is:

```text
lambda = baseline_hazard_per_day * exp(beta dot x)
half_life_days = ln(2) / lambda
```

But the code labels this as a configured heuristic, not a calibrated survival
model.

### What Is Mathematically Defensible

Defensible now:

- discrete freshness states;
- version-triggered invalidation;
- review pressure when degraded;
- explicit non-claims;
- optional heuristic hazard as a policy prior when it carries visible caveats.

Potentially defensible later:

- survival or hazard models only if trained and validated on appropriate claim
  failure histories;
- claim-level-dependent policies;
- reviewer-calibrated risk pressure, presented as review priority rather than
  truth probability.

### What Would Overclaim

Avoid:

- "confidence decays from 91 percent to 74 percent" unless the repo has a
  calibrated model and validation evidence.
- "fresh means safe".
- "expired means false".
- "hazard score means probability of claim failure".
- hiding the configured-heuristic caveat behind a visual gauge.

### What The Frontend Should Show

Show:

- state: `fresh`, `degraded`, or `expired`;
- policy id;
- issued at;
- evaluated at;
- age days;
- degrade/expire thresholds;
- version changes if any;
- non-claims;
- "signed policy, live state computed at verification time."

Do not show:

- a confidence score;
- a safety score;
- a deployment status.

### Future Research

Future research could explore calibrated survival models for evidence
freshness, but only with:

- observed claim-lifecycle event data;
- clear censoring assumptions;
- separate validation;
- model uncertainty;
- claim-level stratification;
- UI that frames output as review pressure, not truth.

## Frontend Architecture Options

| Option | Scope | Speed | Risk | Test burden | Philosophical fit | Dependency burden | Demo value | Maintainability | Enterprise compatibility | Capsule compatibility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A. Extend existing `apps/dashboard` | Add claim-governance types, adapter, and views directly to dashboard | Medium | Medium | Medium: Vitest plus e2e fixture | Strong if `ClaimEnvelope` is canonical | Low: existing Next app | High for unified product | Good: one app | Strong | Strong after adapter |
| B. Standalone static capsule viewer | Add `examples/claim_governance_capsule/viewer` as zero-build HTML/CSS/JS | Fast | Medium: third interface drift | Low to medium | Good only if explicitly tiny and read-only | Lowest | High for offline paper demos | Risky if it grows | Weak | Strong |
| C. Hybrid | Tiny static viewer only if needed; primary bridge into dashboard; later Claim Observatory | Medium | Lowest long-term | Medium to high in stages | Strongest | Low to moderate | High | Best | Strongest | Strongest |

Recommendation: Option C.

Do not create a third disconnected frontend as the main path. Build the bridge
into `apps/dashboard`. If an offline capsule viewer is needed, make it a small
static inspection aid that consumes the same normalized model and carries the
PASS caveat.

## Recommended Implementation Path

### First PR: Dashboard Capsule Adapter, Read-Only

Goal: let the dashboard load the deterministic capsule outputs without changing
schemas.

Files likely touched:

- `apps/dashboard/lib/types.ts`
- new `apps/dashboard/lib/claimGovernance.ts`
- new `apps/dashboard/lib/claimObservatoryModel.ts`
- `apps/dashboard/components/DashboardShell.tsx`
- `apps/dashboard/components/views/VerifyView.tsx`
- new tests under `apps/dashboard/tests`

Work:

1. Add TS types for `CcReport`, `ClaimGovernanceAudit`, `ClaimEnvelope`,
   `CapsuleManifest`.
2. Add a loader that accepts:
   - existing single enterprise bundle JSON;
   - a selected capsule manifest plus related JSON files, or a synthetic
     combined fixture for tests.
3. Produce a normalized read-only `ClaimObservatoryModel`.
4. Add a `Claim` or `Governance` tab with:
   - claim statement;
   - claim level;
   - governance verdict;
   - freshness status;
   - required human review;
   - PASS caveat;
   - support summary.
5. Keep existing enterprise tabs working.
6. Add Vitest fixtures using
   `examples/claim_governance_capsule/outputs/claim_envelope.json`,
   `claim_governance_audit.json`, `cc_report.json`, and
   `capsule_manifest.json`.

Acceptance:

- Existing dashboard smoke still passes.
- New adapter test loads capsule outputs.
- UI labels say `PASS under verifier rules`, not `safe`.
- No generated capsule artifact is edited.

### Second PR: Evidence Body And Non-Claims Wall

Work:

- Artifact cards grouped by role.
- Role permission display: may support / must not support.
- Non-Claims Wall sourced from envelope boundary and audit.
- Missing mandatory non-claims diagnostics.
- Receipt integrity card.

Tests:

- Role cards render forbidden support.
- Non-claims render and do not collapse behind generic JSON.

### Third PR: Decay Clock And Support Graph

Work:

- Decay Clock component.
- Support graph table or compact graph view.
- Confirmatory protocol checklist.
- Scenario endpoint cards.

Tests:

- `fresh`, `degraded`, `expired` states render with correct language.
- Confirmatory checks render as scoped protocol checks.
- Support edges render relation and strength without scalar score.

### Fourth PR: Enterprise Claim Bundle Extension

Work:

- Extend `src/cc/enterprise/aws_reference.py` export to optionally include
  `claim_governance_audit` and `claim_envelope` sections when source artifacts
  exist or a report is supplied.
- Extend enterprise signature checkpoint only after deciding exact signed
  fields.

Tests:

- Enterprise smoke still passes.
- Optional claim sections do not break old bundles.
- Dashboard can load enriched enterprise bundle.

## 2100 Claim Observatory Vision

Design law: the UI must make false confidence uncomfortable.

This should not become a normal SaaS dashboard. It should become a claim
observatory: a place where claims are watched, bounded, challenged, replayed,
and retired.

| Screen | Existing source artifacts | Missing adapter code | Existing dashboard component to reuse | Required new component | Dangerous labels to avoid | Required tests |
| --- | --- | --- | --- | --- | --- | --- |
| Claim Ledger | `capsule_manifest.json`, future Merkle lifecycle records, enterprise chain head | Claim event model; lifecycle event schema | Verify status grid style | `ClaimLedgerView` | approved, certified, trusted | events ordered; caveat visible; no safety status |
| Claim Anatomy | `claim_envelope.identity`, `proposition.fragments`, `cc_report.claim` | `ClaimEnvelope` TS parser | `DashboardShell` tabs | `ClaimAnatomyView` | safe claim, certified claim | statement, level, fragments render |
| Evidence Body | `cc_report.evidence`, `claim_envelope.support_graph.*_refs`, audit artifact statuses | artifact normalizer | Assurance tree node styling | `EvidenceBodyView`, `ArtifactCard` | evidence implies safety | role, hash, bytes, status render |
| Support Graph | `claim_envelope.support_graph.support_edges` | support-edge graph model | tree layout concepts | `SupportGraphView` | proof of safety | edges render relation/strength/non-claims |
| Non-Claims Wall | `claim_envelope.boundary.non_claims`, `audit.non_claims` | non-claim merger and source labels | panel layout | `NonClaimsWall` | limitations hidden in tooltip | all non-claims visible/searchable |
| Decay Clock | `decay_policy.json`, `claim_governance_audit.decay` | decay status adapter | status cards | `DecayClock` | currently safe, valid deployment | states render caveats; version triggers render |
| Challenge Mode | `boundary.defeaters`, `invalidation_conditions`, `scenarios.excluded_evidence_fields`, confirmatory review reasons | challenge model | Assurance defeater rendering | `ChallengeSurface` | resolved means safe | active/unresolved/invalidating triggers render |
| Replay Console | `reproduce.sh`, capsule manifest inputs/files, fixed time/seed | replay metadata parser | Verify upload pattern | `ReplayConsole` | reproducible means representative | command, fixed now, seed, hashes render |
| Human Review Workbench | `required_human_review`, role ontology human-review roles, future review artifacts | review artifact types | Assurance case review status text | `ReviewWorkbench` | approved for deployment | review required/not required/satisfied labels tested |
| Public Claim Receipt Page | `cc_report.receipt`, `claim_envelope.identity`, `capsule_manifest.report_receipt_sha256` | receipt verifier model | Verify root/hash display | `ClaimReceiptPage` | verified safe | receipt hash separate from claim verdict |
| Claim Lifecycle Ledger | Merkle log, KMS chain head, witness anchors, future claim events | claim lifecycle event schema | VerifyView proof status | `LifecycleLedgerView` | compliance-certified | inclusion/consistency/root provenance labels tested |

Allowed labels:

- `PASS under verifier rules`
- `evidence-bound`
- `bounded empirical`
- `internally consistent`
- `requires review`
- `failed governance`
- `non-claim`
- `not a deployment-safety proof`
- `replayable`
- `fresh`
- `degraded`
- `expired`

Banned labels:

- `safe`
- `certified safe`
- `trusted`
- `approved for deployment`
- `deployment safe`
- `compliance-certified`
- `guaranteed`
- `verified safe`

## Concrete PR Plan

### PR 1: Claim Observatory Adapter In Dashboard

Prompt for next Codex run:

```text
Implement the first Claim Observatory bridge in apps/dashboard.

Read docs/frontend_claim_observatory_recon.md, apps/dashboard/lib/types.ts,
apps/dashboard/components/DashboardShell.tsx, apps/dashboard/components/views/*,
examples/claim_governance_capsule/outputs/claim_envelope.json,
examples/claim_governance_capsule/outputs/claim_governance_audit.json,
examples/claim_governance_capsule/outputs/cc_report.json, and
examples/claim_governance_capsule/outputs/capsule_manifest.json.

Add TypeScript types and a pure adapter that converts the capsule artifacts into
a normalized ClaimObservatoryModel. Add a read-only Governance tab to the
existing dashboard showing claim statement, claim level, PASS under verifier
rules, required human review, freshness status, PASS caveat, and support
summary counts. Keep the existing enterprise bundle upload flow working.

Tests:
- existing apps/dashboard smoke still passes with ENTERPRISE_BUNDLE_PATH.
- new Vitest unit test imports capsule output JSON fixtures and verifies the
  adapter model.
- new UI test renders the Governance tab from capsule fixtures and confirms no
  banned labels appear.

Do not edit generated capsule outputs, expected artifacts, schemas, or mkdocs
nav. Use existing CSS patterns and lucide icons.
```

### PR 2: Non-Claims And Evidence Body

- Add `NonClaimsWall`.
- Add `EvidenceBodyView`.
- Show role, source, hash, bytes, status, support permissions, and forbidden
  support.
- Add tests for role boundaries and non-claim visibility.

### PR 3: Decay Clock And Support Graph

- Add `DecayClock`.
- Add support edge table/graph.
- Add confirmatory protocol checklist and scenario cards.
- Add degraded/expired fixture tests.

### PR 4: Enterprise Enrichment

- Add optional claim-governance sections to enterprise export.
- Keep old `EnterpriseBundle` compatible.
- Consider signing report/envelope hash in enterprise KMS checkpoint only after
  bridge semantics are stable.

## Open Questions

1. Should the dashboard load capsule artifacts as multiple uploaded files, a
   directory picker, or a single combined package JSON?
2. Should a `ClaimObservatoryModel` live only in TypeScript, or should Python
   emit a stable view model for dashboard consumption?
3. Should `src/cc/enterprise/aws_reference.py` become the first bridge producer,
   or should `apps/dashboard` parse raw capsule artifacts first?
4. What is the future claim lifecycle event schema for public ledgers?
5. Should report receipt verification happen in the browser for uploaded file
   sets, or only in Python/backend?
6. How should root provenance be communicated when a Merkle root is supplied by
   the same uploaded bundle?
7. What human review artifact should be introduced before building an editable
   workbench?
8. Should `calibration.json` in the capsule get a schema field in a future
   version, or remain report-bound calibration metadata?
9. Should the older `src/cc/core/audit_runner.py` Merkle helper be marked legacy
   in docs to avoid confusion with RFC6962 logs?

## No-Overclaim UI Language Rules

1. Use `PASS under verifier rules`, never standalone `PASS` where users could
   read it as safety approval.
2. Pair every governance pass with: "not a deployment-safety proof."
3. Distinguish:
   - receipt integrity;
   - artifact hash verification;
   - Merkle inclusion;
   - root provenance;
   - governance verdict;
   - human review.
4. Never collapse evidence into a scalar safety score.
5. Show non-claims in the main surface, not in a tooltip.
6. Show evidence role permissions and forbidden support.
7. Label decay as freshness/review state, not truth probability.
8. Label human review as scoped review state, not approval.
9. Label confirmatory protocol checks as scoped protocol/run-separation checks,
   not external validity.
10. When a proof verifies, name the proof: "Merkle inclusion proofs verified",
    "report receipt hash verified", or "artifact hashes verified."

## Final Diagnosis

This repo is not missing a frontend. It is missing a bridge between the
enterprise evidence viewer and the claim-governance semantic witness.

The dashboard should not become a safety dashboard. It should become a claim
observatory: a place where every claim has a body, a boundary, a clock, a
challenge surface, a receipt, and a visible wall of things it does not claim.
