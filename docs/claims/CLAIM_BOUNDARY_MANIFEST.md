# Claim Boundary Manifest

## Purpose

This manifest maps public CC-Framework claims to evidence, validation lanes,
supporting files, tests, and non-claims. It exists to prevent documentation,
demos, receipts, dashboards, or enterprise references from being interpreted
as broader claims than the repository supports.

CC-Framework is not claim avoidance; it is claim discipline.

The manifest is the repository's truth table for public communication. It
does not replace the theorem ledger, validation matrix, release checklist, or
non-claim documents. It connects them so that public language can be checked
against the evidence and assumptions that support it.

## Claim Boundary Levels

- **C0 — Repository Metadata Claim**
  Basic facts about files, docs, package metadata, or repository organization.

- **C1 — Structural Validation Claim**
  A schema, report, or artifact conforms to declared validation rules.

- **C2 — Provenance / Integrity Claim**
  Receipt-covered artifacts can be checked for byte integrity or tamper
  evidence under canonicalization and digest/signature verification.

- **C3 — Mathematical Kernel Claim**
  A mathematical statement about finite binary atom spaces,
  Fréchet-Hoeffding bounds, identified intervals, endpoint witnesses,
  finite-sample outer intervals, or infeasibility detection under stated
  assumptions.

- **C4 — Evidence-Governance Claim**
  A claim about how evidence roles, non-claims, freshness,
  confirmatory/exploratory boundaries, or governance checks are represented
  and validated.

- **C5 — Deployment / Compliance Claim**
  Out of scope for this repository unless a future release explicitly creates
  a separate authority layer. Current status: deferred / not claimed.

## Active Claims

| ID | Claim | Level | Lane | Status | Support | Tests or commands | Authoritative files | Non-claim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `kernel.frechet_bounds` | The strict kernel computes classical Fréchet-Hoeffding bounds for finite binary guardrail failure composition under exact singleton marginals and no additional side constraints. | C3 | Paper Core v0.3 | Active release-candidate boundary | README quickstart, theorem ledger T2, kernel tests. | `tests/unit/kernel/test_classical_frechet_special_cases.py`; `make test-kernel` | `README.md`; `docs/theory/theorem_ledger.md`; `src/cc/kernel/frechet_classes.py`; `src/cc/kernel/sensitivity.py` | Does not assume independence, does not prove deployment safety, does not prove marginals are representative. |
| `kernel.identified_interval` | The finite atom LP computes sharp lower and upper values for declared linear composition queries over feasible finite binary atom distributions. | C3 | Paper Core v0.3 | Active release-candidate boundary | Theorem ledger T1, sensitivity implementation, tests. | `tests/unit/kernel/test_sensitivity.py`; `tests/unit/kernel/test_monotonic_tightening.py`; `make test-kernel` | `docs/theory/theorem_ledger.md`; `src/cc/kernel/sensitivity.py`; `docs/api.md` | Does not validate upstream data collection, semantic safety, or representativeness. |
| `kernel.finite_sample_outer_interval` | Under stated iid Bernoulli sampling assumptions and simultaneous moment coverage, count-derived LP intervals form an outer confidence interval for the target query. | C3 | Paper Core v0.3 | Active release-candidate boundary | `finite_sample_identification.md`, theorem ledger T6, sample complexity tests. | `tests/unit/kernel/test_finite_sample_constraints.py`; `tests/unit/kernel/test_sample_complexity.py`; `make test-kernel` | `docs/theory/finite_sample_identification.md`; `docs/theory/theorem_ledger.md`; `src/cc/kernel/sample_complexity.py`; `src/cc/kernel/sensitivity.py` | Not valid after uncorrected adaptive target selection; not a deployment certificate. |
| `reporting.receipt_integrity` | Receipt/hash verification can provide tamper evidence for recorded bytes under canonical serialization and verification rules. | C2 | Evidence Governance | Active evidence-governance boundary | Theorem ledger T7, validation matrix evidence governance lane, reporting/evidence tests. | `tests/unit/reporting/test_reporting.py`; `tests/unit/evidence/test_transparency_log_adversarial.py`; `PYTHONPATH=src .venv/bin/python -m pytest -q tests/unit/evidence tests/unit/reporting` | `docs/theory/theorem_ledger.md`; `docs/validation_matrix.md`; `src/cc/reporting/canonical.py`; `src/cc/reporting/report.py`; `src/cc/evidence/merkle_log.py`; `src/cc/evidence/anchoring.py` | Does not prove statistical validity, source data truth, representativeness, compliance, or safety. |
| `evidence.non_claim_boundaries` | Documentation and reporting surfaces explicitly separate what each validation result supports from what it does not support. | C4 | Evidence Governance / Shared Docs | Active documentation and governance boundary | Validation matrix, theorem ledger, README What This Is / What This Is Not, disclosure-controls brief. | `.venv/bin/mkdocs build --strict`; `PYTHONPATH=src .venv/bin/python -m pytest -q tests/unit/evidence`; `PYTHONPATH=src .venv/bin/python -m pytest -q tests/integration/test_claim_governance_capsule.py` | `README.md`; `docs/validation_matrix.md`; `docs/theory/theorem_ledger.md`; `docs/briefs/rick_mergenthaler_ai_safety_disclosure_controls.md`; `docs/briefs/three_minute_demo.md`; `docs/research/NON_CLAIMS.md` | Does not prevent bad-faith actors from making misleading claims outside the framework. |
| `release.paper_core_v0_3_rc1` | Paper Core v0.3 is release-candidate quality for the finite atom kernel, canonical metrics, endpoint witnesses, deterministic paper artifacts, and documentation spine. | C0/C3 | Paper Core v0.3 | Active release-candidate boundary | `V0_3_RC1_CHECKLIST.md` validation table. | `make test-kernel`; `make test-release`; `make docs`; `make paper-smoke`; `make verify-paper-artifacts` | `docs/release/V0_3_RC1_CHECKLIST.md`; `docs/validation_matrix.md`; `docs/api.md`; `docs/research/PAPER_CORE.md`; `docs/research/NON_CLAIMS.md` | Does not promote enterprise, dashboard, vendor, cloud, or experimental lanes into paper core. |
| `enterprise.reference_v0_1` | Enterprise Reference v0.1 is an experimental evidence-integrity reference architecture with moto-backed smoke tests. | C0/C2 | Enterprise Reference v0.1 | Experimental reference boundary | Validation matrix, release checklist, enterprise-smoke lane. | `make enterprise-smoke`; `PYTHONPATH=src .venv/bin/pytest tests/integration/test_enterprise_aws_emulation.py tests/e2e/test_enterprise_smoke.py -q` | `docs/validation_matrix.md`; `docs/release/V0_3_RC1_CHECKLIST.md`; `src/cc/enterprise/aws_reference.py`; `infra/lambda/verify_handler.py`; `apps/dashboard/scripts/require-enterprise-bundle.mjs` | Not enterprise-ready, not live-AWS proof, not compliance certification, not deployment safety. |
| `open_core.strategy` | The public research core is intended to remain inspectable for credibility and reproducibility, while possible private/commercial layers may operationalize the primitives. | C0 | Product Strategy / Docs | Active strategy boundary | `OPEN_CORE_STRATEGY.md`. | `.venv/bin/mkdocs build --strict` | `docs/product/OPEN_CORE_STRATEGY.md`; `README.md` | Does not make the project a product today; does not claim enterprise readiness. |

## Forbidden Public Upgrades

- Governance PASS → deployment safety.
- Receipt/hash integrity → statistical validity.
- JSON schema validity → semantic truth.
- Human review → stronger empirical evidence.
- Exploratory red-team discovery → confirmatory proof.
- Product coupling baseline → true independence model.
- Dashboard view → certified assurance result.
- Enterprise smoke test → production readiness.
- Paper Core release candidate → whole-repo production quality.
- Open-core strategy → existing commercial product.

## Review Rule

Any README, paper, demo, release note, website copy, or product-facing document
should be checked against this manifest before publication.
