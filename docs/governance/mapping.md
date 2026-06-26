# Assurance Schema Governance Mapping

This document maps the GSN-inspired assurance-case schema in
`src/cc/evidence/assurance_schema.py` to common AI governance documentation
categories.

**Disclaimer:** This mapping is informational only. It is not legal,
regulatory, audit, or compliance advice. It does not certify conformance with
NIST AI RMF, ISO/IEC 42001, or any other standard. A qualified reviewer must
determine what obligations apply to a specific organization, system, and use
case.

## Reference Frames

- **NIST AI RMF 1.0:** The schema is mapped to the AI RMF Core functions:
  Govern, Map, Measure, and Manage.
- **ISO/IEC 42001-style AI management system:** The schema is mapped to common
  management-system documentation areas: organizational context, leadership and
  accountability, planning and risk treatment, support and documented
  information, operational controls, performance evaluation, and improvement.

## Claim Category Crosswalk

| Schema category | What it means in the assurance case | NIST AI RMF connection | ISO/IEC 42001-style documentation connection |
| --- | --- | --- | --- |
| `system_safety_argument` | Top-level claim that the run evidence forms a reviewable assurance argument for the stated context. It remains `NEEDS HUMAN REVIEW` by default. | Govern for accountability and policy context; Map for intended use and risk context; Manage for risk-response decisions. | Context of the organization; leadership/accountability; planning; performance evaluation; improvement. |
| `composition_risk_bounded` | Sub-claim supported by Frechet-Hoeffding bounds or envelope outputs showing feasible composition-risk ranges. | Measure for quantitative evaluation; Map for risk identification and system context; Manage for deciding whether bounded risk is acceptable. | Planning and risk treatment; operational evaluation controls; performance evaluation records. |
| `dependence_structure_characterized` | Sub-claim supported by cliff certificates, tail-dependence outputs, and CCF/FH consistency checks. | Map for dependence and failure-mode characterization; Measure for diagnostic evidence; Manage for selecting mitigations or additional monitoring. | Risk assessment and operational controls; documented assumptions; monitoring and performance evaluation. |
| `uncertainty_honestly_quantified` | Sub-claim supported by coverage simulations, confidence intervals, tolerance checks, or related uncertainty outputs. | Measure for uncertainty and validation evidence; Govern for transparent documentation; Manage for accepting, rejecting, or improving a control. | Support/documented information; performance evaluation; corrective action and improvement. |

## Node-Type Crosswalk

| Schema node | Governance role | NIST AI RMF connection | ISO/IEC 42001-style connection |
| --- | --- | --- | --- |
| `TopClaim` | A review scaffold for the overall assurance position. The generated top claim is never auto-asserted as true. | Govern, Map, Manage. | Context, leadership, planning, performance evaluation. |
| `SubClaim` | A decomposed assurance position tied to one evidence category. | Map, Measure, Manage. | Operational controls, risk treatment, performance evaluation. |
| `Strategy` | Explains how a claim is decomposed and why the evidence is relevant. | Govern for accountability and traceability. | Planning, documented information, operational planning. |
| `Evidence` | Machine-harvested run output such as FH bounds, cliff certificates, CCF checks, or coverage results. | Measure, with Manage when used in risk decisions. | Performance evaluation records and operational evidence. |
| `Assumption` | Premise that must be reviewed before a claim is relied upon. | Govern and Map. | Planning assumptions, risk criteria, documented information. |
| `Context` | Scope information such as run ID, composition mode, seed, or prompt count. | Map and Govern. | Organizational/system context and documented information. |
| `Defeater` | Explicit challenge that could invalidate or weaken a claim. Empty defeater lists require a written justification. | Manage for risk response; Govern for accountability and review discipline. | Nonconformity/corrective action, improvement, and risk-treatment records. |

## Evidence Bundle Mapping

The auto-population function creates a draft tree from actual run outputs:

- FH bounds and envelope fields support `composition_risk_bounded`.
- Cliff certificates and tail-dependence fields support
  `dependence_structure_characterized`.
- CCF/FH consistency-check fields also support
  `dependence_structure_characterized`.
- Coverage, interval, bootstrap, Wilson/Newcombe, BCA, and tolerance fields
  support `uncertainty_honestly_quantified`.

The function does not decide whether any of those outputs are adequate. Claims,
assumptions, and defeaters default to `NEEDS HUMAN REVIEW`; evidence defaults to
`AUTO-POPULATED` while still requiring review for relevance and sufficiency.

## Review Boundary

Before using an exported assurance case in governance documentation, reviewers
should verify at least:

- the run manifest, versions, datasets, seeds, and transparency-log evidence;
- the intended-use scope and deployment context;
- the harm taxonomy and acceptance thresholds;
- whether missing evidence is justified or must be regenerated;
- whether each defeater has been resolved, accepted, or escalated;
- whether legal, regulatory, customer, or organizational requirements add
  documentation obligations beyond this schema.

Passing this checklist still does not certify conformance. It only improves the
traceability of the evidence-to-claim argument.

## References

- NIST AI Risk Management Framework, AI RMF 1.0 and AI RMF Core:
  <https://www.nist.gov/itl/ai-risk-management-framework>
- NIST AI RMF Knowledge Base:
  <https://airc.nist.gov/AI_RMF_Knowledge_Base/AI_RMF>
- ISO/IEC 42001:2023 overview:
  <https://www.iso.org/standard/81230.html>
