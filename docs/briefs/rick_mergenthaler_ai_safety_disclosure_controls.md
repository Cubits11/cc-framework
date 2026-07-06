# AI Safety Disclosure Controls: CC-Framework Brief

## Core Question

Can AI safety assurance borrow from accounting disclosure logic and internal
control frameworks to reduce the risk that organizations overstate what their
evaluation evidence supports?

## The Problem: AI Safety-Washing As A Disclosure Failure

AI safety reporting can blur distinct evidentiary and operational states:

- **Evidence integrity**: a log, report, receipt, or bundle has not been
  altered under the stated verification procedure.
- **Statistical validity**: an evaluation design supports the inference being
  made for the named population, metric, and time period.
- **Human review**: a reviewer completed a procedural, expert judgment, or
  governance step.
- **Governance PASS**: a report satisfied internal validation rules for its
  declared claim level and evidence roles.
- **Deployment safety**: a real-world system is acceptably safe to operate in
  a specific use context.

CC-Framework treats these as separate categories. Its purpose is not to certify
deployment safety. Its narrower purpose is to reduce the chance that
procedural checks, receipt integrity, or narrow evaluation results are
presented as broad safety guarantees.

## The Prototype: CC-Framework As A Disclosure-Control Substrate

CC-Framework is an open-source research prototype for evidence-bound AI
assurance reporting. It supports a narrow technical claim: composed guardrail
failures should be evaluated under explicit dependence assumptions rather than
silently treated as independent.

The framework currently emphasizes:

- **Dependence-aware risk bounding**: Frechet-Hoeffding and finite atom LP
  methods bound composed guardrail failure probabilities under declared
  assumptions.
- **Evidence-bound reporting**: public claims must remain tied to assumptions,
  evidence roles, validation lanes, and non-claims.
- **Receipt-backed provenance**: canonical serialization, hashes, signatures,
  and Merkle-style logs support byte integrity and provenance, not empirical
  truth or safety.
- **Mandatory non-claims**: documentation and reporting surfaces state what a
  validation result does not prove.
- **Confirmatory/exploratory separation**: exploratory red-team or research
  surfaces are not automatically promoted into paper-core or confirmatory
  claims.
- **Freshness and decay semantics**: evidence may expire or require review when
  time, context, or assumptions change.

## Accounting And Assurance Analogy

| Financial Reporting / Accounting | CC-Framework / AI Assurance |
| --- | --- |
| Financial statements | Public AI assurance reports |
| GAAP and standards design | Claim grammar, schemas, and validation lanes |
| Audit evidence | Evidence artifacts and receipts |
| Internal Control Over Financial Reporting (ICFR) | Documentation and validation controls around AI assurance reports |
| Footnotes and risk factors | Mandatory non-claims and caveats |
| Pro forma adjustments or selective disclosure | Cherry-picked evaluations or overbroad safety narratives |
| Subsequent events and stale disclosures | Freshness, decay, and revalidation boundaries |
| Audit trail | Theorem ledger, tests, artifacts, and provenance records |

The analogy is useful because it shifts attention from a single pass/fail
result to the quality of the claim being made from that result. A control can
expose and structure claim boundaries. It cannot force institutional honesty,
and it cannot prevent a bad-faith actor from making misleading statements
outside the framework.

## Targeted Mentorship Questions

1. In standards design, when do strict rules improve disclosure quality, and
   when do they become box-checking?
2. Do mandatory caveats meaningfully constrain overstatement, or do
   organizations neutralize them with boilerplate?
3. What makes an internal evidence artifact credible to an external auditor,
   regulator, buyer, or investor?
4. How does litigation or liability risk shape what organizations disclose
   about known uncertainty?
5. Where does the analogy between AI safety reporting and financial disclosure
   break down?
6. What would make this framing look naive to an accounting or assurance
   scholar?

## Explicit Non-Claims

CC-Framework does not prove that an AI system is safe.
It does not certify deployment readiness.
It does not prove regulatory compliance.
It does not prove dataset representativeness.
It does not turn cryptographic integrity into statistical validity.
It does not prevent bad-faith actors from making misleading statements outside
the framework.

The narrower goal is to make claim boundaries explicit, reproducible, and
harder to erase accidentally.
