# CC-Framework: Three-Minute Architectural Demo

## Objective

Explain CC-Framework as a prototype for evidence-bound AI assurance reporting,
using three repository artifacts that show dependence-aware bounds, disclosure
controls, and auditability.

## Move 1: The Statistical Illusion

**Open:** `README.md`, 60-second quickstart.

**Say:**

Most AI systems stack multiple guardrails. A common mistake is to assume
independent failures. If two guardrails each fail 10% of the time, an
independence baseline gives a 1% joint failure estimate.

CC-Framework shows why that can be misleading. Without a justified independence
assumption, the Frechet-Hoeffding bounds say the joint failure rate may be
anywhere from 0% to 10%. The point is not that the worst case is always true.
The point is that the narrower 1% claim requires an assumption, and the
framework is designed to make that assumption visible.

**Do not claim:** This proves the system is unsafe or safe.

## Move 2: Disclosure Controls And Non-Claims

**Open:** `docs/validation_matrix.md`.

**Say:**

The validation matrix separates what each lane checks from what it does not
claim. For example, evidence-governance checks can preserve provenance and
tamper-evident records, but they do not prove statistical validity,
compliance, or deployment safety.

That is the disclosure-control idea: a PASS should not silently become a
broader public claim.

**Do not claim:** The framework prevents all institutional overclaiming. It
makes report boundaries explicit and reviewable.

## Move 3: Claim Audit Trail

**Open:** `docs/theory/theorem_ledger.md`.

**Say:**

The theorem ledger links claims to assumptions, implementation files, tests,
artifacts, and failure modes. This is the research equivalent of an audit
trail: every public claim should be traceable to the evidence and assumptions
that support it.

This is why the project may be relevant to accounting and assurance. The core
issue is not only whether an AI system passed a test; it is whether the
organization's public claim remains faithful to the evidence.

**Do not claim:** The theorem ledger proves empirical truth. It documents and
validates scoped mathematical and software claims under stated assumptions.

## Closing Line

CC-Framework is best understood as a prototype disclosure-control layer for AI
assurance reports: it helps prevent narrow evidence from being converted into
broad safety narratives.
