# Prompt 5 - Deterministic Claim Governance Capsule

Use this as a Codex execution prompt.

```text
You are working in the cc-framework repository. Implement Prompt 5:
Deterministic Claim Governance Capsule as a reproducible end-to-end evidence
package, not a hand-edited demo.

Start by reading:
- docs/research/CLAIM_GOVERNANCE_OS.md
- docs/research/CC_REPORTS.md
- docs/reproducibility.md
- examples/minimal/README.md
- examples/reporting/minimal_cc_report.json
- src/cc/reporting/cli.py
- src/cc/evidence/claim_governance.py
- tests/e2e/ and tests/fixtures/reporting/

Objective:
Create a deterministic claim-governance capsule that regenerates a complete
evidence-bound claim package from raw toy inputs under fixed time, fixed seed,
stable ordering, and deterministic output paths.

Required directory:
examples/claim_governance_capsule/
  README.md
  reproduce.sh
  inputs/
  expected/
  outputs/
  manifest.expected.json

The reproduce script should run the full chain that exists in the current repo:
data -> bounds -> scenarios -> decay -> report -> audit -> state/envelope if
available -> package if available.

If later prompt layers are not implemented yet, the capsule should degrade
honestly:
- include TODO or staged sections only in docs,
- do not emit fake envelope/package artifacts,
- make the script verify the strongest implemented chain available.

Determinism requirements:
- fixed --now timestamp
- fixed random seed
- sorted JSON keys and stable serialization
- stable temp/output paths
- no hand-edited generated outputs
- deterministic manifest comparison
- expected hashes where stable and meaningful

Non-negotiable semantics:
- A reproducible capsule is not a safety proof.
- A PASS governance audit means internal consistency under verifier rules.
- The capsule must include a PASS caveat in README.md.
- Do not create a polished demo that cannot be regenerated.

Implementation requirements:
- Build the capsule from minimal checked-in inputs.
- Make reproduce.sh executable.
- Write outputs only under examples/claim_governance_capsule/outputs/.
- Put expected immutable artifacts under expected/.
- Include manifest.expected.json and a generated manifest in outputs.
- Add tests that run the capsule or a lightweight equivalent in CI.
- Avoid broad refactors. Prefer existing CLIs and helpers.

Tests:
- reproduce.sh exits 0.
- A second run produces the same deterministic manifest under fixed --now.
- Governance audit verdict is the expected verdict.
- Mutating one bound artifact or evidence artifact changes audit to FAIL or
  causes the deterministic manifest check to fail.
- README includes the PASS caveat and non-claims.

Docs:
- README.md should explain what the capsule proves and does not prove.
- Include exact commands.
- Include expected outputs and how to re-run verification.

Acceptance checklist:
- Capsule can be regenerated from inputs.
- Generated artifacts are not manually edited.
- Deterministic manifest comparison is tested.
- Existing tests still pass.
- The final response reports files changed, commands run, and remaining risks.
```

