# Prompt 10 - Claim Compiler

Use this as a Codex execution prompt.

```text
You are working in the cc-framework repository. Implement Prompt 10:
Claim Compiler as a portable evidence-bound claim package assembler.

Start by reading:
- docs/research/CLAIM_GOVERNANCE_OS.md
- docs/research/CC_REPORTS.md
- docs/reproducibility.md
- docs/research/prompt-roadmap/02_claim_envelope_boundary_envelope.md if present
- docs/research/prompt-roadmap/08_receipt_verification_upgrade.md if present
- docs/research/prompt-roadmap/09_human_review_artifact.md if present
- src/cc/evidence/claim_governance.py
- src/cc/reporting/cli.py
- existing package, manifest, and e2e tests

Objective:
Build claim_compiler.py as the first claim compiler. It should assemble a
portable package that preserves report bytes, evidence artifacts, audits,
lifecycle/envelope/review artifacts when present, support relations,
invalidation conditions, and non-claims.

Required package shape:
claim_package/
  manifest.json
  report.json
  evidence/
  audits/
  lifecycle/
  envelope/
  reviews/
  README.md

Core manifest:
ClaimPackageManifest(
    package_id,
    subject_report,
    artifacts,
    support_edges,
    verifier_result,
    lifecycle_state,
    human_review_status,
    non_claims,
    reproducibility,
    integrity_checks,
)

Non-negotiable semantics:
- Copying files is not compilation.
- A package is meaningful only if the manifest preserves support relations,
  invalidation conditions, review state, and non-claims.
- The compiler must run verification or consume a fresh verifier result.
- The package must be re-verifiable from the package directory.
- A packaged PASS does not mean deployment safety.

Implementation requirements:
- Add src/cc/evidence/claim_compiler.py or another clearly appropriate module.
- Add a CLI command if it fits existing cc-report CLI patterns, likely
  cc-report compile-claim-package.
- Keep source artifacts unchanged.
- Use deterministic paths, sorted manifests, and fixed --now support for tests.
- Include README.md in the generated package with PASS caveat and non-claims.
- Preserve support edges if ClaimEnvelope exists. If not, preserve a conservative
  support summary and stage a TODO in docs.

Tests:
- Package fails on hash mismatch.
- Source artifacts are unchanged.
- Manifest is deterministic under fixed --now.
- README contains PASS caveat.
- Support edges are preserved when available.
- Package can be re-verified from package directory.
- Missing required artifacts fail closed.

Docs:
- Add docs/design-specs/claim_compiler.md or update docs/research/CC_REPORTS.md
  with packaging semantics.
- Include exact CLI examples.
- Explain what the package proves and does not prove.

Acceptance checklist:
- Package manifest is deterministic and strict.
- Verification is part of compilation or required immediately before it.
- Support semantics and non-claims survive packaging.
- Re-verification from package directory is tested.
- The final response reports files changed, CLI/API added, tests run, and
  remaining risks.
```

