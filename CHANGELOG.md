# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0-rc1] - 2026-06-30

Release classification:

- **Paper Core v0.3** is release-candidate quality for reviewer use.
- **Enterprise Reference v0.1** remains an experimental reference architecture
  and is not promoted into the Paper Core release boundary.

### Added
- Release-candidate checklist for Paper Core v0.3 with required validation
  commands, optional skip policy, and explicit non-claims.
- Reviewer-facing documentation for the stable paper-core surface: finite-atom
  kernel, canonical metrics, endpoint witnesses, deterministic paper artifacts,
  validation matrix, and claim-bounded receipts.
- Clear Enterprise Reference v0.1 lane language for moto-backed AWS evidence
  integrity and dashboard smoke checks, separate from the paper-core lane.

### Changed
- README release framing now distinguishes release-candidate paper-core
  surfaces from experimental enterprise, dashboard, adapter, and legacy
  surfaces.
- Release documentation now records what passed, what was skipped or not run,
  and what the release explicitly does not claim.

### Validation
- Paper Core v0.3 validation evidence is recorded in
  `docs/release/V0_3_RC1_CHECKLIST.md`.
- Skipped optional lanes are treated as exclusions, not hidden passes.

### Non-Claims
- v0.3-rc1 does not prove deployment safety, certify deployed models, infer
  causality without causal assumptions, prove dataset representativeness, or
  establish enterprise production readiness.

### Breaking Changes
- None.

## [0.2.0] - 2026-01-14

### Added
- Evidence bundle runner and audit event override to support compliance workflows.
- Expanded guardrail adapter audits with evidence-based tests.

### Changed
- Made pytest coverage reporting opt-in to reduce default test overhead.
- Removed generated artifacts and log files from the repository to keep the tree clean.

### Breaking Changes
- None.

## [0.1.0] - 2025-08-27

### Added
- Initial release with core constructive-competition (CC) evaluation framework, guardrail adapters, and baseline configuration support.

### Breaking Changes
- None.
