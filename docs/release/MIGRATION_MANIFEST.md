# Repository Migration Manifest

This manifest records disposition decisions for repository surfaces that sit
near a release, archive, or sibling-repository boundary. It complements
[Generated Artifact Boundary](ARTIFACT_BOUNDARY.md): the artifact boundary says
where generated files may be tracked; this document says whether source
surfaces stay in this repository, move, split, or need a human decision.

The governing split is:

- **Paper Core v0.3**: release-candidate research kernel, paper artifacts,
  validation, and documentation spine.
- **Enterprise Reference v0.1**: bounded evidence-integrity reference surfaces.
  These are not paper-core claims and do not certify deployment safety.
- **Ecosystem Bridge**: optional integrations with external guardrail or
  evaluation tooling. These are not enterprise infrastructure and should not be
  treated as first-paper core.

## Category Key

| Category | Meaning |
| --- | --- |
| `CORE_KERNEL` | Finite-atom partial-identification math, assumptions, constraints, LP solving, witnesses, Frechet recovery, metrics, and finite-sample semantics. |
| `CORE_PROTOCOL` | Foundational protocol, data models, reporting, storage, and orchestration that the research system depends on. |
| `CORE_EVIDENCE` | Evidence bundles, claim envelopes, receipts, assurance schema, Merkle/log anchoring, confirmatory protocol, and tamper-evident semantics. |
| `CORE_EVALS` | Two-world evaluation protocol and guardrail composition evaluation. |
| `CORE_DISCOVERY` | Cartographer, red-team, dependence search, and discovered-cliff analysis. |
| `ECOSYSTEM_BRIDGE` | Optional adapters to external guardrail or evaluation tools. |
| `ENTERPRISE_REFERENCE` | AWS/KMS/S3/DynamoDB/API/CDK evidence-integrity reference, not a product. |
| `DASHBOARD_DEMO` | UI, demo, and visualization surfaces that display claims or evidence but do not define the research kernel. |
| `PAPER_CORE` | Manuscript, figures, tables, paper artifacts, theorem validation outputs, and paper-facing docs. |
| `DOC_CORE` | Documentation necessary to explain theory, validation, limitations, release boundaries, reproducibility, API, security, and claim discipline. |
| `EXAMPLE_CORE` | Minimal examples and canonical reviewer quickstarts. |
| `TEST_CORE` | Unit, property, statistical, integration, security, and regression tests. |
| `EXPERIMENTAL` | Valuable but unstable research branches. |
| `GENERATED` | Outputs, runs, checkpoints, summaries, generated plots, and generated reports. |
| `ARCHIVE` | Preserved historical material explicitly removed from the release path. |
| `MIGRATE` | Material that should become a sibling repo or be merged into a canonical surface later. |
| `DELETE` | Secrets, sensitive data, duplicate generated junk, broken binary junk, unexplained root clutter, vendored dependencies, or misleading artifacts with no recoverable value. Do not use without inspecting the file. |

## Current Dispositions

| Path | Classification | Release Lane | Package Status | CI Status | Action | Owner Decision |
| --- | --- | --- | --- | --- | --- | --- |
| `src/cc/kernel/` | `CORE_KERNEL` | Kernel / Paper Core | Packaged stable surface through `cc.kernel.strict`; broader modules remain compatibility support. | Unit, property, focused mypy, focused ruff, paper artifact verifier. | Keep and protect as primary contribution. | No |
| `src/cc/core/` | `CORE_PROTOCOL` | Full Python / Evidence Governance | Packaged support surface. | Full pytest; focused type checks for selected modules; evidence-bundle tests. | Keep, but avoid presenting all modules as Paper Core. | No |
| `src/cc/evidence/` | `CORE_EVIDENCE` | Evidence Governance | Packaged support surface. | Unit evidence tests, transparency-log adversarial tests, claim-governance capsule integration. | Keep with explicit non-claims: integrity is not validity. | No |
| `src/cc/evals/` | `CORE_EVALS` | Paper Core / Evaluation | Packaged paper-facing benchmark surface. | Unit and integration dependence-benchmark tests. | Keep. | No |
| `src/cc/cartographer/` | `CORE_DISCOVERY` | Discovery / Full Python | Packaged CLI and library surface. | Cartographer unit tests, CI type subset, scheduled cartographer workflow. | Keep as discovery support, not first-paper kernel. | No |
| `src/cc/guardrails/` | `CORE_PROTOCOL` | Full Python / Evaluation | Packaged local guardrail support. | Unit guardrail tests and integration examples. | Keep as local protocol support. | No |
| `src/cc/redteam/` | `CORE_DISCOVERY` | Discovery / Full Python | Packaged research support. | Unit red-team tests. | Keep as exploratory dependence-search support. | No |
| `src/cc/reporting/`, `src/cc/io/`, `src/cc/utils/` | `CORE_PROTOCOL` | Reporting / Full Python | Packaged shared support. | Reporting, artifact, storage, and utility tests. | Keep. | No |
| `src/cc/adapters/` | `ECOSYSTEM_BRIDGE` | Optional Vendor | Packaged optional bridge surface; vendor deps are not core deps. | Adapter tests skip when optional vendor packages are absent. | Keep, but do not cite as Paper Core evidence. | No |
| `src/cc/enterprise/` | `ENTERPRISE_REFERENCE` | Enterprise Reference | Packaged helper surface for reference tests. | Moto-backed enterprise integration test. | Keep as evidence-integrity reference only. | No |
| `apps/dashboard/` | `DASHBOARD_DEMO` | Dashboard / Enterprise Reference | Not in Python package; Node workspace local to app. | Dashboard build and smoke run only in dashboard/enterprise lanes. | Keep as demo/display surface. | No |
| `infra/` | `ENTERPRISE_REFERENCE` | Enterprise Reference | Not in Python package; CDK local to `infra/`. | CDK synth/build in enterprise CI lane. | Keep as reference architecture; add IaC security gate before stronger claims. | No |
| `paper/` and `artifacts/paper/` | `PAPER_CORE` | Paper Core | Not packaged. | Paper smoke, artifact reproduction, artifact verifier, integration tests. | Keep under artifact-boundary allowlist. | No |
| `docs/` | `DOC_CORE` | Docs / Release Boundary | Not packaged. | Strict MkDocs build. | Keep as claim-boundary spine. | No |
| `docs/archive/` | `ARCHIVE` | Archive Boundary | Not packaged. | Artifact-boundary checker allows this archive root. | Keep; do not introduce root `archive/`. | No |
| `examples/minimal/` | `EXAMPLE_CORE` | Paper Core | Not packaged. | Minimal example CI step and release lane. | Keep as reviewer quickstart. | No |
| `examples/claim_governance_capsule/` | `EXAMPLE_CORE` | Evidence Governance | Not packaged. | Claim-governance capsule integration test. | Keep as evidence example/fixture generator. | No |
| `tests/` | `TEST_CORE` | All lanes | Not packaged. | Primary CI test surface. | Keep. | No |
| `experiments/` and `notebooks/` | `EXPERIMENTAL` | Experimental Non-Release | Not packaged. | Gated or direct experiment tests only; not release-blocking unless promoted. | Keep outside release claims. | No |
| `deployment/cli/cc/methods.py` | `MIGRATE` | None yet | Not packaged. | Smoke test exists for methods CLI; this path itself is outside package. | Merge into packaged CLI or archive after owner decision. | Yes |
| `tools/two_world_game.py` | `EXPERIMENTAL` | Experimental Non-Release | Not packaged. | Referenced by `experiments/run.py`; no release lane. | Keep as lightweight console game. | No |
| `experiments/two_world_game_phd.py` | `MIGRATE` | Experimental Non-Release | Not packaged. | Not release tested. | Decide whether to merge into `tools/two_world_game.py` or archive. | Yes |
| `visual_identity/claim_observatory/` | `DASHBOARD_DEMO` | Dashboard / Demo | Not packaged. | Not in CI except any explicit visual scripts run by owner. | Keep as visual/demo support; decide sibling-repo boundary before expansion. | Yes |
| `checkpoints/`, `results/`, `runs/`, `figs/`, `figures/` | `GENERATED` | Archive Boundary | Not packaged; README markers only should be tracked. | Artifact-boundary checker rejects tracked runtime payloads. | Keep marker roots only; promote reviewed outputs elsewhere. | No |
| `summaries/week7_summary.json` | `GENERATED` | Archive Boundary | Not packaged. | Not release tested. | Move to `docs/archive/` or convert to fixture if still needed. | Yes |

## Cleanup Decisions Recorded

- The historical root-level `game.csv` was promoted to
  `docs/archive/generated-results/two_world_game/game.csv`.
- The root-level Node package files were removed. JavaScript dependencies now
  live in `apps/dashboard/` or `infra/`; `tools/week6_artifact.html` loads
  chart libraries from CDNs and is not a root npm project.
- `src/cc/core/schema.py` is intentionally retained. Its `SCHEMA_VERSION` is
  imported by `cc.core.models` and covered by model round-trip tests.

## Open Decisions

- Merge or retire `deployment/cli/cc/methods.py` in favor of the packaged
  `cc.cartographer.cli methods` command.
- Decide whether `experiments/two_world_game_phd.py` should become the
  canonical rich two-world game, be archived, or be folded into
  `tools/two_world_game.py`.
- Decide whether `visual_identity/claim_observatory/` remains dashboard support
  in this repository or moves to a sibling repo with an explicit compatibility
  contract.
- Decide whether to consolidate `figs/` and `figures/`. Both are runtime-only
  README-marker roots today, and scripts still write to both names.
- Rename sprint-cadence filenames such as `week3`, `week5`, `week6`, and
  `week7` to content-based names before they become permanent release or paper
  references.
