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
| `PAPER_CORE` | First-paper code, docs, tests, or deterministic artifacts. |
| `CORE_SUPPORT` | Shared protocol, reporting, utility, or test support retained in this repo. |
| `ECOSYSTEM_BRIDGE` | Adapter or interop code for external tools, intentionally outside Paper Core. |
| `ENTERPRISE_REFERENCE` | Dashboard, AWS, or storage reference code for evidence integrity only. |
| `EXPERIMENTAL` | Research scripts, configs, notebooks, and exploratory analyses. |
| `ARCHIVE` | Historical evidence retained under documented archive paths. |
| `GENERATED_RUNTIME` | Local outputs ignored by git unless promoted. |
| `NEEDS_DECISION` | Live enough to keep for now, but requires explicit owner disposition. |

## Current Dispositions

| Path | Category | Disposition |
| --- | --- | --- |
| `src/cc/kernel/` | `PAPER_CORE` | Keep. `cc.kernel.strict` is the stable paper-facing import surface; broader kernel modules remain compatibility or implementation support. |
| `src/cc/core/` | `CORE_SUPPORT` | Keep. Provides models, protocol harnesses, audit runner, evidence-bundle orchestration, logging, and shared statistics. |
| `src/cc/evidence/` | `CORE_SUPPORT` | Keep. Owns claim envelopes, governance, assurance schema, Merkle logs, anchoring, confirmatory protocol, decay, and role ontology. |
| `src/cc/evals/` | `PAPER_CORE` | Keep. Houses the dependence benchmark and paper-facing validation payloads. |
| `src/cc/cartographer/` | `CORE_SUPPORT` | Keep. Dependence-discovery and audit tooling with its own CI lane and CLI. |
| `src/cc/guardrails/` | `CORE_SUPPORT` | Keep. Composable local guardrail implementations used by experiments and tests. |
| `src/cc/redteam/` | `CORE_SUPPORT` | Keep. Adaptive dependence search and red-team analysis remain part of the research workflow. |
| `src/cc/reporting/`, `src/cc/io/`, `src/cc/utils/` | `CORE_SUPPORT` | Keep. Shared reporting, storage, dashboard helpers, validation, and utility code. |
| `src/cc/adapters/` | `ECOSYSTEM_BRIDGE` | Keep as optional bridges. Do not classify these as enterprise infrastructure or paper-core claims. |
| `src/cc/enterprise/` | `ENTERPRISE_REFERENCE` | Keep. Bounded reference helpers for evidence-bundle integrity and dashboard bundle export. |
| `apps/dashboard/` | `ENTERPRISE_REFERENCE` | Keep. Three-view evidence dashboard for generated enterprise bundles and claim-observatory bridge work, not a safety dashboard. |
| `infra/` | `ENTERPRISE_REFERENCE` | Keep. Minimal CDK/Lambda reference architecture for integrity verification only. |
| `paper/` and `artifacts/paper/` | `PAPER_CORE` | Keep. Paper source and deterministic release artifacts remain governed by the artifact boundary. |
| `docs/` | `CORE_SUPPORT` | Keep. Release, theory, architecture, design-spec, and validation documentation remain part of the repo. |
| `docs/archive/` | `ARCHIVE` | Keep. This is the repository's tracked archive location; do not introduce a competing root-level `archive/` convention. |
| `examples/minimal/` | `PAPER_CORE` | Keep. Minimal paper-core example. |
| `examples/claim_governance_capsule/` | `CORE_SUPPORT` | Keep. Evidence-governance example and integration-test fixture generator. |
| `experiments/` and `notebooks/` | `EXPERIMENTAL` | Keep. Experimental or notebook content must not be cited as release evidence unless promoted through the artifact boundary. |
| `deployment/cli/cc/methods.py` | `NEEDS_DECISION` | Keep temporarily. It duplicates `cc.cartographer.cli methods` and should be merged, removed, or documented in a future CLI cleanup. |
| `tools/two_world_game.py` | `EXPERIMENTAL` | Keep as the current lightweight console game entry point. |
| `experiments/two_world_game_phd.py` | `NEEDS_DECISION` | Keep temporarily. It overlaps with `tools/two_world_game.py`; decide whether it is the canonical rich version or historical experimental material. |
| `visual_identity/claim_observatory/` | `NEEDS_DECISION` | Keep temporarily. It appears to feed dashboard claim-observatory work; decide explicitly before any sibling-repo migration. |
| `checkpoints/`, `results/`, `runs/`, `figs/`, `figures/` | `GENERATED_RUNTIME` | Keep README markers only. New payloads stay ignored unless promoted. |
| `summaries/week7_summary.json` | `NEEDS_DECISION` | Keep temporarily. Decide whether this is an archive artifact, fixture, or stale runtime output. |

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
