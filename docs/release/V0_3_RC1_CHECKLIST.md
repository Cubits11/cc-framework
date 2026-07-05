# v0.3-rc1 Release Candidate Checklist

Original record date: 2026-06-30

Current evidence refresh: 2026-07-05

This document is the sober release narrative for v0.3-rc1. It separates the
release-candidate paper core from experimental reference architecture work,
records the validation commands that were run, names skipped or out-of-scope
lanes, and restates the non-claims.

## Release Classification

| Track | Status in v0.3-rc1 | Included scope | Not included |
| --- | --- | --- | --- |
| Paper Core v0.3 | Release-candidate quality | Finite binary atom kernel, canonical metrics, endpoint witnesses, deterministic paper artifacts, documentation spine, validation matrix, and claim-bounded CC report receipts. | Deployment safety, certification, causal validity, dataset representativeness, production operations, vendor integrations, dashboard release, or cloud release. |
| Enterprise Reference v0.1 | Experimental reference architecture | Moto-backed AWS evidence-integrity emulation, signed evidence bundle flow, chain-head update checks, and dashboard smoke path when `make enterprise-smoke` is run. | Paper-core evidence, deployment safety certification, compliance certification, operational readiness, model correctness, or policy correctness. |
| Legacy and exploratory surfaces | Experimental or historical | Older protocol/workflow surfaces, notebooks, experiments, adapters, dashboard views, and performance lanes. | Stable public API guarantees or first-paper claims unless promoted by a future release note. |

## Stable For This RC

These surfaces are inside the Paper Core v0.3 release-candidate boundary:

- `src/cc/kernel/sensitivity.py`: finite binary atom LP, linear assumptions,
  sharp identified intervals, infeasibility detection, and endpoint witnesses.
- `src/cc/kernel/metrics.py`: canonical estimand-layer diagnostics including
  `fh_width`, `fh_position`, `independent_event_probability`,
  `independence_regret`, `cc_gain`, and `cc_shift`.
- `src/cc/kernel/frechet_classes.py`: classical Frechet special cases and
  side-constrained finite Bernoulli bounds.
- `src/cc/kernel/sample_complexity.py`: Hoeffding-style sample-size and radius
  helpers for singleton and pairwise Bernoulli rates, plus count-to-interval
  constraint propagation for finite-sample identification examples.
- `src/cc/kernel/strict.py`: the narrow Paper Core import surface used by
  paper-facing examples, artifact generators, and artifact verifiers.
- Paper artifact scripts and verifiers for deterministic generated outputs
  under `artifacts/paper`.
- Generated artifact boundary documentation and checks in
  `docs/release/ARTIFACT_BOUNDARY.md` and
  `scripts/check_artifact_boundary.py`.
- Documentation that defines the paper-core claim boundary:
  `docs/api.md`, `docs/research/PAPER_CORE.md`,
  `docs/research/NON_CLAIMS.md`, `docs/validation_matrix.md`, and
  `docs/theory/theorem_ledger.md`.

Stable here means suitable for release-candidate review. It does not mean the
project is production-ready or that every non-kernel module is promoted into
the paper-core contract.

## Experimental Or Separate

These surfaces remain separate from the Paper Core v0.3 release candidate:

- Enterprise Reference v0.1: use `make enterprise-smoke` for its own lane.
- Dashboard views and browser verification UI.
- Vendor adapters and optional vendor dependency tests.
- Security, serialization, performance, experiment, notebook, and cloud lanes.
- Historical `src/cc/core`, `src/cc/exp`, `src/cc/cartographer`, notebooks, and
  exploratory scripts unless a specific release note says otherwise.

Skipped optional tests in the full Python suite should be read as lane
exclusions, not proof that those optional lanes passed.

## Required Validation Record

Run commands from the repository root in the listed order. A pass claim in this
section is current only if it appears in the 2026-07-05 table below. Older
results are retained separately as historical evidence.

Current local environment for the 2026-07-05 hardening pass:

- `git status --short` before edits: no output.
- `python3 -V`: Python 3.14.6.
- `.venv/bin/python -V`: Python 3.13.1.
- Working tree after validation: intentional README, API-boundary,
  release-checklist, CLI-help, language-quarantine, and test edits.

| Command | Track | Required for this hardening pass | Current status | Notes |
| --- | --- | --- | --- | --- |
| `git status --short` | Baseline hygiene | Yes | Pass | No output before edits. Final status must contain only intentional release-hardening changes. |
| `python3 -V \|\| true` | Environment | Yes | Pass | Reported Python 3.14.6. |
| `.venv/bin/python -V \|\| true` | Environment | Yes | Pass | Reported Python 3.13.1. |
| `PYTHONPATH=src .venv/bin/python -m pytest -q tests/integration/test_claim_governance_capsule.py` | Claim governance capsule | Yes | Pass | 4 capsule integration tests passed. |
| `PYTHONPATH=src .venv/bin/python -m pytest -q tests/integration/test_claim_governance_capsule.py -vv` | Claim governance capsule | Yes | Pass | 4 passed in 7.14s. |
| `diff -u examples/claim_governance_capsule/manifest.expected.json examples/claim_governance_capsule/outputs/capsule_manifest.json \|\| true` | Claim governance capsule | Yes | Pass | No diff. The generated manifest is `outputs/capsule_manifest.json`; the old root `manifest.json` comparison path is stale. |
| `git diff -- examples/claim_governance_capsule` | Claim governance capsule | Yes | Pass | No tracked capsule diffs. Classification: no semantic drift; expected artifact is current and deterministic. |
| `rg --hidden -n "tests/property/kernel" -g '!/.git' -g '!/.venv' -g '!/infra/node_modules' -g '!/apps/dashboard/node_modules' . \|\| true` | Test-path references | Yes | Pass | No references found. Current kernel invariant/property-style coverage lives under `tests/unit/kernel`. |
| `find tests -maxdepth 3 -type d \| sort` | Test-path references | Yes | Pass | Confirmed `tests/unit/kernel` exists and `tests/property/kernel` does not. No release command currently points at the missing path. |
| `PYTHONPATH=src .venv/bin/python -m pytest -q tests/integration` | Integration tests | Yes | Pass | Integration suite reached 100% with 21 passing tests. |
| `PYTHONPATH=src .venv/bin/python -m pytest -q tests/unit/api/test_public_api_contract.py tests/unit/packaging/test_wheel_boundary.py tests/unit/utils/test_methods_cli_smoke.py` | API, wheel, and CLI boundary | Yes | Pass | 8 tests passed. The wheel test builds into a temporary directory and inspects the wheel contents locally. |
| `PYTHONPATH=src .venv/bin/python -m cc.cartographer.cli --help` | CLI help | Yes | Pass | Top-level help exits 0 and lists subcommands. |
| `source .venv/bin/activate; python - <<'PY' ...; PYTHONPATH=src python -m pytest -q tests/unit/kernel tests/unit/evidence` | README quickstart | Yes | Pass | Produced the documented `[0.00%, 10.00%]` bounds output and the focused kernel/evidence tests reached 100%. Bare `python` is used only after venv activation. |
| `.venv/bin/mkdocs build --strict --site-dir /tmp/cc-framework-mkdocs-site` | Shared docs | Yes | Pass | Strict docs build completed; MkDocs Material printed its upstream MkDocs 2.0 warning and exited 0. |
| `PYTHONPATH=src .venv/bin/python -m pytest -q` | Full Python regression | Yes | Pass with skips | Full pytest reached 100% with 7 optional skips and 3 warnings. |

Unresolved or not-current lanes for this 2026-07-05 refresh:

- Dashboard build, enterprise smoke, security audit, and npm audit were not
  rerun in this local refresh unless separately recorded below.
- The exact unbounded `grep -R "tests/property/kernel" -n .` form was stopped
  because it traversed local dependency directories; the bounded hidden-aware
  `rg` command above is the current repository evidence.

Historical validation record retained from 2026-06-30:

| Command | Track | Required for rc1 | Historical status | Notes |
| --- | --- | --- | --- | --- |
| `make check-artifact-boundary` | Repository hygiene | Yes | Pass | Verified tracked artifact locations, runtime-only roots, fixtures, archive markers, and paper artifact manifest membership. |
| `make check-repro-clean` | Repository hygiene | Yes | Pass | Ran a short reproduction sequence into a temporary directory and checked for new generated diffs. |
| `make test-kernel` | Paper Core v0.3 | Yes | Pass | Kernel unit tests passed; focused strict mypy reported no issues in 11 source files; focused ruff passed. |
| `make test-release` | Paper Core v0.3 | Yes | Pass | Re-ran `make test-kernel`, ran `examples/minimal/run_bounds.py`, and passed 10 paper reproduction / artifact-verifier integration tests. |
| `make test-reporting` | Reporting receipts | Yes | Pass | CC report/receipt unit tests passed. |
| `make docs` | Shared docs | Yes | Pass | Strict MkDocs build completed. MkDocs Material printed its upstream MkDocs 2.0 warning; the build still exited successfully. |
| `PYTHONPATH=src .venv/bin/pytest -q` | Full Python regression | Yes | Pass with skips | Historical full pytest exited successfully with 7 optional skips and 2 expected warnings from tests that drop non-finite bootstrap samples. |
| `npm run build` in `apps/dashboard` | Dashboard | No | Pass | Next 15 production build passed. A later hygiene pass removed the stale root Node package files that caused the earlier multiple-lockfile warning. |
| `make enterprise-smoke` | Enterprise Reference v0.1 | No | Pass | Moto-backed enterprise smoke passed, including the dashboard smoke path. |
| `npm audit --audit-level=high` in `apps/dashboard` | Dashboard dependency hygiene | No | Pass | No critical or high findings remain; 2 moderate transitive `postcss` findings remain through Next. |

## Artifact Verification

The paper artifact lane is part of Paper Core v0.3, but it is tracked
separately from the four rc1 commands above because regeneration can update
tracked artifacts.

Checklist:

- [x] Regenerate artifacts in place: `PYTHONPATH=src .venv/bin/python scripts/reproduce_paper.py --output-dir artifacts/paper`.
- [x] Verify regenerated artifacts: `PYTHONPATH=src .venv/bin/python scripts/verify_paper_artifacts.py --artifact-dir artifacts/paper`.
- [x] Run reproduction and verifier integration tests through `make test-release`.

Status for this release-doc pass: run. The regenerated artifact metadata records
the normalized installed package version `0.3.0rc1`, matching the
`0.3.0-rc1` release candidate after Python package normalization.

## Dashboard Dependency Audit

The dashboard remains outside the Paper Core v0.3 claim boundary, but its
security debt was checked for this pass.

- Initial `npm audit --json` in `apps/dashboard` reported 3 moderate, 2 high,
  and 1 critical vulnerabilities.
- `next` was upgraded from `^14.2.5` to `15.5.19`.
- `vitest` was upgraded from `^2.0.5` to `4.1.9`.
- `vite` was pinned as a dev dependency at `6.4.3` to avoid the `vite@8` Node
  engine requirement on local Node `20.12.2`.
- Follow-up `npm audit --json` reports 0 critical and 0 high vulnerabilities,
  with 2 moderate findings remaining through Next's transitive `postcss`
  dependency path.

Migration risk: both Next and Vitest crossed major versions. The local dashboard
build and `make enterprise-smoke` passed, but this remains dependency hygiene,
not a dashboard release claim.

Expected artifact behavior:

- Manifest hashes, schema versions, metrics, LP witnesses, tables, figures, and
  environment metadata are internally consistent.
- Endpoint witness distributions satisfy the declared constraints and achieve
  the reported LP endpoints.
- Artifact verification does not claim empirical representativeness,
  deployment safety, or certification.

## Optional Skips And Not-Run Lanes

The following lanes are not promoted into Paper Core v0.3 by this checklist:

- `make enterprise-smoke`: separate Enterprise Reference v0.1 lane.
- `make security`: shared package hygiene lane, not paper-core proof.
- Vendor adapter tests requiring packages, credentials, or provider services.
- Serialization-specific tests requiring `fastavro`, `protobuf`, or
  `sqlalchemy`.
- Experiment and performance tests gated by `CC_RUN_EXPERIMENTS=1` or
  `CC_RUN_PERF=1`.
- Notebook, Pandoc/PDF, and LaTeX compilation paths when their optional tools
  are absent. `make paper-smoke` may skip LaTeX compilation when `latexmk` is
  unavailable.

If any optional lane is run and fails, record it as a separate lane result. Do
not relabel a failed optional lane as a Paper Core pass.

Full pytest skipped these optional items in the local validation run:

- `tests/experiments/test_experiment_leak_metrics.py`: requires
  `CC_RUN_EXPERIMENTS=1`.
- `tests/performance/test_adapter_perf.py`: requires `CC_RUN_PERF=1`.
- `tests/unit/adapters/test_guardrails_ai_adapter.py`: requires
  `guardrails-ai`.
- `tests/unit/core/models/test_models_base.py`: one `fastavro` check skipped.
- `tests/unit/core/models/test_models_base.py`: two `protobuf` checks skipped.
- `tests/unit/core/models/test_models_base.py`: one `SQLAlchemy` check skipped.

Not run for this paper-core rc1 record:

- `make security`.
- Direct vendor, performance, experiment, notebook, Pandoc/PDF, and LaTeX-only
  lanes outside the commands listed above.

## Non-Claims

v0.3-rc1 does not claim:

- deployed AI systems are safe,
- deployed models are certified,
- the method infers causality without causal assumptions,
- benchmark or fixture data are representative of future deployments,
- cryptographic hashes or receipts make evidence statistically valid,
- Enterprise Reference v0.1 is production-ready,
- dashboards, adapters, cloud resources, or vendor integrations are part of the
  Paper Core v0.3 release boundary,
- the LaTeX manuscript is complete or archival.

## Release Blockers

Block v0.3-rc1 if any of the following are true:

- A required validation command fails.
- Full pytest creates tracked diffs or new checkpoint artifacts.
- Skipped optional dependencies are presented as successful optional lanes.
- Public docs imply deployment safety, certification, production readiness, or
  dataset representativeness.
- The README blurs Paper Core v0.3 with Enterprise Reference v0.1.
- Mathematical behavior changes without a focused test and release-note entry.
- Paper artifact verification fails before or after regeneration when the
  artifact lane is being claimed.

## Release Decision Checklist

- [x] Required validation record is complete.
- [x] Optional skips and not-run lanes are named.
- [x] Paper Core v0.3 and Enterprise Reference v0.1 remain separate in README,
  changelog, and release docs.
- [x] Non-claims are visible in release-facing docs.
- [x] Final `git status --short` contains only intentional release-doc changes
  and any explicitly accepted artifact updates.
