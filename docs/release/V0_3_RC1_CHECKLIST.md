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
- Node.js/npm: `node -v` reported v22.22.3; `npm -v` reported 10.9.8.
- Working tree after validation: intentional source hardening, public API
  boundary tests, package-boundary tests, claim-governance tests, lint fixes,
  metadata, README wording, and this release checklist.

The request called for a v0.1 gate table. In this repository, the package and
Paper Core release candidate are `0.3.0-rc1`, while Enterprise Reference v0.1
is a separate experimental lane. The table below records both without merging
their claims.

## Current Adversarial Gate Table

| Gate | Command | Current status | Evidence date/context | Notes |
| --- | --- | --- | --- | --- |
| Baseline cleanliness | `git status --short`; `git diff --stat`; targeted `git diff -- README.md docs/api.md docs/release/V0_3_RC1_CHECKLIST.md src/cc/cartographer/cli.py tests/unit/api/test_public_api_contract.py tests/unit/packaging/test_wheel_boundary.py` | Pass before edits | 2026-07-05, repo root | No output before the audit edits. Final status must contain only intentional hardening changes. |
| Full pytest | `PYTHONPATH=src .venv/bin/python -m pytest -q` | Pass with optional skips | 2026-07-05, Python 3.13.1 venv | Full suite reached 100%; 7 optional skips and 3 warnings. |
| Integration pytest | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/integration` | Pass | 2026-07-05, Python 3.13.1 venv | 22 integration tests passed after enterprise dependencies were available. |
| Capsule determinism and governance | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/integration/test_claim_governance_capsule.py` | Pass | 2026-07-05, Python 3.13.1 venv | Current capsule tests include temp-directory deterministic regeneration; expected count is 5 tests. |
| API and wheel boundary | `PYTHONPATH=src .venv/bin/python -m pytest -q tests/unit/api tests/unit/packaging` | Pass | 2026-07-05, Python 3.13.1 venv | Covers strict public API symbols, lazy optional imports, wheel contents, package payload size, and synthetic leaked-wheel rejection. |
| Docs build | `.venv/bin/mkdocs build --strict --site-dir /tmp/cc-framework-mkdocs-site` | Pass | 2026-07-05, Python 3.13.1 venv | MkDocs Material printed its upstream MkDocs 2.0 warning and exited 0. |
| Package build | `.venv/bin/python -m build --outdir /tmp/cc-framework-dist` | Pass | 2026-07-05, Python 3.13.1 venv | Built `cc_framework-0.3.0rc1.tar.gz` and `cc_framework-0.3.0rc1-py3-none-any.whl`. |
| Distribution metadata | `.venv/bin/python -m twine check /tmp/cc-framework-dist/*` | Pass | 2026-07-05, Python 3.13.1 venv | Both sdist and wheel passed. Classifiers now include Python 3.13 to match CI. |
| Fresh wheel install/import | `python3 -m venv /tmp/cc-framework-wheeltest`; install wheel; import `cc`, `cc.kernel.strict`, optional modules; run `cc-cartographer --help` | Pass | 2026-07-05, Python 3.14.6 clean venv | Found and fixed a blocker where CLI help imported matplotlib from the minimal wheel. |
| Artifact boundary checker | `.venv/bin/python scripts/check_artifact_boundary.py --static` | Pass | 2026-07-05, Python 3.13.1 venv | Tracked runtime roots remain marker-only; paper artifacts and manuscript figures are allowlisted. |
| README quickstart smoke | Clean tracked-copy venv, editable install, README Python snippet, and `PYTHONPATH=src python -m pytest -q tests/unit/kernel tests/unit/evidence` | Pass | 2026-07-05, Python 3.14.6 clean copy | Snippet printed `Stacked failure is bounded by [0.00%, 10.00%]`; focused tests reached 100%. |
| CLI help smoke | `PYTHONPATH=src .venv/bin/python -m cc.cartographer.cli --help`; installed-wheel `cc-cartographer --help` | Pass | 2026-07-05 | Top-level help exits 0 without importing optional matplotlib. |
| Lint | `.venv/bin/ruff check .` | Pass | 2026-07-05, Python 3.13.1 venv | Required fixes included import sorting and small Blender-helper lint issues in `visual_identity`. |
| Format | `.venv/bin/ruff format --check .` | Pass | 2026-07-05, Python 3.13.1 venv | All 201 formatted files pass after targeted formatting. |
| Focused type gate | `make type` | Pass | 2026-07-05, Python 3.13.1 venv | Matches the Makefile/CI focused mypy lane; 8 source files reported no issues. |
| Full type audit | `.venv/bin/python -m mypy src` | Fail, not claimed gate | 2026-07-05, Python 3.13.1 venv | 269 errors across 43 broader/legacy/optional files. Do not present the whole `src` tree as typed. |
| Security audit | `make security` | Pass | 2026-07-05, Python 3.13.1 venv | Bandit medium+, detect-secrets, and pip-audit passed; pip-audit found no known vulnerabilities. |
| Python dependency health | `.venv/bin/python -m pip check` | Pass | 2026-07-05, Python 3.13.1 venv | No broken requirements found. |
| Secret grep spot-check | bounded `rg` for AWS/private-key/password/token/API-key patterns | Pass with benign hits | 2026-07-05 | Hits were private-key loader `password=None`, dummy guardrail test text, and environment variable adapter plumbing. |
| Dangerous-pattern grep | bounded `rg` for `eval`, `exec`, `pickle.loads`, `yaml.load`, `shell=True`, destructive paths, and subprocess use | Reviewed | 2026-07-05 | No `eval`, `pickle.loads`, or unsafe YAML hits; remaining subprocess/temp/unlink hits are test, tooling, or scoped utility usage. |
| Dashboard build | `npm --prefix apps/dashboard run build` | Pass | 2026-07-05, Node v22.22.3/npm 10.9.8 | Next 15.5.19 production build, type check, and static page generation passed. |
| Dashboard direct smoke | `npm --prefix apps/dashboard run smoke` | Expected fail without bundle | 2026-07-05 | Fails closed unless `ENTERPRISE_BUNDLE_PATH` points at a generated enterprise dashboard bundle. `make enterprise-smoke` is the authoritative lane. |
| Dashboard npm audit | `npm --prefix apps/dashboard audit --audit-level=moderate` | Pass | 2026-07-05, npm 10.9.8 | Found 0 vulnerabilities. |
| Infra build | `npm --prefix infra run build` | Pass | 2026-07-05, Node v22.22.3/npm 10.9.8 | `tsc --noEmit` passed. |
| Infra synth | `npm --prefix infra run synth` | Pass | 2026-07-05, Node v22.22.3/npm 10.9.8 | CDK synthesized the reference template; CDK printed its feature-flag informational message. |
| Infra npm audit | `npm --prefix infra audit --audit-level=moderate` | Pass | 2026-07-05, npm 10.9.8 | Found 0 vulnerabilities. |
| Enterprise smoke | `make enterprise-smoke` | Pass | 2026-07-05, Python 3.13.1 venv + Node/npm | Installs `.[enterprise,test]`, runs dashboard `npm ci`, verifies boto3/moto, and passes moto-backed AWS emulation plus dashboard e2e smoke (`2 passed`). |

The exact unbounded `grep -R "tests/property/kernel" -n .` form was stopped
because it traversed local dependency and generated directories. The bounded
hidden-aware `rg` commands are the current repository evidence.

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

- `apps/dashboard/package.json` currently uses Next `15.5.19`, React `18.3.1`,
  Vitest `3.2.6`, and Vite `6.4.3`.
- `npm --prefix apps/dashboard audit --audit-level=moderate` found 0
  vulnerabilities.
- `npm --prefix apps/dashboard run build` passed.
- Direct `npm --prefix apps/dashboard run smoke` fails closed when
  `ENTERPRISE_BUNDLE_PATH` is unset. This is intentional; the authoritative
  smoke path is `make enterprise-smoke`.
- `make enterprise-smoke` passed and exercised the generated enterprise bundle
  through the dashboard e2e smoke.

Migration risk: the dashboard remains a separate Enterprise Reference v0.1
surface. Passing these checks is dependency and e2e hygiene, not a dashboard
release claim and not Paper Core evidence.

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

For this 2026-07-05 audit, `make enterprise-smoke`, `make security`, dashboard
build/audit, infra build/synth/audit, and the focused Makefile type gate were
run and recorded in the current adversarial gate table above. They still remain
outside the Paper Core proof boundary.

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

- Direct vendor tests requiring provider credentials or provider services.
- Performance and experiment lanes requiring `CC_RUN_PERF=1` or
  `CC_RUN_EXPERIMENTS=1`.
- Notebook, Pandoc/PDF, and LaTeX-only lanes outside the commands listed
  above.
- Full-repo `mypy src` is not a passing lane; the current full type audit
  failed and is recorded as a residual risk.

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
- [x] Final `git status --short` contains only intentional source, test,
  metadata, README, and release-doc hardening changes.
