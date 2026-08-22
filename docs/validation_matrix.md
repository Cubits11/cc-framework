# Validation Matrix

This matrix separates the claims this repository can validate today from
optional surfaces that are useful but not part of the first-paper core.

The active research track is **Paper Core v0.3**. The cloud, dashboard, and
AWS-specific material is an **Enterprise Reference v0.1** track. Enterprise
Reference evidence checks preserve bundle integrity under a minimum reference
architecture; they are not deployment safety certification, compliance
certification, or a claim that a deployed AI system is safe.

## Runtime Support

- Package runtime floor: Python 3.10 or newer.
- CI matrix for code and docs: Python 3.10, 3.11, 3.12, and 3.13.
- Focused type-check jobs currently run on Python 3.12.
- Enterprise smoke CI currently runs on Python 3.12 plus Node.js 20.

## Lanes

| Lane | Track | What It Checks | Commands | Required Extras | Non-Claims |
| --- | --- | --- | --- | --- | --- |
| Kernel | Paper Core v0.3 | The finite-atom LP, Frechet recovery, canonical metrics, finite-sample helpers, endpoint witnesses, and strict paper-facing import surface behave as tested. | `make test-kernel`; `PYTHONPATH=src .venv/bin/pytest tests/unit/kernel -q` | `.[dev]` | Does not validate datasets, deployed systems, adapters, dashboards, cloud resources, or semantic safety. |
| Paper Core | Paper Core v0.3 | Kernel unit behavior, strict kernel typing, focused lint, minimal example execution, deterministic paper artifact regeneration, and artifact verification including hashes, schemas, metrics, and LP witnesses. | `make test-kernel`; `make test-release`; `rm -rf artifacts/paper && make reproduce-paper && make verify-paper-artifacts`; `make verify-paper-artifacts`; `make paper-smoke` | `.[dev,docs]`; LaTeX compilation inside `make paper-smoke` is optional unless `latexmk` is installed. | Does not prove deployment safety, certification, causal validity, dataset representativeness, or production readiness. |
| E1 Dependence-Evidence | Controlled synthetic research | Five declared three-indicator joint laws, I0–I3 exact bounds, endpoint witnesses, a parity counterexample, count-derived coverage simulation, hashes, and deterministic replay. | `make reproduce-empirical-e1`; `make verify-empirical-e1`; `make test-empirical-e1` | `.[dev]` | No real guardrails, prompts, labels, population, adaptive attack, or deployment outcome is evaluated; I3 is a synthetic reference, not truth about a system. |
| Evidence Governance | Shared evidence-governance track | Evidence bundles, receipts, claim envelopes, confirmatory protocol, Merkle logs, replay resistance, signature checks, and tamper tests behave as tested. | `PYTHONPATH=src .venv/bin/pytest tests/unit/evidence tests/unit/core/test_evidence_bundle.py tests/unit/core/test_audit_runner.py tests/integration/test_claim_governance_capsule.py -q` | `.[dev]`; `cryptography` and `blake3` are core dependencies. | Hashes, signatures, Merkle roots, and receipts prove byte integrity or provenance only; they do not prove statistical validity, dataset representativeness, compliance, or safety. |
| Full Python | Paper Core v0.3 plus broader repository regression coverage | The installed Python package and available optional Python dependencies pass the full local pytest suite. Skipped tests must be read as lane exclusions, not hidden failures. | `PYTHONPATH=src .venv/bin/pytest -q` | `.[dev,docs]` for the normal local suite; use additional extras when testing optional surfaces. | Does not promote every experimental or legacy module into the paper-core claim boundary. |
| Package Build | Release packaging | Source distribution and wheel build successfully and pass metadata checks. | `python -m build`; `python -m twine check dist/*` | `.[dev]` | A clean build does not imply optional lanes were tested or that the package is ready for deployment operations. |
| Enterprise Reference | Enterprise Reference v0.1 | Moto-backed AWS emulation can create the S3, KMS, and DynamoDB surfaces, export and upload one signed evidence bundle, advance the chain head with a conditional write, verify the stored bundle, and pass the dashboard smoke through the e2e pipeline. | `make enterprise-smoke` | `.[enterprise,test]`; Node.js 20/npm; dashboard package dependencies installed by the target. | Evidence integrity checks are not deployment safety certification, compliance certification, model correctness, policy correctness, or operational readiness. |
| Dashboard | Enterprise Reference v0.1 application surface | The dashboard can consume the generated enterprise evidence bundle used by the moto-backed e2e smoke path. | `make enterprise-smoke`; direct `cd apps/dashboard && npm run smoke` requires `ENTERPRISE_BUNDLE_PATH` to point at a generated enterprise dashboard bundle. | Node.js 20/npm; dashboard package dependencies; `.[enterprise,test]` for the e2e pipeline. | Dashboard views do not make the project dashboard-centered or certify deployed systems. |
| Docs | Shared | MkDocs builds strictly across the docs tree. | `make docs` | `.[docs]` or `.[dev,docs]` | A clean docs build does not validate research claims by itself. |
| Security | Shared evidence and package hygiene | Bandit scans source for medium-or-higher findings, `detect-secrets` scans source-controlled material for candidate secrets, and `pip-audit` checks installed dependencies while skipping the editable project package. | `make security`; GitHub `security.yml` also runs the secret scan. | `.[security]` | Security tooling does not certify deployed infrastructure, compliance posture, or AI safety. |
| Archive Boundary | Repository hygiene | Tracked generated artifacts live only in approved release, fixture, manuscript, or archive locations; new runtime output remains ignored. | `make check-artifact-boundary`; `make check-repro-clean` | `.[dev]` | Passing this lane does not validate the contents of an archived artifact. |
| Optional Vendor | Optional vendor/adapters | Vendor-specific adapter tests run only when the relevant vendor packages, credentials, or explicit environment gates are available. | Example: `PYTHONPATH=src .venv/bin/pytest tests/unit/adapters -q`; performance lane: `CC_RUN_PERF=1 PYTHONPATH=src .venv/bin/pytest -m perf tests/performance/test_adapter_perf.py -q` | Vendor packages such as `guardrails-ai`, serialization extras such as `fastavro`, `protobuf`, and `sqlalchemy`, or explicit performance/experiment gates as required by the tests. | Optional vendor results are not part of Paper Core v0.3 unless a release note explicitly promotes them. |
| Experimental Non-Release | Experimental | Gated experiments, notebooks, and legacy research branches can run when their owners opt in. | `CC_RUN_EXPERIMENTS=1 PYTHONPATH=src .venv/bin/pytest tests/experiments -q`; experiment-specific commands in docs or READMEs. | Experiment-specific extras and data. | Failures here do not block Paper Core unless the experiment is promoted into a release claim. |

## Honest Optional Skips

When optional dependencies are not installed, the full local pytest suite may
skip enterprise, dashboard, vendor, serialization, experiment, or performance
tests. These skips are expected only when they match the lane being run:

- `moto` absent: enterprise emulation and dashboard enterprise-smoke tests are
  skipped outside the Enterprise Reference lane. `make enterprise-smoke`
  installs/checks the enterprise extras and fails instead of treating that skip
  as a pass.
- Dashboard dependencies absent: dashboard enterprise-smoke tests may skip
  outside the Enterprise Reference lane. `make enterprise-smoke` runs
  `npm ci` first and fails if the dashboard smoke cannot run.
- `guardrails-ai` or other vendor packages absent: vendor adapter tests are
  skipped outside the Optional Vendor lane.
- `fastavro`, `protobuf`, or `sqlalchemy` absent: serialization-specific model
  tests are skipped outside the matching optional serialization lane.
- `CC_RUN_EXPERIMENTS` unset: gated experiment tests do not run by default.
- `CC_RUN_PERF` unset: performance benchmarks do not run by default.
- `pandoc` or `latexmk` absent: memo/PDF or LaTeX compilation is optional
  unless a release checklist explicitly requires it.

Release evidence should name the lane, command, result, and skipped optional
dependencies. Do not treat skipped optional lanes as proof that those lanes
passed.
