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

| Lane | Track | What It Proves | Commands | Required Extras | Non-Claims |
| --- | --- | --- | --- | --- | --- |
| Paper Core | Paper Core v0.3 | Kernel unit behavior, strict kernel typing, focused lint, minimal example execution, deterministic paper artifact regeneration, and artifact verification including hashes, schemas, metrics, and LP witnesses. | `make test-kernel`; `make test-release`; `rm -rf artifacts/paper && make reproduce-paper && make verify-paper-artifacts`; `make verify-paper-artifacts`; `make paper-smoke` | `.[dev,docs]`; LaTeX compilation inside `make paper-smoke` is optional unless `latexmk` is installed. | Does not prove deployment safety, certification, causal validity, dataset representativeness, or production readiness. |
| Full Python | Paper Core v0.3 plus broader repository regression coverage | The installed Python package and available optional Python dependencies pass the full local pytest suite. Skipped tests must be read as lane exclusions, not hidden failures. | `PYTHONPATH=src .venv/bin/pytest -q` | `.[dev,docs]` for the normal local suite; use additional extras when testing optional surfaces. | Does not promote every experimental or legacy module into the paper-core claim boundary. |
| Enterprise Reference | Enterprise Reference v0.1 | Moto-backed AWS emulation can create the S3, KMS, and DynamoDB surfaces, export and upload one signed evidence bundle, advance the chain head with a conditional write, and verify the stored bundle. | `.venv/bin/pip install -e '.[enterprise,test]'`; `PYTHONPATH=src .venv/bin/pytest tests/integration/test_enterprise_aws_emulation.py -q` | `.[enterprise,test]` | Evidence integrity checks are not deployment safety certification, compliance certification, model correctness, policy correctness, or operational readiness. |
| Dashboard | Enterprise Reference v0.1 application surface | The dashboard builds and the end-to-end enterprise smoke path can exercise the reference verification flow. | `cd apps/dashboard && npm ci && npm run build`; `cd ../.. && PYTHONPATH=src .venv/bin/pytest tests/e2e/test_enterprise_smoke.py -q` | Node.js 20/npm; dashboard package dependencies; `.[enterprise,test]` | Dashboard views do not make the project dashboard-centered or certify deployed systems. |
| Docs | Shared | MkDocs builds strictly across the docs tree. | `make docs` | `.[docs]` or `.[dev,docs]` | A clean docs build does not validate research claims by itself. |
| Security | Shared evidence and package hygiene | Bandit scans source for medium-or-higher findings and `pip-audit` checks installed dependencies while skipping the editable project package. | `make security` | `.[security]` | Security tooling does not certify deployed infrastructure, compliance posture, or AI safety. |
| Optional Vendor | Optional vendor/adapters | Vendor-specific adapter tests run only when the relevant vendor packages, credentials, or explicit environment gates are available. | Example: `PYTHONPATH=src .venv/bin/pytest tests/unit/adapters -q`; performance lane: `CC_RUN_PERF=1 PYTHONPATH=src .venv/bin/pytest -m perf tests/performance/test_adapter_perf.py -q` | Vendor packages such as `guardrails-ai`, serialization extras such as `fastavro`, `protobuf`, and `sqlalchemy`, or explicit performance/experiment gates as required by the tests. | Optional vendor results are not part of Paper Core v0.3 unless a release note explicitly promotes them. |

## Honest Optional Skips

When optional dependencies are not installed, the full local pytest suite may
skip enterprise, dashboard, vendor, serialization, experiment, or performance
tests. These skips are expected only when they match the lane being run:

- `moto` absent: enterprise emulation and dashboard enterprise-smoke tests are
  skipped outside the Enterprise Reference lane.
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
