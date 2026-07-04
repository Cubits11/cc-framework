# CC Framework Enterprise Reference Architecture

This document describes the **Enterprise Reference v0.1** track. It is separate
from **Paper Core v0.3**, which remains the README-centered first-paper research
track for the finite-atom kernel, canonical metrics, paper artifacts, and
documentation spine.

## Non-Claims

This architecture does **not** certify AI safety.

This architecture does **not** provide legal, regulatory, NIST AI RMF, ISO/IEC 42001, SOC 2, or audit-opinion compliance.

This architecture does **not** guarantee guardrail correctness, model correctness, policy correctness, or deployment readiness.

This architecture provides verifiable evidence-integrity checks around
composition-risk evidence bundles under stated assumptions, full stop. Evidence
integrity is not deployment safety certification.

## Minimum Credible AWS Deployment

The reference architecture in `infra/` translates the research kernel into the smallest enterprise deployment surface needed to preserve evidence integrity:

- An S3 evidence bucket with versioning and Object Lock default retention for evidence bundles.
- An asymmetric AWS KMS key used for attestation signing and verification.
- A DynamoDB run metadata table storing bundle metadata and Merkle chain-heads.
- Conditional DynamoDB writes requiring each new chain-head sequence number to be greater than the stored sequence number.
- A single Lambda exposed through `GET /verify/{bundle_id}` for backend verification of stored bundles.
- Separate least-privilege IAM roles for the evidence writer and verifier Lambda.

The dashboard in `apps/dashboard` has exactly three views:

- Composition Risk: plots the Fréchet-Hoeffding feasible envelope, empirical estimate, and cliff-certificate regime.
- Assurance Case: renders the generated GSN-style argument tree with defeaters visibly flagged.
- Verify: uploads an enterprise evidence bundle and recomputes Merkle inclusion and consistency proofs in the browser.

The browser verification path does not trust the backend endpoint. The backend endpoint is operational convenience; the proof check is independently recomputed from the uploaded bundle.

## Known Non-Release Gaps

This reference is intentionally not an enterprise product. Before making any
stronger infrastructure claim, add and verify at least:

- a written threat model for evidence-bundle integrity and verifier misuse,
- CloudTrail or equivalent audit logging for deployed AWS accounts,
- CDK Nag, checkov, or an equivalent IaC security gate,
- explicit KMS key administration and rotation policy,
- replay and tamper tests against deployed verifier payloads, not only local
  moto emulation.

## Local Emulation And Smoke Test

The enterprise tests use moto-backed AWS APIs rather than mocks of the project code. They create the S3 bucket, KMS key, and DynamoDB table through boto3, run one local evaluation, export a signed evidence bundle, upload it, advance the chain head with a conditional write, and verify the bundle.

Run the Python emulation test:

```bash
.venv/bin/pip install -e '.[enterprise,test]'
PYTHONPATH=src .venv/bin/pytest tests/integration/test_enterprise_aws_emulation.py -q
```

Run the full dashboard smoke path:

```bash
cd apps/dashboard
npm ci
npm run build
cd ../..
PYTHONPATH=src .venv/bin/pytest tests/e2e/test_enterprise_smoke.py -q
```

See [docs/validation_matrix.md](docs/validation_matrix.md) for the full lane
matrix and optional dependency skip policy.
