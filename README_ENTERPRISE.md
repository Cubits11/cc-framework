# CC Framework Enterprise Reference Architecture

## Non-Claims

This architecture does **not** certify AI safety.

This architecture does **not** provide legal, regulatory, NIST AI RMF, ISO/IEC 42001, SOC 2, or audit-opinion compliance.

This architecture does **not** guarantee guardrail correctness, model correctness, policy correctness, or deployment readiness.

This architecture provides verifiable evidence about composition risk under stated assumptions, full stop.

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

## Local Emulation And Smoke Test

The enterprise tests use moto-backed AWS APIs rather than mocks of the project code. They create the S3 bucket, KMS key, and DynamoDB table through boto3, run one local evaluation, export a signed evidence bundle, upload it, advance the chain head with a conditional write, and verify the bundle.

Run the Python emulation test:

```bash
python3 -m pip install -e '.[enterprise,test]'
python3 -m pytest tests/integration/test_enterprise_aws_emulation.py
```

Run the full dashboard smoke path:

```bash
cd apps/dashboard
npm ci
cd ../..
python3 -m pytest tests/e2e/test_enterprise_smoke.py
```
