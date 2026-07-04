# Claim Observatory Bridge

This bridge lets `apps/dashboard` understand the deterministic
claim-governance capsule artifacts without replacing the existing enterprise
dashboard data model.

The bridge reads four capsule artifacts:

- `cc_report.json`
- `claim_governance_audit.json`
- `claim_envelope.json`
- `capsule_manifest.json`

It converts them into a read-only `ClaimObservatoryModel` for the dashboard.
The model preserves claim identity, claim text, allowed claim level, governance
verdict, freshness status, human-review requirement, support summary, support
edges, non-claims, evidence artifact audits, manifest files, receipt facts,
decay audit, scenario audit, and confirmatory protocol audit.

## What It Does Not Do

The bridge does not certify an AI system as safe. It does not imply deployment
approval, production readiness, compliance, or external validity. It does not
verify Merkle roots, KMS signatures, Ed25519 signatures, witness anchors, or
artifact bytes in the browser. Those remain separate integrity and transport
surfaces.

The first PR also does not add multi-file capsule upload UX. The new dashboard
view is read-only and test-driven from deterministic capsule fixtures.

## Canonical Boundaries

`ClaimEnvelope` is canonical for claim semantics. It owns the claim identity,
proposition, boundary, support graph, support edges, freshness state, and
governance-state summary that the Claim Observatory renders.

`EnterpriseBundle` remains canonical for enterprise transport and ledger
provenance. It owns the dashboard bundle shape, enterprise attestation,
Merkle inclusion and consistency proof payloads, and enterprise upload path.

The bridge checks continuity between these capsule artifacts instead of
silently repairing mismatches. If verifier schema, verdict, review requirement,
freshness status, support summary, report id, envelope source report id, or
manifest report id disagree, the adapter throws a `Claim observatory continuity
error`.

## UI Language

The dashboard says `PASS under verifier rules`, not `safe`.

That wording is intentional. A `pass` verdict means the claim package is
internally consistent under the claim-governance verifier rules. It is not a
deployment-safety proof, not an approval statement, and not a guarantee.

The bridge surfaces the capsule caveat:

> PASS means internal consistency under verifier rules; it does not mean the AI system is safe in deployment.

## Tests

From the repository root:

```bash
source .venv/bin/activate
ruff check src/cc/evidence src/cc/reporting tests/unit/evidence tests/unit/reporting tests/integration/test_claim_governance_capsule.py
ruff format --check src/cc/evidence src/cc/reporting tests/unit/evidence tests/unit/reporting tests/integration/test_claim_governance_capsule.py
python -m pytest tests/unit/evidence/test_claim_governance.py tests/integration/test_claim_governance_capsule.py -q
make governance-proof-fast
mkdocs build --strict
```

For dashboard tests:

```bash
cd apps/dashboard
npm run smoke
```

The direct dashboard smoke requires `ENTERPRISE_BUNDLE_PATH`. To generate the
fixture and run the strict enterprise/dashboard lane from the repo root, use:

```bash
make enterprise-smoke
```
