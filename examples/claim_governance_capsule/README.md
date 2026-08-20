# Deterministic Claim Governance Capsule

This capsule is a regenerable evidence-bound claim package. It is not a
hand-edited demo: `reproduce.sh` rebuilds every generated artifact from the
checked-in inputs, verifies the governance audit, emits a deterministic
manifest, and compares outputs against checked-in expected artifacts.

## What It Proves

The capsule proves that, under fixed inputs, fixed seed, fixed timestamps,
stable JSON serialization, and the current verifier rules, the evidence-bound
claim package is internally consistent and reproducible.

A PASS governance audit means internal consistency under verifier rules. It
does not mean the AI system is safe in deployment, statistically valid outside
the declared scope, release-ready, representative of future traffic, or
compliant with external policy.

## Chain

```text
inputs/failure_matrix.csv
  -> finite-atom Frechet bounds
  -> lower/upper endpoint scenarios
  -> claim-decay policy
  -> cc.report.v0.3.1 with canonical receipt
  -> claim-governance audit
  -> claim envelope/governance state
  -> deterministic capsule manifest
```

The capsule also attaches a confirmatory protocol artifact and a
confirmatory-failure-matrix artifact because those verifier layers are already
implemented in this repository. It does not emit a fake package/signature layer
beyond the manifest.

## Calibration provenance

`calibration.realized_fpr` has an explicit provenance record in the checked-in
input and generated `calibration.json`. It separates three questions that must
not collapse: what the field means, how the local value was obtained, and
whether an external record binds that declaration. The current fixture has
**unresolved** semantics, an **asserted** origin, and no external anchor. Its
numerical resemblance to `1/24` is not treated as a derivation from the 24-row
failure matrix.

| Axis | Current value | What the builder checks | What it still does not establish |
| --- | --- | --- | --- |
| Semantics | `unresolved` | Requires a reason saying what has not been established. | That the field denotes any particular event or population. |
| Origin | `asserted` | Records the value without inferring a derivation. | That the number was measured, or that it relates to the matrix. |
| Origin (future) | `deterministic_derivation` | A named binary matrix column, source digest, exact numerator/denominator, decimal rendering, and scalar value must agree. | That the declared column is semantically a false-positive rate or that the matrix is representative. |
| External reference | `null` | A future `reference_only/v1` record must bind the exact value, semantics, and origin with an HTTPS URI, issuer, timestamp, and subject hash. The builder does not fetch it. | That the external party measured the value correctly. |

The future origin and anchor paths are independent: an asserted value can be
referenced externally, and a derivation can be referenced externally. Neither
path upgrades this fixture by itself. This is the CH-002 boundary in
[`docs/research/epistemic-program/challenges.md`](../../docs/research/epistemic-program/challenges.md).

## Run

From the repository root:

```bash
examples/claim_governance_capsule/reproduce.sh
```

The script writes generated files only under:

```text
examples/claim_governance_capsule/outputs/
```

Expected immutable artifacts live under:

```text
examples/claim_governance_capsule/expected/
examples/claim_governance_capsule/manifest.expected.json
```

## Expected Outputs

The generated outputs include:

- `bounds.json`: finite-atom Frechet interval from checked-in binary outcomes.
- `extremal_lower.json` and `extremal_upper.json`: feasible endpoint worlds.
- `decay_policy.json`: signed freshness policy, not a live safety state.
- `confirmatory_failure_matrix.json`: fixed failure matrix evidence.
- `confirmatory_protocol.json`: predeclared plan/run separation artifact.
- `cc_report.json`: canonical report receipt binding evidence hashes.
- `claim_governance_audit.json`: verifier verdict and caveats.
- `claim_envelope.json`: typed support graph and governance state.
- `capsule_manifest.json`: deterministic hash manifest for the generated package.

## Verification

Rerun verification without regenerating:

```bash
PYTHONPATH=src python examples/claim_governance_capsule/build_capsule.py --verify-only
```

Verify the report directly:

```bash
PYTHONPATH=src python -m cc.reporting.cli verify-claim-governance \
  examples/claim_governance_capsule/outputs/cc_report.json \
  --base-dir examples/claim_governance_capsule/outputs \
  --now 2026-01-02T00:00:00Z \
  --out examples/claim_governance_capsule/outputs/claim_governance_audit.json
```

Maintainers can refresh the checked-in expected artifacts after intentional
governance changes:

```bash
examples/claim_governance_capsule/reproduce.sh --update-expected
```

## Non-Claims

- The capsule is not a deployment-safety proof.
- The receipt and hashes prove byte integrity only, not statistical validity.
- The endpoint scenarios prove feasibility under constraints, not likelihood.
- The decay artifact defines recheck/degrade/expiry conditions, not current
  operational safety.
- The confirmatory protocol checks fixed-plan separation; it does not certify
  external validity or release readiness.
- A calibrated value marked `asserted` is a declared input, not a measurement
  derived from the checked-in failure matrix.
