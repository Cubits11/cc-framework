# CC Reports and Receipts

CC-Report v0.3.1 is a small machine-readable evidence bundle for a single CC
claim. It does not add a metric or change any estimator. It binds existing run
outputs, calibration metadata, assumptions, audit artifacts, figures, and claim
limits into one deterministic JSON document.

## What It Contains

A report separates:

* schema version: the exact report contract a verifier should apply;
* measured quantity: metric family, point estimate, interval, method, and sample
  sizes;
* assumptions: explicit premises needed to interpret the run;
* calibration: target FPR or alpha cap, realized FPR, calibration window,
  threshold, and pass/fail status;
* evidence: file paths, byte sizes, and SHA-256 hashes for artifacts, audit logs,
  and figure manifests;
* claim boundary: one allowed claim level plus explicit non-claims;
* receipt: a canonical SHA-256 hash over the report.

Allowed claim levels are `diagnostic`, `bounded_empirical`,
`reproducible_run`, and `release_claim`.

## Claim Levels For Review

The claim level is the maximum strength the receipt is allowed to support. A
reviewer may treat the evidence as weaker than the selected level, but should
not infer a stronger claim from the receipt.

| Level | Reviewer/auditor reading | Typical non-claims |
| --- | --- | --- |
| `diagnostic` | Useful for exploratory analysis, debugging, triage, or internal investigation. | Not a release decision, safety finding, compliance finding, or generalization claim. |
| `bounded_empirical` | Supports a measured point estimate and interval only for the named run, evaluation distribution, calibration window, and assumptions. | Does not certify production safety; does not generalize outside the stated evaluation setup. |
| `reproducible_run` | Gives enough run metadata and evidence hashes for rerun or forensic review of the named artifacts. | Does not prove that independent reruns will match unless the environment and inputs are equivalent; does not certify that the evidence is sufficient. |
| `release_claim` | May be attached to a release gate only when an external review process has accepted the criteria, evidence, and residual risk. | Does not by itself certify regulatory compliance, operational safety, data freshness, or deployment fitness. |

Any claim above `diagnostic` must include explicit non-claims. In review, these
non-claims are part of the receipt contract: they say which tempting conclusions
the report is deliberately not making.

## What The Receipt Proves

The receipt proves that a verifier using the same canonicalization method sees
the same report JSON. The report also records SHA-256 hashes of named evidence
files, so a verifier can check whether those files match the report.

The hash deliberately excludes `receipt.canonical_hash` to avoid a circular
definition. All other report fields are included, including `schema_version`,
`claim`, `non_claims`, evidence paths, evidence byte counts, evidence SHA-256
hashes, `hash_algorithm`, `canonicalization_method`, and `previous_hash`.

The canonicalization method for v0.3.1 is:

```text
json.dumps(sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False); receipt.canonical_hash excluded
```

Operationally, verification means:

1. Parse the report as JSON and validate it against `schemas/cc_report.schema.json`.
2. Remove only `receipt.canonical_hash`.
3. Serialize with sorted keys, compact separators, UTF-8 output, NFC-normalized
   strings, and no NaN or Infinity values.
4. SHA-256 hash those bytes and compare the result with
   `receipt.canonical_hash`.
5. For each evidence entry, recompute the file's SHA-256 and compare it with
   the recorded `sha256`.

## What It Does Not Prove

A receipt does not prove production safety, deployment fitness, data freshness,
statistical validity, or generalization outside the named evaluation
distribution. It also does not prove that the evidence is sufficient for a
regulatory or release decision. Those claims require human review and external
context.

For that reason, any claim above `diagnostic` must include non-claims. The report
should make the permitted claim easy to audit and the forbidden claims hard to
miss.

## Fixture Command

```bash
PYTHONPATH=src python -m cc.reporting.cli build-report \
  --run-id smoke-example \
  --measurement-json tests/fixtures/reporting/measurement.json \
  --calibration-json tests/fixtures/reporting/calibration_summary.json \
  --evidence tests/fixtures/reporting/artifact.txt \
  --audit-log tests/fixtures/reporting/audit.jsonl \
  --figure-manifest tests/fixtures/reporting/figure_manifest.json \
  --claim "At the pinned operating point, the composed guardrail has a bounded empirical CC interval under the stated assumptions." \
  --claim-level bounded_empirical \
  --non-claim "This report does not certify production safety." \
  --non-claim "This report does not generalize outside the named evaluation distribution." \
  --assumption "The fixture data are treated as a fixed evaluation distribution." \
  --assumption "The calibration window is the named fixture operating window." \
  --config-path experiments/configs/smoke.yaml \
  --config-hash 1111111111111111111111111111111111111111111111111111111111111111 \
  --seed 123 \
  --command "fixture build-report" \
  --report-id cc-report-smoke-example \
  --created-at 2026-01-01T00:00:00Z \
  --framework-version 0.3.1-fixture \
  --git-commit 0000000000000000000000000000000000000000 \
  --git-dirty false \
  --git-branch main \
  --python-version 3.12.0 \
  --platform fixture-platform \
  --dependency-hash 2222222222222222222222222222222222222222222222222222222222222222 \
  --out results/reports/smoke_cc_report.json
```

The CLI creates the output directory, refuses missing evidence files, writes the
report JSON, and prints the receipt hash. The fixture report is expected to
validate against `schemas/cc_report.schema.json`, and its receipt hash should be
stable when the same inputs and metadata are supplied.

A deterministic example report is checked in at
`examples/reporting/minimal_cc_report.json`.

## Reproducibility Context

CC reports complement the existing audit logs and run manifests. Calibration
windows define the operating point that the measurement is allowed to support.
Audit logs and figure manifests provide reproducibility evidence. The report
receipt binds those pieces together without changing the underlying math.
