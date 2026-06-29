# CC Reports and Receipts

CC-Report v0.3.1 is a small machine-readable evidence bundle for a single CC
claim. It does not add a metric or change any estimator. It binds existing run
outputs, calibration metadata, assumptions, audit artifacts, figures, and claim
limits into one deterministic JSON document.

## What It Contains

A report separates:

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

## What The Receipt Proves

The receipt proves that a verifier using the same canonicalization method sees
the same report JSON. The report also records SHA-256 hashes of named evidence
files, so a verifier can check whether those files match the report.

The hash deliberately excludes `receipt.canonical_hash` to avoid a circular
definition. All other receipt fields are included.

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
  --out results/reports/smoke_cc_report.json
```

The CLI creates the output directory, refuses missing evidence files, writes the
report JSON, and prints the receipt hash.

A deterministic example report is checked in at
`examples/reporting/minimal_cc_report.json`.

## Reproducibility Context

CC reports complement the existing audit logs and run manifests. Calibration
windows define the operating point that the measurement is allowed to support.
Audit logs and figure manifests provide reproducibility evidence. The report
receipt binds those pieces together without changing the underlying math.
