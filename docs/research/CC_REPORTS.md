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
* claim boundary: one allowed claim level plus explicit non-claims and, when an
  interval relationship must be machine-checked, an optional explicitly
  structured quantitative proposition;
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

### Optional structured quantitative proposition

`claim.statement` remains free text. A receipt binds that text but does not
establish what it means. A producer may instead include
`claim.quantitative_proposition` with an exact metric family, relation
(`upper_bound` or `lower_bound`), and threshold. The claim-package compiler can
compare that small structure with `measurement.interval`; it does not infer a
proposition from prose or validate population, denominator, confidence, or
source-data meaning. See
[`CLAIM_PACKAGE_COMPILER.md`](CLAIM_PACKAGE_COMPILER.md) for the separate
integrity, entailment, and independence results.

## What The Receipt Verifies

The receipt verifies that a checker using the same canonicalization method sees
the same report JSON. The report also records SHA-256 hashes of named evidence
files, so a verifier can check whether those files match the report. This is a
byte-integrity statement, not a claim that the evidence is statistically valid
or deployment-safety sufficient.

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

## Optional Evidence Roles

`cc.report.v0.3.1` can attach optional evidence artifacts without changing the
report schema. The role is stored on each item in `evidence.artifacts`, so the
artifact path, byte count, role, and SHA-256 hash are included in the canonical
receipt.

Supported optional roles include:

* `claim_decay`: a signed decay policy that says when a claim should be
  rechecked, degraded, or expired. It is not the live claim state.
* `extremal_scenario`: an endpoint or fitted scenario artifact with atom-table
  and feasibility diagnostics. It is not a deployment approval.
* `confirmatory_protocol`: a pre-registered plan plus separate run reference
  used to check confirmatory timing, provenance, endpoints, analysis plans,
  stopping rules, and cluster-blocking obligations. It is not a deployment
  approval or external-validity proof.

A signed decay policy is the artifact attached to the receipt. Verification-time
decay state is computed later by evaluating that policy against a verifier's
clock and observed versions. A configured statistical hazard score, when present,
is a configured heuristic risk score, not a fitted or calibrated survival model.
None of these prove actual deployment validity.

```bash
cc-report build-report \
  --decay-policy decay.json \
  --extremal-scenario upper.json \
  --extremal-scenario lower.json
```

Generic role attachment is also available:

```bash
cc-report build-report \
  --evidence-role decay.json=claim_decay
```

Exploratory red-team discovery can find candidate dependence cliffs. It does not
by itself certify a confidence interval. Confirmatory failure-matrix evidence
must be generated separately.

## Executable Claim Governance Verifier

`cc-report verify-claim-governance` is a read-only verifier over an existing
`cc.report.v0.3.1` report and its attached evidence artifacts. It does not
change the report schema and does not add a new mandatory report field.

The verifier checks:

* the report JSON can be loaded and has the expected core fields;
* the canonical report receipt verifies, when possible;
* every report-bound evidence artifact has the recorded SHA-256 and byte count;
* `claim_decay` artifacts evaluate to `fresh`, `degraded`, or `expired` at the
  verifier's clock;
* `extremal_scenario` artifacts parse, expose feasible endpoint/fitted worlds,
  and record excluded evidence fields;
* `confirmatory_protocol` artifacts bind a pre-registered plan to a separate
  run and enforce the exploratory/confirmatory firewall;
* exploratory red-team intervals are not surfaced as confirmatory evidence;
* mandatory non-claims implied by evidence roles are present.

The verifier does not prove production safety, statistical validity, deployment
fitness, dataset representativeness, label correctness, or regulatory
sufficiency. A PASS verdict means the evidence-bound claim package is internally
consistent under the verifier rules. It does not mean the AI system is safe in
deployment.

```bash
cc-report verify-claim-governance report.json \
  --now 2026-01-02T00:00:00Z \
  --out claim_governance_audit.json
```

Exit codes are intended for CI:

* `0`: `PASS`
* `1`: `NEEDS_REVIEW`
* `2`: `FAIL`

Interpretation:

* `PASS`: hashes, decay state, scenario artifacts, non-claims, and firewall
  checks are internally consistent under the v0 rules.
* `NEEDS_REVIEW`: the package is readable but has conservative review triggers,
  such as degraded decay, unknown roles, missing mandatory non-claims, scenario
  exclusions, clustered confirmatory data without blocking, or evidence weaker
  than the claim level.
* `FAIL`: the package cannot be trusted as an internally consistent evidence
  bundle because the report is unreadable, a hash mismatches, decay has expired,
  scenario evidence is malformed or infeasible, a confirmatory protocol violates
  timing/provenance requirements, or exploratory evidence leaked into a
  confirmatory surface.

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
