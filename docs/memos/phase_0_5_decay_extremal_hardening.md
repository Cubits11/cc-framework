# Phase 0.5 Decay and Extremal Scenario Hardening Memo

## Scope

Read-only audit covered:

- `src/cc/evidence/decay.py`
- `src/cc/evidence/extremal_scenario.py`
- `src/cc/evidence/assurance_schema.py`
- `src/cc/evidence/__init__.py`
- `src/cc/reporting/cli.py`
- `src/cc/redteam/dependence_search.py`
- `tests/unit/evidence/test_decay.py`
- `tests/unit/evidence/test_extremal_scenario.py`
- `tests/unit/reporting/test_reporting.py`
- `docs/research/CC_REPORTS.md`
- `docs/research/NON_CLAIMS.md`

The requested `docs/CC_REPORTS.md` and `docs/NON_CLAIMS.md` paths do not exist in
this checkout; the corresponding checked-in files are under `docs/research/`.

## What Was Implemented Correctly

- Claim decay records are strict Pydantic objects and reject unknown fields.
- `ClaimDecayRecord` records policy, covariates, version watch sets, evidence refs,
  and notes, while `evaluate_claim_decay()` computes live verification-time state.
- `issued_at` and `as_of` are required to be timezone-aware.
- TTL thresholds and version-watch changes are evaluated deterministically for a
  fixed verification time.
- Extremal scenarios serialize full atom tables, top outcomes, feasibility
  residuals, narratives, non-claims, and explicit exclusions for adaptive
  interval fields found in source payloads.
- `cc-report build-report` attaches decay and extremal scenario paths as
  `evidence.artifacts`, so their bytes, byte counts, roles, and SHA-256 hashes are
  receipt-bound without changing `cc.report.v0.3.1`.
- Assurance-case generation keeps claims, assumptions, defeaters, and top-level
  approval status in `NEEDS HUMAN REVIEW`.

## What Is Over-Scoped Or Dangerous

- Hazard scoring language is not yet armored enough. `HazardDecayPolicy`,
  `HazardCovariates`, `hazard_rate_per_day()`, and `half_life_days()` read like
  survival-analysis machinery even though the implementation is configured only.
- Hazard scoring can be configured without an explicit rationale. That makes it
  too easy for a JSON artifact to look statistically motivated when it is only a
  policy heuristic.
- Hazard-enabled decay artifacts do not yet carry mandatory non-claims explaining
  that the score is not calibrated and does not prove deployment validity.
- `--evidence-role PATH=ROLE` parsing is loose: it trims strings but does not
  reject extra separators, whitespace-heavy role names, or non-identifier roles.
- The public evidence package exports helper-ish internals such as hazard
  covariates, version watch sets, scenario feasibility, and exclusion helpers
  without a clear API-surface test.
- Extremal scenarios have useful current fields, but the object does not yet use
  the requested explicit names `schema`, `kind`, `source`, `guardrail_ids`,
  `event_definition` or `objective`, and `bound_value`.
- Feasibility diagnostics exist, but they do not yet include the requested audit
  names such as `probability_sum`, `min_probability`, `max_probability`,
  `negative_probability_count`, and `marginal_residual_linf`.

## What Could Mislead A Reader

- A reader could mistake configured hazard scoring for fitted Cox/survival
  analysis because the code uses proportional-hazard and half-life terminology
  without enough repeated non-calibration warnings.
- A serialized decay record could be mistaken for a current validity decision
  unless the docs and artifact non-claims keep separating signed policy from
  verification-time state.
- A reader could mistake an extremal scenario narrative for a likely deployment
  scenario unless feasibility diagnostics, atom tables, and non-claims remain
  prominent.
- A reader could mistake adaptive red-team `certificate_ci` output for a
  confirmatory interval because `DiscoveredCliffReport.to_dict()` still serializes
  `certificate_ci`, even though role fields and `exploratory_certificate_ci`
  currently label it as adaptive exploratory output.

## Required Audit Answers

1. Proportional-hazard scoring is partly marked as configured and non-Cox, but it
   is not clearly and repeatedly marked as a configured heuristic,
   non-calibrated risk score across class names, fields, docstrings, serialized
   records, and non-claims.
2. `certificate_ci` is not impossible to surface. It is quarantined in
   `DiscoveredCliffReport` by role fields and excluded by
   `ExtremalScenario.from_confirmatory_failure_matrix()` when passed as source
   payload, but additional tests and exclusion wording are needed to make the
   confirmatory firewall harder to misuse.
3. New report artifacts are hash-bound when attached through
   `evidence.artifacts`, including dedicated `--decay-policy`,
   `--extremal-scenario`, and generic `--evidence-role` paths.
4. Current docs do not yet explain the difference among signed decay policy,
   verification-time decay state, statistical hazard score, and actual
   deployment validity.

## Initial Recommendation

Do not add new reporting features or change the CC report schema. Harden the
existing objects by tightening names, serialized disclaimers, CLI validation,
feasibility diagnostics, confirmatory/exploratory exclusions, conservative
assurance harvesting, public exports, and focused tests.
