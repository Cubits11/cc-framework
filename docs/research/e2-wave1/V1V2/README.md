# V1/V2: structured validator failures and checked timestamps

Base: CONTRACT E1 `d060fee`; applied after SITE E5 `1fe36e6`.

`pre.log`: **exit 1, 24 failed / 19 passed**. The corpus reproduces malformed
replicate/outcome tracebacks, unstructured malformed-file errors, and invalid
timestamps accepted with exit 0. `nonfinite.pre.log`: **exit 1, 3 failed**;
non-JSON NaN/Infinity tokens nested in raw detail were accepted before strict
JSON parsing.

`scripts/validate_e2_observations.py` now reports input/schema violations before
running design checks or audit calculations. Malformed rows never reach the
replicate hash/sort or outcome integer coercion. Input read/decode errors and
empty inputs produce structured violations with exit 1. JSON mode writes one
JSON document to stdout. Valid datasets retain the same audit and warnings
(`audit.log`, including strict JSON decoding of both reference runs).

The schema validator receives a `FormatChecker`. If jsonschema's optional
date-time checker is absent, a registered stdlib fallback checks the timestamp
shape, calendar date, time, and explicit UTC/numeric offset. No validator
dependency was added. Syntax reference:
[RFC 3339 sections 5.6–5.7](https://www.rfc-editor.org/rfc/rfc3339#section-5.6).

`post.log`: **exit 0, 82 passed**, with no skips. This includes the malformed
corpus, E1 interoperability, existing observation-contract tests, and dry-run
tests. Ruff and whitespace checks pass (`audit.log`). No scientific constants,
thresholds, outcomes, sample sizes, or frozen preregistration text changed.

```sh
E2_SITE_ROOT=/Users/pranavbhave/Developer/GitHub/e2-wave1-site \
PYTHONPATH=/tmp/e2-wave1-20261024/test-dependencies:src:. \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/tmp/cc-validation-20261023/bin/python -B -m pytest \
  -o addopts= -p no:cacheprovider -q -rs \
  tests/unit/research/test_e2_validator_malformed.py \
  tests/unit/research/test_e2_stratum.py \
  tests/unit/research/test_e2_observation_contract.py \
  tests/unit/research/test_e2_dryrun.py
```

Repository release checks are recorded separately from the frozen E2 GO
criteria. `CONTRACT.test-release.log` records the required command **exit 2**:
kernel unit tests pass; strict mypy reports nine errors in four unchanged
kernel files. Kernel source equality to base `bf5740c` is checked in `audit.log`.
No unrelated fixes were made. `CONTRACT.release-remainder.log` records a
diagnostic run with that failed prerequisite explicitly skipped: the minimal
example and ten release integration tests pass, **exit 0**. The required
combined release check remains failed.
