# E1 / B-04: evaluation stratum schema compatibility

Base: CONTRACT `bf5740c266ff495319442402e0fa7d6aa23ccafa`.
Applied after SITE E4 commit `c949b2b`.

The schema now requires a string `stratum` with exactly `harmful`,
`benign_eval`, `harmful_jbb`, or `benign_jbb`
(`schemas/cc.e2_observation_row.v1.json`). These are the four evaluation
populations frozen in SITE `experiments/e2/freeze/PREREG_SELECTION.md:25-49`;
calibration remains separate. The frozen preregistration and measurement
contract text are unchanged.

`pre.log` records the reproduced failure: **exit 1, 7 failed, 7 passed**.
The governing validator accepts three original reference rows, then the actual
SITE analyzer raises `KeyError: 'stratum'`. Adding a valid stratum is rejected
as an additional property by that original schema.

`post.log` records **exit 0, 36 passed**, including required/type/enum checks,
the actual SITE analyzer load/complete-case round trip, existing observation
contract tests, and the dry-run tests. The full SITE
collection → validation → analysis round trip remains E2's responsibility.

`audit.log` records **exit 0** for both reference datasets' real validators,
Ruff, and `git diff --check`. Its metadata-only comparison proves that all
existing fields in the 24 example rows and 66 dry-run rows equal HEAD after
removing the new `stratum: harmful` field. The dry-run emitter and existing test
row factory now supply that field. These are implementation fixtures, not E2
measurements.

Focused post-fix command, run from CONTRACT:

```sh
E2_SITE_ROOT=/Users/pranavbhave/Developer/GitHub/e2-wave1-site \
PYTHONPATH=/tmp/e2-wave1-20261024/test-dependencies:src:. \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/tmp/cc-validation-20261023/bin/python -B -m pytest \
  -o addopts= -p no:cacheprovider -q -rs \
  tests/unit/research/test_e2_stratum.py \
  tests/unit/research/test_e2_observation_contract.py \
  tests/unit/research/test_e2_dryrun.py
```

The isolated test dependency directory supplies scikit-learn 1.9.1, joblib
1.6.0, threadpoolctl 3.7.0, cloudpickle 3.1.2, and narwhals 2.27.1; repository
dependency declarations were not changed.
