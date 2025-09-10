# Readiness Report

- ✅ Criterion 1: `make reproduce-smoke` and `make ccc GAMMA=0.5` completed without error.
- ✅ Criterion 2: Required artifacts were generated (summary CSV, phase diagram, CCC addendum, AND figure).
- ✅ Criterion 3: `ccc_addendum.csv` records real outcomes with non-zero `successes` and `CCC` computed correctly.
- ✅ Criterion 4: `pytest tests/unit -q` passed.
- ✅ Criterion 5: `GuardrailAdapter.evaluate` returns `(blocked, score)` consistent with `blocks` and `score`.

## Recent Artifacts
- evaluation/ccc/addenda/example__AND.png
- evaluation/ccc/addenda/ccc_addendum.csv
- results/aggregates/summary.csv

## Sample CCC Rows
| rail_a | rail_b | mode | L | I | U | C_hat | CCC | (C_hat-L)/(U-L) | headroom | lift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | B | AND | 0.0 | 0.5 | 1.0 | 0.6 | 0.6 | 0.6 | 0.4 | 0.6 |
| C | D | OR | 0.0 | 0.5 | 1.0 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| E | F | AND | 0.0 | 0.5 | 1.0 | 0.25 | 0.25 | 0.25 | 0.75 | 0.25 |
