# Week 5 Scan Fixture

`scan.csv` is the golden Week 5 scan fixture consumed by
`tests/regression/week5/test_week5.py`.

Regenerate it by running `CC_REFRESH_GOLDEN_ARTIFACTS=1 make week5-pilot`, then
promote the resulting `results/week5_scan/scan.csv` here after reviewing the
diff. Companion generated files belong in `docs/archive/generated-results/` if
they are retained as historical evidence.
