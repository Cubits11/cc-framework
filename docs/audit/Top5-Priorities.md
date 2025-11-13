# CC-Framework: Top 5 Priorities (Start Here)

**Timeline:** This week (9–10 hours)  
**Outcome:** Resolve all blocking issues for publication readiness.

---

## Why These 5 Matter

| Priority | Risk if Ignored | Reviewer Reaction |
|----------|----------------|-------------------|
| FH correction | CI coverage invalid | ❌ "Statistical rigor concerns" |
| Frozen dependencies | Irreproducible results | ❌ Immediate reject |
| Baseline suite | CC_max uninterpretable | ⚠️ "Unfair comparison" |
| Data provenance | Unknown corpus bias | ⚠️ Credibility issue |
| Power analysis | Neutrality band arbitrary | ⚠️ Methodology challenged |

Fix all five to unlock the rest of the roadmap.

---

## Priority #1 — Freedman-Hilton Correction (Gap 1.1)

1. Create `src/cc/core/dependence_correction.py` with:
   - `estimate_intra_class_correlation()`
   - `apply_fh_correction()`
   - `validate_ci_coverage()`
2. Update `stats.bootstrap_ci()` to call FH functions and log rho/effective sample size.
3. Run coverage simulation (100 datasets) to confirm 95% CI coverage.

**Validation Snippet**
```python
from src.cc.core.dependence_correction import apply_fh_correction
assert apply_fh_correction(0.40, 0.60, 0.50, 1.5) == (0.25, 0.75)
```

---

## Priority #2 — Frozen Dependencies (Gap 2.1)

1. Run `pip freeze > requirements-frozen.txt`.
2. Commit the file and update README reproduction instructions:
   ```bash
   pip install -r requirements-frozen.txt
   python -m cc.exp.run_two_world --config configs/main.yaml
   ```
3. Configure CI to install via frozen requirements.

**Validation:** `grep "==" requirements-frozen.txt | wc -l` should show ≥20 packages.

---

## Priority #3 — Baseline Suite (Gaps 1.4–1.5)

1. Implement `src/cc/exp/baseline_suite.py` (BaselineResult dataclass + BaselineSuite API).
2. Collect and report:
   - `J_none`
   - `J_guardrail_A`, `J_guardrail_B`, ...
   - `J_composed` for each ordering (A→B, B→A)
3. Print formatted baseline table inside experiments and log CC_max metrics.

**Validation:** `python -m cc.exp.run_two_world --report-baselines` prints table with all entries.

---

## Priority #4 — Data Provenance (Gap 2.3)

1. Add `data/generation_scripts/generate_synthetic_corpus.py`:
   - Deterministic benign corpus (500) + harmful templates (100) with `random.seed(42)`.
   - Save `synthetic.csv`, `benign_corpus.txt`, `harmful_templates.json`, and `generation_metadata.json`.
2. Document sources (Wei et al. 2023, Zou et al. 2023) in README.
3. Commit generated files and script.

**Validation:** Run script twice and compare `md5sum data/examples/synthetic.csv`; hashes must match.

---

## Priority #5 — Power Analysis (Gap 1.2)

1. Create `src/cc/analysis/power_analysis.py` with:
   - `simulate_cc_power()`
   - `power_curve_analysis()`
   - `recommend_band_width()`
2. Generate power curves showing ≥80% detection power for CC ≤0.90 or ≥1.10.
3. Update README with table + sensitivity analysis to justify [0.95, 1.05] neutrality band.

**Validation:**
```python
from src.cc.analysis.power_analysis import power_curve_analysis
results = power_curve_analysis()
assert results['power_constructive'][0] > 0.7
```

---

## Implementation Timeline (This Week)

| Day | Task | Time |
|-----|------|------|
| Mon | FH correction + frozen deps | 2.5 h |
| Tue | Baseline suite | 2.5 h |
| Wed | Data provenance script + metadata | 1.5 h |
| Thu | Power analysis code + docs | 1.5 h |
| Fri | Validation + documentation touch-ups | 1 h |

---

## All-in-One Verification Script

```bash
#!/bin/bash
python - <<'PY'
from src.cc.core.dependence_correction import apply_fh_correction
print('FH OK', apply_fh_correction(0.4, 0.6, 0.5, 1.5))
PY

grep "==" requirements-frozen.txt | wc -l

python - <<'PY'
from src.cc.exp.baseline_suite import BaselineSuite
print('BaselineSuite import OK')
PY

md5sum data/examples/synthetic.csv

python - <<'PY'
from src.cc.analysis.power_analysis import power_curve_analysis
print('Power curve sample', power_curve_analysis()['power_constructive'][0])
PY
```

Run this script after completing all five priorities; investigate any failures immediately.

---

## Next Steps After Completion

- Move to Priority 2 checklist (tooling + documentation).
- Share progress update with advisor, citing experiment IDs and commit hashes.
- Plan OSF preregistration and paper artifacts for Weeks 3–4.

Stay focused—these five deliverables unlock publication viability.
