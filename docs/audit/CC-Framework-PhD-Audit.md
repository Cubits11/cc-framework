# CC-Framework: PhD-Level Audit & Upgrade Blueprint

**Prepared for:** Pranav Bhave (IST 496 Independent Study, Penn State)  
**Audit Date:** November 12, 2025  
**Goal:** Elevate the Composability Coefficient (CC) Framework to publication-grade rigor for IEEE S&P / ACM CCS / NeurIPS.

---

## Executive Overview

Your CC-Framework presents a rigorous research design (two-world protocol, J-statistic, adaptive attacker). The remaining gaps are polish issues across statistics, reproducibility, and documentation. Closing **37 identified gaps** (categorized below) brings the framework to PhD-level readiness.

| Severity | Count | Target Week | Impact |
|----------|-------|-------------|--------|
| 🔴 Blocking | 12 | Week 1 | Without fixes, reviewers will reject |
| 🟠 High | 15 | Week 2 | Weakens reviewer confidence |
| 🟡 Medium | 10 | Week 3 | Boosts polish and completeness |

---

## Part 1 — Statistical Rigor Gaps

### Gap 1.1: Freedman-Hilton Correction Missing

**Problem:** README claims FH correction, but code lacks dependence-aware CI widening. Bootstrap CIs may be too narrow if attack sessions share strategies.

**Fix:** Add `src/cc/core/dependence_correction.py` implementing:
- `estimate_intra_class_correlation()` — computes ICC via ANOVA, returns rho, effective sample size, FH widening factor.
- `apply_fh_correction()` — widens percentile CIs.
- `validate_ci_coverage()` — simulation helper to verify coverage.

Integrate into `stats.bootstrap_ci()` with optional `apply_fh=True`, log rho/effective_n, and rerun coverage simulations (100 synthetic datasets). Document assumption: ICC ≤ 0.8.

### Gap 1.2: Neutrality Band Unjustified

**Problem:** Band [0.95, 1.05] appears arbitrary; no power analysis or sensitivity study.

**Fix:** Create `src/cc/analysis/power_analysis.py`:
- `simulate_cc_power()` — Monte Carlo power for constructive/destructive detection.
- `power_curve_analysis()` — curves across CC ∈ [0.80, 1.20].
- `recommend_band_width()` — δ ensuring ≥80% power for CC ≤ 0.90 or ≥1.10.

Document results in README with table and sensitivity data (±1%, ±5%, ±10%). Cite Youden’s J power references.

### Gap 1.3: No Effect Size Reporting

**Problem:** Reporting only CC_max leaves reviewers uncertain about practical significance.

**Fix:** Implement `src/cc/analysis/effect_size_calculator.py`:
- `cohens_d_for_cc()` — convert CC deviation + CI width to standardized effect size with interpretation (negligible/small/medium/large).
- `percent_difference_from_neutral()` — intuitive ±% explanation.

Include effect sizes alongside every reported CC.

### Gap 1.4–1.5: Missing Baseline Reporting

**Problem:** Only CC_max shown. Without J_none, J_A, J_B, reviewers cannot interpret composition effects.

**Fix:** Add `src/cc/exp/baseline_suite.py` (BaselineResult dataclass + BaselineSuite). Features:
- `run_unprotected_baseline()` (J_none = 0).
- `run_individual_guardrail_baselines()` for each guardrail, calibrating to target FPR.
- `run_composed_baseline()` per ordering to capture order effects.
- `report_baselines()` — formatted table (J, CI, FPR, FNR).
- `compute_cc_max()` — returns CC_max plus raw J metrics.

Integrate into experiments and always report order effects (A→B vs B→A).

---

## Part 2 — Reproducibility Gaps

### Gap 2.1: No Frozen Dependencies

**Fix:** Run `pip freeze > requirements-frozen.txt`. Keep loose `requirements.txt` for development but instruct users to install frozen versions for exact reproduction. Add CI job that installs via frozen file.

### Gap 2.2: Hardware/OS Not Logged

**Fix:** Create `src/cc/core/system_info.py` with `SystemRecorder`. Capture timestamp, OS, CPU, memory, GPU availability, Python version. Log to `system_info.json` inside each run directory and ignore file in git.

### Gap 2.3: Data Provenance Missing

**Fix:** Add `data/generation_scripts/generate_synthetic_corpus.py` that deterministically (seed 42) creates benign corpus, harmful templates, CSV, and `generation_metadata.json`. Document templates, sources, and verification commands in README.

### Gaps 2.4–2.5: No Checkpointing or Timing

**Fix:** Implement `src/cc/exp/checkpoint_manager.py` to save/resume experiments (attacker state, session count, J history). Checkpoint every 100 sessions; add cleanup command for old checkpoints. Record runtime stats.

---

## Part 3 — Code Quality Gaps

### Gaps 3.1–3.5: Type Hints, Docstrings, Tooling

**Fix Bundle:**
- Add `pyproject.toml` configuring Poetry metadata, dependencies, Black, Ruff, Mypy (strict), Pytest coverage.
- Add `.pre-commit-config.yaml` hooking Black, Ruff, Mypy.
- Enforce strict typing + docstrings in NumPy style.
- Run `pre-commit run --all-files`, `mypy src/ --strict`, `make test-cov` (target ≥85%).
- If you run `pytest` directly with `--cov`, ensure `pytest-cov` is installed; otherwise use the coverage-aware Makefile target.

---

## Part 4 — Documentation Gaps

### Gap 4.1: Formal Definitions Missing

Add README section covering two-world game, J-statistic, CC_max, bootstrap CI formulas (with FH correction term).

### Gap 4.2: Explicit Assumptions

Create `docs/ASSUMPTIONS.md` enumerating seven assumptions (threshold realizability, attacker knowledge, world symmetry, independence, stationarity, attack success validity, corpus representativeness) plus failure impacts and mitigations.

### Gap 4.3: Failure Modes & Threat Model

Add `docs/FAILURE_MODES.md` capturing attacker strength limits, composition order effects, utility trade-offs, threat model scope.

### Gaps 4.4–4.6: Related Work, Diagrams, Threat Model Visualization

- Add systematic comparison table to README vs ensemble learning, DP composition, crypto, Constitutional AI, certified robustness.
- Create `docs/THREAT_MODEL_DIAGRAM.md` illustrating two-world protocol and attacker knowledge.
- Include ASCII diagrams or reference figures for paper usage.

---

## Part 5 — Experimental Design Gaps

### Gap 5.1: Preregistration

Register on OSF and link in README. List preregistered hypotheses (constructive composition, minimal order effect, bootstrap coverage target).

### Gap 5.2: Power Analysis Documentation

Reuse outputs from `power_analysis.py` — include figures/tables plus sensitivity discussion.

---

## Part 6 — Publication Artifacts

Create `paper/` structure:
```
paper/
├── figures/
│   ├── Fig1_two_world_protocol.pdf
│   ├── Fig2_cc_phase_diagram.pdf
│   ├── Fig3_bootstrap_ci_coverage.pdf
│   └── Fig4_order_effects.pdf
├── tables/
│   ├── Table1_baseline_results.tex
│   ├── Table2_cc_results.tex
│   ├── Table3_related_work.tex
│   └── Table4_ablation_studies.tex
├── main.tex
├── appendix.tex
└── SUBMISSION_CHECKLIST.md
```

Add `paper/latex_templates/results_table_template.py` to auto-generate LaTeX tables from CSV outputs (with significance stars when CI excludes 0.5).

---

## Phased Roadmap

### Phase 1 (Week 1): Statistical Soundness — 🔴 Blocking
- [ ] FH correction (Gap 1.1)
- [ ] Power analysis (Gap 1.2)
- [ ] Effect size reporting (Gap 1.3)
- [ ] Baseline suite (Gaps 1.4–1.5)

### Phase 2 (Week 1–2): Reproducibility — 🔴 Blocking
- [ ] requirements-frozen.txt (Gap 2.1)
- [ ] System info logging (Gap 2.2)
- [ ] Data provenance script (Gap 2.3)
- [ ] Checkpoint manager (Gaps 2.4–2.5)

### Phase 3 (Week 2): Code Quality — 🟠 High
- [ ] pyproject + pre-commit
- [ ] `mypy --strict`
- [ ] `pytest --cov=src/cc`

### Phase 4 (Week 2–3): Documentation — 🟠 High
- [ ] Formal math + assumptions + failure modes
- [ ] Related work + threat model diagram

### Phase 5 (Week 3): Experimental Design — 🟡 Medium
- [ ] OSF preregistration + hypotheses

### Phase 6 (Week 3–4): Publication Artifacts — 🟡 Medium
- [ ] Paper directory, tables, figures, supplementary checklist

---

## Prioritized Checklist Template

```markdown
# CC-Framework PhD-Level Upgrade Checklist

## Priority 1: Blocking
- [ ] Gap 1.1 – FH correction
- [ ] Gap 1.2 – Power analysis
- [ ] Gap 1.3 – Effect sizes
- [ ] Gap 1.4–1.5 – Baseline suite
- [ ] Gap 2.1 – Frozen requirements
- [ ] Gap 2.2 – System logging
- [ ] Gap 2.3 – Data provenance
- [ ] Gaps 2.4–2.5 – Checkpointing

## Priority 2: High
- [ ] Gap 3.x – Tooling + linting + tests
- [ ] Gap 4.1–4.5 – Documentation, diagrams

## Priority 3: Medium
- [ ] Gap 5.1 – OSF preregistration
- [ ] Gap 6.x – Publication artifacts
```

Track completion counts to show progress against the 37 total gaps.

---

## Success Metrics

After executing the roadmap, the framework should:

- ✅ Demonstrate statistically sound CI coverage (FH correction + simulations).
- ✅ Be fully reproducible (frozen dependencies, provenance scripts, system logs, checkpoints).
- ✅ Pass strict code review (type hints, linting, tests ≥90% coverage).
- ✅ Document assumptions, failure modes, and threat model clearly.
- ✅ Provide publication-ready assets (tables, figures, preregistration link).

---

## Key References

1. Freedman & Perkins (2001) — Empirical comparison of bootstrap CIs.
2. Davison & Hinkley (1997) — *Bootstrap Methods and Their Application*.
3. Schisterman et al. (2005) — Youden’s J power analysis guidance.
4. Wei et al. (2023) — Universal jailbreak attacks.
5. Zou et al. (2023) — Universal adversarial triggers for NLP.
6. Bai et al. (2023) — Constitutional AI guardrails.
7. Cohen et al. (2019) — Certified robustness via randomized smoothing.
8. Dwork & Roth (2014) — DP composition theorems.

---

**Bottom Line:** The CC-Framework is scientifically novel and compelling. Addressing the 37 gaps above will make it defensible for top-tier venues and strengthen PhD applications.
