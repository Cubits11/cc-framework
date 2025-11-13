# CC-Framework: Quick Reference Checklist

Use this document as an executive guide. Each block contains actionable steps, commands, and validation checks derived from the full audit.

---

## Week-by-Week Plan

| Week | Focus | Deliverables | Validation |
|------|-------|-------------|------------|
| 1 | Statistical rigor + reproducibility | FH correction, baseline suite, frozen deps, system logging, data provenance | `pytest stats`, CI coverage simulation, `pip freeze` diff |
| 2 | Code quality + documentation | pyproject, pre-commit, assumptions/failure docs, threat model diagrams | `pre-commit run --all-files`, `mypy --strict`, README updates |
| 3 | Experimental design | OSF preregistration, power analysis documentation, sensitivity tables | OSF link live, regenerated plots committed |
| 4 | Publication artifacts | Paper figures/tables, submission checklist | `make paper` (or equivalent), artifact validation |

---

## Daily Sprint Template (Use per week)

```markdown
### Day X Goals
- [ ] Task A (ref Gap #)
- [ ] Task B (ref Gap #)

**Commands**
- `python -m cc.exp.run_two_world --config configs/main.yaml`
- `pip freeze > requirements-frozen.txt`
- `python data/generation_scripts/generate_synthetic_corpus.py`

**Validation**
- [ ] Coverage simulation passes 95% target
- [ ] New artifacts committed (screenshots, plots)
- [ ] README updated with results
```

---

## Priority Checklist (Copy into Issues)

```markdown
# Priority 1 — Blocking (Week 1)
- [ ] FH correction (dependence_correction.py, stats.py integration)
- [ ] Neutrality band power analysis (power_analysis.py)
- [ ] Effect size metrics (effect_size_calculator.py)
- [ ] Baseline suite (baseline_suite.py + experiment integration)
- [ ] Frozen dependencies (requirements-frozen.txt)
- [ ] System info logging (system_info.py)
- [ ] Data provenance (generate_synthetic_corpus.py + metadata)
- [ ] Checkpoint manager (checkpoint_manager.py + resume hooks)

# Priority 2 — High (Week 2)
- [ ] Tooling (pyproject.toml, .pre-commit-config.yaml, strict lint/test)
- [ ] Formal documentation (README math, ASSUMPTIONS.md, FAILURE_MODES.md)
- [ ] Threat model diagram + related work table

# Priority 3 — Medium (Week 3–4)
- [ ] OSF preregistration + hypotheses section
- [ ] Paper artifacts (figures, tables, latex templates, submission checklist)
```

---

## Testing & Validation Commands

| Purpose | Command | Expectation |
|---------|---------|-------------|
| FH correction smoke test | `python - <<'PY' ...` (apply_fh_correction) | Corrected CI widens proportionally |
| Frozen deps snapshot | `pip freeze > requirements-frozen.txt` | File contains exact versions (≥20 entries) |
| Baseline reporting | `python -m cc.exp.run_two_world --report-baselines` | Table shows J_none, each guardrail, compositions |
| Power analysis | `python -m src.cc.analysis.power_analysis` | Outputs power ≥80% for CC ≤0.90 |
| System logging | Inspect `runs/<exp>/system_info.json` | Contains OS, CPU, GPU metadata |
| Checkpoint resume | Interrupt run at 500 sessions, rerun | Continues from checkpoint |

---

## Git Workflow

1. Branch per phase (`feature/fh-correction`, `feature/reproducibility`, etc.).
2. Commit after each major doc/code addition (reference gap numbers in commit message).
3. Run `pre-commit run --all-files` before pushing once tooling exists.
4. Tag milestone `v0.9-phd-audit` after Phase 2 to snapshot reproducible baseline.

---

## Testing Matrix Template

| Component | Test | Command | Frequency |
|-----------|------|---------|-----------|
| Statistical routines | Bootstrap coverage sim | `python tests/stats/test_bootstrap.py` | Weekly |
| Baseline suite | Regression test w/ mocks | `pytest tests/exp/test_baseline_suite.py` | On change |
| Data provenance | Determinism | `md5sum data/examples/synthetic.csv` | After regeneration |
| Checkpointing | Resume integration test | `pytest tests/exp/test_checkpoint_manager.py` | On change |
| CLI | Smoke test | `python -m cc.cli --help` | Release |

---

## Escalation Thresholds

- If FH correction shows ICC > 0.8 → document limitation + consider block bootstrapping.
- If power < 70% for CC=0.90 → increase session count or narrow CI via variance reduction.
- If baseline suite exposes CC_max > 1.1 (destructive) → schedule ablation to explain root cause.

---

## Communication Cadence

- **Monday:** Status sync — confirm prior week’s checklist completion.
- **Wednesday:** Mid-week review — verify FH/power/requirements progress.
- **Friday:** Demo results — share plots, tables, checkpoint logs.

Use this cadence to keep advisors and collaborators aligned.

---

## Success Criteria Recap

- All blocking tasks checked before Week 2 begins.
- Frozen dependencies + provenance allow third parties to replicate results in ≤1 hour setup.
- README and docs clearly articulate mathematical foundation, assumptions, and limitations.
- Paper artifacts ready for quick assembly by Week 4.

Stay disciplined: update this checklist daily, linking commits and experiment IDs for traceability.
