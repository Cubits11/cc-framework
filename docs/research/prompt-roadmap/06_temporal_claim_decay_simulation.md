# Prompt 6 - Temporal Claim Decay Research Simulation

Use this as a Codex execution prompt.

```text
You are working in the cc-framework repository. Implement Prompt 6:
Temporal Claim Decay Research Simulation as deterministic configured-risk
research infrastructure.

Start by reading:
- docs/research/CLAIM_GOVERNANCE_OS.md, especially Section 5 and Lane A
- docs/research/CC_REPORTS.md
- src/cc/evidence/decay.py
- src/cc/evidence/claim_governance.py
- scripts/calibrate_anytime_valid.py, scripts/make_week7_figs.py, and other
  script patterns if useful
- tests that cover claim decay

Objective:
Build a deterministic simulation showing how configured review pressure can
depend on dependence uncertainty, drift, policy age, and adversary shift.
This must be framed as governance policy simulation, not calibrated survival
analysis.

Core object:
ConfiguredDecayExperiment(
    frechet_width_grid,
    drift_grid,
    policy_age_grid,
    adversary_shift_grid,
    coefficients,
    rationale,
)

Output object:
DecaySimulationResult(
    scenario_id,
    configured_risk_rate,
    configured_half_life,
    degraded_at,
    expired_at,
    non_claims,
)

Non-negotiable semantics:
- configured governance risk is not calibrated probability of failure.
- Wide Frechet interval does not mean the claim is false.
- Dependence width is a governance covariate, not empirical truth.
- Every artifact must include the non-calibration caveat.
- A negative coefficient on dependence width requires explicit rationale or is
  rejected.

Implementation requirements:
- Add a small simulation module or script consistent with the repo's existing
  layout. Prefer src/cc/evidence/decay_simulation.py if it is part of the
  library, or scripts/ if it is only artifact generation.
- Reuse ConfiguredHazardPolicy and related decay primitives where possible.
- Emit deterministic JSON and, if useful, CSV/PNG artifacts under docs/theory/
  figures or another established research-output directory.
- Include fixed seed/time where relevant.
- Do not claim empirical calibration without data.

Tests:
- Wider dependence uncertainty increases configured review pressure when the
  coefficient is positive.
- Zero coefficient makes dependence width irrelevant.
- Negative coefficient on dependence width without explicit rationale is
  rejected.
- Outputs include non-calibrated warning.
- Generated results are deterministic under fixed inputs.

Docs:
- Add docs/research/claim_decay_theory.md or update it if already present.
- Explain proof meaning and non-proof meaning.
- Include one figure/table only if it can be regenerated.

Acceptance checklist:
- Simulation artifacts are deterministic.
- Non-calibration caveat is present in code outputs and docs.
- Tests cover coefficient behavior.
- The final response reports files changed, commands run, and remaining risks.
```

