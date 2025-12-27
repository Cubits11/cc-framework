# Correlation Cliff Experiment — Agent Directives

## Project Goal
Identify the critical correlation threshold ρ* where guardrail compositions transition from constructive (CC < 1.0) to destructive (CC > 1.0) using controlled Gaussian copula simulation.

## Code Style & Patterns
- **Language:** Python 3.10+ (type hints required)
- **Testing:** pytest (every module has test_*.py file)
- **Assertions:** Use `assert` for hard constraints (FH bounds, feasibility)
- **Reproducibility:** All RNG must use np.random.seed(42) or explicit seed parameter
- **Output:** CSV for numerical data, PDF for figures, JSON for configs

## Marginals Setup (Theory Constraints)
- World 0 (protected): p_A=0.20, p_B=0.15
- World 1 (unprotected): p_A=0.70, p_B=0.55
- Both worlds: Verify FH bounds are non-degenerate (L < U)

## Data Generation: Gaussian Copula
- Import scipy.stats.multivariate_normal
- For each ρ in linspace(-1.0, 1.0, 21):
  - Sample (Z1, Z2) from bivariate normal with correlation ρ
  - Transform via normal CDF to [0,1]
  - Threshold to binary failures: failures_i = (U_i > 1 - f_i)
- Save outputs: data/synthetic/failures_pair_{i:02d}_guard{1|2}.npy

## Composability Coefficient (CC) Computation
- CC := J_actual / J_indep where J = |p_C^(1) - p_C^(0)|
- J_indep = p_A * p_B (independence assumption)
- J_actual = |(p_C^(1) - p_C^(0))| (empirical joint failure rate)
- All points MUST satisfy FH bounds: J_best ≤ J_actual ≤ J_worst
- Bootstrap CI: n_bootstrap=1000, BCa method, α=0.05

## Critical Threshold (ρ*)
- Fit cubic spline to CC(ρ) with smoothing s=0.01
- Find zero-crossing of CC(ρ) - 1.0 via brentq
- Report: ρ* ± precision (where precision = grid spacing / 2)

## FH Envelope (Sanity Constraints)
- For each (λ, w): compute L(w) = max(0, p_A + p_B - 1), U(w) = min(p_A, p_B)
- Verify: L(w) ≤ p_11(λ) ≤ U(w) for all λ
- Hard assertion: If violated, exit with "FH bounds violated" message

## Visualization (3-Panel Figure)
- Panel A: CC(ρ) with 95% CI band + critical threshold marker
- Panel B: FH bounds (J_best, J_worst) with J_actual scatter
- Panel C: Risk inflation landscape (fill between 1.0 and CC(ρ))
- All axes: clear labels, grid on, legend
- Output: figures/correlation_cliff_main_figure.pdf (300 dpi)

## File Outputs (Non-Negotiable)
- CSV: data/synthetic/cliff_dataset_metadata.csv (metadata for all 21 pairs)
- CSV: results/correlation_cliff_raw_results.csv (CC, CI, FH validation)
- JSON: results/critical_threshold.json (ρ*, ρ* precision, cc_at_critical)
- PDF: figures/correlation_cliff_main_figure.pdf (main result)

## Escalation Rules
- If any FH bound violated: STOP and print diagnostic table
- If no zero-crossing found: Print "CC is monotonic" + which direction
- If CI too wide (range > 1.0): Print "Sample size insufficient" + recommend n*
- For PR: Add human review gate before "figures/" commit (visualization approval)

## Success Metrics (All Must Pass)
1. ✅ ρ* identified with precision ±0.02
2. ✅ All 21 points within FH bounds (0 violations)
3. ✅ 95% CIs exclude CC=1.0 except near ρ*
4. ✅ Main figure generation without error
5. ✅ All output files exist and have non-zero size

## Reproducibility
- Entry point: python run_all.py (from experiments/correlation_cliff/)
- Config: config_s1.yaml (YAML with all hyperparameters)
- Logging: All RNG seeds + code versions printed to stdout
- One-command check: pytest tests/test_correlation_cliff.py -v
