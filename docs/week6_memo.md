# Week-6 Ablation @ α-cap (0.05)

**Protocol**  
- Calibrate guardrails to α-cap window [0.04, 0.06] on benign corpus (deterministic seed=123).  
- Two-world experiment (World-0 baseline, World-1 with guardrail stack), 150 episodes/rail.  
- Derived configs written post-calibration (no manual `--set`).  
- Fail-fast policy: run aborts if FPR ∉ [0.04, 0.06] or if required artifacts are missing.  
- Audit chain: `runs/audit_week6.jsonl` (calibration + run entries).

## Ablation Results (Keyword, Regex, Semantic, AND, OR)
Artifacts per rail live under `results/week6/<rail>/` with figures under `figures/week6/<rail>/`.

**Figures**  
- Δ bars: `figures/week6/<rail>/delta_bar.png`  
- ROC grid: `figures/week6/<rail>/roc_grid.png`

**Table (per rail)**  
See `figures/week6/<rail>/table.csv` with columns: `rail, TPR, FPR, Δ, CI_lo, CI_hi, CC_max`.

## Utility under α-cap
For each rail, compute and inspect per-world utility distributions: