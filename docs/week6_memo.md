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
python scripts/summarize_results.py –utility 
–final-results results/week6//final_results.json 
–out-csv results/week6//utility_summary.csv 
–out-fig figures/week6//utility_hist.png

> Interpretation: discuss any shift in mean±sd for World-1 vs World-0 and how the utility trade-off correlates with Δ and CC_max. *(Fill in commentary after runs.)*

## Power sizing (target half-width t = 0.10 @ 95%)
Use `scripts/make_power_curve.py` to recommend sample sizes for observed Week-6 intervals:

Example:
python scripts/make_power_curve.py 
–I1 0.30,0.50 –I0 0.05,0.05 –D 1.0 –delta 0.05 –target-t 0.10 
–out figures/week6/power_curve.png

Record the recommended (n1*, n0*) and include the figure: `figures/week6/power_curve.png`.

## Audit & Reproducibility
- Git SHA: *(auto-captured by runner)*  
- Config SHA: *(auto-captured)*  
- Seeds: `seed=123`  
- To reproduce a single rail end-to-end:

make week6-rail-keyword   # replace with regex/semantic/and/or

## Takeaways
- **Best Δ @ α-cap:** *(fill after runs)*  
- **Composition insights (AND vs OR):** *(fill)*  
- **Next steps:** refine guardrail combo, extend to non-toy rails (regex/semantic embeddings), scan Δ vs sample size.