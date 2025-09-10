# CC Framework Research & Development Guide

Comprehensive, step-by-step manual for engineers and researchers working on the Composability Coefficient (CC) Framework. This document explains the architecture, outlines open research problems, and codifies patterns for extending the codebase.

## 1. Architectural Overview

### 1.1 Repository Topology

- `experiments/` – configuration files (`configs/`), experiment runner (`run.py`), grids, and two-world game utilities.
- `src/cc/`
  - `analysis/` – composability estimation, cartographer, reporting, figure generation.
  - `core/` – attackers, guardrail API, composition theory, metrics, logging.
  - `guardrails/` – base classes plus keyword, regex, semantic, and composite guardrails.
  - `io/` – seed management and persistent storage helpers.
  - `utils/` – plotting helpers, dashboard, validation utilities.
- `tests/` – unit, integration, and end‑to‑end tests.
- `docs/` – design specs, memos, and additional guides.
- `scripts/` – support tooling (calibration, plotting, summarization).

### 1.2 Data Flow

1. **Experiment Run** – `python experiments/run.py --config <yaml> --seed <int> --out results/<ID>/raw.jsonl` generates raw detections and metadata.
2. **Metric Computation** – `python -m src.cc.analysis.cc_estimation` derives J, ΔJ, CC, and confidence intervals; outputs `metrics_*.json` files.
3. **Null Shuffle** – same module with `--shuffle.k` for label-shuffle baseline; outputs `null_shuffle.json`.
4. **Figure Generation** – `python -m src.cc.analysis.generate_figures` creates FH region, ROC overlay, CC convergence, and CI bar charts under `results/<ID>/figs/`.
5. **Audit Trail** – all runs append SHA-chained logs to `runs/audit.jsonl`.
6. **Decision** – classification (constructive/neutral/destructive) written to `decision.json`.

## 2. Key Research Questions

- **Guardrail Interaction** – When do guardrails compose constructively vs. destructively? Explore pairwise and higher-order stacks.
- **Attacker Modeling** – Which attack strategies best expose weaknesses? Investigate RL-based, genetic, or side‑channel attackers.
- **Statistical Reliability** – Required sample sizes for stable CC and ΔJ estimates; variance reduction techniques.
- **Utility Trade-offs** – Impact of stacking on latency and false-positive rates beyond policy caps.
- **Auto-Calibration** – Can guardrails adapt to maintain target FPR over time without manual tuning?

## 3. Technical Improvements & Best Practices

- **Testing & CI**
  - Maintain high coverage via `pytest -q tests/unit tests/integration`.
  - Consider enabling `pre-commit` for linting/formatting.
- **Configuration Management**
  - Keep YAML configs minimal; prefer code defaults and documented overrides.
  - Store experiment plans (PLAN JSON) alongside results for reproducibility.
- **Type Safety & Documentation**
  - Use type hints and detailed docstrings throughout `src/cc`.
  - Update `docs/` whenever APIs change.
- **Performance**
  - Profile attacker generation and guardrail evaluation loops.
  - Apply vectorization or caching for large runs.
- **Reproducibility**
  - All experiments record seeds and git SHAs; ensure deterministic execution.

## 4. Implementation & Extension Recipes

### 4.1 Adding a New Guardrail

1. Subclass `Guardrail` in `src/cc/guardrails/`:
   ```python
   from cc.guardrails.base import Guardrail

   class LengthGuardrail(Guardrail):
       def score(self, text: str) -> float:
           return len(text)

       def blocks(self, text: str) -> bool:
           return len(text) > 500
   ```
2. Register the guardrail in the creation mapping (e.g., `TwoWorldProtocol._create_guardrail`).
3. Provide example configuration in `experiments/configs/`.
4. Add unit tests under `tests/unit/` and run `pytest`.

### 4.2 Creating a Custom Attacker

1. Subclass `AttackStrategy` in `src/cc/core/attackers.py`.
2. Implement `generate_attack`, `update_strategy`, and `reset`.
3. Register in `experiments/run.py` (or `two_world_game_phd.py`) so configs can reference it.
4. Document usage and add example config.

### 4.3 Extending Analysis

- Implement new metrics or visualizations in `src/cc/analysis/`.
- Ensure functions accept file paths and output JSON/figures under `results/<ID>/`.
- Add corresponding tests under `tests/analysis/` and update documentation.

### 4.4 Running Experiments

```bash
python experiments/run.py \
  --config experiments/configs/toy.yaml \
  --seed 123 \
  --out results/my_run/raw.jsonl

python -m src.cc.analysis.cc_estimation \
  --raw results/my_run/raw.jsonl \
  --policy.fpr_cap 0.05 \
  --bootstrap.fixed 1000 \
  --out results/my_run/metrics_fixed.json
```

## 5. Action Items & Roadmap

- [ ] Benchmark existing guardrails on larger datasets and log CC statistics.
- [ ] Prototype RL-based attackers and compare with genetic/random baselines.
- [ ] Implement utility-aware scoring to capture latency and false-positive trade-offs.
- [ ] Automate experiment dashboard generation from results files.
- [ ] Prepare publication-ready experiments using the full pipeline.

---
This guide should evolve with the codebase. Contributions and updates are welcome.
