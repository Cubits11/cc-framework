# CC Framework — Research & Development Guide

This manual introduces the cc-framework repository and provides step-by-step guidance for researchers and engineers.

## 1. Architectural Overview

- **Experiments (`experiments/`)**
  - `run.py`: entry point for executing guardrail experiments.
  - `configs/`: canonical YAML configurations; `grids/` for parameter sweeps.
  - `two_world_game_phd.py`: reference environment used in docs and memos.
- **Core Library (`src/cc/`)**
  - `core/`: composition theory, attackers, metrics, and guardrail APIs.
  - `guardrails/`: concrete guardrail implementations (keyword, regex, semantic, etc.).
  - `analysis/`: CC estimation, reporting utilities, and figure generation.
  - `cartographer/`: statistical bounds, auditing helpers, and command‑line interfaces.
  - `io/`: seed management and storage utilities; `utils/`: plotting and validation helpers.
- **Auxiliary Areas**
  - `scripts/`: calibration, plotting, and summarisation utilities.
  - `runs/`: append-only audit logs; `results/`: experiment outputs and figures.
  - `docs/`, `paper/`: documentation, memos, and manuscript sources.

## 2. Development Environment

1. **Clone and install**
   ```bash
   git clone https://github.com/<org>/cc-framework.git
   cd cc-framework
   pip install -e .
   ```
2. **Optional tools**
   - Activate `deployment/conda/env.yaml` or Dockerfile for reproducible setups.
3. **Quality checks**
   ```bash
   pre-commit run --files <changed_files>
   pytest -q tests/unit tests/integration
   bash tests/e2e/test_reproduce_mvp.sh
   ```

## 3. Running Experiments

1. **Plan an experiment** – choose guardrail pair, config, seed, and output directory.
2. **Execute**
   ```bash
   python experiments/run.py \
     --config experiments/configs/toy.yaml \
     --seed 123 \
     --out results/demo/raw.jsonl
   ```
3. **Compute metrics (policy FPR≤5%)**
   ```bash
   python -m src.cc.analysis.cc_estimation \
     --raw results/demo/raw.jsonl \
     --policy.fpr_cap 0.05 --bootstrap.fixed 1000 \
     --out results/demo/metrics_fixed.json
   ```
4. **Shuffle baseline & refit sensitivity** (recommended)
   ```bash
   python -m src.cc.analysis.cc_estimation \
     --raw results/demo/raw.jsonl --shuffle.k 1000 \
     --out results/demo/null_shuffle.json

   python -m src.cc.analysis.cc_estimation \
     --raw results/demo/raw.jsonl --bootstrap.refit 100 --grid 51 \
     --out results/demo/metrics_refit.json
   ```
5. **Generate figures**
   ```bash
   python -m src.cc.analysis.generate_figures \
     --raw results/demo/raw.jsonl \
     --metrics results/demo/metrics_fixed.json \
     --out results/demo/figs
   ```
6. **Audit** – append run metadata to `runs/audit.jsonl`.

## 4. Key Research Questions

- How do guardrail combinations shift Youden's J and the composability coefficient (CC)?
- What attacker strategies most effectively bypass current guardrails?
- How many samples are required for stable CC estimates under bootstrap CIs?
- What utility trade-offs (latency, false positives) emerge when stacking rails?
- Can guardrails self-calibrate to maintain target false-positive rates over time?

## 5. Technical Improvements & Best Practices

- Maintain unit, integration, and end-to-end tests under `tests/`.
- Keep YAML configs minimal; prefer defaults in code and document overrides.
- Use type hints and docstrings across modules; update `docs/` alongside new APIs.
- Profile attacker generation and guardrail evaluation loops for performance.
- Record random seeds and git SHAs for every run via the audit logger.

## 6. Implementing & Extending the Framework

### Adding a Guardrail
1. Subclass `Guardrail` in `src/cc/guardrails/`:
   ```python
   from cc.guardrails.base import Guardrail

   class LengthGuardrail(Guardrail):
       def score(self, text: str) -> float:
           return len(text)

       def blocks(self, text: str) -> bool:
           return len(text) > 500
   ```
2. Register it in the guardrail factory (e.g., `TwoWorldProtocol._create_guardrail`).
3. Provide a sample config and write unit tests.

### Creating an Attacker
1. Subclass `AttackStrategy` in `src/cc/core/attackers.py`.
2. Implement `generate_attack`, `update_strategy`, and `reset` methods.
3. Register the attacker in the experiment runner.
4. Document usage and add tests.

### Adding Analysis Utilities
1. Place new metrics or visualisations in `src/cc/analysis/`.
2. Expose CLIs or scripts under `scripts/` if user-facing.
3. Include examples in `docs/` and extend tests.

## 7. Action Items & Roadmap

- [ ] Benchmark guardrail stacks on larger datasets and record CC statistics.
- [ ] Prototype RL-based attackers to compare with genetic and random baselines.
- [ ] Implement utility-aware scoring to capture latency/precision trade-offs.
- [ ] Automate dashboards from `results/` for rapid experiment review.
- [ ] Prepare publication-quality experiments and figures.

## 8. Memos

- [Week 1 Memo](memos/2025-08-27-week1.md)
- [Week 2 Memo](memos/2025-09-03-week2.md)
- [Week 3 Methods Memo](memos/2025-09-12-week3_methods.md)

---
This guide should evolve with the project. Contributions and updates are encouraged.
