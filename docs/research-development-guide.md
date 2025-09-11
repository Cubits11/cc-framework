# CC Framework — Research & Development Guide

This manual acts as an internal developer handbook for the cc-framework project. It explains the architecture, provides a research agenda, and codifies how to extend the framework.

## 1. Architectural Overview

- **Core Library (`src/cc/`)
  - `core/`: composition theory primitives, attacker interfaces, metrics, and experiment orchestration.
  - `guardrails/`: implementations for keyword, regex, semantic, and other guardrails.
  - `analysis/`: composability coefficient (CC) estimation, reporting utilities, and figure generation.
  - `cartographer/`: statistical bounds, auditing helpers, and command‑line interfaces.
  - `io/`: seed management and storage utilities; `utils/`: plotting and validation helpers.
- **Experiments (`experiments/`)
  - `run.py`: entry point for running guardrail experiments.
  - `configs/`: canonical YAML configurations; `grids/` support parameter sweeps.
  - `two_world_game_phd.py`: reference environment used in docs and memos.
- **Support Assets**
  - `scripts/`: calibration, plotting, summarisation, and migration scripts.
  - `runs/`: append-only audit logs; `results/`: experiment outputs and figures.
  - Documentation lives under `docs/`; `paper/` contains manuscript sources.

## 2. Development Environment & Workflow

1. **Setup**
   ```bash
   git clone https://github.com/<org>/cc-framework.git
   cd cc-framework
   pip install -e .
   ```
2. **Optional tooling**
   - Use `deployment/conda/env.yaml` or the Dockerfile for reproducible builds.
3. **Quality checks**
   ```bash
   pre-commit run --files <changed_files>
   pytest -q tests/unit tests/integration
   ```
4. **Workflow tips**
   - Keep commits small and descriptive.
   - Record seeds and git SHAs via the audit logger.
   - Update relevant docs when adding new features.

## 3. Research Agenda & Key Questions

- How do guardrail combinations affect Youden's J and the CC across domains?
- Which attacker strategies most effectively bypass current guardrail stacks?
- What sample sizes are required for stable CC estimates with bootstrap CIs?
- How do utility trade-offs (latency, false positives) change with additional rails?
- Can guardrails self-calibrate to maintain target false-positive rates over time?
- What formal guarantees can we provide about composition under adaptive attacks?

## 4. Experiment Workflow

1. **Plan** – choose guardrail pair, config, seed, and output directory.
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

## 5. Extension Guide

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
3. Provide a sample config and unit tests.

### Implementing an Attacker
1. Subclass `AttackStrategy` in `src/cc/core/attackers.py`.
2. Implement `generate_attack`, `update_strategy`, and `reset` methods.
3. Register the attacker in the experiment runner.
4. Document usage and add tests.

### Adding Analysis Utilities
1. Place new metrics or visualisations in `src/cc/analysis/`.
2. Expose CLIs or scripts under `scripts/` if user-facing.
3. Include examples in `docs/` and extend tests.

## 6. Technical Improvements & Best Practices

- Maintain unit, integration, and end-to-end tests under `tests/`.
- Use type hints and docstrings across modules; prefer minimal YAML configs.
- Profile attacker generation and guardrail evaluation loops for performance.
- Automate dashboards from `results/` for rapid experiment review.
- Ensure datasets and outputs are versioned and stored with metadata.

## 7. Roadmap & Action Items

- [ ] Benchmark guardrail stacks on larger datasets and record CC statistics.
- [ ] Prototype RL-based attackers to compare with genetic and random baselines.
- [ ] Implement utility-aware scoring to capture latency/precision trade-offs.
- [ ] Add self-calibrating guardrails that adjust thresholds automatically.
- [ ] Prepare publication-quality experiments and figures.

---
This guide should evolve with the project. Contributions and updates are encouraged.
