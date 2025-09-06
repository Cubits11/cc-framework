# CC Framework Research & Development Guide

## Architectural Overview
- **Core Modules**: `cc.core` handles models, logging, metrics, and protocol logic.
- **Guardrails**: Implemented under `cc.guardrails`, providing defensive components like keyword blockers.
- **Analysis & Metrics**: `cc.analysis` and `cc.core.metrics` compute compositional capabilities (CC) statistics.
- **I/O Utilities**: `cc.io` manages data loading and serialization.
- **Legacy Modules**: `_legacy` contains reference implementations for migration or comparison.

## Key Research Questions
1. How can guardrail stacking reduce attack success rates without degrading utility?
2. What statistical methods best estimate CC under limited session counts?
3. How does attacker strategy diversity impact CC metrics?
4. Which data representations ensure reproducible experiments across environments?

## Technical Improvement Suggestions
- **Type Safety**: Adopt `mypy` to enforce type checking across modules.
- **Configuration Management**: Centralize experiment configs using `pydantic` models for validation.
- **Testing**: Expand coverage for protocol edge cases and guardrail calibration routines.
- **Continuous Integration**: Add linting and test jobs to CI to catch regressions early.

## Implementation & Extension Guidelines
- Use dataclasses for all configuration objects and provide `to_dict` methods for serialization.
- Prefer dependency injection for guardrails and attackers to ease experimentation.
- Maintain reproducibility by logging environment hashes and RNG seeds.
- When adding guardrails, implement both scoring and calibration interfaces.

## Action Items
- [ ] Integrate static analysis (`mypy`, `ruff`).
- [ ] Document attack strategy schemas in `docs/`.
- [ ] Provide examples for multi-guardrail experimentation.
- [ ] Benchmark alternative bootstrapping methods for CC estimation.
