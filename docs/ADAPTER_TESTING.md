# Adapter Testing & Reproducibility

This document describes how to run adapter tests, experiments, and performance benchmarks
in a deterministic, audit-safe way.

## Unit Tests

Run unit tests with:

```bash
pytest tests/unit -v
```

## Hypothesis Property Tests

Property tests are optional and run only when `hypothesis` is installed:

```bash
pip install hypothesis
pytest tests/unit/test_output_parsing_properties.py -v
```

The hypothesis-based tests use an explicit seed to ensure determinism.

## Experiments (Artifacts + Plots)

Experiments are gated behind the `experiment` marker and `CC_RUN_EXPERIMENTS=1`.
Artifacts are written to `artifacts/adapters/<run-id>/` and include:

- `metrics.json`
- `leak_score_hist.png`
- `manifest.json`

Run with:

```bash
CC_RUN_EXPERIMENTS=1 pytest -m experiment tests/experiments/test_experiment_leak_metrics.py -v
```

## Performance Benchmarks

Performance tests are gated behind the `perf` marker and `CC_RUN_PERF=1`.
Artifacts are written to `artifacts/perf/<run-id>/` and include:

- `metrics.json`
- `latency_hist.png`
- `manifest.json`

Run with:

```bash
CC_RUN_PERF=1 pytest -m perf tests/performance/test_adapter_perf.py -v
```

## Notes

- All audit payloads are leak-safe (hash/length/type summaries only).
- Experiments and benchmarks never run by default in CI.
