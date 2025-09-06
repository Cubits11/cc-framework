# Experiments Guide

This guide explains how to run and extend experiments in the **cc‑framework**.

## 1. Environment Setup

* **Python**: 3.10 or later.
* **Install**: `make install` installs runtime and development dependencies.
* **Pre‑commit**: `pre-commit install` enables linting and type checks.
* **Reproducibility tip**: use `python -m venv .venv && source .venv/bin/activate` to isolate the environment.

## 2. Running Provided Experiments

```bash
# quick smoke test (<5 min)
make reproduce-smoke

# full MVP experiment (2‑4 hours)
make reproduce-mvp
```

Results are written to `results/` and figures to `paper/figures/`.

## 3. Custom Experiments

1. Copy a configuration from `experiments/configs/`.
2. Modify guardrails, attacker types, or session counts.
3. Execute the run:

```bash
python -m cc.exp.run_two_world --config experiments/configs/custom.yaml
```

## 4. Interpreting Output

* `results/aggregates/summary.csv` – aggregate metrics including `cc_max`.
* `results/**/audit.jsonl` – JSONL audit trail of interactions.
* `figs/` – diagnostic plots and protocol diagrams.

## 5. Troubleshooting

* Missing figures → ensure dependencies `matplotlib` and `seaborn` are installed.
* Long runtimes → decrease `n_sessions` in the configuration.
* Non‑deterministic results → set the `seed` field in configs.

For a deeper architectural overview see [docs/index.md](index.md).