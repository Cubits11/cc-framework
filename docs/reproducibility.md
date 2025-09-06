# Reproducibility Notes

Reliable results require a controlled environment and transparent logging.

## 1. Software Environment

| Component | Version/Notes |
|-----------|---------------|
| Python    | 3.10+ |
| OS        | Linux (Ubuntu 22.04 tested) |
| Dependencies | `pip install -e .[dev]` |

Freeze dependencies with:

```bash
pip freeze > runs/requirements.lock
```

## 2. Hardware Assumptions

* CPU: 4 cores minimum for MVP runs.
* Memory: 8‑16 GB recommended.
* GPU: optional; only required for advanced attackers.

## 3. Determinism

* Set `seed` in configuration files for RNG control.
* Use `PYTHONHASHSEED=0` when invoking Python for bit‑wise reproducibility.
* Archive `audit.jsonl` and generated figures with each run.

## 4. Data and Artifacts

* `datasets/` holds fixed input corpora; avoid modifying in‑place.
* Results under `results/` are timestamped; keep raw outputs for peer review.
* Use `make verify-statistics` to validate statistical assumptions.

## 5. Sharing Environments

The repository provides a `.devcontainer` specification for containerised development.
Run the project inside Docker to mirror CI:

```bash
devcontainer open .
```

For more on running experiments see [experiments-guide.md](experiments-guide.md).
