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
* Archive `audit.jsonl`, run manifests, and generated figures with each run.

## 4. Data and Artifacts

* `datasets/` holds fixed input corpora; avoid modifying in‑place.
* Run manifests live under `runs/<shard>/<hash>/manifest.json` and include config + dataset hashes.
* Results live under `results/<shard>/<hash>/` with figures under `figures/<shard>/<hash>/`.
* Deterministic hashes mean identical inputs land in the same storage path.
* Use `make verify-statistics` to validate statistical assumptions.

### Synthetic demo dataset provenance

The lightweight baseline demo uses a synthetic rails dataset at
`datasets/examples/rails_tiny.csv`. It is generated via
`scripts/generate_rails_demo.py` with a fixed seed (`--seed 7`) and a
simple latent-factor model that controls label prevalence, signal strength,
and noise. Regenerate the dataset with:

```bash
python scripts/generate_rails_demo.py \
  --out datasets/examples/rails_tiny.csv \
  --n 200 \
  --seed 7 \
  --pos-rate 0.45 \
  --signal 1.2 \
  --noise 1.0 \
  --latent-scale 0.7
```

## 5. Sharing Environments

The repository provides a `.devcontainer` specification for containerised development.
Run the project inside Docker to mirror CI:

```bash
devcontainer open .
```

For more on running experiments see [experiments-guide.md](experiments-guide.md).
