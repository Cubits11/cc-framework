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

## 6. Run Lineage Manifests

Each run emits a manifest JSON plus a hash-chain companion under `runs/manifest/`.
The manifest captures reproducibility-critical lineage metadata:

```json
{
  "run_id": "run_2f9c9e4c1d2a",
  "created_at": 1735087361.123,
  "config_hashes": {
    "audit_config_blake3": "b8f7..."
  },
  "dataset_ids": [
    "datasets/prompts.jsonl",
    "datasets/benign_prompts.jsonl"
  ],
  "guardrail_versions": {
    "KeywordBlocker": "unknown",
    "RegexFilter": "unknown"
  },
  "git_sha": "9e1a1d4..."
}
```

The hash chain (`runs/manifest/<run_id>.jsonl`) is append-only and links each
record with `prev_sha256` → `sha256` pointers for tamper evidence. To query
manifests:

```bash
python -m cc.cli.manifest list
python -m cc.cli.manifest show --run-id run_2f9c9e4c1d2a
python -m cc.cli.manifest show --hash <chain_head_sha256>
```

For more on running experiments see [experiments-guide.md](experiments-guide.md).
