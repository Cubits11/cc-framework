# experiments/run.py
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import yaml

from cc.io.storage import LocalStorageBackend, dataset_hash_from_config


REQUIRED_SECTIONS = ["experiment", "protocol", "worlds", "attacker"]


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration has required fields and two worlds."""
    if not isinstance(config, dict):
        raise ValueError("Top-level YAML must be a mapping (dictionary).")

    missing = [k for k in REQUIRED_SECTIONS if k not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    worlds = config.get("worlds", {})
    if not isinstance(worlds, dict) or "0" not in worlds or "1" not in worlds:
        raise ValueError('Config must define "worlds" with keys "0" and "1".')

    # Provide safe defaults if experiment subkeys are missing
    exp = config.setdefault("experiment", {})
    exp.setdefault("name", "unnamed")
    exp.setdefault("n_sessions", 100)
    exp.setdefault("results_dir", "results")
    exp.setdefault("seed", 1337)

    # Normalize minimal protocol section
    proto = config.setdefault("protocol", {})
    proto.setdefault("seed", exp["seed"])


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate YAML config."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Config file not found: {path}") from e

    if config is None:
        raise ValueError(f"Empty or invalid YAML file: {path}")
    validate_config(config)
    return config


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run two-world experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--n", type=int, help="Override n_sessions")
    parser.add_argument("--seed", type=int, help="Override seed")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}", file=sys.stderr)
        return 1

    # CLI overrides
    if args.n is not None:
        config["experiment"]["n_sessions"] = int(args.n)
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
        config["protocol"]["seed"] = int(args.seed)

    # Import here so that config errors are shown before any heavy imports.
    try:
        from cc.exp.run_two_world import run_experiment  # your runner
    except Exception as e:
        # Friendly guidance for Week-2 demo if heavy deps bite
        print(
            "ERROR: Could not import cc.exp.run_two_world. "
            "For today’s demo, run the lightweight console games:\n"
            "  python tools/two_world_game_pro.py -n 20 --bot\n"
            "  python tools/two_world_game_phd.py -n 20 --bot --online\n\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 2

    print(f"Starting experiment: {config['experiment']['name']}")
    print(f"Sessions: {config['experiment']['n_sessions']} | Seed: {config['experiment']['seed']}")

    repo_root = _repo_root()
    git_sha = _git_commit(repo_root)
    cfg_sha = _sha256_hex(_stable_json(config))
    dataset_sha = dataset_hash_from_config(config, base_dir=repo_root)
    storage = LocalStorageBackend(base_dir=repo_root)
    started_at = time.time()

    try:
        results = run_experiment(config)
    except ImportError as ie:
        # Typical cause: sklearn/pandas compiled against NumPy 1.x with NumPy 2.x present
        print(
            "ImportError during run_experiment. This is often a NumPy/sklearn/pandas binary "
            "compatibility issue. For quick relief:\n"
            "  • Option A (recommended today): use the PRO/PhD console demos (no heavy deps)\n"
            "  • Option B: pin NumPy<2 and reinstall sklearn/pandas (conda or pip)\n"
            f"Details: {ie}",
            file=sys.stderr,
        )
        return 3
    except Exception as e:
        print(f"ERROR: Experiment crashed: {e}", file=sys.stderr)
        return 4

    completed_at = time.time()
    manifest = {
        "experiment_name": config["experiment"]["name"],
        "config_sha256": cfg_sha,
        "dataset_sha256": dataset_sha,
        "git_sha": git_sha,
        "started_at_unix": started_at,
        "completed_at_unix": completed_at,
        "sessions_completed": len(results),
    }
    manifest_path = storage.save_json(manifest, category="runs", filename="manifest.json")

    print(f"✓ Completed {len(results)} sessions")
    print(f"Wrote manifest → {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
