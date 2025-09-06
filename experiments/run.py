# experiments/run.py
import argparse
import sys
from typing import Any, Dict

import yaml


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

    print(f"✓ Completed {len(results)} sessions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
