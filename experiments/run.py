"""
Simple entrypoint to run experiments from YAML configs while delegating to
the existing runner at src/cc/exp/run_two_world.py. Keeps backwards compatibility.
"""

import argparse
import importlib
import sys

import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--n", type=int, default=None, help="Override n_sessions")
    p.add_argument("--results-dir", default=None, help="Override results dir")
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides (optional)
    if args.n is not None:
        cfg.setdefault("experiment", {})["n_sessions"] = args.n
    if args.results_dir is not None:
        cfg.setdefault("experiment", {})["results_dir"] = args.results_dir
    if args.seed is not None:
        cfg.setdefault("experiment", {})["seed"] = args.seed

    # Delegate to your existing runner
    mod = importlib.import_module("cc.exp.run_two_world")
    if not hasattr(mod, "main"):
        print("Expected cc.exp.run_two_world.main(config_dict) to exist.", file=sys.stderr)
        sys.exit(2)
    return mod.main(cfg)


if __name__ == "__main__":
    sys.exit(main())
