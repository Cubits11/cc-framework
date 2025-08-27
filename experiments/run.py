# experiments/run.py - FIXED CONFIG LOADER
import argparse
import yaml
import sys
from pathlib import Path
from typing import Dict, Any
import json

def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration has required fields"""
    required = ["experiment", "protocol", "worlds", "attacker"]
    missing = [k for k in required if k not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")
    
    # Ensure we have both worlds
    if "0" not in config["worlds"] or "1" not in config["worlds"]:
        raise ValueError("Config must define worlds 0 and 1")

def load_config(path: str) -> Dict[str, Any]:
    """Load and validate YAML config"""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Empty or invalid YAML file: {path}")
    
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config)}")
    
    validate_config(config)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config YAML')
    parser.add_argument('--n', type=int, help='Override n_sessions')
    parser.add_argument('--seed', type=int, help='Override seed')
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"ERROR: Failed to load config: {e}", file=sys.stderr)
        return 1
    
    # Apply overrides
    if args.n:
        config['experiment']['n_sessions'] = args.n
    if args.seed:
        config['experiment']['seed'] = args.seed
    
    # Delegate to two-world runner
    from cc.exp.run_two_world import run_experiment
    
    print(f"Starting experiment: {config['experiment']['name']}")
    print(f"Sessions: {config['experiment']['n_sessions']}")
    
    results = run_experiment(config)
    
    print(f"✓ Completed {len(results)} sessions")
    return 0

if __name__ == "__main__":
    sys.exit(main())
