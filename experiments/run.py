# experiments/run.py - Enhanced delegation
"""
Main experiment runner with proper delegation to two-world protocol
"""
import argparse
import yaml
import sys
from pathlib import Path
from typing import Dict, Any, Optional

def main():
    parser = argparse.ArgumentParser(description='CC Framework experiment runner')
    parser.add_argument('--config', required=True, help='YAML configuration file')
    parser.add_argument('--n', type=int, help='Override number of sessions')
    parser.add_argument('--seed', type=int, help='Override random seed')
    parser.add_argument('--results-dir', default='results', help='Results directory')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Load and merge configurations
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.n is not None:
        config['experiment']['n_sessions'] = args.n
    if args.seed is not None:
        config['protocol']['seed'] = args.seed
    
    # Delegate to two-world runner with proper config structure
    from cc.exp.run_two_world import run_experiment
    
    try:
        results = run_experiment(
            config=config,
            output_dir=Path(args.results_dir),
            verbose=args.log_level == 'DEBUG'
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Experiment Complete: {config['experiment']['name']}")
        print(f"{'='*60}")
        print(f"Sessions completed: {results['n_sessions']}")
        print(f"J-statistic: {results['j_statistic']:.4f}")
        print(f"CC_max: {results['cc_max']:.4f}")
        print(f"95% CI width: {results['ci_width']:.4f}")
        
        if results['cc_max'] < 0.95:
            print("✓ Constructive interaction detected")
        elif results['cc_max'] > 1.05:
            print("⚠ Destructive interaction detected")
        else:
            print("= Independent operation")
            
        return 0
        
    except Exception as e:
        print(f"Experiment failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

def load_config(path: str) -> Dict[str, Any]:
    """Load and validate configuration"""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Ensure required sections exist
    required_sections = ['experiment', 'protocol', 'attacker', 'guardrails']
    for section in required_sections:
        if section not in config:
            config[section] = {}
    
    # Set defaults
    config['experiment'].setdefault('n_sessions', 5000)
    config['protocol'].setdefault('seed', 1337)
    config['protocol'].setdefault('episode_length', 10)
    
    return config

if __name__ == "__main__":
    sys.exit(main())
