# src/cc/exp/run_two_world.py
"""Main experiment runner for two-world protocol"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from ..core.attackers import GeneticAlgorithmAttacker, RandomInjectionAttacker
from ..core.logging import ChainedJSONLLogger
from ..core.models import AttackResult, GuardrailSpec, WorldConfig
from ..core.protocol import TwoWorldProtocol
from ..core.stats import bootstrap_ci_j_statistic


def load_config(config_path: str) -> dict:
    """Load experiment configuration from YAML"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_attacker(config: dict):
    """Create attacker from configuration"""
    attacker_config = config.get("attacker", {})
    attacker_type = attacker_config.get("type", "random_injection")

    if attacker_type == "random_injection":
        return RandomInjectionAttacker(
            vocab_harmful=config.get("vocab_harmful", []),
            vocab_benign=config.get("vocab_benign", []),
            base_success=attacker_config.get("base_success", 0.6),
            seed=config.get("seed", 42),
        )
    elif attacker_type == "genetic_algorithm":
        vocab = config.get("vocab_harmful", []) + config.get("vocab_benign", [])
        return GeneticAlgorithmAttacker(
            vocab=vocab,
            population_size=attacker_config.get("population_size", 50),
            mutation_rate=attacker_config.get("mutation_rate", 0.1),
            seed=config.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown attacker type: {attacker_type}")


def create_world_configs(config: dict) -> Dict[int, WorldConfig]:
    """Create world configurations from config"""
    worlds = {}

    # World 0: No protection (baseline)
    worlds[0] = WorldConfig(
        world_id=0,
        guardrail_stack=[],  # No guardrails
        utility_profile={},
        baseline_success_rate=config.get("baseline_success_rate", 0.6),
    )

    # World 1: With guardrails
    guardrail_specs = []
    for g_config in config.get("guardrails", []):
        spec = GuardrailSpec(
            name=g_config["name"],
            params=g_config.get("params", {}),
            calibration_fpr_target=g_config.get("fpr_target", 0.05),
        )
        guardrail_specs.append(spec)

    worlds[1] = WorldConfig(
        world_id=1,
        guardrail_stack=guardrail_specs,
        utility_profile={},
        baseline_success_rate=config.get("baseline_success_rate", 0.6),
    )

    return worlds


def analyze_results(results: List[AttackResult]) -> dict:
    """Analyze experimental results and compute CC metrics"""
    # Separate by world
    world_0_results = [r.success for r in results if r.world_bit == 0]
    world_1_results = [r.success for r in results if r.world_bit == 1]

    if not world_0_results or not world_1_results:
        raise ValueError("Need results from both worlds")

    # Convert to numpy arrays
    w0_array = np.array(world_0_results, dtype=float)
    w1_array = np.array(world_1_results, dtype=float)

    # Compute empirical success rates
    p0_hat = np.mean(w0_array)  # Success rate without protection
    p1_hat = np.mean(w1_array)  # Success rate with protection

    # Compute J-statistic (protection effectiveness)
    j_empirical = p0_hat - p1_hat

    # Bootstrap confidence interval
    bootstrap_result = bootstrap_ci_j_statistic(w0_array, w1_array, B=2000, alpha=0.05)

    # For single guardrail analysis, CC_max = J_empirical / J_max
    # where J_max = p0_hat (theoretical maximum)
    j_theoretical_max = p0_hat
    cc_max = j_empirical / j_theoretical_max if j_theoretical_max > 0 else 0.0

    # Compose results
    analysis = {
        "sample_sizes": {
            "world_0": len(world_0_results),
            "world_1": len(world_1_results),
            "total": len(results),
        },
        "success_rates": {"world_0_empirical": float(p0_hat), "world_1_empirical": float(p1_hat)},
        "j_statistic": {
            "empirical": float(j_empirical),
            "theoretical_max": float(j_theoretical_max),
            "confidence_interval": {
                "lower": float(bootstrap_result.ci_lower),
                "upper": float(bootstrap_result.ci_upper),
                "width": float(bootstrap_result.ci_width),
                "method": bootstrap_result.method,
            },
        },
        "composability_metrics": {
            "cc_max": float(cc_max),
            "delta_add": float(j_empirical - j_theoretical_max),
            "protection_effectiveness": float(j_empirical / p0_hat) if p0_hat > 0 else 0.0,
        },
        "statistical_tests": {
            "j_significantly_positive": bootstrap_result.ci_lower > 0,
            "j_significantly_negative": bootstrap_result.ci_upper < 0,
            "effect_size_category": _categorize_effect_size(abs(j_empirical)),
        },
    }

    return analysis


def _categorize_effect_size(effect_size: float) -> str:
    """Categorize effect size using Cohen's conventions"""
    if effect_size < 0.01:
        return "negligible"
    elif effect_size < 0.05:
        return "small"
    elif effect_size < 0.15:
        return "medium"
    else:
        return "large"


def main():
    """Main experiment runner"""
    parser = argparse.ArgumentParser(description="Run CC Framework two-world experiment")
    parser.add_argument("--config", type=str, required=True, help="Configuration YAML file")
    parser.add_argument("--n", type=int, default=5000, help="Number of sessions")
    parser.add_argument("--log", type=str, default="logs/experiment.jsonl", help="Log file path")
    parser.add_argument(
        "--output", type=str, default="results/analysis.json", help="Output analysis file"
    )
    parser.add_argument("--seed", type=int, help="Override random seed")

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Override seed if provided
    if args.seed is not None:
        config["seed"] = args.seed

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = ChainedJSONLLogger(args.log)

    # Create attacker and world configurations
    try:
        attacker = create_attacker(config)
        world_configs = create_world_configs(config)
        print(f"Created {type(attacker).__name__} attacker")
        print(f"World 0: {len(world_configs[0].guardrail_stack)} guardrails")
        print(f"World 1: {len(world_configs[1].guardrail_stack)} guardrails")
    except Exception as e:
        print(f"Error creating experiment components: {e}")
        sys.exit(1)

    # Run experiment
    protocol = TwoWorldProtocol(
        logger=logger,
        base_success_rate=config.get("baseline_success_rate", 0.6),
        episode_length=config.get("episode_length", 10),
        random_seed=config.get("seed", 42),
    )

    print(f"Starting experiment with {args.n} sessions...")

    try:
        results = protocol.run_experiment(
            attacker=attacker,
            world_configs=world_configs,
            n_sessions=args.n,
            experiment_id=f"exp_{int(time.time())}",
        )

        print(f"Completed {len(results)} sessions")

        # Analyze results
        analysis = analyze_results(results)

        # Add metadata
        analysis["metadata"] = {
            "config_file": args.config,
            "n_sessions_requested": args.n,
            "n_sessions_completed": len(results),
            "attacker_type": type(attacker).__name__,
            "timestamp": time.time(),
            "git_commit": _get_git_commit(),
            "configuration": config,
        }

        # Save analysis
        with open(args.output, "w") as f:
            json.dump(analysis, f, indent=2)

        print(f"Analysis saved to {args.output}")

        # Print summary
        print("\n=== EXPERIMENT SUMMARY ===")
        print(
            f"J-statistic: {analysis['j_statistic']['empirical']:.4f} "
            f"(95% CI: {analysis['j_statistic']['confidence_interval']['lower']:.4f}, "
            f"{analysis['j_statistic']['confidence_interval']['upper']:.4f})"
        )
        print(f"CC_max: {analysis['composability_metrics']['cc_max']:.4f}")
        print(f"Effect size: {analysis['statistical_tests']['effect_size_category']}")

        if analysis["statistical_tests"]["j_significantly_positive"]:
            print("✓ Guardrails show statistically significant protection")
        elif analysis["statistical_tests"]["j_significantly_negative"]:
            print("⚠ Guardrails show statistically significant vulnerability (unexpected)")
        else:
            print("- No statistically significant protection effect")

    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _get_git_commit() -> str:
    """Get current git commit hash"""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except:
        return "unknown"


if __name__ == "__main__":
    import time

    main()
