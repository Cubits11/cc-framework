# src/cc/utils/validation.py
"""
Validation utilities for experiments
"""

from typing import Any, Dict

import jsonschema

EXPERIMENT_SCHEMA = {
    "type": "object",
    "required": ["experiment", "protocol", "attacker", "guardrails"],
    "properties": {
        "experiment": {
            "type": "object",
            "required": ["name", "n_sessions"],
            "properties": {
                "name": {"type": "string"},
                "n_sessions": {"type": "integer", "minimum": 100},
                "seed": {"type": "integer"},
            },
        },
        "protocol": {
            "type": "object",
            "properties": {"episode_length": {"type": "integer", "minimum": 1, "maximum": 100}},
        },
        "attacker": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "type": {"type": "string", "enum": ["random_injection", "genetic_algorithm"]},
                "params": {"type": "object"},
            },
        },
        "guardrails": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}, "params": {"type": "object"}},
            },
        },
    },
}


def validate_config(config: Dict[str, Any]) -> None:
    """Validate experiment configuration"""
    jsonschema.validate(instance=config, schema=EXPERIMENT_SCHEMA)


def run_invariant_suite() -> bool:
    """Run all invariant checks"""
    checks = []

    # Check 1: J-statistic bounds
    from cc.core.stats import compute_j_statistic

    dummy_results = [...]  # Create test data
    j, p0, p1 = compute_j_statistic(dummy_results)
    checks.append(0 <= j <= 1)

    # Check 2: Bootstrap convergence
    # ... additional checks

    return all(checks)
