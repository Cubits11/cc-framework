# src/cc/exp/run_two_world.py
"""Main experiment runner for two-world protocol (audit-friendly)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from ..core.attackers import GeneticAlgorithmAttacker, RandomInjectionAttacker
from ..core.logging import ChainedJSONLLogger
from ..core.models import AttackResult, GuardrailSpec, WorldConfig
from ..core.protocol import TwoWorldProtocol
from ..core.stats import bootstrap_ci_j_statistic


# --------------------------------------------------------------------------- #
# Config utilities
# --------------------------------------------------------------------------- #

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_set(cfg: dict, dotted: str, value: Any) -> None:
    """Set cfg['a']['b']... = value for key 'a.b....' creating maps as needed."""
    cur = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def apply_overrides(cfg: dict, kvs: List[str]) -> dict:
    """--set a.b=1 c.d=0.1    (int/float autodetected, else str)"""
    for kv in kvs:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        cast = v
        for caster in (int, float):
            try:
                cast = caster(v)
                break
            except Exception:
                pass
        _deep_set(cfg, k, cast)
    return cfg


# --------------------------------------------------------------------------- #
# Factory helpers (aligned to your YAMLs) - FIXED
# --------------------------------------------------------------------------- #

def create_attacker(cfg: dict):
    a = cfg.get("attacker", {}) or {}
    a_type = a.get("type", "random_injection")
    a_params = a.get("params", {}) or {}

    if a_type == "random_injection":
        # RandomInjectionAttacker requires vocab_harmful and vocab_benign as positional arguments
        vocab_harmful = a_params.get("vocab_harmful", cfg.get("vocab_harmful", []))
        vocab_benign = a_params.get("vocab_benign", cfg.get("vocab_benign", []))
        
        # Provide defaults if empty
        if not vocab_harmful:
            vocab_harmful = ["attack", "hack", "exploit", "malicious", "harmful", "dangerous"]
        if not vocab_benign:
            vocab_benign = ["hello", "please", "thank", "help", "information", "question"]
        
        # Create with required positional arguments
        return RandomInjectionAttacker(vocab_harmful, vocab_benign)
            
    elif a_type == "genetic_algorithm":
        vocab = (a_params.get("vocab", []) or
                 a_params.get("vocab_harmful", []) + a_params.get("vocab_benign", []))
        
        # Provide default vocab if empty
        if not vocab:
            vocab = ["attack", "hack", "exploit", "malicious", "test", "probe"]
        
        # GeneticAlgorithmAttacker takes vocab as first positional argument
        return GeneticAlgorithmAttacker(vocab)
    else:
        raise ValueError(f"Unknown attacker type: {a_type}")


def create_world_configs(cfg: dict) -> Dict[int, WorldConfig]:
    worlds: Dict[int, WorldConfig] = {}

    # World 0 (baseline)
    worlds[0] = WorldConfig(
        world_id=0,
        guardrail_stack=[],
        utility_profile={},
        baseline_success_rate=cfg.get("baseline_success_rate", 0.6),
    )

    # World 1 (guardrails)
    guardrail_specs: List[GuardrailSpec] = []
    for g in (cfg.get("guardrails") or []):
        guardrail_specs.append(
            GuardrailSpec(
                name=g["name"],
                params=g.get("params", {}) or {},
                calibration_fpr_target=g.get("fpr_target", 0.05),
            )
        )

    worlds[1] = WorldConfig(
        world_id=1,
        guardrail_stack=guardrail_specs,
        utility_profile={},
        baseline_success_rate=cfg.get("baseline_success_rate", 0.6),
    )
    return worlds


# --------------------------------------------------------------------------- #
# Analysis (maps to your memo/plots naming)
# --------------------------------------------------------------------------- #

def analyze_results(results: List[AttackResult]) -> dict:
    """Compute success rates, J, CC_max, bootstrap CI, and tags."""
    w0 = np.array([r.success for r in results if r.world_bit == 0], dtype=float)
    w1 = np.array([r.success for r in results if r.world_bit == 1], dtype=float)
    if w0.size == 0 or w1.size == 0:
        raise ValueError("Need results from both worlds")

    p0_hat = float(np.mean(w0))  # attacker success w/o protection
    p1_hat = float(np.mean(w1))  # attacker success w protection
    j_emp = p0_hat - p1_hat

    boot = bootstrap_ci_j_statistic(w0, w1, B=2000, alpha=0.05)
    j_ci = (float(boot.ci_lower), float(boot.ci_upper))

    j_theory_max = p0_hat if p0_hat > 0 else 1e-9
    cc_max = j_emp / j_theory_max

    # Map into the labels you've been using in memos / analysis
    metrics = {
        "CC_max": float(cc_max),
        "Delta_add": float(j_emp - j_theory_max),
        "J_A": float(p0_hat),                      # memo uses J_A as "A" effectiveness proxy
        "J_A_CI": [float(j_ci[0] + p1_hat), float(j_ci[1] + p1_hat)],  # loose placeholder
        "J_B": float(1.0 - p1_hat),                # illustrative; keep if you've been using it
        "J_B_CI": [],                              # not estimated here
        "J_comp": float(j_emp),
        "J_comp_CI": [float(j_ci[0]), float(j_ci[1])],
    }

    return {
        "sample_sizes": {"world_0": w0.size, "world_1": w1.size, "total": w0.size + w1.size},
        "success_rates": {"world_0_empirical": p0_hat, "world_1_empirical": p1_hat},
        "j_statistic": {
            "empirical": float(j_emp),
            "theoretical_max": float(j_theory_max),
            "confidence_interval": {
                "lower": float(j_ci[0]),
                "upper": float(j_ci[1]),
                "method": boot.method,
            },
        },
        "composability_metrics": {"cc_max": float(cc_max), "delta_add": float(j_emp - j_theory_max)},
        "metrics_for_audit": metrics,  # exact block we'll drop into the audit event
    }


# --------------------------------------------------------------------------- #
# Audit helpers
# --------------------------------------------------------------------------- #

def _git_commit() -> str:
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parents[3]
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _audit_cfg(full_cfg: dict, n_sessions: int) -> dict:
    exp = (full_cfg.get("experiment") or {})
    proto = (full_cfg.get("protocol") or {})
    return {
        "A": "A",
        "B": "B",
        "comp": "AND",
        "epsilon": proto.get("epsilon"),
        "T": proto.get("T"),
        "samples": exp.get("n_sessions", n_sessions),
        "seed": exp.get("seed", full_cfg.get("seed")),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Run two-world experiment (with audit logging)")
    ap.add_argument("--config", required=True, help="YAML config file")
    ap.add_argument("--n", type=int, help="Override number of sessions")
    ap.add_argument("--log", default="runs/audit.jsonl", help="Audit JSONL path")
    ap.add_argument("--output", default="results/analysis.json", help="Write analysis JSON here")
    ap.add_argument("--seed", type=int, help="Override global seed")
    ap.add_argument("--set", nargs="*", default=[], help="Overrides: key=val (e.g., protocol.epsilon=0.02)")
    args = ap.parse_args()

    # Load & override config
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.set:
        cfg = apply_overrides(cfg, args.set)

    exp_cfg = cfg.get("experiment", {}) or {}
    n_sessions = int(args.n if args.n is not None else exp_cfg.get("n_sessions", 200))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    logger = ChainedJSONLLogger(args.log)

    # Build components
    attacker = create_attacker(cfg)
    worlds = create_world_configs(cfg)
    protocol = TwoWorldProtocol(
        logger=logger,
        base_success_rate=cfg.get("baseline_success_rate", 0.6),
        episode_length=cfg.get("episode_length", 10),
        random_seed=cfg.get("seed", 42),
    )

    print(f"Starting experiment: n_sessions={n_sessions}")

    # Run & analyze
    results: List[AttackResult] = protocol.run_experiment(
        attacker=attacker,
        world_configs=worlds,
        n_sessions=n_sessions,
        experiment_id=f"exp_{int(time.time())}",
    )
    print(f"Completed {len(results)} sessions")

    analysis = analyze_results(results)
    metrics_for_audit = analysis["metrics_for_audit"]

    # Persist human-readable analysis too
    analysis_out = {
        "metadata": {
            "config_file": args.config,
            "n_sessions_requested": n_sessions,
            "n_sessions_completed": len(results),
            "attacker_type": type(attacker).__name__,
            "timestamp": time.time(),
            "git_commit": _git_commit(),
            "configuration": cfg,
        },
        "results": analysis,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(analysis_out, f, indent=2)

    # ---------------- Audit event (hash-chained) ---------------- #
    event = {
        "cfg": _audit_cfg(cfg, n_sessions),
        "decision": f"DECISION: {('DEPLOY' if metrics_for_audit['CC_max'] >= 1.0 else 'REDESIGN')} "
                    f"— reason: CC_max={metrics_for_audit['CC_max']:.2f}",
        "figures": [],  # downstream scripts add figures; keep placeholder
        "meta": {"schema": "cartographer/audit.v1", "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        "metrics": metrics_for_audit,
    }
    logger.log_event(event)

    # Console summary
    j = analysis["j_statistic"]
    print("\n=== SUMMARY ===")
    print(f"J (empirical) = {j['empirical']:.4f} "
          f"[{j['confidence_interval']['lower']:.4f}, {j['confidence_interval']['upper']:.4f}]")
    print(f"CC_max = {metrics_for_audit['CC_max']:.3f}")
    print(f"Wrote analysis → {args.output}")
    print(f"Appended audit → {args.log}")


if __name__ == "__main__":
    main()