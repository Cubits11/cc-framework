# src/cc/exp/run_two_world.py
"""Main experiment runner for the two-world protocol (deterministic + audit-friendly)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

from ..analysis.ci import bootstrap_ci, wilson_ci

# Cartographer audit (tamper-evident chain helpers)
from ..cartographer import audit as cart_audit

# Framework imports
from ..core.attackers import GeneticAlgorithmAttacker, RandomInjectionAttacker
from ..core.logging import ChainedJSONLLogger, audit_context
from ..core.metrics import cc_max as cc_max_metric
from ..core.metrics import delta_add as delta_add_metric
from ..core.metrics import youden_j
from ..core.models import AttackResult, GuardrailSpec, WorldConfig
from ..core.protocol import TwoWorldProtocol
from ..io.storage import LocalStorageBackend, dataset_hash_from_config

logger = logging.getLogger(__name__)

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
        # Try int, then float; fallback to string
        try:
            cast = int(v)
        except Exception:
            try:
                cast = float(v)
            except Exception:
                cast = v
        _deep_set(cfg, k, cast)
    return cfg


def _stable_json(obj: Any) -> str:
    """Canonical JSON for hashing / provenance."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Determinism & Env
# --------------------------------------------------------------------------- #


def _set_determinism(seed: int) -> None:
    """
    Make the run as deterministic as feasible:
      - Python, NumPy RNG
      - Hash seed
      - Thread caps for BLAS libs (best-effort)
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    # Thread caps (avoid nondeterministic scheduling)
    for var in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


# --------------------------------------------------------------------------- #
# Factories (aligned to your YAML)
# --------------------------------------------------------------------------- #


def create_attacker(cfg: dict):
    a = cfg.get("attacker", {}) or {}
    a_type = a.get("type", "random_injection")
    a_params = a.get("params", {}) or {}

    if a_type == "random_injection":
        # RandomInjectionAttacker(vocab_harmful, vocab_benign)
        vocab_harmful = a_params.get("vocab_harmful", cfg.get("vocab_harmful", []))
        vocab_benign = a_params.get("vocab_benign", cfg.get("vocab_benign", []))
        if not vocab_harmful:
            vocab_harmful = ["attack", "hack", "exploit", "malicious", "harmful", "dangerous"]
        if not vocab_benign:
            vocab_benign = ["hello", "please", "thank", "help", "information", "question"]
        return RandomInjectionAttacker(vocab_harmful, vocab_benign)

    if a_type == "genetic_algorithm":
        # GeneticAlgorithmAttacker(vocab)
        vocab = a_params.get("vocab", []) or a_params.get("vocab_harmful", []) + a_params.get(
            "vocab_benign", []
        )
        if not vocab:
            vocab = ["attack", "hack", "exploit", "malicious", "test", "probe"]
        return GeneticAlgorithmAttacker(vocab)

    raise ValueError(f"Unknown attacker type: {a_type!r}")


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
    for g in cfg.get("guardrails") or []:
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


def _resolve_world_fprs(calibration_summary: Dict[str, Any] | None) -> Dict[int, float]:
    """Return world-indexed FPRs using calibration metadata when available."""

    if not calibration_summary:
        return {0: 0.0, 1: 0.0}

    composed_fpr = None
    for key in ("stack_fpr", "composition_fpr", "composed_fpr"):
        value = calibration_summary.get(key)
        if isinstance(value, (int, float)):
            composed_fpr = float(value)
            break

    # Base world (no guardrail stack in our two-world protocol).
    world_fprs = {0: 0.0}

    if composed_fpr is not None:
        world_fprs[1] = max(0.0, min(1.0, composed_fpr))
        return world_fprs

    guardrails = calibration_summary.get("guardrails") or []
    if not guardrails:
        return {0: 0.0, 1: 0.0}

    # Composition world uses the stack of guardrails; approximate stack FPR via
    # complementary probability assuming independence between guardrails.
    logger.warning(
        "Calibration summary missing composed FPR; using independence approximation for stack FPR."
    )
    comp_survival = 1.0
    for entry in guardrails:
        fpr = float(entry.get("fpr", 0.0))
        fpr = max(0.0, min(1.0, fpr))
        comp_survival *= 1.0 - fpr
    world_fprs[1] = 1.0 - comp_survival

    return world_fprs


def analyze_results(
    results: List[AttackResult], calibration_summary: Dict[str, Any] | None = None
) -> dict:
    """Compute success rates, operating points, and CC metrics for the two worlds."""
    w0 = np.array([r.success for r in results if r.world_bit == 0], dtype=float)
    w1 = np.array([r.success for r in results if r.world_bit == 1], dtype=float)
    if w0.size == 0 or w1.size == 0:
        raise ValueError("Need results from both worlds")

    p0_hat = float(np.mean(w0))  # attacker success w/o protection
    p1_hat = float(np.mean(w1))  # attacker success w/ protection

    tpr_world0 = max(0.0, min(1.0, 1.0 - p0_hat))
    tpr_world1 = max(0.0, min(1.0, 1.0 - p1_hat))

    world_fprs = _resolve_world_fprs(calibration_summary)
    fpr_world0 = world_fprs.get(0, 0.0)
    fpr_world1 = world_fprs.get(1, 0.0)

    j_emp = p0_hat - p1_hat  # reduction in attacker success (higher is better)

    rng = np.random.default_rng(123)
    delta_samples = []
    for _ in range(200):
        sample0 = rng.choice(w0, size=w0.size, replace=True)
        sample1 = rng.choice(w1, size=w1.size, replace=True)
        delta_samples.append(float(np.mean(sample0) - np.mean(sample1)))
    j_ci = bootstrap_ci(delta_samples, n_resamples=0, alpha=0.05)

    j_a = youden_j(tpr_world0, fpr_world0)
    j_b = youden_j(tpr_world1, fpr_world1)

    cc_max_value = cc_max_metric(j_emp, j_a, j_b)
    delta_add_value = delta_add_metric(j_emp, j_a, j_b)
    j_theory_max = max(j_a, j_b)

    metrics = {
        "J_A": j_a,
        "J_A_CI": None,
        "J_B": j_b,
        "J_B_CI": None,
        "J_comp": j_emp,
        "J_comp_CI": [j_ci[0], j_ci[1]],
        "CC_max": cc_max_value,
        "Delta_add": delta_add_value,
        "TPR_A": tpr_world0,
        "FPR_A": fpr_world0,
        "TPR_B": tpr_world1,
        "FPR_B": fpr_world1,
    }

    return {
        "sample_sizes": {
            "world_0": int(w0.size),
            "world_1": int(w1.size),
            "total": int(w0.size + w1.size),
        },
        "success_rates": {"world_0_empirical": p0_hat, "world_1_empirical": p1_hat},
        "operating_points": {
            "world_0": {"tpr": tpr_world0, "fpr": fpr_world0},
            "world_1": {"tpr": tpr_world1, "fpr": fpr_world1},
        },
        "j_statistic": {
            "empirical": j_emp,
            "theoretical_max": j_theory_max,
            "confidence_interval": {"lower": j_ci[0], "upper": j_ci[1], "method": "bootstrap"},
        },
        "composability_metrics": {"cc_max": cc_max_value, "delta_add": delta_add_value},
        "metrics_for_audit": metrics,  # exact block to drop into the audit event
    }


def _bootstrap_delta(
    arr0: np.ndarray, arr1: np.ndarray, alpha: float, seed: int = 2025
) -> Tuple[float, float, float]:
    """Bootstrap CI for difference in success rates."""
    if arr0.size == 0 or arr1.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(200):
        sample0 = rng.choice(arr0, size=arr0.size, replace=True)
        sample1 = rng.choice(arr1, size=arr1.size, replace=True)
        reps.append(float(np.mean(sample0) - np.mean(sample1)))
    ci_lo, ci_hi = bootstrap_ci(reps, n_resamples=0, alpha=alpha)
    return float(np.mean(reps)), float(ci_lo), float(ci_hi)


def _build_csv_rows(
    results: List[AttackResult], alpha_cap: float, calibration: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Construct per-prefix metrics for scan.csv."""
    rows: List[Dict[str, Any]] = []
    world_fprs = _resolve_world_fprs(calibration)
    fpr_a_default = world_fprs.get(0)
    fpr_b_default = world_fprs.get(1)
    arr_results = list(results)
    for idx in range(1, len(arr_results) + 1):
        subset = arr_results[:idx]
        w0 = np.array([1.0 if r.success else 0.0 for r in subset if r.world_bit == 0], dtype=float)
        w1 = np.array([1.0 if r.success else 0.0 for r in subset if r.world_bit == 1], dtype=float)
        row: Dict[str, Any] = {}
        row["tpr_a"] = ""
        row["tpr_b"] = ""
        row["fpr_a"] = fpr_a_default if fpr_a_default is not None else ""
        row["fpr_b"] = fpr_b_default if fpr_b_default is not None else ""
        row["I1_lo"] = ""
        row["I1_hi"] = ""
        row["I0_lo"] = ""
        row["I0_hi"] = ""
        row["vbar1"] = ""
        row["vbar0"] = ""
        row["cc_hat"] = ""
        row["ci_lo"] = ""
        row["ci_hi"] = ""
        row["ci_width"] = ""
        row["D"] = ""
        row["D_lamp"] = ""
        row["bonferroni_call"] = "hold"
        row["bhy_call"] = "hold"

        if w0.size > 0 and w1.size > 0:
            vbar0 = float(np.mean(w0))
            vbar1 = float(np.mean(w1))
            row["vbar0"] = vbar0
            row["vbar1"] = vbar1

            tpr_a = max(0.0, min(1.0, 1.0 - vbar0))
            tpr_b = max(0.0, min(1.0, 1.0 - vbar1))
            row["tpr_a"] = tpr_a
            row["tpr_b"] = tpr_b

            w1_lo, w1_hi = wilson_ci(int(np.sum(w1)), int(w1.size), alpha=alpha_cap)
            w0_lo, w0_hi = wilson_ci(int(np.sum(w0)), int(w0.size), alpha=alpha_cap)
            row["I1_lo"] = w1_lo
            row["I1_hi"] = w1_hi
            row["I0_lo"] = w0_lo
            row["I0_hi"] = w0_hi

            j_comp = vbar0 - vbar1
            fpr_a = row["fpr_a"] if row["fpr_a"] != "" else 0.0
            fpr_b = row["fpr_b"] if row["fpr_b"] != "" else 0.0
            j_a = youden_j(tpr_a, fpr_a)
            j_b = youden_j(tpr_b, fpr_b)
            row["cc_hat"] = cc_max_metric(j_comp, j_a, j_b)
            _, ci_lo, ci_hi = _bootstrap_delta(w0, w1, alpha=alpha_cap, seed=idx + 1000)
            if not np.isnan(ci_lo) and not np.isnan(ci_hi):
                row["ci_lo"] = ci_lo
                row["ci_hi"] = ci_hi
                row["ci_width"] = ci_hi - ci_lo
                delta_value = delta_add_metric(j_comp, j_a, j_b)
                row["D"] = delta_value
                row["D_lamp"] = delta_value
                if ci_lo > 0:
                    row["bonferroni_call"] = "reject"
                    row["bhy_call"] = "reject"
                else:
                    row["bonferroni_call"] = "hold"
                    row["bhy_call"] = "hold"

        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Provenance & audit helpers
# --------------------------------------------------------------------------- #


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(repo_root)
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _repo_root() -> Path:
    # repo_root ≈ project root (…/src/cc/exp/run_two_world.py → repo_root=parents[3])
    return Path(__file__).resolve().parents[3]


def _audit_cfg(full_cfg: dict, n_sessions: int, seed: int | None) -> dict:
    exp = full_cfg.get("experiment") or {}
    proto = full_cfg.get("protocol") or {}
    return {
        "A": "A",
        "B": "B",
        "comp": "AND",
        "epsilon": proto.get("epsilon"),
        "T": proto.get("T"),
        "samples": exp.get("n_sessions", n_sessions),
        "seed": exp.get("seed", seed),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run two-world experiment (deterministic, audit-logged)"
    )
    ap.add_argument("--config", required=True, help="YAML config file")
    ap.add_argument("--n", type=int, help="Override number of sessions")
    ap.add_argument("--audit", help="Audit JSONL path (default: runs/audit.jsonl)")
    ap.add_argument("--log", help="Deprecated alias for --audit")
    ap.add_argument("--output", default="results/analysis.json", help="Write analysis JSON here")
    ap.add_argument("--out-csv", help="Write per-session metrics CSV here")
    ap.add_argument("--calibration-summary", help="Optional JSON file from calibration script")
    ap.add_argument("--seed", type=int, help="Override global seed")
    ap.add_argument(
        "--set", nargs="*", default=[], help="Overrides: key=val (e.g., protocol.epsilon=0.02)"
    )
    ap.add_argument("--experiment-id", default=None, help="Optional tag for this run")
    args = ap.parse_args()

    calibration_summary: Dict[str, Any] = {}
    if args.calibration_summary:
        calib_path = Path(args.calibration_summary)
        if calib_path.exists():
            with open(calib_path, "r", encoding="utf-8") as f:
                calibration_summary = json.load(f)

    # Load & override config
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.set:
        cfg = apply_overrides(cfg, args.set)

    # Determinism + paths
    seed = int(cfg.get("seed", 42))
    _set_determinism(seed)

    if calibration_summary:
        summary_map = {}
        for entry in calibration_summary.get("guardrails", []):
            name = entry.get("name")
            if name:
                summary_map[name] = entry
        for guardrail_cfg in cfg.get("guardrails", []) or []:
            name = guardrail_cfg.get("name")
            if name in summary_map:
                params = guardrail_cfg.setdefault("params", {})
                if "threshold" in summary_map[name]:
                    params["threshold"] = summary_map[name]["threshold"]

    exp_cfg = cfg.get("experiment", {}) or {}
    n_sessions = int(args.n if args.n is not None else exp_cfg.get("n_sessions", 200))

    out_path = Path(args.output)
    audit_arg = args.audit or args.log
    if audit_arg is None:
        audit_arg = "runs/audit.jsonl"
    log_path = Path(audit_arg)
    out_csv = Path(args.out_csv) if args.out_csv else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Provenance
    repo_root = _repo_root()
    git_sha = _git_commit(repo_root)
    cfg_stable = _stable_json(cfg)
    cfg_sha = _sha256_hex(cfg_stable)
    dataset_sha = dataset_hash_from_config(cfg, base_dir=repo_root)
    exp_id = args.experiment_id or f"exp_{int(time.time())}"
    storage = LocalStorageBackend(base_dir=repo_root)
    started_at = time.time()

    # Logger
    logger = ChainedJSONLLogger(str(log_path))

    # Build components
    attacker = create_attacker(cfg)
    worlds = create_world_configs(cfg)
    protocol = TwoWorldProtocol(
        logger=logger,
        base_success_rate=cfg.get("baseline_success_rate", 0.6),
        episode_length=cfg.get("episode_length", 10),
        random_seed=seed,
    )

    # Audited operation
    with audit_context(
        logger,
        "two_world_run",
        exp_id=exp_id,
        config_file=args.config,
        config_sha256=cfg_sha,
        git_commit=git_sha,
        n_sessions=n_sessions,
        seed=seed,
        calibration_summary=calibration_summary,
    ):
        print(f"Starting experiment: n_sessions={n_sessions} (exp_id={exp_id})")

        # Run
        results: List[AttackResult] = protocol.run_experiment(
            attacker=attacker,
            world_configs=worlds,
            n_sessions=n_sessions,
            experiment_id=exp_id,
        )
        print(f"Completed {len(results)} sessions")

        # Analyze
        analysis = analyze_results(results, calibration_summary)
        metrics_for_audit = analysis["metrics_for_audit"]

        # Persist human-readable analysis
        analysis_out = {
            "metadata": {
                "experiment_id": exp_id,
                "config_file": args.config,
                "config_sha256": cfg_sha,
                "dataset_sha256": dataset_sha,
                "n_sessions_requested": n_sessions,
                "n_sessions_completed": len(results),
                "attacker_type": type(attacker).__name__,
                "timestamp_unix": time.time(),
                "git_commit": git_sha,
                "configuration": cfg,  # full (as run)
                "calibration_summary": calibration_summary,
            },
            "results": analysis,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(analysis_out, f, indent=2)

        # CSV metrics for Week 5 pilot
        if out_csv:
            rows = _build_csv_rows(results, cfg.get("alpha_cap", 0.05), calibration_summary)
            header = [
                "tpr_a",
                "tpr_b",
                "fpr_a",
                "fpr_b",
                "I1_lo",
                "I1_hi",
                "I0_lo",
                "I0_hi",
                "vbar1",
                "vbar0",
                "cc_hat",
                "ci_lo",
                "ci_hi",
                "ci_width",
                "D",
                "D_lamp",
                "bonferroni_call",
                "bhy_call",
            ]
            with open(out_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=header)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k, "") for k in header})

        # Cartographer audit record (tamper-evident)
        decision = (
            f"DECISION: {'DEPLOY' if metrics_for_audit['CC_max'] >= 1.0 else 'REDESIGN'} "
            f"(CC_max={metrics_for_audit['CC_max']:.3f})"
        )

        # Build the cfg slice expected by cart_audit.make_record (fields it uses)
        cfg_slice = _audit_cfg(cfg, n_sessions, seed)
        cfg_full = {
            "guardrails": cfg.get("guardrails", []),
            "attacker": cfg.get("attacker", {}),
            "calibration_summary": calibration_summary,
            "config_hashes": {
                "config_sha256": cfg_sha,
                "config_sha256_source": "stable_json",
            },
        }
        # Append to the same chain file; figures left empty (downstream scripts may add)
        cart_audit.append_record(
            path=str(log_path),
            cfg=cfg_slice,
            cfg_full=cfg_full,
            j_a=float(metrics_for_audit["J_A"]),
            j_a_ci=None
            if metrics_for_audit.get("J_A_CI") in (None, [], ())
            else tuple(metrics_for_audit["J_A_CI"]),  # type: ignore
            j_b=float(metrics_for_audit["J_B"]),
            j_b_ci=None
            if metrics_for_audit.get("J_B_CI") in (None, [], ())
            else tuple(metrics_for_audit["J_B_CI"]),  # type: ignore
            j_comp=float(metrics_for_audit["J_comp"]),
            j_comp_ci=None
            if metrics_for_audit.get("J_comp_CI") in (None, [], ())
            else tuple(metrics_for_audit["J_comp_CI"]),  # type: ignore
            cc_max=float(metrics_for_audit["CC_max"]),
            delta_add=float(metrics_for_audit["Delta_add"]),
            decision=decision,
            figures=[],
            fsync=True,
        )

    completed_at = time.time()
    manifest = {
        "experiment_id": exp_id,
        "config_file": args.config,
        "config_sha256": cfg_sha,
        "dataset_sha256": dataset_sha,
        "git_sha": git_sha,
        "started_at_unix": started_at,
        "completed_at_unix": completed_at,
        "artifacts": {
            "analysis_json": str(out_path),
            "audit_log": str(log_path),
            "metrics_csv": str(out_csv) if out_csv else None,
        },
    }
    manifest_path = storage.save_json(manifest, category="runs", filename="manifest.json")

    # Console summary
    j = analysis["j_statistic"]
    print("\n=== SUMMARY ===")
    print(f"Git SHA       : {git_sha}")
    print(f"Config SHA256 : {cfg_sha[:12]}…")
    print(
        f"J (empirical) : {j['empirical']:.4f}  "
        f"[{j['confidence_interval']['lower']:.4f}, {j['confidence_interval']['upper']:.4f}]  ({j['confidence_interval']['method']})"
    )
    print(f"CC_max        : {metrics_for_audit['CC_max']:.3f}")
    print(f"Wrote analysis → {out_path}")
    print(f"Appended audit → {log_path}")
    print(f"Wrote manifest → {manifest_path}")
    if out_csv:
        print(f"Wrote scan CSV → {out_csv}")


if __name__ == "__main__":
    main()
