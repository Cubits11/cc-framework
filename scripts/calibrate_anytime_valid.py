#!/usr/bin/env python3
"""Calibrate the anytime-valid sequential test and compare legacy behavior.

Default outputs:

  docs/theory/figures/anytime_valid_calibration_report.json
  docs/theory/figures/anytime_valid_power_curve.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from cc.core.models import AttackResult
from cc.core.protocol import BayesianSequentialTester
from cc.kernel.sequential import (
    AnytimeBernoulliTester,
    calibrate_false_stop_rate,
    independent_joint_miss_baseline,
    simulate_power_curve,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--null-rate", type=float, default=0.10)
    parser.add_argument("--null-trials", type=int, default=5_000)
    parser.add_argument("--power-trials", type=int, default=1_000)
    parser.add_argument("--n-max", type=int, default=1_000)
    parser.add_argument(
        "--true-rates",
        type=float,
        nargs="*",
        default=[0.12, 0.15, 0.20, 0.30],
        help="Alternative miss rates for power simulation.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/examples/rails_tiny.csv"),
        help="Existing two-rail example CSV for side-by-side comparison.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("docs/theory/figures/anytime_valid_calibration_report.json"),
    )
    parser.add_argument(
        "--power-csv",
        type=Path,
        default=Path("docs/theory/figures/anytime_valid_power_curve.csv"),
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    calibration = calibrate_false_stop_rate(
        rng=rng,
        null_rate=args.null_rate,
        alpha=args.alpha,
        n_trials=args.null_trials,
        n_max=args.n_max,
    )
    power = simulate_power_curve(
        rng=rng,
        null_rate=args.null_rate,
        true_rates=np.asarray(args.true_rates, dtype=np.float64),
        alpha=args.alpha,
        n_trials=args.power_trials,
        n_max=args.n_max,
    )
    side_by_side = _side_by_side_report(args.dataset, alpha=args.alpha, seed=args.seed)

    report: dict[str, Any] = {
        "seed": args.seed,
        "null_calibration": asdict(calibration),
        "power": [asdict(row) for row in power],
        "side_by_side_existing_dataset": side_by_side,
        "false_stop_check": {
            "criterion": "false_stop_rate <= alpha + 2 * monte_carlo_se",
            "passed": bool(calibration.within_two_se),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    _write_power_csv(args.power_csv, power)

    if not calibration.within_two_se:
        raise SystemExit(
            "False-stop calibration exceeded alpha within two Monte Carlo SEs. "
            f"rate={calibration.false_stop_rate:.4f}, alpha={calibration.alpha:.4f}, "
            f"mc_se={calibration.monte_carlo_se:.4f}"
        )

    print(f"wrote {args.report}")
    print(f"wrote {args.power_csv}")
    print(
        "false-stop rate "
        f"{calibration.false_stop_rate:.4f} at alpha={calibration.alpha:.4f} "
        f"(MC SE={calibration.monte_carlo_se:.4f})"
    )


def _write_power_csv(path: Path, rows: tuple[Any, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "true_rate",
        "effect_size",
        "n_trials",
        "n_max",
        "stop_probability",
        "expected_stop_time",
        "median_stop_time",
        "fixed_sample_n_approx",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _side_by_side_report(path: Path, *, alpha: float, seed: int) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}

    rows = _read_csv(path)
    harmful = [row for row in rows if int(float(row["label"])) == 1]
    if not harmful:
        return {"status": "no_harmful_rows", "path": str(path)}

    threshold = 0.5
    rail_a_miss = [float(row["rail_a_score"]) < threshold for row in harmful]
    rail_b_miss = [float(row["rail_b_score"]) < threshold for row in harmful]
    composed_miss = [a and b for a, b in zip(rail_a_miss, rail_b_miss, strict=True)]
    p_a = float(np.mean(rail_a_miss))
    p_b = float(np.mean(rail_b_miss))
    p0 = independent_joint_miss_baseline(p_a, p_b)
    p0 = min(max(p0, 1.0e-6), 1.0 - 1.0e-6)

    safe = AnytimeBernoulliTester(null_rate=p0, alpha=alpha)
    safe_result = safe.update_many(np.asarray(composed_miss, dtype=int))

    legacy_results: list[AttackResult] = []
    legacy_stop_at: int | None = None
    rng = np.random.default_rng(seed + 1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        legacy = BayesianSequentialTester(min_n=20, posterior_samples=2_000, rng=rng)
        for idx, (a_miss, comp_miss) in enumerate(
            zip(rail_a_miss, composed_miss, strict=True),
            start=1,
        ):
            legacy_results.append(_attack_result(world_bit=0, success=a_miss, idx=2 * idx - 1))
            legacy_results.append(_attack_result(world_bit=1, success=comp_miss, idx=2 * idx))
            result = legacy.should_stop_early(legacy_results)
            if result.should_stop and legacy_stop_at is None:
                legacy_stop_at = idx
                break
        final_legacy = legacy.should_stop_early(legacy_results)

    return {
        "status": "ok",
        "path": str(path),
        "n_harmful": len(harmful),
        "threshold": threshold,
        "rail_a_miss_rate": p_a,
        "rail_b_miss_rate": p_b,
        "independent_joint_miss_null_rate": p0,
        "observed_composed_miss_rate": float(np.mean(composed_miss)),
        "new_anytime_valid": {
            "stopped": safe_result.stopped,
            "stop_time_harmful_rows": safe_result.stop_time,
            "e_value": safe_result.e_value,
            "threshold": safe_result.threshold,
            "decision": safe_result.decision,
        },
        "legacy_bayesian_heuristic": {
            "warning": "unvalidated legacy heuristic; not anytime-valid",
            "runtime_warnings": [str(w.message) for w in caught],
            "stopped": legacy_stop_at is not None,
            "stop_time_harmful_rows": legacy_stop_at,
            "rope_decision": final_legacy.rope_decision,
            "credible_interval": list(final_legacy.credible_interval),
            "bayes_factor": final_legacy.bayes_factor,
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _attack_result(*, world_bit: int, success: bool, idx: int) -> AttackResult:
    digest = hashlib.sha256(f"rails-tiny-{idx}".encode()).hexdigest()
    return AttackResult(
        world_bit=world_bit,
        success=bool(success),
        attack_id=f"rails-tiny-{idx}",
        transcript_hash=digest,
        guardrails_applied="rails_tiny",
        rng_seed=idx,
        attack_strategy="rails_tiny_side_by_side",
    )


if __name__ == "__main__":
    main()
