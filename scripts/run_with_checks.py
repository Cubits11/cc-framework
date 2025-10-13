#!/usr/bin/env python3
"""
Two-world execution with Week-6 validation gates.

Quality gates enforced
1) Structural integrity of analysis.json
2) Statistical sanity (ranges, sample sizes)
3) Calibration fidelity (threshold equality, 1e-9 tolerance)
4) Alpha-cap compliance via window check with small-sample tolerance
5) Reproducibility cues (config hash in logs; optional audit append handled by the runner)

Typical usage
-------------
# Produce analysis.json, then validate
python scripts/run_with_checks.py \
  --config checkpoints/example_config_calibrated.yaml \
  --out-json results/week6/keyword/analysis.json \
  --calibration runs/week6/keyword/calibration_summary.json

# Force running the experiment (if your project exposes cc.exp.run_two_world)
python scripts/run_with_checks.py \
  --config checkpoints/example_config_calibrated.yaml \
  --out-json results/week6/keyword/analysis.json \
  --calibration runs/week6/keyword/calibration_summary.json \
  --force-rerun -v
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------
# Constants & exceptions
# ---------------------------------------------------------------------

TOL = 1e-9  # threshold equality tolerance


class ValidationError(Exception):
    """Fatal validation error (causes non-zero exit)."""
    pass


class ValidationWarning(Warning):
    """Non-fatal validation warning (printed, execution continues)."""
    pass


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class ExperimentMetrics:
    """Minimal metrics we validate from analysis.json."""
    fpr: float
    tpr: float
    threshold: float
    delta: float
    n_world0: int
    n_world1: int
    confidence_interval: Tuple[float, float]

    def __post_init__(self) -> None:
        if not (0.0 <= self.fpr <= 1.0):
            raise ValidationError(f"FPR {self.fpr} outside [0,1]")
        if not (0.0 <= self.tpr <= 1.0):
            raise ValidationError(f"TPR {self.tpr} outside [0,1]")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValidationError(f"Threshold {self.threshold} outside [0,1]")
        if self.n_world0 < 1 or self.n_world1 < 1:
            raise ValidationError("Sample sizes must be positive")


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ValidationError(f"File not found: {path}")
    if path.stat().st_size == 0:
        raise ValidationError(f"Empty file: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValidationError(f"{path} must contain a JSON object")
        return data
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in {path}: {e}") from e


def compute_file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ---------------------------------------------------------------------
# Extraction from analysis/calibration
# ---------------------------------------------------------------------

def extract_metrics(analysis: Dict[str, Any]) -> ExperimentMetrics:
    try:
        sizes = analysis["results"]["sample_sizes"]
        n0 = int(sizes["world_0"])
        n1 = int(sizes["world_1"])

        op_w1 = analysis["results"]["operating_points"]["world_1"]
        fpr = float(op_w1["fpr"])
        tpr = float(op_w1["tpr"])

        guardrails = analysis["metadata"]["configuration"]["guardrails"]
        if not guardrails or not isinstance(guardrails, list):
            raise ValidationError("No guardrails in configuration")
        threshold = float(guardrails[0]["params"]["threshold"])

        j = analysis["results"]["j_statistic"]
        delta = float(j["empirical"])
        ci = j["confidence_interval"]
        ci_lo = float(ci["lower"])
        ci_hi = float(ci["upper"])

        # soft sanity warnings
        if tpr < fpr and fpr > 0.1:
            warnings.warn(
                f"TPR ({tpr:.4f}) < FPR ({fpr:.4f}); classifier may be miscalibrated.",
                ValidationWarning,
            )
        if (ci_hi - ci_lo) > 0.5:
            warnings.warn(
                f"Wide CI [{ci_lo:.4f}, {ci_hi:.4f}] — consider increasing sample size.",
                ValidationWarning,
            )

        return ExperimentMetrics(
            fpr=fpr,
            tpr=tpr,
            threshold=threshold,
            delta=delta,
            n_world0=n0,
            n_world1=n1,
            confidence_interval=(ci_lo, ci_hi),
        )
    except KeyError as e:
        raise ValidationError(f"Missing required field in analysis.json: {e}") from e
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Bad types in analysis.json: {e}") from e


def extract_calibration_threshold(cal: Dict[str, Any]) -> Tuple[float, int, Optional[float]]:
    """
    Supports:
      - flat:   {"threshold": 0.12, "n_texts": 200, "fpr": 0.05}
      - nested: {"guardrails": [{"threshold": 0.12, "n_samples": 200}], "fpr": 0.05}
    Returns: (threshold, n_samples, target_fpr_or_None)
    """
    # nested then flat
    if "guardrails" in cal and isinstance(cal["guardrails"], list) and cal["guardrails"]:
        g = cal["guardrails"][0]
        thr = float(g["threshold"])
        n = int(g.get("n_samples", g.get("n_texts", 0)))
        return thr, n, cal.get("fpr", None)
    if "threshold" in cal:
        thr = float(cal["threshold"])
        n = int(cal.get("n_texts", cal.get("n_samples", 0)))
        return thr, n, cal.get("fpr", None)
    raise ValidationError("Calibration JSON lacks a threshold.")


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def compute_expected_false_positives(fpr: float, n: int) -> Tuple[float, float]:
    expected = n * fpr
    var = n * fpr * (1 - fpr)
    ci_width = 1.96 * np.sqrt(var)
    return expected, ci_width


def validate_threshold_equality(run_thr: float, cal_thr: float, tol: float = TOL) -> None:
    diff = abs(run_thr - cal_thr)
    if diff > tol:
        rel = diff / cal_thr if cal_thr else float("inf")
        raise ValidationError(
            "Threshold mismatch\n"
            f"  run:         {run_thr:.12f}\n"
            f"  calibration: {cal_thr:.12f}\n"
            f"  abs diff:    {diff:.3e}\n"
            f"  rel error:   {rel:.4%}\n"
            f"  tolerance:   {tol:.1e}\n"
            "Likely causes: wrong config, failed write-back, or stale files."
        )
    print(f"✓ Threshold equality verified (diff={diff:.2e})")


def validate_fpr_window_adaptive(
    fpr: float,
    n_samples: int,
    lo: float,
    hi: float,
    target_from_cal: Optional[float] = None,
) -> None:
    """
    Large n (>=200): strict window.
    Small n: allow (a) zero-FP when expected FP < 5, or (b) ±(2/n) slack.
    """
    if n_samples >= 200:
        if not (lo <= fpr <= hi):
            raise ValidationError(
                f"FPR {fpr:.6f} outside window [{lo:.4f},{hi:.4f}] (n={n_samples})"
            )
        print(f"✓ FPR {fpr:.4f} within window [{lo:.4f},{hi:.4f}] (n={n_samples})")
        return

    # small-n tolerance
    target = float(target_from_cal) if target_from_cal is not None else (lo + hi) / 2.0
    expected_fp, ciw = compute_expected_false_positives(target, n_samples)

    if fpr == 0.0 and expected_fp < 5.0:
        warnings.warn(
            f"FPR=0 with n={n_samples} is plausible (target≈{target:.3f}, "
            f"expected FP={expected_fp:.1f}±{ciw:.1f}). Consider increasing n.",
            ValidationWarning,
        )
        print(f"⚠ FPR {fpr:.4f} accepted under small-sample rule (n={n_samples}).")
        return

    slack = 2.0 / max(n_samples, 1)
    if lo - slack <= fpr <= hi + slack:
        warnings.warn(
            f"FPR {fpr:.4f} slightly outside [{lo:.4f},{hi:.4f}] but within ±{slack:.4f} slack for n={n_samples}.",
            ValidationWarning,
        )
        print(f"⚠ FPR {fpr:.4f} accepted with small-n slack (n={n_samples}).")
        return

    raise ValidationError(
        f"FPR {fpr:.6f} outside window [{lo:.4f},{hi:.4f}] and not covered by small-n tolerance "
        f"(n={n_samples}, target≈{target:.3f}, expected FP={expected_fp:.1f}±{ciw:.1f})."
    )


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def run_experiment(config: Path, output: Path, audit: Optional[Path], seed: int, verbose: bool) -> None:
    """
    Calls your project runner if available:
        python -m cc.exp.run_two_world --config ... --output ... --log ... --seed ...
    If your pipeline already produced `analysis.json`, just skip --force-rerun.
    """
    if not config.exists():
        raise ValidationError(f"Config not found: {config}")

    cmd = [sys.executable, "-m", "cc.exp.run_two_world", "--config", str(config), "--output", str(output)]
    if audit is not None:
        cmd += ["--log", str(audit)]
    cmd += ["--seed", str(seed)]

    print(f"Executing: {' '.join(cmd)}")
    print(f"  config sha256: {compute_file_hash(config)[:16]}…")

    res = subprocess.run(cmd, text=True, capture_output=True)
    if verbose and res.stdout:
        print("\n--- runner stdout ---\n" + res.stdout)
    if res.returncode != 0:
        if res.stderr:
            print("\n--- runner stderr ---\n" + res.stderr, file=sys.stderr)
        raise ValidationError(f"Experiment process exited with {res.returncode}")


def print_summary(m: ExperimentMetrics, cal: Optional[Dict[str, Any]]) -> None:
    print("\n" + "=" * 68)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 68)
    print("\n[Operating (World 1)]")
    print(f"  TPR: {m.tpr:8.4f}")
    print(f"  FPR: {m.fpr:8.4f}")
    print(f"  J  : {m.tpr - m.fpr:8.4f}")
    print("\n[Samples]")
    print(f"  world_0: {m.n_world0:6d}")
    print(f"  world_1: {m.n_world1:6d}")
    print("\n[Guardrail]")
    print(f"  threshold: {m.threshold:0.9f}")
    if cal:
        print(f"  calibration FPR: {cal.get('fpr', 'N/A')}")
        print(f"  calibration n  : {cal.get('n_texts', cal.get('n_samples', 'N/A'))}")
    print("\n[Δ and 95% CI]")
    lo, hi = m.confidence_interval
    print(f"  Δ (empirical): {m.delta:8.4f}")
    print(f"  95% CI       : [{lo:.4f}, {hi:.4f}]")
    print("=" * 68 + "\n")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Week-6 two-world validator with α-window + threshold equality checks"
    )
    ap.add_argument("--config", type=Path, required=True, help="Calibrated YAML used to run the experiment")
    ap.add_argument("--out-json", type=Path, required=True, help="Path to analysis.json to validate (and/or write)")
    ap.add_argument("--calibration", type=Path, help="Calibration summary JSON (for threshold equality & target FPR)")
    ap.add_argument("--audit", type=Path, help="Audit JSONL path (forwarded to the runner if --force-rerun)")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed forwarded to runner")
    ap.add_argument("--fpr-lo", type=float, default=0.04, help="Lower bound of α-window (default 0.04)")
    ap.add_argument("--fpr-hi", type=float, default=0.06, help="Upper bound of α-window (default 0.06)")
    ap.add_argument("--force-rerun", action="store_true", help="Call the project runner to regenerate analysis.json")
    ap.add_argument("--skip-fpr-check", action="store_true", help="Skip FPR window validation (not recommended)")
    ap.add_argument("--verbose", "-v", action="store_true", help="Verbose subprocess output")
    return ap.parse_args()


def main() -> None:
    warnings.simplefilter("always", ValidationWarning)
    args = parse_args()

    # 1) Optionally run the experiment to produce analysis.json
    if args.force_rerun or not args.out_json.exists():
        run_experiment(args.config, args.out_json, args.audit, args.seed, args.verbose)
    else:
        print("Analysis already exists; running in validation-only mode.")

    # 2) Load analysis & extract metrics
    analysis = load_json(args.out_json)
    metrics = extract_metrics(analysis)
    print(f"✓ Loaded analysis: TPR={metrics.tpr:.4f}, FPR={metrics.fpr:.4f}")

    # 3) Load calibration (if provided) and validate threshold equality
    cal_info: Optional[Dict[str, Any]] = None
    cal_target_fpr: Optional[float] = None
    if args.calibration:
        calibration = load_json(args.calibration)
        cal_thr, cal_n, cal_target_fpr = extract_calibration_threshold(calibration)
        cal_info = {"threshold": cal_thr, "n_texts": cal_n, "fpr": cal_target_fpr}
        validate_threshold_equality(metrics.threshold, cal_thr)

    # 4) Validate α-window (adaptive small-n rule)
    if not args.skip_fpr_check:
        validate_fpr_window_adaptive(
            fpr=metrics.fpr,
            n_samples=metrics.n_world1,
            lo=args.fpr_lo,
            hi=args.fpr_hi,
            target_from_cal=cal_target_fpr,
        )
    else:
        warnings.warn("FPR window validation skipped (--skip-fpr-check).", ValidationWarning)

    # 5) Print comprehensive summary
    print_summary(metrics, cal_info)

    print("✓ ALL WEEK-6 VALIDATION CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
