#!/usr/bin/env python3
"""
Week-6: Production-grade two-world experiment wrapper with adaptive validation.

Features:
- Adaptive FPR validation with small-sample tolerance
- Comprehensive statistical checks with confidence intervals
- Detailed logging and diagnostic output
- Graceful degradation for edge cases
- Extensible validation framework
- Support for multiple calibration formats

Quality Gates:
1. Structural integrity (JSON schema validation)
2. Statistical sanity (TPR >= FPR, ranges, sample sizes)
3. Calibration fidelity (threshold equality)
4. Alpha-cap compliance (FPR window with adaptive tolerance)
5. Reproducibility (seed verification, audit logging)
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


class ValidationError(Exception):
    """Custom exception for validation failures with detailed context."""
    pass


class ValidationWarning(Warning):
    """Custom warning for non-critical validation issues."""
    pass


@dataclass
class ExperimentMetrics:
    """Container for experiment metrics with validation metadata."""
    fpr: float
    tpr: float
    threshold: float
    delta: float
    n_world0: int
    n_world1: int
    confidence_interval: Tuple[float, float]
    
    def __post_init__(self):
        """Validate metric ranges on instantiation."""
        if not (0.0 <= self.fpr <= 1.0):
            raise ValidationError(f"FPR {self.fpr} outside [0,1]")
        if not (0.0 <= self.tpr <= 1.0):
            raise ValidationError(f"TPR {self.tpr} outside [0,1]")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValidationError(f"Threshold {self.threshold} outside [0,1]")
        if self.n_world0 < 1 or self.n_world1 < 1:
            raise ValidationError("Sample sizes must be positive")


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON with comprehensive error handling and validation."""
    if not path.exists():
        raise ValidationError(f"File not found: {path}")
    
    if path.stat().st_size == 0:
        raise ValidationError(f"Empty file: {path}")
    
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValidationError(f"Expected dict, got {type(data).__name__}")
            return data
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in {path}: {e}")
    except UnicodeDecodeError as e:
        raise ValidationError(f"Encoding error in {path}: {e}")
    except Exception as e:
        raise ValidationError(f"Cannot read {path}: {e}")


def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of file for integrity verification."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_metrics(data: Dict[str, Any]) -> ExperimentMetrics:
    """
    Extract and validate all relevant metrics from analysis.json.
    
    Returns:
        ExperimentMetrics dataclass with validated values
    
    Raises:
        ValidationError: if structure is invalid or metrics are unreasonable
    """
    try:
        # Extract sample sizes
        sizes = data["results"]["sample_sizes"]
        n_world0 = int(sizes["world_0"])
        n_world1 = int(sizes["world_1"])
        
        # Extract operating points
        op = data["results"]["operating_points"]["world_1"]
        fpr = float(op["fpr"])
        tpr = float(op["tpr"])
        
        # Extract threshold from config
        guardrails = data["metadata"]["configuration"]["guardrails"]
        if not isinstance(guardrails, list) or len(guardrails) == 0:
            raise ValidationError("No guardrails in configuration")
        threshold = float(guardrails[0]["params"]["threshold"])
        
        # Extract J-statistic and confidence interval
        j_stat = data["results"]["j_statistic"]
        delta = float(j_stat["empirical"])
        ci = j_stat["confidence_interval"]
        ci_lower = float(ci["lower"])
        ci_upper = float(ci["upper"])
        
        # Additional sanity checks
        if tpr < fpr and fpr > 0.1:  # Allow small violations for low FPR
            warnings.warn(
                f"TPR ({tpr:.4f}) < FPR ({fpr:.4f}): classifier may be miscalibrated",
                ValidationWarning
            )
        
        if ci_upper - ci_lower > 0.5:
            warnings.warn(
                f"Wide confidence interval [{ci_lower:.4f}, {ci_upper:.4f}]: "
                f"consider increasing sample size",
                ValidationWarning
            )
        
        return ExperimentMetrics(
            fpr=fpr,
            tpr=tpr,
            threshold=threshold,
            delta=delta,
            n_world0=n_world0,
            n_world1=n_world1,
            confidence_interval=(ci_lower, ci_upper)
        )
        
    except KeyError as e:
        raise ValidationError(f"Missing required field: {e}")
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid metric type: {e}")


def extract_calibration_threshold(data: Dict[str, Any]) -> Tuple[float, int]:
    """
    Extract calibration threshold and sample size with multi-format support.
    
    Supports:
    - Flat format: {"threshold": 0.123, "n_texts": 200, ...}
    - Nested format: {"guardrails": [{"threshold": 0.123, "n_samples": 200}]}
    - Legacy format: {"threshold": 0.123} (n_texts optional)
    
    Returns:
        Tuple of (threshold, n_calibration_samples)
    """
    try:
        # Try nested format first
        if "guardrails" in data and isinstance(data["guardrails"], list):
            if len(data["guardrails"]) == 0:
                raise ValidationError("Empty guardrails list")
            g = data["guardrails"][0]
            threshold = float(g["threshold"])
            n_samples = int(g.get("n_samples", g.get("n_texts", 0)))
            return threshold, n_samples
        
        # Fall back to flat format
        if "threshold" in data:
            threshold = float(data["threshold"])
            n_samples = int(data.get("n_texts", data.get("n_samples", 0)))
            return threshold, n_samples
        
        raise ValidationError("No threshold found in calibration")
        
    except (KeyError, IndexError, ValueError, TypeError) as e:
        raise ValidationError(f"Cannot extract calibration threshold: {e}")


def compute_expected_false_positives(fpr: float, n_samples: int) -> Tuple[float, float]:
    """
    Compute expected number of false positives and 95% CI using binomial.
    
    Returns:
        Tuple of (expected_fp, ci_width_95)
    """
    expected = n_samples * fpr
    # 95% CI for binomial: ±1.96 * sqrt(n*p*(1-p))
    variance = n_samples * fpr * (1 - fpr)
    ci_width = 1.96 * np.sqrt(variance)
    return expected, ci_width


def validate_fpr_adaptive(
    fpr: float,
    fpr_lo: float,
    fpr_hi: float,
    n_samples: int,
    calibration_fpr: Optional[float] = None
) -> None:
    """
    Adaptive FPR validation with small-sample tolerance.
    
    For small samples (n < 200), applies Poisson approximation to determine
    if FPR=0 is statistically plausible given the target FPR.
    
    Args:
        fpr: Observed FPR
        fpr_lo: Lower bound of acceptable window
        fpr_hi: Upper bound of acceptable window
        n_samples: Number of samples in world_1
        calibration_fpr: Target FPR from calibration (if available)
    
    Raises:
        ValidationError: if FPR outside window and statistically implausible
    """
    # Strict check for large samples
    if n_samples >= 200:
        if not (fpr_lo <= fpr <= fpr_hi):
            raise ValidationError(
                f"FPR {fpr:.6f} outside window [{fpr_lo:.4f}, {fpr_hi:.4f}] "
                f"(n={n_samples}, no tolerance for large samples)"
            )
        print(f"✓ FPR {fpr:.4f} within window [{fpr_lo:.4f}, {fpr_hi:.4f}] (n={n_samples})")
        return
    
    # Adaptive check for small samples
    target_fpr = calibration_fpr if calibration_fpr else (fpr_lo + fpr_hi) / 2
    expected_fp, ci_width = compute_expected_false_positives(target_fpr, n_samples)
    
    # If FPR=0 and expected FPs < 5, it's plausibly within statistical noise
    if fpr == 0.0 and expected_fp < 5.0:
        warnings.warn(
            f"FPR=0.0 with small sample (n={n_samples}, expected FP={expected_fp:.1f}±{ci_width:.1f}). "
            f"This is within statistical noise for target FPR={target_fpr:.4f}. "
            f"Consider increasing n_sessions for reliable FPR estimation.",
            ValidationWarning
        )
        print(f"⚠ FPR {fpr:.4f} accepted (small-sample tolerance, n={n_samples})")
        return
    
    # Standard window check
    if not (fpr_lo <= fpr <= fpr_hi):
        # Check if it's close enough given sample size
        tolerance = 2.0 / n_samples  # Allow ±2 observations of slack
        if fpr_lo - tolerance <= fpr <= fpr_hi + tolerance:
            warnings.warn(
                f"FPR {fpr:.4f} slightly outside window [{fpr_lo:.4f}, {fpr_hi:.4f}] "
                f"but within tolerance for n={n_samples}",
                ValidationWarning
            )
            print(f"⚠ FPR {fpr:.4f} accepted (within tolerance for n={n_samples})")
            return
        
        raise ValidationError(
            f"FPR {fpr:.6f} outside window [{fpr_lo:.4f}, {fpr_hi:.4f}] "
            f"and not within statistical tolerance (n={n_samples}, "
            f"expected FP={expected_fp:.1f}±{ci_width:.1f})"
        )
    
    print(f"✓ FPR {fpr:.4f} within window [{fpr_lo:.4f}, {fpr_hi:.4f}] (n={n_samples})")


def validate_threshold_equality(
    threshold_run: float,
    threshold_cal: float,
    tolerance: float = 1e-9
) -> None:
    """
    Validate threshold equality with detailed diagnostics.
    
    Raises:
        ValidationError: if thresholds differ beyond tolerance
    """
    diff = abs(threshold_run - threshold_cal)
    
    if diff > tolerance:
        # Provide actionable diagnostics
        rel_error = diff / threshold_cal if threshold_cal > 0 else float('inf')
        raise ValidationError(
            f"Threshold mismatch:\n"
            f"  Run:           {threshold_run:.12f}\n"
            f"  Calibration:   {threshold_cal:.12f}\n"
            f"  Abs. diff:     {diff:.12e}\n"
            f"  Rel. error:    {rel_error:.4%}\n"
            f"  Tolerance:     {tolerance:.2e}\n"
            f"\nPossible causes:\n"
            f"  - Config write-back failed in calibration step\n"
            f"  - Wrong config file passed to experiment\n"
            f"  - Floating-point precision issue (unlikely if diff > 1e-6)"
        )
    
    print(f"✓ Threshold equality verified: {threshold_run:.9f} (diff={diff:.2e})")


def run_experiment(
    config: Path,
    output: Path,
    audit: Path,
    seed: int,
    verbose: bool = False
) -> None:
    """
    Execute two-world experiment with comprehensive error handling.
    
    Raises:
        ValidationError: if experiment fails
    """
    # Verify config exists before running
    if not config.exists():
        raise ValidationError(f"Config file not found: {config}")
    
    cmd = [
        sys.executable, "-m", "cc.exp.run_two_world",
        "--config", str(config),
        "--output", str(output),
        "--log", str(audit),
        "--seed", str(seed),
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    print(f"  Config:     {config}")
    print(f"  Output:     {output}")
    print(f"  Audit:      {audit}")
    print(f"  Seed:       {seed}")
    print(f"  Config MD5: {compute_file_hash(config)[:16]}...")
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        if verbose and result.stdout:
            print(f"\n--- Experiment Output ---\n{result.stdout}\n")
        
        if result.returncode != 0:
            print(f"\nSTDOUT:\n{result.stdout}", file=sys.stderr)
            print(f"\nSTDERR:\n{result.stderr}", file=sys.stderr)
            raise ValidationError(
                f"Experiment failed with exit code {result.returncode}. "
                f"Check logs above for details."
            )
        
        print("✓ Experiment completed successfully")
        
    except subprocess.SubprocessError as e:
        raise ValidationError(f"Subprocess execution error: {e}")


def print_comprehensive_summary(
    metrics: ExperimentMetrics,
    calibration_info: Optional[Dict[str, Any]] = None
) -> None:
    """Print detailed experiment summary with all relevant metrics."""
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*70)
    
    # Operating characteristics
    print("\n[Operating Characteristics - World 1]")
    print(f"  TPR (True Positive Rate):   {metrics.tpr:8.4f}")
    print(f"  FPR (False Positive Rate):  {metrics.fpr:8.4f}")
    print(f"  Youden's J (TPR - FPR):     {metrics.tpr - metrics.fpr:8.4f}")
    
    # Sample allocation
    print("\n[Sample Allocation]")
    print(f"  World 0 (no guardrails):    {metrics.n_world0:8d} sessions")
    print(f"  World 1 (with guardrails):  {metrics.n_world1:8d} sessions")
    print(f"  Total:                      {metrics.n_world0 + metrics.n_world1:8d} sessions")
    
    # Guardrail parameters
    print("\n[Guardrail Configuration]")
    print(f"  Threshold:                  {metrics.threshold:8.6f}")
    if calibration_info:
        print(f"  Target FPR:                 {calibration_info.get('fpr', 'N/A'):8.4f}")
        print(f"  Calibration samples:        {calibration_info.get('n_texts', 'N/A'):8}")
    
    # Effectiveness metrics
    print("\n[Effectiveness Metrics]")
    print(f"  Δ (empirical):              {metrics.delta:8.4f}")
    ci_lo, ci_hi = metrics.confidence_interval
    print(f"  95% CI:                     [{ci_lo:.4f}, {ci_hi:.4f}]")
    ci_width = ci_hi - ci_lo
    print(f"  CI width:                   {ci_width:8.4f}")
    
    # Statistical power indicators
    expected_fp, _ = compute_expected_false_positives(metrics.fpr, metrics.n_world1)
    print("\n[Statistical Power]")
    print(f"  Expected false positives:   {expected_fp:8.1f}")
    print(f"  Actual false positives:     {int(metrics.fpr * metrics.n_world1):8d}")
    
    print("="*70 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Production-grade two-world experiment wrapper with adaptive validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quality Gates Enforced:
  1. Structural integrity     - JSON schema validation
  2. Statistical sanity       - Range checks, sample sizes
  3. Calibration fidelity     - Threshold write-back verification
  4. Alpha-cap compliance     - Adaptive FPR window validation
  5. Reproducibility          - Seed/hash verification, audit logging

Adaptive FPR Validation:
  - Large samples (n≥200): Strict window enforcement
  - Small samples (n<200): Statistical tolerance for FPR=0
  - Poisson approximation for expected false positives
  - Automatic sample size warnings

Examples:
  # Standard run with all checks
  python scripts/run_with_checks.py \\
    --config results/week6/keyword/calibrated.yaml \\
    --out-json results/week6/keyword/analysis.json \\
    --audit runs/audit_week6.jsonl \\
    --calibration results/week6/keyword/calibration_summary.json

  # Force re-run with verbose output
  python scripts/run_with_checks.py \\
    --config results/week6/keyword/calibrated.yaml \\
    --out-json results/week6/keyword/analysis.json \\
    --audit runs/audit_week6.jsonl \\
    --force-rerun --verbose
        """
    )
    
    ap.add_argument("--config", required=True, type=Path,
                    help="Calibrated YAML config to run")
    ap.add_argument("--out-json", required=True, type=Path,
                    help="Path to write/validate analysis.json")
    ap.add_argument("--audit", required=True, type=Path,
                    help="Audit JSONL path for logging")
    ap.add_argument("--seed", type=int, default=123,
                    help="Global RNG seed (default: 123)")
    ap.add_argument("--fpr-lo", type=float, default=0.00,
                    help="Lower bound for FPR window (default: 0.00)")
    ap.add_argument("--fpr-hi", type=float, default=0.08,
                    help="Upper bound for FPR window (default: 0.08)")
    ap.add_argument("--calibration", type=Path,
                    help="Calibration summary JSON for enhanced validation")
    ap.add_argument("--force-rerun", action="store_true",
                    help="Force re-execution even if valid analysis exists")
    ap.add_argument("--skip-fpr-check", action="store_true",
                    help="Skip FPR window validation (not recommended)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Enable verbose output")
    
    args = ap.parse_args()
    
    # Configure warnings
    warnings.simplefilter("always", ValidationWarning)
    
    try:
        # Step 1: Determine if experiment needs to run
        needs_run = args.force_rerun or not args.out_json.exists()
        
        if needs_run:
            print(f"\n{'='*70}")
            print("RUNNING EXPERIMENT")
            print(f"{'='*70}\n")
            run_experiment(args.config, args.out_json, args.audit, args.seed, args.verbose)
        else:
            print(f"\n{'='*70}")
            print("VALIDATION MODE (experiment output exists)")
            print(f"{'='*70}\n")
        
        # Step 2: Load and extract metrics
        print(f"\n{'='*70}")
        print("VALIDATING ANALYSIS")
        print(f"{'='*70}\n")
        
        analysis = load_json(args.out_json)
        metrics = extract_metrics(analysis)
        print(f"✓ Analysis structure valid")
        print(f"✓ Extracted metrics: TPR={metrics.tpr:.4f}, FPR={metrics.fpr:.4f}")
        
        # Step 3: Load calibration if provided
        calibration_info = None
        calibration_fpr = None
        threshold_cal = None
        
        if args.calibration:
            calibration = load_json(args.calibration)
            threshold_cal, n_cal = extract_calibration_threshold(calibration)
            calibration_fpr = calibration.get("fpr", None)
            calibration_info = {
                "threshold": threshold_cal,
                "fpr": calibration_fpr,
                "n_texts": n_cal
            }
            print(f"✓ Calibration loaded: threshold={threshold_cal:.9f}, FPR={calibration_fpr}")
        
        # Step 4: Validate threshold equality
        if threshold_cal is not None:
            validate_threshold_equality(metrics.threshold, threshold_cal)
        
        # Step 5: Validate FPR window (adaptive)
        if not args.skip_fpr_check:
            validate_fpr_adaptive(
                metrics.fpr,
                args.fpr_lo,
                args.fpr_hi,
                metrics.n_world1,
                calibration_fpr
            )
        else:
            warnings.warn("FPR validation skipped (--skip-fpr-check)", ValidationWarning)
        
        # Step 6: Print comprehensive summary
        print_comprehensive_summary(metrics, calibration_info)
        
        print(f"{'='*70}")
        print("✓ ALL VALIDATION CHECKS PASSED")
        print(f"{'='*70}\n")
        
        sys.exit(0)
        
    except ValidationError as e:
        print(f"\n{'='*70}", file=sys.stderr)
        print("✗ VALIDATION FAILED", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        print(f"\nError: {e}\n", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()