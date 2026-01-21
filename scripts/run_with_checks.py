#!/usr/bin/env python3
"""
scripts/run_with_checks.py

Two-world execution with Week-6 validation gates.

Design goals (PhD-grade):
- **Fail loudly on true contract violations**, but **degrade gracefully** when optional fields
  are absent (e.g., minimal analysis.json in integration tests).
- Make every failure *actionable*: print what was expected, what was found, and likely causes.
- Support both "run + validate" and "validate-only" workflows without requiring the runner.

Quality gates enforced
1) Structural integrity of analysis.json (required paths exist; sane types)
2) Statistical sanity (ranges; CI coherence; optional sample size checks)
3) Calibration fidelity (threshold equality; tolerance controlled)
4) Alpha-cap compliance (FPR window), with **adaptive small-n tolerance** when n is known
5) Reproducibility cues (config sha256 printed when relevant)

Typical usage
-------------
# Validate an existing analysis.json
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
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------
# Constants & exceptions
# ---------------------------------------------------------------------

TOL = 1e-9  # default threshold equality tolerance (abs)


class ValidationError(Exception):
    """Fatal validation error (causes non-zero exit)."""


class ValidationWarning(Warning):
    """Non-fatal validation warning (printed, execution continues)."""


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------


def _fmt_path(p: Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)


def _get(d: dict[str, Any], path: str) -> Any:
    """
    Safe dotted-path getter for dicts.
    Raises KeyError with a precise failing segment.
    """
    cur: Any = d
    for seg in path.split("."):
        if not isinstance(cur, dict):
            raise KeyError(f"{path} (stopped at '{seg}': not an object)")
        if seg not in cur:
            raise KeyError(f"{path} (missing '{seg}')")
        cur = cur[seg]
    return cur


def _as_float(x: Any, *, where: str) -> float:
    try:
        return float(x)
    except Exception as e:
        raise ValidationError(f"Expected numeric at {where}, got {type(x).__name__}: {x!r}") from e


def _as_int(x: Any, *, where: str) -> int:
    try:
        v = int(x)
        return v
    except Exception as e:
        raise ValidationError(f"Expected int at {where}, got {type(x).__name__}: {x!r}") from e


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentMetrics:
    """Minimal metrics validated from analysis.json."""

    fpr: float
    tpr: float
    threshold: float
    delta: float
    n_world0: int | None
    n_world1: int | None
    confidence_interval: tuple[float, float]

    def __post_init__(self) -> None:
        if not (0.0 <= self.fpr <= 1.0):
            raise ValidationError(f"FPR {self.fpr} outside [0,1]")
        if not (0.0 <= self.tpr <= 1.0):
            raise ValidationError(f"TPR {self.tpr} outside [0,1]")

        # Threshold range depends on score scale; Week-6 contracts typically assume [0,1].
        # Keep strict (fail) because your tests and calibration summary assume it.
        if not (0.0 <= self.threshold <= 1.0):
            raise ValidationError(f"Threshold {self.threshold} outside [0,1]")

        lo, hi = self.confidence_interval
        if lo > hi:
            raise ValidationError(f"Invalid CI: lower {lo} > upper {hi}")

        # Only enforce positivity if sample sizes are present.
        if self.n_world0 is not None and self.n_world0 < 1:
            raise ValidationError("Sample sizes must be positive (world_0)")
        if self.n_world1 is not None and self.n_world1 < 1:
            raise ValidationError("Sample sizes must be positive (world_1)")


@dataclass(frozen=True)
class CalibrationInfo:
    name: str | None
    threshold: float
    n_samples: int | None
    target_fpr: float | None
    target_window: tuple[float, float] | None


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValidationError(f"File not found: {_fmt_path(path)}")
    if path.stat().st_size == 0:
        raise ValidationError(f"Empty file: {_fmt_path(path)}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in {_fmt_path(path)}: {e}") from e
    except OSError as e:
        raise ValidationError(f"Could not read {_fmt_path(path)}: {e}") from e

    if not isinstance(data, dict):
        raise ValidationError(f"{_fmt_path(path)} must contain a JSON object at top-level")
    return data


def compute_file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ---------------------------------------------------------------------
# Extraction from analysis/calibration
# ---------------------------------------------------------------------


def extract_metrics(
    analysis: dict[str, Any], *, calibration_name: str | None = None
) -> ExperimentMetrics:
    """
    Required minimal contract (must exist):
      - results.operating_points.world_1.fpr
      - results.operating_points.world_1.tpr
      - results.j_statistic.empirical
      - results.j_statistic.confidence_interval.{lower,upper}
      - metadata.configuration.guardrails[...].params.threshold

    Optional:
      - results.sample_sizes.world_0 / world_1 (used for small-n tolerance rules)
    """
    # --- sample sizes (optional) ---
    n0: int | None = None
    n1: int | None = None
    sizes = analysis.get("results", {}).get("sample_sizes")
    if isinstance(sizes, dict):
        # present -> must be well-formed
        n0 = _as_int(sizes.get("world_0"), where="results.sample_sizes.world_0")
        n1 = _as_int(sizes.get("world_1"), where="results.sample_sizes.world_1")
    elif sizes is None:
        warnings.warn(
            "analysis.json missing results.sample_sizes; proceeding with strict FPR window checks (no small-n tolerance).",
            ValidationWarning,
            stacklevel=2,
        )
    else:
        warnings.warn(
            f"analysis.json results.sample_sizes is not an object (got {type(sizes).__name__}); ignoring it.",
            ValidationWarning,
            stacklevel=2,
        )

    # --- operating point ---
    op_w1 = _get(analysis, "results.operating_points.world_1")
    fpr = _as_float(op_w1.get("fpr"), where="results.operating_points.world_1.fpr")
    tpr = _as_float(op_w1.get("tpr"), where="results.operating_points.world_1.tpr")

    # --- guardrail threshold ---
    guardrails = _get(analysis, "metadata.configuration.guardrails")
    if not isinstance(guardrails, list) or not guardrails:
        raise ValidationError("metadata.configuration.guardrails must be a non-empty list")

    chosen = None
    if calibration_name:
        for g in guardrails:
            if isinstance(g, dict) and g.get("name") == calibration_name:
                chosen = g
                break
        if chosen is None:
            warnings.warn(
                f"Calibration name '{calibration_name}' not found in analysis guardrails; using guardrails[0].",
                ValidationWarning,
                stacklevel=2,
            )
    if chosen is None:
        chosen = guardrails[0]

    if not isinstance(chosen, dict):
        raise ValidationError("Guardrail entry is not an object; cannot read threshold")

    params = chosen.get("params")
    if not isinstance(params, dict) or "threshold" not in params:
        raise ValidationError("Guardrail params.threshold missing in analysis.json")

    threshold = _as_float(
        params["threshold"], where="metadata.configuration.guardrails[i].params.threshold"
    )

    # --- j statistic + CI ---
    j = _get(analysis, "results.j_statistic")
    delta = _as_float(j.get("empirical"), where="results.j_statistic.empirical")

    ci = j.get("confidence_interval")
    if not isinstance(ci, dict):
        raise ValidationError("results.j_statistic.confidence_interval must be an object")

    ci_lo = _as_float(ci.get("lower"), where="results.j_statistic.confidence_interval.lower")
    ci_hi = _as_float(ci.get("upper"), where="results.j_statistic.confidence_interval.upper")

    # --- soft sanity warnings ---
    if tpr < fpr and fpr > 0.1:
        warnings.warn(
            f"TPR ({tpr:.4f}) < FPR ({fpr:.4f}); classifier may be inverted or miscalibrated.",
            ValidationWarning,
            stacklevel=2,
        )
    if (ci_hi - ci_lo) > 0.5:
        warnings.warn(
            f"Wide CI [{ci_lo:.4f}, {ci_hi:.4f}] — increase sample size or bootstrap reps.",
            ValidationWarning,
            stacklevel=2,
        )

    return ExperimentMetrics(
        fpr=float(fpr),
        tpr=float(tpr),
        threshold=float(threshold),
        delta=float(delta),
        n_world0=n0,
        n_world1=n1,
        confidence_interval=(float(ci_lo), float(ci_hi)),
    )


def extract_calibration(cal: dict[str, Any]) -> CalibrationInfo:
    """
    Accepts multiple calibration summary shapes.

    Supported (flat):
      {"name": "...", "threshold": 0.12, "n_texts": 200, "fpr": 0.05, "target_window":[0.04,0.06]}

    Supported (nested):
      {"name":"...", "guardrails":[{"threshold": 0.12, "n_samples": 200}], "fpr": 0.05, "target_window":[...]}
    """
    name = cal.get("name")
    thr: float | None = None
    n: int | None = None

    # nested first
    if isinstance(cal.get("guardrails"), list) and cal["guardrails"]:
        g0 = cal["guardrails"][0]
        if isinstance(g0, dict) and "threshold" in g0:
            thr = _as_float(g0["threshold"], where="calibration.guardrails[0].threshold")
            if "n_samples" in g0:
                n = _as_int(g0["n_samples"], where="calibration.guardrails[0].n_samples")
            elif "n_texts" in g0:
                n = _as_int(g0["n_texts"], where="calibration.guardrails[0].n_texts")

    # flat fallback
    if thr is None and "threshold" in cal:
        thr = _as_float(cal["threshold"], where="calibration.threshold")
        if "n_texts" in cal:
            n = _as_int(cal["n_texts"], where="calibration.n_texts")
        elif "n_samples" in cal:
            n = _as_int(cal["n_samples"], where="calibration.n_samples")

    if thr is None:
        raise ValidationError(
            "Calibration JSON lacks a threshold (expected 'threshold' or guardrails[0].threshold)."
        )

    target_fpr = cal.get("fpr")
    if target_fpr is not None:
        target_fpr = _as_float(target_fpr, where="calibration.fpr")

    tw = cal.get("target_window")
    target_window: tuple[float, float] | None = None
    if isinstance(tw, (list, tuple)) and len(tw) == 2:
        lo = _as_float(tw[0], where="calibration.target_window[0]")
        hi = _as_float(tw[1], where="calibration.target_window[1]")
        target_window = (lo, hi)

    return CalibrationInfo(
        name=str(name) if name is not None else None,
        threshold=float(thr),
        n_samples=n,
        target_fpr=target_fpr if target_fpr is None else float(target_fpr),
        target_window=target_window,
    )


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------


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
            "Likely causes: wrong config, failed write-back, stale analysis.json, or wrong guardrail selected."
        )
    print(f"✓ Threshold equality verified (diff={diff:.2e})")


def compute_expected_false_positives(target_fpr: float, n: int) -> tuple[float, float]:
    """
    Normal approximation for FP count. Returns (expected, ~95% half-width).
    Useful for explaining why FPR=0 can be plausible for small n.
    """
    expected = n * target_fpr
    var = n * target_fpr * (1.0 - target_fpr)
    ci_half_width = 1.96 * float(np.sqrt(max(var, 0.0)))
    return float(expected), float(ci_half_width)


def validate_fpr_window_adaptive(
    *,
    fpr: float,
    n_samples: int | None,
    lo: float,
    hi: float,
    target_from_cal: float | None = None,
) -> None:
    """
    Alpha-cap / operating point window validation.

    If sample size is unknown:
      - apply STRICT window check (no slack), because we cannot justify tolerance.

    If sample size is known:
      - n >= 200: strict window check
      - n < 200 : allow small-n tolerance:
          (a) accept FPR=0 when expected FP < 5 under target fpr
          (b) allow ±(2/n) slack around window (a conservative discrete-rate cushion)
    """
    # Unknown n => strict.
    if n_samples is None or n_samples <= 0:
        if not (lo <= fpr <= hi):
            raise ValidationError(
                f"FPR {fpr:.6f} outside window [{lo:.4f},{hi:.4f}] (sample size unknown)"
            )
        print(f"✓ FPR {fpr:.4f} within window [{lo:.4f},{hi:.4f}] (n unknown)")
        return

    n = int(n_samples)

    # Large n => strict.
    if n >= 200:
        if not (lo <= fpr <= hi):
            raise ValidationError(f"FPR {fpr:.6f} outside window [{lo:.4f},{hi:.4f}] (n={n})")
        print(f"✓ FPR {fpr:.4f} within window [{lo:.4f},{hi:.4f}] (n={n})")
        return

    # Small n tolerance regime.
    target = float(target_from_cal) if target_from_cal is not None else (lo + hi) / 2.0
    expected_fp, ciw = compute_expected_false_positives(target, n)

    if fpr == 0.0 and expected_fp < 5.0:
        warnings.warn(
            f"FPR=0 with n={n} is plausible under target≈{target:.3f} "
            f"(expected FP={expected_fp:.1f}±{ciw:.1f}). Consider increasing n.",
            ValidationWarning,
            stacklevel=2,
        )
        print(f"⚠ FPR {fpr:.4f} accepted under small-sample rule (n={n}).")
        return

    slack = 2.0 / n
    if (lo - slack) <= fpr <= (hi + slack):
        warnings.warn(
            f"FPR {fpr:.4f} slightly outside [{lo:.4f},{hi:.4f}] but within ±{slack:.4f} slack for n={n}.",
            ValidationWarning,
            stacklevel=2,
        )
        print(f"⚠ FPR {fpr:.4f} accepted with small-n slack (n={n}).")
        return

    raise ValidationError(
        f"FPR {fpr:.6f} outside window [{lo:.4f},{hi:.4f}] and not covered by small-n tolerance "
        f"(n={n}, target≈{target:.3f}, expected FP={expected_fp:.1f}±{ciw:.1f})."
    )


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------


def run_experiment(
    *,
    config: Path,
    output: Path,
    audit: Path | None,
    seed: int,
    verbose: bool,
    runner_module: str,
) -> None:
    """
    Calls your project runner:
        python -m <runner_module> --config ... --output ... --log ... --seed ...
    """
    if not config.exists():
        raise ValidationError(f"Config not found: {_fmt_path(config)}")

    cmd = [
        sys.executable,
        "-m",
        runner_module,
        "--config",
        str(config),
        "--output",
        str(output),
        "--seed",
        str(seed),
    ]
    if audit is not None:
        cmd += ["--log", str(audit)]

    print(f"Executing: {' '.join(cmd)}")
    print(f"  config sha256: {compute_file_hash(config)[:16]}…")

    res = subprocess.run(cmd, text=True, capture_output=True)
    if verbose and res.stdout:
        print("\n--- runner stdout ---\n" + res.stdout)
    if res.returncode != 0:
        if res.stderr:
            print("\n--- runner stderr ---\n" + res.stderr, file=sys.stderr)
        raise ValidationError(f"Experiment process exited with {res.returncode}")


def print_summary(m: ExperimentMetrics, cal: CalibrationInfo | None) -> None:
    print("\n" + "=" * 72)
    print("WEEK-6 VALIDATION SUMMARY")
    print("=" * 72)

    print("\n[Operating (World 1)]")
    print(f"  TPR: {m.tpr:8.4f}")
    print(f"  FPR: {m.fpr:8.4f}")
    print(f"  J  : {m.tpr - m.fpr:8.4f}")

    print("\n[Samples]")
    w0 = m.n_world0 if m.n_world0 is not None else "N/A"
    w1 = m.n_world1 if m.n_world1 is not None else "N/A"
    print(f"  world_0: {w0}")
    print(f"  world_1: {w1}")

    print("\n[Guardrail]")
    print(f"  threshold: {m.threshold:0.9f}")
    if cal is not None:
        print(f"  calibration name : {cal.name if cal.name is not None else 'N/A'}")
        print(f"  calibration thr  : {cal.threshold:0.9f}")
        print(f"  calibration n    : {cal.n_samples if cal.n_samples is not None else 'N/A'}")
        if cal.target_fpr is not None:
            print(f"  calibration FPR  : {cal.target_fpr:.6f}")
        if cal.target_window is not None:
            lo, hi = cal.target_window
            print(f"  target window    : [{lo:.4f}, {hi:.4f}]")

    print("\n[Δ and 95% CI]")
    lo, hi = m.confidence_interval
    print(f"  Δ (empirical): {m.delta:8.4f}")
    print(f"  95% CI       : [{lo:.4f}, {hi:.4f}]")

    print("=" * 72 + "\n")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Week-6 two-world validator with α-window + threshold equality checks"
    )
    ap.add_argument(
        "--config", type=Path, required=True, help="Calibrated YAML used to run the experiment"
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        required=True,
        help="Path to analysis.json to validate (and/or write)",
    )
    ap.add_argument(
        "--calibration",
        type=Path,
        help="Calibration summary JSON (threshold equality & target FPR/window)",
    )
    ap.add_argument(
        "--audit", type=Path, help="Audit JSONL path (forwarded to runner if --force-rerun)"
    )
    ap.add_argument("--seed", type=int, default=123, help="RNG seed forwarded to runner")
    ap.add_argument(
        "--fpr-lo", type=float, default=0.04, help="Lower bound of α-window (default 0.04)"
    )
    ap.add_argument(
        "--fpr-hi", type=float, default=0.06, help="Upper bound of α-window (default 0.06)"
    )
    ap.add_argument(
        "--force-rerun", action="store_true", help="Run the experiment to regenerate analysis.json"
    )
    ap.add_argument(
        "--skip-fpr-check", action="store_true", help="Skip FPR window validation (not recommended)"
    )
    ap.add_argument(
        "--runner-module", default="cc.exp.run_two_world", help="Python module to execute with -m"
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose subprocess output / debug traceback"
    )
    ap.add_argument(
        "--tol", type=float, default=TOL, help=f"Threshold equality tolerance (default {TOL:g})"
    )
    return ap.parse_args()


def main() -> None:
    warnings.simplefilter("always", ValidationWarning)
    args = parse_args()

    # Validate bounds early.
    if not (0.0 <= args.fpr_lo <= 1.0 and 0.0 <= args.fpr_hi <= 1.0 and args.fpr_lo <= args.fpr_hi):
        raise ValidationError("Invalid FPR window: need 0 <= fpr-lo <= fpr-hi <= 1")

    # Load calibration first (optional) so we can choose the right guardrail by name.
    cal: CalibrationInfo | None = None
    if args.calibration:
        cal_obj = load_json(args.calibration)
        cal = extract_calibration(cal_obj)

    # Decide whether to run.
    if args.force_rerun or not args.out_json.exists():
        run_experiment(
            config=args.config,
            output=args.out_json,
            audit=args.audit,
            seed=args.seed,
            verbose=args.verbose,
            runner_module=args.runner_module,
        )
    else:
        print("Analysis already exists; running in validation-only mode.")
        # If config doesn't exist in validation-only, warn (do not fail).
        if not args.config.exists():
            warnings.warn(
                f"--config does not exist ({_fmt_path(args.config)}), but analysis.json exists; "
                "skipping runner and proceeding with validation-only.",
                ValidationWarning,
                stacklevel=2,
            )

    # Load analysis.
    analysis = load_json(args.out_json)
    metrics = extract_metrics(analysis, calibration_name=(cal.name if cal else None))
    print(f"✓ Loaded analysis: TPR={metrics.tpr:.4f}, FPR={metrics.fpr:.4f}")

    # Threshold equality gate.
    if cal is not None:
        validate_threshold_equality(metrics.threshold, cal.threshold, tol=float(args.tol))

    # FPR window gate.
    if not args.skip_fpr_check:
        validate_fpr_window_adaptive(
            fpr=metrics.fpr,
            n_samples=metrics.n_world1,  # may be None; function handles it
            lo=float(args.fpr_lo),
            hi=float(args.fpr_hi),
            target_from_cal=(cal.target_fpr if cal else None),
        )
    else:
        warnings.warn(
            "FPR window validation skipped (--skip-fpr-check).", ValidationWarning, stacklevel=2
        )

    # Summary.
    print_summary(metrics, cal)

    print("✓ ALL WEEK-6 VALIDATION CHECKS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except ValidationError as e:
        print(f"\nVALIDATION FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # For unexpected bugs, still give a clean failure, but optionally show traceback.
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        if "-v" in sys.argv or "--verbose" in sys.argv:
            traceback.print_exc()
        sys.exit(2)
