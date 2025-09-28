#!/usr/bin/env python3
"""
Week-6: wrapper that runs the two-world experiment *and* enforces quality gates.

- Runs: python -m cc.exp.run_two_world ...
- Then validates:
  * FPR for world_1 lies within [--fpr-lo, --fpr-hi]
  * If --calibration is given, the run-time threshold equals the calibration threshold.
- Short-circuit: if --out-json already exists and is valid, skip the execution
  step and only perform validations (useful for tests and quick rechecks).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_valid_analysis(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        d = load_json(path)
        _ = d["results"]["operating_points"]["world_1"]["fpr"]
        _ = d["metadata"]["configuration"]["guardrails"][0]["params"]["threshold"]
        return True
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Run two-world with FPR/threshold checks (fail-fast).")
    ap.add_argument("--config", required=True, help="Calibrated YAML config to run")
    ap.add_argument("--out-json", required=True, help="Path to write analysis.json")
    ap.add_argument("--audit", required=True, help="Audit JSONL path")
    ap.add_argument("--seed", type=int, default=123, help="Global RNG seed for the run")
    ap.add_argument("--fpr-lo", type=float, default=0.04, help="Lower bound for allowed FPR window")
    ap.add_argument("--fpr-hi", type=float, default=0.06, help="Upper bound for allowed FPR window")
    ap.add_argument("--calibration", help="Optional calibration summary JSON to compare threshold equality")
    args = ap.parse_args()

    out_path = Path(args.out_json)

    # 1) Possibly execute the experiment unless a valid analysis already exists.
    if not is_valid_analysis(out_path):
        cmd = [
            sys.executable,
            "-m",
            "cc.exp.run_two_world",
            "--config",
            args.config,
            "--output",
            str(out_path),
            "--log",
            args.audit,
            "--seed",
            str(args.seed),
        ]
        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"ERROR: Experiment process returned {proc.returncode}.", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Skip run: found valid analysis at {out_path}")

    # 2) Load analysis and extract values
    try:
        d = load_json(out_path)
    except Exception as e:
        print(f"ERROR: Cannot read analysis JSON: {out_path} ({e})", file=sys.stderr)
        sys.exit(1)

    try:
        fpr = float(d["results"]["operating_points"]["world_1"]["fpr"])
    except Exception:
        print("ERROR: analysis.json missing results.operating_points.world_1.fpr", file=sys.stderr)
        sys.exit(1)

    try:
        thr_run = float(d["metadata"]["configuration"]["guardrails"][0]["params"]["threshold"])
    except Exception:
        print("ERROR: analysis.json missing configuration.guardrails[0].params.threshold", file=sys.stderr)
        sys.exit(1)

    # 3) Check FPR window
    if not (args.fpr_lo <= fpr <= args.fpr_hi):
        print(
            f"ERROR: FPR {fpr:.6f} outside [{args.fpr_lo:.2f},{args.fpr_hi:.2f}].",
            file=sys.stderr,
        )
        sys.exit(2)

    # 4) Check calibrated threshold equality, if provided
    if args.calibration:
        try:
            cal = load_json(Path(args.calibration))
            # Accept either a flat summary {name, threshold, ...} OR the Week-5 array style
            if isinstance(cal.get("guardrails"), list) and cal["guardrails"]:
                thr_cal = float(cal["guardrails"][0].get("threshold"))
            else:
                thr_cal = float(cal.get("threshold"))
        except Exception as e:
            print(f"ERROR: Cannot read calibration JSON: {args.calibration} ({e})", file=sys.stderr)
            sys.exit(1)

        if abs(thr_run - thr_cal) > 1e-9:
            print(
                f"ERROR: Threshold mismatch: run={thr_run:.9f} vs calibration={thr_cal:.9f}",
                file=sys.stderr,
            )
            sys.exit(3)

    print(f"OK: FPR {fpr:.4f} in-window and threshold checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()