#!/usr/bin/env python
"""Reproduce the deterministic E1 Dependence-Evidence Study artifacts."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from cc.evals.dependence_benchmark import (
    E1_DEFAULT_DELTA,
    E1_DEFAULT_REPLICATES,
    E1_DEFAULT_SAMPLE_SIZES,
    E1_DEFAULT_SEED,
    write_e1_artifacts,
)

DEFAULT_OUTPUT_DIR = Path("artifacts/empirical/e1")


def main() -> int:
    """Parse arguments, write small regenerable artifacts, and report the path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=E1_DEFAULT_SEED)
    parser.add_argument("--delta", type=float, default=E1_DEFAULT_DELTA)
    parser.add_argument("--replicates", type=int, default=E1_DEFAULT_REPLICATES)
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=E1_DEFAULT_SAMPLE_SIZES,
        metavar="N",
    )
    args = parser.parse_args()
    write_e1_artifacts(
        args.output_dir,
        generation_command=" ".join(shlex.quote(value) for value in sys.argv),
        seed=args.seed,
        delta=args.delta,
        replicates=args.replicates,
        sample_sizes=tuple(args.sample_sizes),
    )
    print(f"Wrote E1 artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
