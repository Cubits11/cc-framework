#!/usr/bin/env python
"""CLI + helper access for the BCa bootstrap implementation.

The heavy lifting lives in :mod:`cc.analysis.week7_utils`.  This thin wrapper
exists so that the Week 7 pipeline can import ``bca_bootstrap`` or execute the
script directly for ad-hoc diagnostics.

Example
-------

``python scripts/bca_bootstrap.py --samples 0.1 0.2 0.3``

will print the BCa interval for the mean of the provided samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from cc.analysis.week7_utils import BCaInterval, bca_bootstrap


def compute_interval(values: Sequence[float], alpha: float, n_bootstrap: int, seed: int | None) -> BCaInterval:
    arr = np.asarray(values, dtype=float)
    return bca_bootstrap(arr, lambda xs: float(np.mean(xs[0])), alpha=alpha, n_bootstrap=n_bootstrap, rng=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BCa intervals for toy samples")
    parser.add_argument("--samples", nargs="+", type=float, required=True, help="Sample values")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided error rate (default 0.05)")
    parser.add_argument("--bootstrap", type=int, default=2000, help="Number of bootstrap replicates")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--json", type=Path, default=None, help="Optional output path for JSON interval")
    args = parser.parse_args()

    interval = compute_interval(args.samples, args.alpha, args.bootstrap, args.seed)
    result = {"lower": interval.lower, "upper": interval.upper, "width": interval.width}
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
