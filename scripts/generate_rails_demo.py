#!/usr/bin/env python3
"""
Generate a tiny synthetic rail score dataset for the lightweight demo.

Outputs columns expected by scripts/rails_compare.py:
- id
- label
- rail_a_score
- rail_b_score
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_demo(
    n: int,
    seed: int,
    pos_rate: float,
    signal: float,
    noise: float,
    latent_scale: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.binomial(1, pos_rate, size=n)
    latent = rng.normal(loc=0.0, scale=latent_scale, size=n)

    base_a = latent + rng.normal(loc=0.0, scale=noise, size=n)
    base_b = 0.6 * latent + rng.normal(loc=0.0, scale=noise, size=n)

    score_a = sigmoid(base_a + signal * labels)
    score_b = sigmoid(base_b + 0.8 * signal * labels)

    return pd.DataFrame(
        {
            "id": [f"ex{i:04d}" for i in range(1, n + 1)],
            "label": labels.astype(int),
            "rail_a_score": score_a.astype(float),
            "rail_b_score": score_b.astype(float),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic rails demo data.")
    parser.add_argument("--out", type=Path, default=Path("datasets/examples/rails_tiny.csv"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--pos-rate", type=float, default=0.45)
    parser.add_argument("--signal", type=float, default=1.2)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--latent-scale", type=float, default=0.7)
    args = parser.parse_args()

    if not 0.0 < args.pos_rate < 1.0:
        raise SystemExit("--pos-rate must be in (0, 1).")
    if args.n <= 0:
        raise SystemExit("--n must be positive.")

    df = generate_demo(
        n=args.n,
        seed=args.seed,
        pos_rate=args.pos_rate,
        signal=args.signal,
        noise=args.noise,
        latent_scale=args.latent_scale,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
