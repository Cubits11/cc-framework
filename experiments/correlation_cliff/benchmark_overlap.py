"""
Benchmark: empirical overlap between independently defined labeling schemas.

This benchmark simulates two labeling schemas (e.g., harm vs autonomy), measures
empirical overlap, and evaluates whether the OCP-predicted bounds (FH bounds)
hold. It also includes negative controls (independent and shuffled labels).
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from typing import Iterable, List, Optional

import numpy as np

from .theory_core import CONFIG, fh_bounds


@dataclass(frozen=True)
class BenchmarkResult:
    scenario: str
    n: int
    pA_emp: float
    pB_emp: float
    p11_emp: float
    fh_lower: float
    fh_upper: float
    within_bounds: bool

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "n": self.n,
            "pA_emp": self.pA_emp,
            "pB_emp": self.pB_emp,
            "p11_emp": self.p11_emp,
            "fh_lower": self.fh_lower,
            "fh_upper": self.fh_upper,
            "within_bounds": self.within_bounds,
        }


def _simulate_joint_labels(
    n: int,
    pA: float,
    pB: float,
    p11: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    p00 = 1.0 - pA - pB + p11
    p10 = pA - p11
    p01 = pB - p11
    probs = np.array([p00, p01, p10, p11], dtype=float)
    if np.any(probs < -CONFIG.eps_prob):
        raise ValueError("Infeasible joint probabilities; check pA, pB, p11.")
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / probs.sum()

    draws = rng.choice(4, size=n, p=probs)
    a = (draws == 2) | (draws == 3)
    b = (draws == 1) | (draws == 3)
    return a.astype(int), b.astype(int)


def _empirical_overlap(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    pA = float(np.mean(a))
    pB = float(np.mean(b))
    p11 = float(np.mean((a == 1) & (b == 1)))
    return pA, pB, p11


def _evaluate_bounds(pA: float, pB: float, p11: float) -> tuple[float, float, bool]:
    lower, upper = fh_bounds(pA, pB)
    within = (lower - CONFIG.eps_prob) <= p11 <= (upper + CONFIG.eps_prob)
    return float(lower), float(upper), bool(within)


def run_benchmark(
    *,
    n: int,
    pA: float,
    pB: float,
    p11: Optional[float],
    seed: Optional[int],
) -> List[BenchmarkResult]:
    rng = np.random.default_rng(seed)
    if p11 is None:
        p11 = pA * pB

    a, b = _simulate_joint_labels(n, pA, pB, p11, rng)
    base_result = _summarize("empirical", a, b)

    # Negative control 1: independent labels with same marginals.
    a_ind = rng.binomial(1, pA, size=n).astype(int)
    b_ind = rng.binomial(1, pB, size=n).astype(int)
    independent_result = _summarize("neg_control_independent", a_ind, b_ind)

    # Negative control 2: shuffled pairing (breaks any dependence structure).
    b_shuffled = rng.permutation(b)
    shuffled_result = _summarize("neg_control_shuffled", a, b_shuffled)

    return [base_result, independent_result, shuffled_result]


def _summarize(name: str, a: np.ndarray, b: np.ndarray) -> BenchmarkResult:
    pA_emp, pB_emp, p11_emp = _empirical_overlap(a, b)
    lower, upper, within = _evaluate_bounds(pA_emp, pB_emp, p11_emp)
    return BenchmarkResult(
        scenario=name,
        n=int(a.size),
        pA_emp=pA_emp,
        pB_emp=pB_emp,
        p11_emp=p11_emp,
        fh_lower=lower,
        fh_upper=upper,
        within_bounds=within,
    )


def _format_results(results: Iterable[BenchmarkResult]) -> str:
    lines = [
        "scenario\tn\tpA_emp\tpB_emp\tp11_emp\tfh_lower\tfh_upper\twithin_bounds",
    ]
    for r in results:
        lines.append(
            f"{r.scenario}\t{r.n}\t{r.pA_emp:.6f}\t{r.pB_emp:.6f}\t"
            f"{r.p11_emp:.6f}\t{r.fh_lower:.6f}\t{r.fh_upper:.6f}\t{r.within_bounds}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000, help="Number of samples")
    parser.add_argument("--pA", type=float, default=0.2, help="Schema A marginal rate")
    parser.add_argument("--pB", type=float, default=0.3, help="Schema B marginal rate")
    parser.add_argument(
        "--p11",
        type=float,
        default=None,
        help="Joint overlap rate (default: independence pA*pB)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()

    results = run_benchmark(n=args.n, pA=args.pA, pB=args.pB, p11=args.p11, seed=args.seed)
    print(_format_results(results))


if __name__ == "__main__":
    main()
