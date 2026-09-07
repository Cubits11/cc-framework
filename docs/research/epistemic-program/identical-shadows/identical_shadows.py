#!/usr/bin/env python3
"""Identical Shadows — reproduce every table in FINDING.md from scratch.

Re-derives the CC-Framework E1 reference intervals and fails loudly on mismatch,
then computes the fiber geometry, the identification-gap scaling, and the flat
composition bound.

    python3 identical_shadows.py            # run everything
    python3 identical_shadows.py --check    # reference check only; exit 1 on drift

Requires numpy and scipy.
"""

from __future__ import annotations

import sys
from itertools import combinations
from math import comb

import numpy as np
from scipy.optimize import linprog

TOL = 1e-9


def bits(i: int, k: int) -> list[int]:
    """Little-endian: bit t of atom index i is 1 iff guardrail t+1 fails."""
    return [(i >> t) & 1 for t in range(k)]


def moment_rows(k: int, marginals=None, pairs=None):
    """Constraint system: total mass, then each supplied marginal and pair."""
    N = 1 << k
    A: list[list[float]] = [[1.0] * N]
    b: list[float] = [1.0]
    if marginals is not None:
        for t in range(k):
            if marginals[t] is None:
                continue
            A.append([1.0 if bits(i, k)[t] else 0.0 for i in range(N)])
            b.append(float(marginals[t]))
    if pairs is not None:
        for idx, (u, v) in enumerate(combinations(range(k), 2)):
            if pairs[idx] is None:
                continue
            A.append([1.0 if (bits(i, k)[u] and bits(i, k)[v]) else 0.0 for i in range(N)])
            b.append(float(pairs[idx]))
    return np.array(A), np.array(b)


def all_fail_interval(k: int, marginals, pairs=None):
    """Sharp [lower, upper] for P(every guardrail fails) under the given evidence."""
    A, b = moment_rows(k, marginals, pairs)
    N = 1 << k
    c = np.zeros(N)
    c[N - 1] = 1.0
    lo = linprog(c, A_eq=A, b_eq=b, bounds=[(0, 1)] * N, method="highs")
    hi = linprog(-c, A_eq=A, b_eq=b, bounds=[(0, 1)] * N, method="highs")
    if lo.status != 0 or hi.status != 0:
        raise ValueError(f"infeasible evidence: {lo.message}")
    return max(0.0, lo.fun), -hi.fun


# --------------------------------------------------------------------------- #
# 1. reference check against the frozen E1 artifact
# --------------------------------------------------------------------------- #
E1 = {
    # name:            atoms (little-endian),                    I0,          I2
    "S1  independent": ([0.125] * 8, (0, 0.5), (0, 0.25)),
    "S2  common cause": ([0.5, 0, 0, 0, 0, 0, 0, 0.5], (0, 0.5), (0.5, 0.5)),
    "S3  mutually exclusive": ([0.25, 0.25, 0.25, 0, 0.25, 0, 0, 0], (0, 0.25), (0, 0)),
    "S4a even parity": ([0.25, 0, 0, 0.25, 0, 0.25, 0.25, 0], (0, 0.5), (0, 0.25)),
    "S4b odd parity": ([0, 0.25, 0.25, 0, 0.25, 0, 0, 0.25], (0, 0.5), (0, 0.25)),
}


def moments_of(atoms, k=3):
    x = np.asarray(atoms, float)
    m = [float(sum(x[i] for i in range(1 << k) if bits(i, k)[t])) for t in range(k)]
    p = [
        float(sum(x[i] for i in range(1 << k) if bits(i, k)[u] and bits(i, k)[v]))
        for u, v in combinations(range(k), 2)
    ]
    return m, p


def check_reference() -> bool:
    print("REFERENCE CHECK — CC-Framework E1, regimes I0 (marginals) and I2 (+ all pairs)")
    print(f"{'scenario':<24}{'truth':>8}{'product':>10}{'I0 sharp':>18}{'I2 sharp':>18}   verdict")
    ok = True
    for name, (atoms, e_i0, e_i2) in E1.items():
        m, p = moments_of(atoms)
        i0 = all_fail_interval(3, m, None)
        i2 = all_fail_interval(3, m, p)
        good = (
            abs(i0[0] - e_i0[0]) < TOL
            and abs(i0[1] - e_i0[1]) < TOL
            and abs(i2[0] - e_i2[0]) < TOL
            and abs(i2[1] - e_i2[1]) < TOL
        )
        ok &= good

        def f(t):
            return f"[{t[0]:.4g}, {t[1]:.4g}]"

        print(
            f"{name:<24}{atoms[7]:>8.4g}{m[0] * m[1] * m[2]:>10.4g}"
            f"{f(i0):>18}{f(i2):>18}   {'MATCH' if good else 'DIFFER'}"
        )
    print()
    print("  S1, S4a and S4b submit identical evidence — singletons (0.5, 0.5, 0.5),")
    print("  pairs (0.25, 0.25, 0.25) — and hold truths 0.125, 0 and 0.25.")
    return ok


# --------------------------------------------------------------------------- #
# 2. fiber geometry
# --------------------------------------------------------------------------- #
def fiber_geometry() -> None:
    print("\nFIBER GEOMETRY (k = 3, equal rates)")
    A, _ = moment_rows(3, [0.5, 0.5, 0.5], [0.25, 0.25, 0.25])
    rank = np.linalg.matrix_rank(A)
    print(f"  atoms 8, moment rows {A.shape[0]}, rank {rank} -> affine fiber dimension {8 - rank}")
    even = np.array([0.25, 0, 0, 0.25, 0, 0.25, 0.25, 0])
    odd = np.array([0, 0.25, 0.25, 0, 0.25, 0, 0, 0.25])
    indep = np.array([0.125] * 8)
    print(f"  even and odd parity share a moment vector: {np.allclose(A @ even, A @ odd)}")
    print(f"  independence law is their exact midpoint:  {np.allclose((even + odd) / 2, indep)}")
    _, _, vt = np.linalg.svd(A)
    d = vt[rank:][0]
    print(f"  null direction (normalised): {np.round(d / np.abs(d).max(), 3)}")
    print(f"  P(all three fail) at even end / midpoint / odd end: {even[7]}, {indep[7]}, {odd[7]}")


# --------------------------------------------------------------------------- #
# 3. how much of the joint law singles + pairs pin down
# --------------------------------------------------------------------------- #
def identification_gap(kmax: int = 10) -> None:
    print("\nIDENTIFICATION GAP")
    print("Affine dimension shares only; these percentages are not information fractions.")
    print(f"{'layers':>7}{'atoms':>8}{'free':>7}{'pinned':>8}{'unmeasured':>12}{'dim share':>10}")
    for k in range(2, kmax + 1):
        N = 1 << k
        A, _ = moment_rows(k, [0.05] * k, [0.0025] * comb(k, 2))
        rank = np.linalg.matrix_rank(A)
        pinned = k + comb(k, 2)
        print(f"{k:>7}{N:>8}{N - 1:>7}{pinned:>8}{N - rank:>12}{100 * pinned / (N - 1):>9.2f}%")


# --------------------------------------------------------------------------- #
# 4. the bound does not move with layer count
# --------------------------------------------------------------------------- #
def flat_bound(m: float = 0.05, kmax: int = 8) -> None:
    print(f"\nCOMPOSITION BOUND, per-layer miss rate {m}, every pair measured at m^2")
    print(
        f"{'layers':>7}{'independence':>16}{'sharp lower':>14}{'sharp upper':>14}{'overstatement':>16}"
    )
    for k in range(2, kmax + 1):
        lo, hi = all_fail_interval(k, [m] * k, [m * m] * comb(k, 2))
        prod = m**k
        over = "—" if abs(hi / prod - 1) < 0.01 else f"x{hi / prod:,.0f}"
        print(f"{k:>7}{prod:>16.6g}{lo:>14.6g}{hi:>14.6g}{over:>16}")
    print("\n  In this displayed symmetric regime, the upper bound equals min over pairs.")
    print("  This is not the general sharp bound; the counterexample search below tests it.")


def main() -> int:
    ok = check_reference()
    if "--check" in sys.argv:
        return 0 if ok else 1
    fiber_geometry()
    identification_gap()
    flat_bound()
    where_independence_sits()
    min_pair_is_not_general()
    external_cases()
    ensemble_channel()
    disclosure_ladder()
    if not ok:
        print("\nREFERENCE CHECK FAILED — the solver disagrees with the frozen E1 artifact.")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# 5. revision 2 additions
# --------------------------------------------------------------------------- #
def where_independence_sits(m_values=(0.5, 0.3, 0.1, 0.05, 0.01)) -> None:
    """Corrects the revision-1 'midpoint' claim: it sits at exactly m of the way across."""
    print("\nWHERE THE INDEPENDENCE ASSUMPTION SITS IN THE INTERVAL")
    print(f"{'m':>8}{'sharp interval':>26}{'product':>14}{'position':>14}")
    for m in m_values:
        lo, hi = all_fail_interval(3, [m] * 3, [m * m] * 3)
        prod = m**3
        pos = (prod - lo) / (hi - lo) * 100 if hi > lo else float("nan")
        print(f"{m:>8}{f'[{lo:.6g}, {hi:.6g}]':>26}{prod:>14.6g}{pos:>13.1f}%")
    print("  => exactly m of the way across. Better guardrails make the assumption")
    print("     MORE optimistic relative to what the evidence permits, not less.")


def min_pair_is_not_general(trials: int = 20_000, seed: int = 7, stop_after: int = 3) -> None:
    """Counterexamples to 'sharp ceiling = min over pairs'."""
    from itertools import combinations as _c

    rng = np.random.default_rng(seed)
    print("\nIS 'CEILING = MIN OVER PAIRS' THE GENERAL SHARP BOUND?")
    found, tested = 0, 0
    for _ in range(trials):
        m = rng.uniform(0, 1, 3)
        p = [rng.uniform(max(0.0, m[u] + m[v] - 1), min(m[u], m[v])) for u, v in _c(range(3), 2)]
        try:
            _lo, hi = all_fail_interval(3, list(m), p)
        except ValueError:
            continue
        tested += 1
        if hi < min(p) - 1e-7:
            found += 1
            print(f"  m={np.round(m, 4)} p={np.round(p, 4)}")
            print(
                f"     LP upper={hi:.6f}  min-pair={min(p):.6f}  1-Sm+Sp={1 - sum(m) + sum(p):.6f}"
            )
            if found >= stop_after:
                break
    print(f"  feasible tested={tested}, strict counterexamples={found}")
    print("  => No. Inclusion-exclusion binds instead. The LP is the authority.")


def external_pair(name, n, det_a, det_b, det_union):
    """Reconstruct a 2x2 from published marginals + a block-on-either rate."""
    from scipy.stats import fisher_exact

    ma, mb = 1 - det_a, 1 - det_b
    both = 1 - det_union
    a = round(both * n)
    b = round(ma * n) - a
    c = round(mb * n) - a
    d = n - a - b - c
    phi = (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    _, pv = fisher_exact([[a, b], [c, d]])
    print(f"\n  {name}")
    print(f"    published: {det_a:.1%} / {det_b:.1%} / union {det_union:.1%}, n={n}")
    print(f"    misses {round(ma * n)} and {round(mb * n)}; BOTH MISS {a} ({both:.2%})")
    print(
        f"    independence predicts {ma * mb:.4%} ({ma * mb * n:.2f} items) -> ratio x{both / (ma * mb):.2f}"
    )
    print(f"    phi={phi:.4f}  Fisher exact p={pv:.3g}")
    print(
        f"    residual coverage of the addition: {(round(ma * n) - a)}/{round(ma * n)} = {(round(ma * n) - a) / round(ma * n):.1%}"
    )


def external_cases() -> None:
    print("\nEXTERNAL CASES (Multimodal Safeguard Bench, published aggregates only)")
    external_pair("A  LG4 + LG3V, harmful text", 200, 0.925, 0.890, 0.960)
    external_pair("B  LG4 + SG2,  harmful image", 200, 0.820, 0.870, 0.970)


def ensemble_channel() -> None:
    """Every 'block on either' row over a subset S reports P(all of S miss)."""
    from itertools import combinations as _c

    print("\nWHAT A PUBLISHED ENSEMBLE TABLE DETERMINES")

    def free_dims(k, subsets):
        N = 1 << k
        rows = [[1.0] * N]
        for S in subsets:
            rows.append([1.0 if all(bits(i, k)[t] for t in S) else 0.0 for i in range(N)])
        return N - np.linalg.matrix_rank(np.array(rows))

    singles, pairs, triple = [(0,), (1,), (2,)], [(0, 1), (0, 2), (1, 2)], [(0, 1, 2)]
    for label, subs in [
        ("detection rates only", singles),
        ("+ one pairwise ensemble", [*singles, (0, 1)]),
        ("+ all three pairwise", singles + pairs),
        ("+ all pairs and the 3-way", singles + pairs + triple),
    ]:
        d = free_dims(3, subs)
        print(f"  {label:<28} free dims={d}{'   FULLY DETERMINED' if d == 0 else ''}")
    print("  full subset table, all k:", end=" ")
    print(
        ", ".join(
            f"k={k}:{free_dims(k, [S for n in range(1, k + 1) for S in _c(range(k), n)])}"
            for k in range(2, 7)
        ),
        "(all zero: the transform is invertible)",
    )


def disclosure_ladder() -> None:
    """Which aggregate disclosures identify which decision quantity.

    Key point: the all-miss target needs ONE row, not 2^k - 1.
    P(all k miss) = 1 - P(at least one catches) = 1 - (full-stack block-on-any).
    """
    from itertools import combinations as _c

    print("\nDISCLOSURE LADDER — what each published aggregate identifies (k = 5)")
    k = 5
    N = 1 << k

    def free_dims(subsets):
        rows = [[1.0] * N]
        for S in subsets:
            rows.append([1.0 if all(bits(i, k)[t] for t in S) else 0.0 for i in range(N)])
        return N - np.linalg.matrix_rank(np.array(rows))

    singles = [(t,) for t in range(k)]
    full = [tuple(range(k))]
    prefixes = [tuple(range(j)) for j in range(2, k + 1)]
    pairs = list(_c(range(k), 2))
    allsub = [S for n in range(1, k + 1) for S in _c(range(k), n)]

    rows = [
        ("each guard's own rate", singles, "standalone performance only"),
        (
            "+ block-on-any for the whole stack",
            singles + full,
            "P(all k miss) EXACTLY — the target",
        ),
        ("+ ordered prefix rates", singles + prefixes, "residual coverage of each added guard"),
        ("+ all pairwise rates", singles + pairs, "pairwise complementarity structure"),
        ("+ every non-empty subset", allsub, "entire joint law (Mobius inversion)"),
    ]
    print(f"{'published':<38}{'rows':>6}{'joint dims left':>18}   identifies")
    for label, subs, what in rows:
        print(f"{label:<38}{len(subs) + 1:>6}{free_dims(subs):>18}   {what}")
    print("\n  The decision quantity does not require the full joint law. One row does it.")
    print("  Reconstructing the whole joint is a different, larger ask.")


if __name__ == "__main__":
    raise SystemExit(main())
