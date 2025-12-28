# tests/unit/test_cc_metrics.py
"""
Unit tests for CC metrics computation.

Why this file exists
--------------------
These tests verify the *statistical core* of the CC framework:

- compute_j_statistic: estimates (J, p0, p1) from two-world outcomes
- bootstrap_ci_with_cc: produces stable bootstrap estimates + CC metrics

Design constraints (important)
------------------------------
Your core data spine (cc.core.models.AttackResult) enforces hardened invariants,
including:

- transcript_hash must be a 64-character hex digest (BLAKE3 hexdigest length)
- success is StrictBool (must be True/False, no coercion)
- world_bit is an IntEnum (WorldBit), but Pydantic can coerce ints safely

Therefore, tests MUST NOT construct AttackResult with transcript_hash="".

Instead, tests should use AttackResult.from_transcript(...) which generates a
valid transcript_hash in a deterministic way.

Determinism guarantees
----------------------
- We set timestamp=0.0 for all synthetic results so hashes don’t become time-dependent.
- We generate transcript strings deterministically from (world_bit, attack_id, seed).
- The bootstrap stability test expects bootstrap_ci_with_cc to be internally deterministic
  (e.g., fixed RNG seed inside the function). If you later change bootstrap seeding,
  this test will correctly catch the regression.
"""

from __future__ import annotations

import numpy as np

from cc.core.models import AttackResult, WorldBit
from cc.core.stats import bootstrap_ci_with_cc, compute_j_statistic


def _mk_result(
    *,
    world_bit: WorldBit,
    success: bool,
    attack_id: str,
    guardrails_applied: str,
    rng_seed: int,
    timestamp: float = 0.0,
) -> AttackResult:
    """
    Create a valid AttackResult using the *official* constructor.

    Notes:
    - We build a deterministic transcript that is unique per record.
    - transcript_hash is derived from transcript (no salt) => stable across runs.
    - timestamp is pinned to 0.0 to avoid accidental time-dependence.

    If you later decide that tests should include salted hashes, do it here
    (and store the salt in the test fixture), NOT scattered across test bodies.
    """
    transcript = f"world={int(world_bit)}|attack_id={attack_id}|seed={rng_seed}|gr={guardrails_applied}"
    return AttackResult.from_transcript(
        world_bit=world_bit,
        success=success,
        attack_id=attack_id,
        transcript=transcript,
        guardrails_applied=guardrails_applied,
        rng_seed=rng_seed,
        timestamp=timestamp,
    )


class TestCCMetrics:
    def test_perfect_defense(self):
        """
        Perfect defense case:
          - World 0 (baseline): success always True  => p0 = 1.0
          - World 1 (protected): success always False => p1 = 0.0
          => J = p0 - p1 = 1.0
        """
        results = [
            _mk_result(
                world_bit=WorldBit.BASELINE,
                success=True,
                attack_id=f"a{i}",
                guardrails_applied="",
                rng_seed=i,
                timestamp=0.0,
            )
            for i in range(100)
        ] + [
            _mk_result(
                world_bit=WorldBit.PROTECTED,
                success=False,
                attack_id=f"a{i}",
                guardrails_applied="g1",
                rng_seed=i,
                timestamp=0.0,
            )
            for i in range(100, 200)
        ]

        j_stat, p0, p1 = compute_j_statistic(results)

        assert np.isclose(j_stat, 1.0)
        assert np.isclose(p0, 1.0)
        assert np.isclose(p1, 0.0)

    def test_no_effect(self):
        """
        No-effect case:
          - Same success rate in both worlds (≈0.6)
          => J should be near 0.

        This is a stochastic test but uses a fixed RNG seed and a large n
        to make failure very unlikely unless logic changes.
        """
        rng = np.random.default_rng(42)
        results: list[AttackResult] = []

        for i in range(1000):
            world_int = int(rng.integers(0, 2))
            world = WorldBit(world_int)

            # Same success rate regardless of world
            success = bool(rng.random() < 0.6)

            results.append(
                _mk_result(
                    world_bit=world,
                    success=success,
                    attack_id=f"a{i}",
                    guardrails_applied="g1" if world == WorldBit.PROTECTED else "",
                    rng_seed=i,
                    timestamp=0.0,
                )
            )

        j_stat, p0, p1 = compute_j_statistic(results)

        # Near-zero check: robust but not absurdly tight.
        assert abs(j_stat) < 0.1

    def test_bootstrap_stability(self):
        """
        Bootstrap stability:
        Run bootstrap twice with identical inputs. If bootstrap_ci_with_cc uses a fixed
        internal RNG seed (or is purely deterministic), outputs must match exactly.

        If you later make bootstrap RNG externally-seeded (recommended for research
        control), update this test to pass a seed explicitly and assert stability
        under that seed.
        """
        results: list[AttackResult] = []
        for i in range(500):
            world = WorldBit(i % 2)

            # Different success patterns per world to ensure non-trivial J
            success = ((i % 3) == 0) if world == WorldBit.BASELINE else ((i % 5) == 0)

            results.append(
                _mk_result(
                    world_bit=world,
                    success=bool(success),
                    attack_id=f"a{i}",
                    guardrails_applied="",
                    rng_seed=i,
                    timestamp=0.0,
                )
            )

        j_individual = {"g1": 0.2}

        result1 = bootstrap_ci_with_cc(results, j_individual, B=500)
        result2 = bootstrap_ci_with_cc(results, j_individual, B=500)

        assert abs(result1.j_statistic - result2.j_statistic) < 1e-10
        assert abs(result1.cc_max - result2.cc_max) < 1e-10
        assert result1.ci_j == result2.ci_j
