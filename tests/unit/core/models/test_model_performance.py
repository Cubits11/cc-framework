# tests/unit/models/test_model_performance.py

import time
import timeit

import pytest

from cc.core.models import (
    NUMPY_AVAILABLE,
    AttackResult,
    CCResult,
    ModelBase,
    WorldBit,
    _hash_json,
)

if NUMPY_AVAILABLE:  # type: ignore[truthy-bool]
    import numpy as np  # type: ignore[import]


def _measure_runtime(func, *, number: int = 1000) -> float:
    """Helper to measure wall-clock runtime of a callable."""
    start = time.perf_counter()
    for _ in range(number):
        func()
    return time.perf_counter() - start


# ---------------------------------------------------------------------
# blake3_hash cache performance
# ---------------------------------------------------------------------


@pytest.mark.slow
def test_blake3_hash_cache_speedup_relative():
    """Cached blake3_hash calls must be faster than recomputing each time.

    WHAT:
        Compare repeated cached vs uncached hashing on the same ModelBase instance.
    WHY:
        blake3_hash is a hot path; caching must provide a measurable speedup.
    THREAT:
        A regression disables caching or accidentally recomputes, multiplying costs
        across large experiments.
    """

    class PerfModel(ModelBase):
        x: int = 1
        y: str = "abc"

    m = PerfModel()

    # Warm up both code paths to avoid one-time costs dominating measurements.
    m.blake3_hash(use_cache=False)
    m.blake3_hash()  # populate cache

    def uncached():
        m.blake3_hash(use_cache=False)

    def cached():
        m.blake3_hash()

    n_calls = 3000
    uncached_time = _measure_runtime(uncached, number=n_calls)
    cached_time = _measure_runtime(cached, number=n_calls)

    # Cached must be faster in absolute terms.
    assert cached_time < uncached_time

    # And the ratio should indicate a meaningful (though not brittle) speedup.
    speedup = uncached_time / cached_time
    assert speedup > 1.1, f"Expected >10% speedup from cache, got {speedup:.2f}x"


# ---------------------------------------------------------------------
# _hash_json bulk throughput
# ---------------------------------------------------------------------


@pytest.mark.slow
def test_hash_json_bulk_10k_under_reasonable_budget():
    """Hashing 10k small JSON-like dicts must stay within a reasonable budget.

    WHAT:
        Measure time to hash 10,000 small dictionaries.
    WHY:
        _hash_json is used extensively in experiment tracking and deduplication.
    THREAT:
        An accidental O(n^2) operation (e.g., repeated sorting, copying) would
        silently blow up runtime on realistic experiment sizes.
    """
    objs: list[dict[str, int]] = [{"k": i, "v": i * 2} for i in range(10_000)]

    def worker():
        for obj in objs:
            _hash_json(obj)

    duration = timeit.timeit(worker, number=1)

    # This threshold is generous but catches catastrophic regressions.
    assert duration < 2.0, f"_hash_json too slow: {duration:.3f}s for 10k objects"


# ---------------------------------------------------------------------
# AttackResult.from_transcript inner-loop cost
# ---------------------------------------------------------------------


@pytest.mark.slow
def test_attack_result_from_transcript_scaling_1k():
    """Creating 1k AttackResult objects from transcripts must be fast enough.

    WHAT:
        Call AttackResult.from_transcript 1,000 times on small payloads.
    WHY:
        This is the core inner loop of many experiments.
    THREAT:
        Adding heavy per-call overhead (e.g. JSON dumps, deep copies) would make
        experiments unusably slow.
    """

    transcripts = [f"payload-{i}" for i in range(1000)]

    def worker():
        for i, payload in enumerate(transcripts):
            AttackResult.from_transcript(
                world_bit=WorldBit.BASELINE if i % 2 == 0 else WorldBit.PROTECTED,
                success=bool(i % 2),
                attack_id=f"id-{i}",
                transcript=payload,
                guardrails_applied="none",
                rng_seed=i,
            )

    duration = timeit.timeit(worker, number=1)
    # Loose upper bound - on a modern laptop this should be far below this.
    assert duration < 2.0, (
        f"AttackResult.from_transcript too slow: {duration:.3f}s for 1k "
        "creations; check for accidental heavy work in the hot path."
    )


# ---------------------------------------------------------------------
# CCResult bootstrap-array construction cost
# ---------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not NUMPY_AVAILABLE, reason="numpy not available")
def test_ccresult_large_bootstrap_samples_init_under_budget():
    """Initializing CCResult with 20k bootstrap samples should be reasonably fast.

    WHAT:
        Construct a CCResult with 20,000 bootstrap samples.
    WHY:
        CI estimation workflows may pass large bootstrap arrays.
    THREAT:
        If CCResult does per-element Python work beyond simple validation,
        it can dominate runtime for statistical analysis.
    """
    # Use a deterministic, NaN/Inf-free sequence for stability.
    arr = np.linspace(0.0, 1.0, 20_000, dtype=float)  # type: ignore[attr-defined]

    def worker():
        CCResult(j_empirical=0.1, cc_max=0.2, delta_add=0.1, bootstrap_samples=arr)

    duration = timeit.timeit(worker, number=1)
    assert duration < 2.0, (
        f"CCResult init too slow: {duration:.3f}s for 20k bootstrap samples; "
        "check for unnecessary per-element overhead."
    )
