# tests/conftest.py
from __future__ import annotations

import time
import pytest


def _has_pytest_benchmark() -> bool:
    try:
        import pytest_benchmark  # noqa: F401
        return True
    except Exception:
        return False


class _NoOpBenchmark:
    """
    Minimal stub compatible with common pytest-benchmark usage patterns.
    - benchmark(fn, *args, **kwargs)
    - benchmark.pedantic(fn, args=..., kwargs=..., rounds=..., iterations=...)
    - benchmark.extra_info (dict)
    - benchmark.stats (None)
    """

    def __init__(self) -> None:
        self.extra_info: dict[str, object] = {}
        self.stats = None

    def __call__(self, func, *args, **kwargs):
        return func(*args, **kwargs)

    def pedantic(
        self,
        func,
        args=(),
        kwargs=None,
        rounds: int = 1,
        iterations: int = 1,
        warmup_rounds: int = 0,
    ):
        if kwargs is None:
            kwargs = {}

        # Warmup (do nothing special, just execute)
        for _ in range(max(0, int(warmup_rounds))):
            for _ in range(max(1, int(iterations))):
                func(*args, **kwargs)

        # Timed runs (kept lightweight)
        start = time.perf_counter()
        last = None
        for _ in range(max(1, int(rounds))):
            for _ in range(max(1, int(iterations))):
                last = func(*args, **kwargs)
        end = time.perf_counter()

        self.extra_info["noop_total_seconds"] = end - start
        self.extra_info["noop_rounds"] = int(rounds)
        self.extra_info["noop_iterations"] = int(iterations)
        return last


if not _has_pytest_benchmark():
    @pytest.fixture
    def benchmark():
        return _NoOpBenchmark()