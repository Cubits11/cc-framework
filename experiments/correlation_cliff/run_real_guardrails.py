"""Run real-guardrail dependence summaries through the Paper 1 benchmark CLI.

Correlation-cliff experiments remain a follow-on research direction. This file
is kept as a compatibility entry point for older experiment docs, but it now
delegates to ``cc.evals.dependence_benchmark`` so real guardrail runs use the
same failure-event convention and artifact schema as Paper 1.
"""

from __future__ import annotations

from collections.abc import Iterable

from cc.evals.dependence_benchmark import main as dependence_benchmark_main


def main(argv: Iterable[str] | None = None) -> int:
    """Delegate to ``cc.evals.dependence_benchmark.main``."""

    return dependence_benchmark_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
