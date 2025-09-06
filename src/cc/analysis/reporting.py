"""cc/analysis/reporting.py
===========================

Lightweight helpers for transforming metric dictionaries produced by
:mod:`cc.analysis.cc_estimation` into human‑readable reports.  No external
reporting dependencies are required which keeps the utilities easy to use in
small scripts and notebooks.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

# Default precision for numeric values in reports
_PRECISION = 4


def summarize_metrics(
    metrics: Dict[str, float], precision: int = _PRECISION
) -> List[Tuple[str, float]]:
    """Return a stable, rounded list of metric key/value pairs.

    Args:
        metrics: Raw metrics dictionary.
        precision: Number of decimal places to retain.

    Returns:
        List of ``(metric, value)`` tuples sorted alphabetically with numerical
        values rounded for readability.  The stable ordering is helpful for
        deterministic report generation and unit testing.
    """

    return [
        (k, round(float(v), precision)) for k, v in sorted(metrics.items())
    ]


def metrics_to_markdown(summary: Iterable[Tuple[str, float]]) -> str:
    """Render a list of metrics as a Markdown table."""

    lines = ["| Metric | Value |", "| --- | --- |"]
    for metric, value in summary:
        lines.append(f"| {metric} | {value} |")
    return "\n".join(lines) + "\n"


def metrics_to_csv(summary: Iterable[Tuple[str, float]]) -> str:
    """Render a list of metrics as comma‑separated values."""

    lines = ["metric,value"]
    for metric, value in summary:
        lines.append(f"{metric},{value}")
    return "\n".join(lines) + "\n"