"""
experiments.correlation_cliff.simulate

Public import surface for the simulation subpackage (backwards-compatible).
"""

from importlib import import_module

from .config import SimConfig  # alias guaranteed by config.py patch
from .paths import p11_from_path


def _export(name: str, modules: tuple[str, ...]) -> object:
    for m in modules:
        mod = import_module(f"{__name__}.{m}")
        if hasattr(mod, name):
            return getattr(mod, name)
    raise ImportError(f"{__name__}: could not find '{name}' in modules {modules}")


# These historically existed and are used by run_all.py + tests.
simulate_grid = _export("simulate_grid", ("core", "utils"))
summarize_simulation = _export("summarize_simulation", ("core", "utils"))
simulate_replicate_at_lambda = _export("simulate_replicate_at_lambda", ("core", "utils"))

__all__ = [
    "SimConfig",
    "p11_from_path",
    "simulate_replicate_at_lambda",
    "simulate_grid",
    "summarize_simulation",
]
