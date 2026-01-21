"""Top-level package for the cc framework."""

from importlib import metadata as _metadata

from . import adapters, analysis, cartographer, core, exp, guardrails, io, utils

try:
    __version__ = _metadata.version("cc-framework")
except _metadata.PackageNotFoundError:  # pragma: no cover - during local usage
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "adapters",
    "analysis",
    "cartographer",
    "core",
    "exp",
    "guardrails",
    "io",
    "utils",
]
