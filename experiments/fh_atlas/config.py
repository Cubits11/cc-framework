from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any, Dict, List


VALID_COMPOSITIONS = {"serial_or", "parallel_and"}


@dataclass
class FHAtlasConfig:
    """Configuration for the FH dependence atlas experiments."""

    seed: int = 7
    k_values: List[int] = field(default_factory=lambda: [2, 3])
    miss_rate_grid: List[float] = field(default_factory=lambda: [0.05, 0.1])
    fpr_rate_grid: List[float] = field(default_factory=lambda: [0.01, 0.05])
    copula_families: List[str] = field(
        default_factory=lambda: [
            "independence",
            "comonotonic",
            "countermonotonic",
            "clayton",
            "gumbel",
            "frank",
        ]
    )
    theta_grid: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    composition_types: List[str] = field(default_factory=lambda: ["serial_or"])
    sample_sizes: List[int] = field(default_factory=lambda: [500])
    bootstrap_iterations: int = 500
    output_dir: str = "outputs/fh_atlas"

    def validate(self) -> None:
        if not self.k_values or any(k <= 0 for k in self.k_values):
            raise ValueError("k_values must contain positive integers")
        if not self.miss_rate_grid or not self.fpr_rate_grid:
            raise ValueError("Rate grids must be non-empty")
        for rate in self.miss_rate_grid + self.fpr_rate_grid:
            if rate < 0.0 or rate > 1.0:
                raise ValueError("Rates must be in [0, 1]")
        if not self.sample_sizes or any(n <= 0 for n in self.sample_sizes):
            raise ValueError("sample_sizes must contain positive integers")
        if any(name not in VALID_COMPOSITIONS for name in self.composition_types):
            raise ValueError(f"composition_types must be subset of {VALID_COMPOSITIONS}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "FHAtlasConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        config = cls(**data)
        config.validate()
        return config
