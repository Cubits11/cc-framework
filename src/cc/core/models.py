"""Core data models for the CC framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AttackResult:
    """Result of a single attack session in two-world protocol"""

    world_bit: int  # 0 or 1
    success: bool
    attack_id: str
    transcript_hash: str
    guardrails_applied: str
    rng_seed: int
    timestamp: float
    session_id: str = ""
    attack_strategy: str = ""
    utility_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary including all fields."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class GuardrailSpec:
    """Specification for a guardrail configuration"""

    name: str
    params: Dict[str, Any]
    calibration_fpr_target: float = 0.05
    calibration_data_hash: str = ""
    version: str = "1.0"

    def __post_init__(self):
        if not self.params:
            self.params = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary including all fields."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class WorldConfig:
    """Configuration for a world in two-world protocol"""

    world_id: int  # 0 or 1
    guardrail_stack: List[GuardrailSpec]
    utility_profile: Dict[str, Any]
    env_hash: str = ""
    baseline_success_rate: float = 0.6

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary including all fields."""
        return {
            k: [gr.to_dict() for gr in getattr(self, k)]
            if k == "guardrail_stack"
            else getattr(self, k)
            for k in self.__dataclass_fields__
        }


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    experiment_id: str
    n_sessions: int
    attack_strategies: List[str]
    guardrail_configs: Dict[str, List[GuardrailSpec]]
    utility_target: float = 0.9
    utility_tolerance: float = 0.02
    random_seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary including all fields."""
        return {
            "experiment_id": self.experiment_id,
            "n_sessions": self.n_sessions,
            "attack_strategies": list(self.attack_strategies),
            "guardrail_configs": {
                k: [gr.to_dict() for gr in v]
                for k, v in self.guardrail_configs.items()
            },
            "utility_target": self.utility_target,
            "utility_tolerance": self.utility_tolerance,
            "random_seed": self.random_seed,
        }


@dataclass
class CCResult:
    """Results of CC analysis."""

    j_empirical: float
    cc_max: float
    delta_add: float
    cc_multiplicative: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    bootstrap_samples: Optional[np.ndarray] = None
    n_sessions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary including all fields."""
        data: Dict[str, Any] = {
            "j_empirical": self.j_empirical,
            "cc_max": self.cc_max,
            "delta_add": self.delta_add,
            "cc_multiplicative": self.cc_multiplicative,
            "confidence_interval": list(self.confidence_interval)
            if self.confidence_interval is not None
            else None,
            "bootstrap_samples": self.bootstrap_samples.tolist()
            if self.bootstrap_samples is not None
            else None,
            "n_sessions": self.n_sessions,
        }
        return data


@dataclass
class AttackStrategy:
    """Configuration for attack strategy."""

    name: str
    params: Dict[str, Any]
    vocabulary: List[str] = field(default_factory=list)
    success_threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary including all fields."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
