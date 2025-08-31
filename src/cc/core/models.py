# src/cc/core/models.py
"""
Core data models for CC framework
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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
        """Convert to dictionary for serialization"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


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


@dataclass
class WorldConfig:
    """Configuration for a world in two-world protocol"""

    world_id: int  # 0 or 1
    guardrail_stack: List[GuardrailSpec]
    utility_profile: Dict[str, Any]
    env_hash: str = ""
    baseline_success_rate: float = 0.6
