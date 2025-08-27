# src/cc/core/models.py
"""
Module: models
Purpose: Dataclasses and types for CC framework
Dependencies: dataclasses, typing
Author: Pranav Bhave
Date: 2025-08-27
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Literal
import numpy as np
from datetime import datetime

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

@dataclass
class GuardrailSpec:
    """Specification for a guardrail configuration"""
    name: str
    params: Dict[str, Any]
    calibration_fpr_target: float = 0.05
    calibration_data_hash: str = ""
    version: str = "1.0"
    
@dataclass
class WorldConfig:
    """Configuration for a world in two-world protocol"""
    world_id: int  # 0 or 1
    guardrail_stack: List[GuardrailSpec]
    utility_profile: Dict[str, Any]
    env_hash: str = ""
    baseline_success_rate: float = 0.6

@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_id: str
    n_sessions: int
    attack_strategies: List[str]
    guardrail_configs: Dict[str, List[GuardrailSpec]]
    utility_target: float = 0.9
    utility_tolerance: float = 0.02
    random_seed: int = 42
    
@dataclass
class CCResult:
    """Results of CC analysis"""
    j_empirical: float
    cc_max: float
    delta_add: float
    cc_multiplicative: Optional[float] = None
    confidence_interval: Optional[tuple] = None
    bootstrap_samples: Optional[np.ndarray] = None
    n_sessions: int = 0
    
@dataclass
class AttackStrategy:
    """Configuration for attack strategy"""
    name: str
    params: Dict[str, Any]
    vocabulary: List[str] = field(default_factory=list)
    success_threshold: float = 0.5