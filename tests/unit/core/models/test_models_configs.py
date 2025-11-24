# tests/unit/models/test_models_configs.py

"""
Configuration-style models: GuardrailSpec, WorldConfig, ExperimentConfig,
and AttackStrategy.

Scope:
- GuardrailSpec: params normalization, config_hash semantics, FPR validation
- WorldConfig: guardrail_stack normalization, env_hash semantics, baselines
- ExperimentConfig: attack_strategies/guardrail_configs normalization +
  basic validation and iso_time semantics
- AttackStrategy: simple roundtrip + params behaviour
"""

from typing import Any, Dict, Mapping, Sequence

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from cc.core.models import (
    AttackStrategy,
    ExperimentConfig,
    GuardrailSpec,
    WorldBit,
    WorldConfig,
    _iso_from_unix,
)


# ---------------------------------------------------------------------
# AttackStrategy
# ---------------------------------------------------------------------


def test_attack_strategy_basic_roundtrip():
    """
    AttackStrategy should be a thin name/params container with a clean dump.
    """
    strat = AttackStrategy(name="strat", params={"foo": "bar"})
    dumped = strat.model_dump()

    assert dumped["name"] == "strat"
    assert dumped["params"] == {"foo": "bar"}


@given(
    name=st.text(min_size=1, max_size=16),
    params=st.dictionaries(st.text(min_size=1, max_size=8), st.integers(), max_size=4),
)
def test_attack_strategy_params_can_be_empty_or_mapping(name: str, params: Dict[str, int]):
    """
    AttackStrategy.params must accept arbitrary JSON-like mappings and roundtrip.
    """
    strat = AttackStrategy(name=name, params=params)
    dumped = strat.model_dump()
    assert dumped["name"] == name
    assert dumped["params"] == params


# ---------------------------------------------------------------------
# GuardrailSpec
# ---------------------------------------------------------------------


def test_guardrail_spec_params_normalization_and_config_hash():
    """
    GuardrailSpec:
    - params can be Mapping-like and will be normalized to dict
    - config_hash must ignore calibration tuning and depend only on params/name
    """
    spec1 = GuardrailSpec(name="g1", params={"a": 1}, calibration_fpr_target=0.1)
    spec2 = GuardrailSpec(name="g1", params={"a": 1}, calibration_fpr_target=0.5)
    spec3 = GuardrailSpec(name="g1", params={"a": 2})

    class MyMap(Mapping[str, Any]):
        def __init__(self):
            self._data = {"a": 1}

        def __getitem__(self, k: str) -> Any:
            return self._data[k]

        def __iter__(self):
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

    spec4 = GuardrailSpec(name="g1", params=MyMap())

    assert spec1.params == {"a": 1}
    assert spec4.params == {"a": 1}

    # config_hash must ignore calibration_fpr_target and depend on params/name
    assert spec1.config_hash == spec2.config_hash
    assert spec1.config_hash != spec3.config_hash


def test_guardrail_spec_invalid_params_type_rejected():
    """
    params must be mapping-like; non-mapping types are rejected eagerly.
    """
    with pytest.raises(TypeError):
        GuardrailSpec(name="g", params=123)  # type: ignore[arg-type]


def test_guardrail_spec_fpr_bounds_validation():
    """
    calibration_fpr_target must live in [0, 1]; we allow the endpoints.
    """
    GuardrailSpec(name="g", params={}, calibration_fpr_target=0.0)
    GuardrailSpec(name="g", params={}, calibration_fpr_target=1.0)

    with pytest.raises(ValidationError):
        GuardrailSpec(name="g", params={}, calibration_fpr_target=-0.1)
    with pytest.raises(ValidationError):
        GuardrailSpec(name="g", params={}, calibration_fpr_target=1.1)


# ---------------------------------------------------------------------
# WorldConfig
# ---------------------------------------------------------------------


def test_world_config_guardrail_stack_normalization_variants():
    """
    WorldConfig.guardrail_stack should accept:
    - None (→ [])
    - single GuardrailSpec
    - single mapping (→ GuardrailSpec)
    - list of mixture
    """
    spec = GuardrailSpec(name="g", params={})

    wc_none = WorldConfig(world_id=WorldBit.BASELINE, guardrail_stack=None)  # type: ignore[arg-type]
    assert wc_none.guardrail_stack == []

    wc_single = WorldConfig(world_id=WorldBit.BASELINE, guardrail_stack=spec)  # type: ignore[arg-type]
    assert wc_single.guardrail_stack == [spec]

    wc_map = WorldConfig(world_id=WorldBit.BASELINE, guardrail_stack={"name": "g2", "params": {}})  # type: ignore[arg-type]
    assert isinstance(wc_map.guardrail_stack[0], GuardrailSpec)

    wc_list = WorldConfig(
        world_id=WorldBit.BASELINE,
        guardrail_stack=[spec, {"name": "g3", "params": {}}],
    )
    assert len(wc_list.guardrail_stack) == 2
    assert all(isinstance(x, GuardrailSpec) for x in wc_list.guardrail_stack)


def test_world_config_env_hash_changes_with_stack_or_world_and_equal_when_same():
    """
    env_hash must change when:
    - world_id changes
    - guardrail_stack changes
    and remain equal for identical environments.
    """
    spec = GuardrailSpec(name="g", params={})
    wc1 = WorldConfig(world_id=WorldBit.BASELINE, guardrail_stack=[spec])
    wc2 = WorldConfig(world_id=WorldBit.PROTECTED, guardrail_stack=[spec])
    wc3 = WorldConfig(world_id=WorldBit.BASELINE, guardrail_stack=[])
    wc4 = WorldConfig(world_id=WorldBit.BASELINE, guardrail_stack=[spec])

    assert wc1.env_hash != wc2.env_hash
    assert wc1.env_hash != wc3.env_hash
    # identical env → identical hash
    assert wc1.env_hash == wc4.env_hash


def test_world_config_baseline_success_rate_bounds():
    """
    baseline_success_rate is treated as a probability in [0, 1].
    """
    WorldConfig(
        world_id=WorldBit.BASELINE,
        guardrail_stack=[],
        baseline_success_rate=0.0,
    )
    WorldConfig(
        world_id=WorldBit.BASELINE,
        guardrail_stack=[],
        baseline_success_rate=1.0,
    )
    with pytest.raises(ValidationError):
        WorldConfig(
            world_id=WorldBit.BASELINE,
            guardrail_stack=[],
            baseline_success_rate=-0.1,
        )
    with pytest.raises(ValidationError):
        WorldConfig(
            world_id=WorldBit.BASELINE,
            guardrail_stack=[],
            baseline_success_rate=1.1,
        )


# ---------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------


def _minimal_guardrail_cfg() -> Dict[str, Sequence[GuardrailSpec]]:
    return {"cfg": [GuardrailSpec(name="g", params={})]}


def test_experiment_config_basic_and_iso_and_seed_normalization():
    """
    Core invariants:

    - guardrail_configs normalized to mapping[str, list[GuardrailSpec]]
    - attack_strategies preserved
    - iso_time derived from created_at
    - random_seed coerced to int even if provided as string
    """
    cfg = ExperimentConfig(
        experiment_id="exp1",
        n_sessions=10,
        attack_strategies=["a1", "a2"],
        guardrail_configs=_minimal_guardrail_cfg(),
        random_seed="123",
    )
    assert cfg.n_sessions == 10
    assert cfg.attack_strategies == ["a1", "a2"]
    assert list(cfg.guardrail_configs.keys()) == ["cfg"]
    assert isinstance(cfg.guardrail_configs["cfg"][0], GuardrailSpec)

    assert cfg.iso_time == _iso_from_unix(cfg.created_at)
    assert cfg.random_seed == 123  # coercion from string → int


def test_experiment_config_attack_strategies_normalization_and_errors():
    """
    attack_strategies:
    - string → single-element list
    - empty list is rejected
    """
    cfg = ExperimentConfig(
        experiment_id="exp2",
        n_sessions=5,
        attack_strategies="single",
        guardrail_configs=_minimal_guardrail_cfg(),
    )
    assert cfg.attack_strategies == ["single"]

    with pytest.raises(ValueError):
        ExperimentConfig(
            experiment_id="exp3",
            n_sessions=5,
            attack_strategies=[],  # type: ignore[arg-type]
            guardrail_configs=_minimal_guardrail_cfg(),
        )


def test_experiment_config_guardrail_configs_type_and_empty_errors():
    """
    guardrail_configs:
    - must be a non-empty mapping
    - None, non-mapping, or {} are rejected
    """
    with pytest.raises(ValueError):
        ExperimentConfig(
            experiment_id="exp",
            n_sessions=1,
            attack_strategies=["a"],
            guardrail_configs=None,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError):
        ExperimentConfig(
            experiment_id="exp",
            n_sessions=1,
            attack_strategies=["a"],
            guardrail_configs=["not", "mapping"],  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        ExperimentConfig(
            experiment_id="exp",
            n_sessions=1,
            attack_strategies=["a"],
            guardrail_configs={},
        )
