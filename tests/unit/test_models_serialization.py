import numpy as np

from cc.core.models import (
    AttackResult,
    GuardrailSpec,
    WorldConfig,
    ExperimentConfig,
    CCResult,
    AttackStrategy,
)


def test_attack_result_to_dict_includes_all_fields():
    result = AttackResult(
        world_bit=1,
        success=True,
        attack_id="a1",
        transcript_hash="h",
        guardrails_applied="g",
        rng_seed=0,
        timestamp=0.0,
    )
    data = result.to_dict()
    assert set(data.keys()) == set(result.__dataclass_fields__.keys())
    assert "utility_score" in data and data["utility_score"] is None


def test_guardrail_spec_to_dict_includes_all_fields():
    spec = GuardrailSpec(name="k", params={"p": 1})
    data = spec.to_dict()
    assert set(data.keys()) == set(spec.__dataclass_fields__.keys())


def test_world_config_to_dict_serializes_nested_spec():
    spec = GuardrailSpec(name="k", params={})
    wc = WorldConfig(world_id=0, guardrail_stack=[spec], utility_profile={"u": 1.0})
    data = wc.to_dict()
    assert set(data.keys()) == set(wc.__dataclass_fields__.keys())
    assert isinstance(data["guardrail_stack"][0], dict)


def test_experiment_config_to_dict_serializes_nested():
    spec = GuardrailSpec(name="k", params={})
    ec = ExperimentConfig(
        experiment_id="exp",
        n_sessions=1,
        attack_strategies=["s"],
        guardrail_configs={"a": [spec]},
    )
    data = ec.to_dict()
    assert set(data.keys()) == set(ec.__dataclass_fields__.keys())
    assert isinstance(data["guardrail_configs"]["a"][0], dict)


def test_ccresult_to_dict_converts_arrays_and_tuples():
    arr = np.array([1, 2, 3])
    cc = CCResult(
        j_empirical=0.1,
        cc_max=0.2,
        delta_add=0.01,
        cc_multiplicative=0.5,
        confidence_interval=(0.0, 0.1),
        bootstrap_samples=arr,
        n_sessions=10,
    )
    data = cc.to_dict()
    assert set(data.keys()) == set(cc.__dataclass_fields__.keys())
    assert data["bootstrap_samples"] == [1, 2, 3]
    assert data["confidence_interval"] == [0.0, 0.1]


def test_attack_strategy_to_dict_includes_all_fields():
    strat = AttackStrategy(
        name="s",
        params={"a": 1},
        vocabulary=["x", "y"],
        success_threshold=0.7,
    )
    data = strat.to_dict()
    assert set(data.keys()) == set(strat.__dataclass_fields__.keys())