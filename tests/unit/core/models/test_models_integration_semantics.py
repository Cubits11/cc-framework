# tests/unit/models/test_models_integration_semantics.py

"""
Integration-style semantics across model types.

Scope:
- ExperimentConfig ↔ AttackResult temporal consistency
- GuardrailSpec / WorldConfig ↔ ExperimentConfig consistency on stacks
- AttackStrategy naming consistency with ExperimentConfig.attack_strategies
"""

import time
from typing import List

from cc.core.models import (
    AttackResult,
    AttackStrategy,
    ExperimentConfig,
    GuardrailSpec,
    WorldBit,
    WorldConfig,
)
from tests._factories import mk_attack_result


# ---------------------------------------------------------------------
# ExperimentConfig ↔ AttackResult temporal semantics
# ---------------------------------------------------------------------


def test_integration_experiment_config_to_attack_results_workflow():
    """
    END-TO-END: ExperimentConfig → JSON → ExperimentConfig →
                AttackResult.from_transcript

    Invariants:
    - Config JSON roundtrips.
    - AttackResult timestamps are >= config.created_at (created later).
    """
    # Two guardrail configs just to exercise the mapping
    baseline_stack: List[GuardrailSpec] = []
    protected_stack: List[GuardrailSpec] = [GuardrailSpec(name="kw", params={"k": "v"})]

    cfg = ExperimentConfig(
        experiment_id="exp-integration",
        n_sessions=5,
        attack_strategies=["basic"],
        guardrail_configs={
            "baseline": baseline_stack,
            "protected": protected_stack,
        },
    )

    json_str = cfg.model_dump_json()
    cfg2 = ExperimentConfig.model_validate_json(json_str)
    assert cfg2.model_dump() == cfg.model_dump()

    # Small sleep to make ordering robust even on coarse timers
    time.sleep(0.001)

    results: List[AttackResult] = []
    for i in range(cfg.n_sessions):
        world_bit = WorldBit.BASELINE if i == 0 else WorldBit.PROTECTED
        guardrails_applied = "baseline" if world_bit is WorldBit.BASELINE else "protected"

        ar = AttackResult.from_transcript(
            world_bit=world_bit,
            success=bool(i % 2),
            attack_id=f"{cfg.experiment_id}-{i}",
            transcript=f"prompt-{i}",
            guardrails_applied=guardrails_applied,
            rng_seed=i,
        )
        results.append(ar)

    # All results should have timestamps >= config.created_at
    for ar in results:
        assert ar.timestamp >= cfg.created_at
        assert ar.attack_id.startswith(cfg.experiment_id)


# ---------------------------------------------------------------------
# GuardrailSpec / WorldConfig ↔ ExperimentConfig consistency
# ---------------------------------------------------------------------


def test_integration_guardrail_configs_worldconfig_env_hash_consistency():
    """
    Given a guardrail stack used in ExperimentConfig, constructing a WorldConfig
    with the same stack and world_id should yield the same env_hash every time.
    """
    spec_a = GuardrailSpec(name="ga", params={"threshold": 0.1})
    spec_b = GuardrailSpec(name="gb", params={"threshold": 0.2})
    stack = [spec_a, spec_b]

    cfg = ExperimentConfig(
        experiment_id="exp-env",
        n_sessions=1,
        attack_strategies=["basic"],
        guardrail_configs={"protected": stack},
    )

    wc1 = WorldConfig(world_id=WorldBit.PROTECTED, guardrail_stack=stack)
    wc2 = WorldConfig(
        world_id=WorldBit.PROTECTED,
        guardrail_stack=list(cfg.guardrail_configs["protected"]),
    )

    assert wc1.env_hash == wc2.env_hash


# ---------------------------------------------------------------------
# AttackStrategy naming consistency with ExperimentConfig
# ---------------------------------------------------------------------


def test_integration_attack_strategy_and_experiment_config_consistency():
    """
    attack_strategies in ExperimentConfig should align with the AttackStrategy
    names used in the experiment orchestration layer.
    """
    strategies = [
        AttackStrategy(name="strat_a", params={"id": 0}),
        AttackStrategy(name="strat_b", params={"id": 1}),
    ]
    cfg = ExperimentConfig(
        experiment_id="exp-strat",
        n_sessions=3,
        attack_strategies=[s.name for s in strategies],
        guardrail_configs={"baseline": []},
    )

    assert set(cfg.attack_strategies) == {s.name for s in strategies}


def test_integration_attack_results_mark_world_bit_consistently():
    """
    Sanity: AttackResults produced for baseline vs protected worlds should
    have world_bit flags matching the intended world.
    """
    baseline_result = mk_attack_result(
        world_bit=WorldBit.BASELINE,
        success=True,
        attack_id="baseline-1",
        guardrails_applied="baseline",
        rng_seed=1,
    )
    protected_result = mk_attack_result(
        world_bit=WorldBit.PROTECTED,
        success=False,
        attack_id="protected-1",
        guardrails_applied="protected",
        rng_seed=2,
    )

    assert baseline_result.world_bit is WorldBit.BASELINE
    assert protected_result.world_bit is WorldBit.PROTECTED
