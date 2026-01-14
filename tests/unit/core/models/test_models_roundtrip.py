# tests/unit/core/models/test_models_roundtrip.py

"""
Roundtrip and JSON / schema robustness tests.

Scope:
- Basic model_dump_json / model_validate_json roundtrip for core models
- Adversarial JSON behaviour for AttackResult (extra fields, wrong types,
  missing required)
- Cross-version schema_version handling for AttackResult
"""

from typing import Any, Dict, Type

from pydantic import ValidationError
import pytest

from cc.core.models import (
    AttackResult,
    AttackStrategy,
    CCResult,
    ExperimentConfig,
    GuardrailSpec,
    ModelBase,
    WorldBit,
    WorldConfig,
)
from cc.core.schema import SCHEMA_VERSION as _SCHEMA_VERSION
from tests._factories import mk_attack_result


# ---------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------


def roundtrip_via_json(model_cls: Type[ModelBase], data: Dict[str, Any]) -> None:
    """
    Generic JSON roundtrip for ModelBase subclasses.

    Invariants:
    - The model can be constructed from the given data.
    - model_dump_json() produces JSON that can be parsed by model_validate_json().
    - The second instance has identical model_dump() payload to the first.
    """
    inst = model_cls(**data)
    json_str = inst.model_dump_json()
    inst2 = model_cls.model_validate_json(json_str)
    assert inst.model_dump() == inst2.model_dump()


# ---------------------------------------------------------------------
# Basic roundtrip for all major models
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_cls, data",
    [
        (
            AttackResult,
            mk_attack_result(
                world_bit=WorldBit.BASELINE,
                success=True,
                attack_id="id",
                guardrails_applied="ga",
                rng_seed=42,
            ).model_dump(exclude={"iso_time"}),
        ),
        (
            GuardrailSpec,
            {
                "name": "spec",
                "params": {"k": "v"},
            },
        ),
        (
            WorldConfig,
            {
                "world_id": WorldBit.BASELINE,
                "guardrail_stack": [],
            },
        ),
        (
            ExperimentConfig,
            {
                "experiment_id": "exp1",
                "n_sessions": 10,
                "attack_strategies": ["basic"],
                # Empty list for a label is explicitly allowed:
                # represents a "no-guardrail" configuration for that label.
                "guardrail_configs": {"cfg": []},
            },
        ),
        (
            CCResult,
            {
                "j_empirical": 0.4,
                "cc_max": 0.6,
                "delta_add": 0.2,
            },
        ),
        (
            AttackStrategy,
            {
                "name": "strat",
                "params": {"foo": "bar"},
            },
        ),
    ],
)
def test_roundtrip_all_models_via_json(
    model_cls: Type[ModelBase],
    data: Dict[str, Any],
) -> None:
    """
    Simple sanity: all major models should support JSON roundtrip without
    losing information.
    """
    roundtrip_via_json(model_cls, data)


# ---------------------------------------------------------------------
# Adversarial JSON for AttackResult
# ---------------------------------------------------------------------


def _valid_attack_result_fields() -> Dict[str, Any]:
    """
    Baseline valid field set for AttackResult.

    NOTE: We intentionally omit schema_version here because ModelBase
    provides a default; schema_version-specific tests below exercise
    explicit setting and roundtrip behaviour.
    """
    return mk_attack_result(
        world_bit=WorldBit.BASELINE,
        success=True,
        attack_id="id",
        guardrails_applied="ga",
        rng_seed=42,
    ).model_dump(exclude={"iso_time"})


def test_roundtrip_attack_result_extra_fields_ignored() -> None:
    """
    Extra JSON fields should be ignored, not stored, for backward compatibility.

    This relies on ModelBase.model_config.extra = "ignore".
    """
    json_with_extra = {
        **_valid_attack_result_fields(),
        "extra_field": "injection",  # should be dropped
    }
    ar = AttackResult.model_validate(json_with_extra)
    assert not hasattr(ar, "extra_field")


def test_roundtrip_attack_result_wrong_type_rejected() -> None:
    """
    Wrong types on core fields (e.g., success='true') should be rejected,
    not silently coerced.

    This documents that AttackResult.success is a *strict* boolean:
    only True/False are accepted; strings like "true" or integers are not.
    """
    wrong_type = {
        **_valid_attack_result_fields(),
        "success": "true",  # string, not bool
    }
    with pytest.raises(ValidationError):
        AttackResult.model_validate(wrong_type)


def test_roundtrip_attack_result_missing_required_field_rejected() -> None:
    """
    Missing required fields (e.g., attack_id) must raise ValidationError.
    """
    missing = {**_valid_attack_result_fields()}
    del missing["attack_id"]
    with pytest.raises(ValidationError, match="attack_id"):
        AttackResult.model_validate(missing)


# ---------------------------------------------------------------------
# Cross-version / schema_version semantics
# ---------------------------------------------------------------------


def test_roundtrip_attack_result_schema_version_preserved_from_json() -> None:
    """
    If JSON specifies a schema_version, it should be preserved rather than
    silently overwritten by the default.

    This guards against accidental schema_version clobbering when loading
    older payloads that are still considered compatible.
    """
    data = {
        **_valid_attack_result_fields(),
        "schema_version": _SCHEMA_VERSION,
    }
    ar = AttackResult.model_validate(data)
    assert ar.schema_version == _SCHEMA_VERSION


def test_roundtrip_attack_result_schema_version_roundtrip_through_json() -> None:
    """
    schema_version specified in the instance should roundtrip through JSON.
    """
    ar = mk_attack_result(
        world_bit=WorldBit.BASELINE,
        success=True,
        attack_id="id",
        guardrails_applied="ga",
        rng_seed=42,
    ).model_copy(update={"schema_version": _SCHEMA_VERSION})
    dumped = ar.model_dump_json()
    loaded = AttackResult.model_validate_json(dumped)
    assert loaded.schema_version == _SCHEMA_VERSION
    assert loaded.attack_id == "id"
